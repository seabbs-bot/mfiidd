# Fits for the time-varying transmission session.
#
# One model, two samplers. SEIT4L is solved as an ODE, one day at a time, with
# `log β` following a Gaussian random walk over the 59 observation days. Given
# the whole transmission trajectory the epidemic is deterministic, so the walk
# is the only latent randomness, and it can be handled either way:
#
#   arm A: the walk lives in a particle filter's state, the filter integrates it
#          out, and PMMH samples the seven parameters
#   arm B: the 59 increments are parameters, non-centred, and NUTS samples all
#          66 unknowns directly
#
# Both arms call `tvbeta_incidence`/`seit4l_day_step` from the course package,
# so the likelihood is shared rather than written twice.
#
# The priors on the six SEIT4L parameters are the course's, and must match
# sessions/time_varying_transmission.qmd and scripts/pmmh_setup.jl.
#
# Run with `julia --project=. --threads=auto scripts/generate_tvbeta_comparison.jl`.
# Takes a couple of hours. Pass a stage name to run one part:
#   noise   the filter's log-likelihood noise against σ and the particle count
#   nuts    arm B
#   pmmh    arm A
#   paths   the smoothed transmission trajectory from arm A

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Distributions
using DataFrames
using CSV
using DrWatson
using Turing
using MCMCChains
using AdvancedMH
using ForwardDiff
using ADTypes

using MFIIDD

const INIT_STATE = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
const PARAMETERS = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ, :σ]

flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
const OBS = flu.obs
const N_OBS = length(OBS)

# ---------------------------------------------------------------------------
# arm B: the increments as parameters
# ---------------------------------------------------------------------------

@model function tvbeta_nuts(n_obs, init_state)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)
    σ ~ truncated(Normal(0.0, 0.2), lower = 0.0)
    z ~ filldist(Normal(0, 1), n_obs)

    logβ = log(R_0 / D_inf) .+ σ .* cumsum(z)
    inc = tvbeta_incidence(logβ, D_lat, D_inf, α, D_imm, init_state)
    λ = [max(ρ * inc[i], 1e-10) for i in 1:n_obs]
    obs ~ arraydist(Poisson.(λ))
    return nothing
end

# ---------------------------------------------------------------------------
# arm A: the walk in the filter's state
# ---------------------------------------------------------------------------

@model function tvbeta_pmmh(obs, n_particles, init_state, nchunks)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)
    σ ~ truncated(Normal(0.0, 0.2), lower = 0.0)

    ## the filter is not differentiable, so the Duals that Turing's
    ## initialisation probe pushes through have to be stripped
    θ = Dict(
        :R_0 => ForwardDiff.value(R_0),
        :D_lat => ForwardDiff.value(D_lat),
        :D_inf => ForwardDiff.value(D_inf),
        :α => ForwardDiff.value(α),
        :D_imm => ForwardDiff.value(D_imm),
        :ρ => ForwardDiff.value(ρ),
        :σ => ForwardDiff.value(σ),
    )

    Turing.@addlogprob! run_filter_tvbeta(
        θ,
        obs,
        n_particles;
        init_state = init_state,
        threaded = true,
        nchunks = nchunks,
    )
    return nothing
end

function chain_frame(chain)
    df = DataFrame(chain)
    return select(df, Not(intersect(["iteration", "iter", "chain"], names(df))))
end

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

"""
How noisy the filter's log-likelihood estimate is, over a grid of random-walk
scales and particle counts. PMMH needs the standard deviation near 1, so this
table is what says whether arm A is available at a given σ.
"""
function stage_noise(; n_reps = 12, nchunks = 4)
    base = Dict(
        :R_0 => 3.0,
        :D_lat => 2.0,
        :D_inf => 3.0,
        :α => 0.5,
        :D_imm => 15.0,
        :ρ => 0.7,
    )
    rows = DataFrame(
        σ = Float64[],
        n_particles = Int[],
        mean_ll = Float64[],
        sd_ll = Float64[],
        ms_per_run = Float64[],
    )
    for σ in (0.02, 0.05, 0.1, 0.15, 0.25, 0.35), n in (128, 512, 2048, 8192)
        θ = merge(base, Dict(:σ => σ))
        lls = Float64[]
        t = @elapsed for i in 1:n_reps
            Random.seed!(3000 + i)
            push!(
                lls,
                run_filter_tvbeta(
                    θ,
                    OBS,
                    n;
                    init_state = INIT_STATE,
                    threaded = true,
                    nchunks = nchunks,
                ),
            )
        end
        push!(rows, (σ, n, mean(lls), std(lls), t / n_reps * 1000))
        println(
            "σ $σ, N $n: sd ",
            round(std(lls), digits = 2),
            ", ",
            round(t / n_reps * 1000, digits = 0),
            " ms",
        )
        flush(stdout)
    end
    CSV.write(datadir("tvbeta_filter_noise.csv"), rows)
    return rows
end

"""
Arm B. Four chains, so the comparison with arm A has an R-hat behind it.
"""
function stage_nuts(;
    n_warmup = parse(Int, get(ENV, "TVBETA_WARMUP", "400")),
    n_samples = parse(Int, get(ENV, "TVBETA_SAMPLES", "400")),
    n_chains = parse(Int, get(ENV, "TVBETA_CHAINS", "4")),
)
    model = tvbeta_nuts(N_OBS, INIT_STATE) | (; obs = OBS)
    Random.seed!(1234)
    t = @elapsed chain = sample(
        model,
        NUTS(n_warmup, 0.8; adtype = AutoForwardDiff()),
        MCMCThreads(),
        n_samples,
        n_chains;
        progress = false,
    )
    println(
        "NUTS: $(n_chains) x $(n_samples) draws in ",
        round(t / 60, digits = 1),
        " minutes",
    )
    show(stdout, MIME("text/plain"), summarystats(chain[PARAMETERS]))
    println()

    df = chain_frame(chain)
    ## rebuild the transmission trajectory each draw implies, so the two arms
    ## can be compared on β rather than on the increments
    logβ = mapreduce(vcat, eachrow(df)) do r
        z = [r[Symbol("z[$i]")] for i in 1:N_OBS]
        reshape(log(r.R_0 / r.D_inf) .+ r.σ .* cumsum(z), 1, :)
    end
    out = hcat(select(df, PARAMETERS), DataFrame(logβ, ["logbeta[$i]" for i in 1:N_OBS]))
    CSV.write(datadir("tvbeta_nuts_chain.csv"), out)

    diag = DataFrame(summarystats(chain[PARAMETERS]))
    CSV.write(datadir("tvbeta_nuts_diagnostics.csv"), insertcols!(diag, :seconds => t))
    return chain
end

"""
Arm A. RAM adapts in the warmup iterations and those draws are discarded, as in
`scripts/pmmh_setup.jl`.
"""
function stage_pmmh(;
    n_particles = 2048,
    n_warmup = 4_000,
    n_samples = 40_000,
    thinning = 20,
    nchunks = Threads.nthreads(),
)
    model = tvbeta_pmmh(OBS, n_particles, INIT_STATE, nchunks)
    Random.seed!(1234)
    t = @elapsed chain_full = sample(
        model,
        externalsampler(AdvancedMH.RobustAdaptiveMetropolis()),
        n_samples;
        num_warmup = n_warmup,
        check_model = false,
        progress = false,
    )
    println(
        "PMMH: $(n_samples) iterations at $(n_particles) particles in ",
        round(t / 60, digits = 1),
        " minutes",
    )

    chain = chain_full[1:thinning:end]
    x = chain_frame(chain_full)[!, :R_0]
    println("acceptance ", round(mean(x[2:end] .!= x[1:(end - 1)]) * 100, digits = 1), "%")
    show(stdout, MIME("text/plain"), summarystats(chain[PARAMETERS]))
    println()

    CSV.write(datadir("tvbeta_pmmh_chain.csv"), select(chain_frame(chain), PARAMETERS))
    diag = DataFrame(summarystats(chain[PARAMETERS]))
    CSV.write(
        datadir("tvbeta_pmmh_diagnostics.csv"),
        insertcols!(diag, :seconds => t, :n_particles => n_particles),
    )
    return chain
end

"""
Smoothed transmission trajectories from arm A, one per posterior draw, so the
two arms can be compared on β day by day.
"""
function stage_paths(; n_paths = 200, n_particles = 2048)
    df = CSV.read(datadir("tvbeta_pmmh_chain.csv"), DataFrame)
    Random.seed!(99)
    paths = map(1:n_paths) do _
        r = df[rand(1:nrow(df)), :]
        θ = Dict(
            :R_0 => r.R_0,
            :D_lat => r.D_lat,
            :D_inf => r.D_inf,
            :α => r.α,
            :D_imm => r.D_imm,
            :ρ => r.ρ,
            :σ => r.σ,
        )
        first(filtered_tvbeta(θ, OBS, n_particles; init_state = INIT_STATE))
    end
    out = DataFrame(reduce(hcat, paths)', ["logbeta[$i]" for i in 1:N_OBS])
    CSV.write(datadir("tvbeta_pmmh_paths.csv"), out)
    return out
end

stages = isempty(ARGS) ? ["noise", "nuts", "pmmh", "paths"] : ARGS
for s in stages
    println("="^60, "\n", s, "\n", "="^60)
    flush(stdout)
    s == "noise" && stage_noise()
    s == "nuts" && stage_nuts()
    s == "pmmh" && stage_pmmh()
    s == "paths" && stage_paths()
end
