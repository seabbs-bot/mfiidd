# Chains for the time-varying transmission session.
#
# Two fits of the same generative model, differing only in where the
# transmission trajectory lives:
#
#   arm A  log β is a component of the latent state, so the filter integrates
#          the trajectory out and the outer sampler sees seven parameters
#   arm B  the 59 increments are outer parameters, so a filter run is a
#          likelihood conditional on a trajectory and the outer sampler sees 66
#
# The six SEIT4L priors are the ones in CONTRIBUTING.md and in
# scripts/pmmh_setup.jl, unchanged. σ is the only addition. The random walk
# starts at log(R_0 / D_inf), so σ = 0 recovers the constant-rate model the
# particle MCMC session fits, and R_0 keeps its meaning as the reproduction
# number on day zero.
#
# Arm A is run as two chains from different seeds, because the session's claim
# about σ rests on where its posterior sits and one chain cannot support an
# R-hat. Arm B is run once and at a fifth of the length: it is not a fit anyone
# would use, and it is here for its effective sample size, which a longer run
# would not improve enough to be worth the hours.
#
# Run with `julia --project=. --threads=auto scripts/generate_tvbeta_chains.jl`.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Distributions
using DataFrames
using Turing
using MCMCChains
using CSV
using DrWatson
using AdvancedMH
using ForwardDiff
using SSMProblems
using GeneralisedFilters

using MFIIDD

const N_PARTICLES = 128
const PARAMETERS = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ, :σ]

flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)

"""
Arm A: `log β` in the latent state. Seven parameters for the outer sampler.
"""
@model function pmmh_tvbeta_state(obs, n_particles)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)
    σ ~ truncated(Normal(0.0, 0.2), lower = 0.0)

    v = ForwardDiff.value
    θ = Dict(
        :D_lat => v(D_lat),
        :D_inf => v(D_inf),
        :α => v(α),
        :D_imm => v(D_imm),
        :ρ => v(ρ),
        :σ => v(σ),
        :logβ0 => log(v(R_0) / v(D_inf)),
    )
    Turing.@addlogprob! run_filter_rw(θ, obs, n_particles)
end

"""
Arm B: the 59 increments as outer parameters, moved by the same random-walk
proposal as everything else. 66 dimensions with no gradient available.
"""
@model function pmmh_tvbeta_outer(obs, n_particles)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)
    σ ~ truncated(Normal(0.0, 0.2), lower = 0.0)
    ε ~ filldist(Normal(0, 1), length(obs))

    v = ForwardDiff.value
    logβ = log(v(R_0) / v(D_inf)) .+ v(σ) .* cumsum(v.(ε))
    θ = Dict(
        :D_lat => v(D_lat),
        :D_inf => v(D_inf),
        :α => v(α),
        :D_imm => v(D_imm),
        :ρ => v(ρ),
    )
    Turing.@addlogprob! run_filter_path(θ, logβ, obs, n_particles)
end

"""
    run_chain(model, seed; n_warmup, n_samples, thinning)

Sample `model` with Robust Adaptive Metropolis and return the thinned draws,
the wall clock, and the acceptance rate. RAM adapts in the warmup iterations
only, so `n_warmup` is the whole adaptation budget.
"""
function run_chain(model, seed; n_warmup, n_samples, thinning)
    Random.seed!(seed)
    t = @elapsed chain = sample(
        model,
        externalsampler(AdvancedMH.RobustAdaptiveMetropolis()),
        n_samples;
        num_warmup = n_warmup,
        check_model = false,
        progress = true,
    )
    df = DataFrame(chain)
    df = select(df, Not(intersect(["iteration", "iter", "chain"], names(df))))
    acc = mean(df.R_0[2:end] .!= df.R_0[1:(end - 1)])
    return df[1:thinning:end, :], t, acc
end

"""
    trim(df)

Keep the seven parameters, and summarise the 59 increments rather than storing
them. Arm B carries one column per increment, which is around four megabytes of
CSV for a chain nobody reads the increments of. What is worth keeping about them
is their effective sample size, so that goes in as two constant columns and the
columns themselves are dropped.
"""
function trim(df)
    cols = names(df)
    inc = filter(n -> startswith(n, "ε"), cols)
    out = select(df, string.(PARAMETERS))
    if !isempty(inc)
        e = ess(Chains(Matrix(df[:, inc]), Symbol.(inc)))[:, :ess]
        out.increment_min_ess .= minimum(e)
        out.increment_median_ess .= median(e)
    end
    return out
end

"""
    run_and_save(model, name, path, seeds; ...)

Run one chain per seed, stack them with a `chain` column, and save. The wall
clock is saved with the draws because the session reports effective sample size
per second, which cannot be recovered from the draws alone.
"""
function run_and_save(model, name, path, seeds; n_warmup, n_samples, thinning)
    println("="^60)
    println("$name: $(length(seeds)) chain(s), warmup $n_warmup, kept $n_samples")
    parts, total = DataFrame[], 0.0
    for (i, seed) in enumerate(seeds)
        df, t, acc = run_chain(model, seed; n_warmup, n_samples, thinning)
        kept = trim(df)
        kept.chain .= i
        push!(parts, kept)
        total += t
        println(
            "  chain $i: $(round(t / 60, digits = 1)) min, " *
            "acceptance $(round(100 * acc, digits = 2))%",
        )
    end
    out = vcat(parts...)
    out.wall_seconds .= total
    CSV.write(path, out)
    println("  wrote $(nrow(out)) draws to $path")
    return out
end

run_and_save(
    pmmh_tvbeta_state(flu.obs, N_PARTICLES),
    "arm A, log β in the state",
    datadir("pmmh_tvbeta_state.csv"),
    [20260909, 20260910];
    n_warmup = 20_000,
    n_samples = 60_000,
    thinning = 20,
)

run_and_save(
    pmmh_tvbeta_outer(flu.obs, N_PARTICLES),
    "arm B, the increments in the outer sampler",
    datadir("pmmh_tvbeta_outer.csv"),
    [20260909];
    n_warmup = 20_000,
    n_samples = 60_000,
    thinning = 20,
)
