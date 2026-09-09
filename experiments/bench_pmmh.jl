## Baseline: PMMH with a threaded bootstrap particle filter, exactly the sampler
## `scripts/pmmh_setup.jl` uses to generate the committed SEIT4L chain. Run with
##
##     julia --project=. --threads=12 experiments/bench_pmmh.jl [n_warmup] [n_samples]

include(joinpath(@__DIR__, "common.jl"))

using Turing
using AdvancedMH
using ForwardDiff
using JLD2

const N_PARTICLES = 128

n_warmup = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 20_000
n_samples = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 100_000

@model function pmmh(obs, n_particles)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)

    θ = Dict(
        :R_0 => ForwardDiff.value(R_0),
        :D_lat => ForwardDiff.value(D_lat),
        :D_inf => ForwardDiff.value(D_inf),
        :α => ForwardDiff.value(α),
        :D_imm => ForwardDiff.value(D_imm),
        :ρ => ForwardDiff.value(ρ),
    )

    Turing.@addlogprob! run_particle_filter(
        θ,
        obs,
        n_particles;
        init_state = INIT_STATE,
        threaded = true,
    )
end

obs = flu_observations()

## one short run first, so the timed run measures sampling rather than compilation
sample(
    pmmh(obs, N_PARTICLES),
    externalsampler(AdvancedMH.RobustAdaptiveMetropolis()),
    20;
    num_warmup = 10,
    check_model = false,
    progress = false,
)

println("threads: $(Threads.nthreads())")
println("particles: $N_PARTICLES, warmup: $n_warmup, samples: $n_samples")

Random.seed!(20260909)
t0 = time()
chain = sample(
    pmmh(obs, N_PARTICLES),
    externalsampler(AdvancedMH.RobustAdaptiveMetropolis()),
    n_samples;
    num_warmup = n_warmup,
    check_model = false,
    progress = false,
)
elapsed = time() - t0

df = DataFrame(chain)
draws = Matrix(df[:, PARAMETERS])

accept = mean(draws[2:end, 1] .!= draws[1:(end - 1), 1])
println("acceptance rate: $(round(accept * 100, digits = 1))%")

ess_report(draws, "PMMH (bootstrap filter, $N_PARTICLES particles, RAM)", elapsed)
posterior_table(draws, "PMMH")
posterior_table(committed_chain(), "committed chain")

jldsave(joinpath(@__DIR__, "pmmh_draws.jld2"); draws, elapsed, accept)
