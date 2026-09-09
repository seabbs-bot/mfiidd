## Particle Gibbs against the PMMH baseline.
##
##     julia --project=. --threads=12 experiments/bench_pg.jl [n_particles] [n_iter]
##
## The particle count is swept because it trades differently in particle Gibbs
## than in PMMH. PMMH needs enough particles to keep the variance of the
## log-likelihood estimate near one, or the chain sticks. Particle Gibbs never
## uses that estimate, so its particle count only has to buy a path move.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))
include(joinpath(@__DIR__, "pg.jl"))

using Printf
using JLD2

obs = flu_observations()

particle_counts = if length(ARGS) ≥ 1
    [parse(Int, a) for a in split(ARGS[1], ",")]
else
    [32, 64, 128, 256]
end
n_iter = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 20_000
n_theta = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 200

println("threads: $(Threads.nthreads())")
println("iterations: $n_iter, parameter steps per sweep: $n_theta")

## a short run first, so the timed runs measure sampling rather than compilation
particle_gibbs(obs; n_particles = 32, n_iter = 20, n_warmup = 10, n_theta_steps = 10)

results = Dict{Int, Any}()
for N in particle_counts
    r = particle_gibbs(
        obs;
        n_particles = N,
        n_iter = n_iter,
        n_warmup = max(500, n_iter ÷ 10),
        n_theta_steps = n_theta,
    )
    results[N] = r

    ess_report(r.draws, "Particle Gibbs, $N particles", r.elapsed)
    @printf("  parameter-step acceptance: %.1f%%\n", 100 * r.theta_accept)
    @printf(
        "  path departs from the reference at day: median %d, mean %.1f\n",
        median(r.coalescence),
        mean(r.coalescence)
    )
    @printf(
        "  sweeps that changed nothing: %.1f%%\n",
        100 * mean(r.coalescence .== length(obs) + 1)
    )
    posterior_table(r.draws, "Particle Gibbs, $N particles")
end

posterior_table(committed_chain(), "committed PMMH chain")

jldsave(
    joinpath(@__DIR__, "pg_draws.jld2");
    draws = Dict(N => results[N].draws for N in particle_counts),
    elapsed = Dict(N => results[N].elapsed for N in particle_counts),
    coalescence = Dict(N => results[N].coalescence for N in particle_counts),
)
