## Does the parameter step limit particle Gibbs here?
##
## The composite sampler worth testing is one whose parameter update uses
## gradients, since the complete-data likelihood is differentiable in θ. That is
## only worth building if the parameter update is what holds the sampler back.
## This sweeps the number of random-walk steps taken per sweep, from one to a
## thousand. A thousand steps on a smooth six-parameter target with an adapted
## covariance is an all but exact draw from θ given the path, and it still costs
## a small fraction of the sweep that produced the path. If the effective sample
## size stops improving well before then, the parameter step is already exact
## enough and no gradient method can recover anything from it.
##
##     julia --project=. --threads=12 experiments/bench_theta_step.jl [n_iter]

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))
include(joinpath(@__DIR__, "pg.jl"))

using Printf
using MCMCChains: Chains, ess

obs = flu_observations()
n_iter = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10_000

println("threads: $(Threads.nthreads())")
println("particles: 128, iterations: $n_iter")

particle_gibbs(obs; n_particles = 32, n_iter = 20, n_warmup = 10, n_theta_steps = 10)

println(
    "\n",
    rpad("θ steps", 10),
    rpad("seconds", 12),
    rpad("worst ESS", 12),
    rpad("ESS/s", 12),
    "accept",
)
for n_theta in (1, 10, 50, 200, 1000)
    r = particle_gibbs(
        obs;
        n_particles = 128,
        n_iter = n_iter,
        n_warmup = max(500, n_iter ÷ 10),
        n_theta_steps = n_theta,
    )
    chn = Chains(reshape(r.draws, size(r.draws, 1), 6, 1), PARAMETERS)
    e = DataFrame(ess(chn))
    worst = minimum(e.ess)
    @printf(
        "%-10d%-12.1f%-12.1f%-12.4f%.1f%%\n",
        n_theta,
        r.elapsed,
        worst,
        worst / r.elapsed,
        100 * r.theta_accept
    )
end
