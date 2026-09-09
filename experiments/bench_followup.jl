## Two follow-up questions the main comparison raises.
##
## 1. Would a gradient-based parameter step help particle Gibbs? The number of
##    random-walk steps taken per sweep is swept from one to a thousand. A
##    thousand steps on a smooth six-parameter target with an adapted covariance
##    is an all but exact draw from θ given the path, and it still costs a small
##    fraction of the sweep. If the effective sample size stops improving well
##    before then, the parameter step is not what holds the sampler back and no
##    gradient method can recover anything from it.
##
## 2. Why does this PMMH mix several times better than the one the course runs,
##    at the same particle count and the same filter? The candidate explanation
##    is that it keeps adapting, where `RobustAdaptiveMetropolis` under Turing
##    only adapts during the warmup it is given. Freezing the adaptation at the
##    end of warmup tests that directly. The runs are also repeated under a
##    second seed, because a recommendation should not rest on one chain.
##
##     julia --project=. --threads=12 experiments/bench_followup.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))
include(joinpath(@__DIR__, "pg.jl"))
include(joinpath(@__DIR__, "pmmh_plain.jl"))

using Printf
using MCMCChains: Chains, ess

say(args...) = (println(args...); flush(stdout))

obs = flu_observations()
say("threads: $(Threads.nthreads())")

particle_gibbs(obs; n_particles = 32, n_iter = 20, n_warmup = 10, n_theta_steps = 10)
pmmh_plain(obs; n_particles = 32, n_iter = 20, n_warmup = 10)

function worst_ess(draws)
    chn = Chains(reshape(draws, size(draws, 1), 6, 1), PARAMETERS)
    e = DataFrame(ess(chn))
    i = argmin(e.ess)
    return e[i, :ess], string(e[i, :parameters])
end

## ------------------------------ 1. does the parameter step limit the sampler
say("\nparticle Gibbs, 128 particles, 8000 iterations, varying parameter steps")
say(rpad("θ steps", 10), rpad("s/iter", 10), rpad("worst", 8), rpad("ESS", 10),
    rpad("ESS/s", 10), "accept")
for n_theta in (1, 10, 200, 1000)
    r = particle_gibbs(
        obs;
        n_particles = 128,
        n_iter = 8_000,
        n_warmup = 1_000,
        n_theta_steps = n_theta,
    )
    e, p = worst_ess(r.draws)
    @printf(
        "%-10d%-10.5f%-8s%-10.1f%-10.4f%.1f%%\n",
        n_theta,
        r.elapsed / 8_000,
        p,
        e,
        e / r.elapsed,
        100 * r.theta_accept
    )
    flush(stdout)
end

## ------------------------------------- 2. what makes the PMMH parameter step work
say("\nPMMH variants, 40000 iterations")
say(rpad("variant", 46), rpad("s/iter", 10), rpad("worst", 8), rpad("ESS", 10),
    rpad("ESS/s", 10), "min to 400")

for (label, N, seed, freeze) in (
    ("128 particles, keeps adapting, seed A", 128, 20260909, false),
    ("128 particles, adaptation frozen after warmup", 128, 20260909, true),
    ("128 particles, keeps adapting, seed B", 128, 771, false),
    ("64 particles, keeps adapting, seed B", 64, 771, false),
    ("96 particles, keeps adapting, seed A", 96, 20260909, false),
)
    r = pmmh_plain(
        obs;
        n_particles = N,
        n_iter = 40_000,
        n_warmup = 15_000,
        seed = seed,
        freeze_after_warmup = freeze,
    )
    e, p = worst_ess(r.draws)
    @printf(
        "%-46s%-10.5f%-8s%-10.1f%-10.4f%.1f\n",
        label,
        r.elapsed / 40_000,
        p,
        e,
        e / r.elapsed,
        400 / (e / r.elapsed) / 60
    )
    posterior_table(r.draws, label)
    flush(stdout)
end

posterior_table(committed_chain(), "committed PMMH chain")
