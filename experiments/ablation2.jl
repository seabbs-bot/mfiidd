## Follow-up to the ablation: is RAM's deficit its initial scale or its
## adaptation, and does the best variant hold under other seeds?
##
## The first ablation left two loose ends. RAM on the constrained scale
## collapsed to 0.1% acceptance, which is what a proposal of unit standard
## deviation does to parameters that live in the unit interval, so that run says
## nothing about the adaptation. And the best variant rested on one chain.
##
## The initial factor matters for the recommendation. If handing RAM a sensible
## starting scale closes the gap, the fix is an argument. If it does not, the
## rank-one adaptation itself is what costs the course its mixing, and the fix
## is a different proposal.
##
##     julia --project=. --threads=12 experiments/ablation2.jl

include(joinpath(@__DIR__, "pmmh_variants.jl"))

## Roughly the posterior standard deviations, halved. This is deliberately
## generous information to give RAM: if it cannot make use of a starting scale
## this good, no realistic starting scale will help it.
const SD_CONSTRAINED = [1.17, 0.41, 0.81, 0.056, 1.80, 0.048] ./ 2
const SD_TRANSFORMED = [0.22, 0.30, 0.50, 0.23, 0.17, 0.22] ./ 2

runs = [
    ("full cov, RM scale, constrained, seed B", :full, :constrained, nothing, 771),
    ("full cov, RM scale, constrained, seed C", :full, :constrained, nothing, 4242),
    ("full cov, RM scale, log/logit, seed B", :full, :transformed, nothing, 771),
    (
        "RAM given a good initial scale, constrained",
        :ram,
        :constrained,
        SD_CONSTRAINED,
        20260909,
    ),
    (
        "RAM given a good initial scale, log/logit",
        :ram,
        :transformed,
        SD_TRANSFORMED,
        20260909,
    ),
]

say("threads: $(Threads.nthreads())")
say("$N_PARTICLES particles, $N_WARMUP warmup, $N_KEPT kept")

obs = flu_observations()

## warm the code paths so the timed runs measure sampling
run_variant(obs; n_warmup = 5, n_iter = 5)
run_variant(obs; shape = :ram, n_warmup = 5, n_iter = 5)

say(
    "\n",
    rpad("variant", 46),
    rpad("s/iter", 10),
    rpad("worst", 8),
    rpad("ESS", 9),
    rpad("iters/eff", 11),
    "accept",
)
for (label, shape, transform, S0, seed) in runs
    r = run_variant(obs; shape, transform, initial_S = S0, seed)
    chn = Chains(reshape(r.draws, N_KEPT, 6, 1), PARAMETERS)
    e = DataFrame(ess(chn))
    i = argmin(e.ess)
    @printf(
        "%-46s%-10.5f%-8s%-9.1f%-11.1f%.1f%%\n",
        label,
        r.elapsed / N_KEPT,
        string(e[i, :parameters]),
        e[i, :ess],
        N_KEPT / e[i, :ess],
        100 * r.accept
    )
    posterior_table(r.draws, label)
    flush(stdout)
end

posterior_table(committed_chain(), "committed PMMH chain")
