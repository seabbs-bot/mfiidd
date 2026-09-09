## Which part of the parameter step carries the difference?
##
## The plain-loop PMMH mixes about three times better than the course's
## Turing + `RobustAdaptiveMetropolis`, at the same particle count and the same
## filter. Three things differ, and this varies them one at a time so that a
## variant differs from another in exactly one respect:
##
##   1. the shape of the proposal covariance, full against diagonal
##   2. how it is adapted, an empirical covariance with a separate
##      Robbins-Monro scale against RAM's rank-one update
##   3. the scale it is proposed on, log and logit against the constrained
##      parameters themselves
##
##     julia --project=. --threads=12 experiments/ablation.jl

include(joinpath(@__DIR__, "pmmh_variants.jl"))

## -------------------------------------------------------------------- the runs

obs = flu_observations()
say("threads: $(Threads.nthreads())")
say("$N_PARTICLES particles, $N_WARMUP warmup, $N_KEPT kept, one seed")

## warm the code paths
run_variant(obs; n_warmup = 5, n_iter = 5)
run_variant(obs; shape = :ram, n_warmup = 5, n_iter = 5)

variants = [
    ("full covariance, RM scale, log/logit", :full, :transformed, false),
    ("diagonal covariance, RM scale, log/logit", :diagonal, :transformed, false),
    ("full covariance, RM scale, constrained scale", :full, :constrained, false),
    ("RAM, log/logit, adapting throughout", :ram, :transformed, false),
    ("RAM, log/logit, frozen after warmup", :ram, :transformed, true),
    ("RAM, constrained scale, adapting throughout", :ram, :constrained, false),
]

results = []
say(
    "\n",
    rpad("variant", 46),
    rpad("s/iter", 10),
    rpad("worst", 8),
    rpad("ESS", 9),
    rpad("iters/eff", 11),
    "accept",
)
for (label, shape, transform, freeze) in variants
    r = run_variant(obs; shape, transform, freeze_after_warmup = freeze)
    chn = Chains(reshape(r.draws, N_KEPT, 6, 1), PARAMETERS)
    e = DataFrame(ess(chn))
    i = argmin(e.ess)
    push!(results, (label = label, draws = r.draws, ess = e[i, :ess]))
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

jldsave(
    joinpath(@__DIR__, "ablation_draws.jld2");
    draws = Dict(r.label => r.draws for r in results),
)
