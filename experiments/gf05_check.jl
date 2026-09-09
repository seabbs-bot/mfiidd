## What GeneralisedFilters 0.5 gives over a hand-rolled `ref_state` loop.
##
## 0.5 is unreleased at the time of writing: it exists on the SSMProblems.jl
## main branch and is pinned here by commit, so nothing that depends on it can
## reach the course until 0.5 is released. This is a scouting run, not a route
## to shipping.
##
## Four questions:
##
##   1. Does the course's own filter code still run? `ThreadedBF` overrides
##      `predict` and forwards four other methods, which is the kind of thing a
##      minor version breaks. This is the cost of an eventual upgrade.
##   2. Does `ConditionalSMC` reproduce the hand-rolled conditional SMC sweep,
##      and is it faster or slower?
##   3. Does `AncestorSampling()` work against these dynamics? It should not.
##      Ancestor sampling needs the density of the reference's state at time t
##      given a candidate ancestor at t-1, and a Gillespie simulator can sample
##      that transition without being able to evaluate it. Confirmed here rather
##      than assumed.
##   4. Same for `BackwardSimulation()`.
##
##     julia --project=. --threads=12 experiments/gf05_check.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))

using Printf
using GeneralisedFilters
using SSMProblems: StateSpaceModel
using AbstractMCMC

say(args...) = (println(args...); flush(stdout))

obs = flu_observations()
θ = Dict{Symbol, Float64}(
    :R_0 => 6.36,
    :D_lat => 1.33,
    :D_inf => 2.14,
    :α => 0.47,
    :D_imm => 11.77,
    :ρ => 0.69,
)

say("GeneralisedFilters version: ", pkgversion(GeneralisedFilters))
say("threads: $(Threads.nthreads())")

## ------------------------------------- 1. does the course's filter still run?
say("\n1. the course's own filter code under 0.5")
try
    Random.seed!(1)
    lls = [
        run_particle_filter(θ, obs, 128; init_state = INIT_STATE, threaded = false) for
        _ in 1:30
    ]
    @printf("  plain bootstrap filter: mean %.2f, sd %.2f  OK\n", mean(lls), std(lls))
catch e
    say("  plain bootstrap filter FAILED: ", sprint(showerror, e))
end

try
    Random.seed!(1)
    lls = [
        run_particle_filter(θ, obs, 128; init_state = INIT_STATE, threaded = true) for
        _ in 1:30
    ]
    @printf(
        "  ThreadedBF (overrides predict): mean %.2f, sd %.2f  OK\n",
        mean(lls),
        std(lls)
    )
catch e
    say("  ThreadedBF FAILED: ", sprint(showerror, e))
end

## --------------------------------------------------- 2. packaged conditional SMC
build_model(θ) = StateSpaceModel(
    SEIT4LStatInitial(INIT_STATE),
    SEIT4LStatDynamics(θ),
    PoissonStatObservation(θ[:ρ]),
)

say("\n2. ConditionalSMC with no refreshment")
csmc_model = GeneralisedFilters.CSMCModel(build_model(θ), obs)

state = nothing
try
    Random.seed!(2)
    algo = GeneralisedFilters.ConditionalSMC(GeneralisedFilters.BF(128))
    _, state = AbstractMCMC.step(Random.default_rng(), csmc_model, algo)

    ## does the path move, and where does it stop agreeing with the reference?
    first_diffs = Int[]
    for _ in 1:100
        prev = state.trajectory
        _, state = AbstractMCMC.step(Random.default_rng(), csmc_model, algo, state)
        fd = length(obs) + 1
        for t in 1:length(obs)
            if state.trajectory[t] != prev[t]
                fd = t
                break
            end
        end
        push!(first_diffs, fd)
    end
    @printf(
        "  OK. path departs from the reference at day: median %d, mean %.1f; %.0f%% of sweeps changed nothing\n",
        median(first_diffs),
        mean(first_diffs),
        100 * mean(first_diffs .== length(obs) + 1)
    )

    ## cost per sweep, against the hand-rolled loop
    t0 = time()
    for _ in 1:100
        _, state = AbstractMCMC.step(Random.default_rng(), csmc_model, algo, state)
    end
    @printf("  seconds per sweep, packaged: %.5f\n", (time() - t0) / 100)
catch e
    say("  ConditionalSMC FAILED: ", sprint(showerror, e))
end

## ------------------------------------- 3 and 4. the refreshment strategies
for (name, refresh) in (
    ("AncestorSampling", GeneralisedFilters.AncestorSampling()),
    ("BackwardSimulation", GeneralisedFilters.BackwardSimulation()),
)
    say("\n$name against Gillespie dynamics")
    try
        Random.seed!(3)
        algo = GeneralisedFilters.ConditionalSMC(GeneralisedFilters.BF(128), refresh)
        _, st = AbstractMCMC.step(Random.default_rng(), csmc_model, algo)
        _, st = AbstractMCMC.step(Random.default_rng(), csmc_model, algo, st)
        say(
            "  it ran. That is unexpected; the sampler needs a transition density ",
            "these dynamics cannot evaluate, so check what it actually called.",
        )
    catch e
        msg = sprint(showerror, e)
        say("  failed as expected, with:")
        say("    ", first(split(msg, "\n"), 4) |> x -> join(x, "\n    "))
    end
end
