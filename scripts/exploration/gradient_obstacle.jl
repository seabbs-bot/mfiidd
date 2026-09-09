## Arm B: the 59 increments as outer parameters. Establish the gradient obstacle.
using MFIIDD, CSV, DataFrames, DrWatson, Random, Statistics
using Turing, Distributions, SSMProblems, GeneralisedFilters, ForwardDiff

flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
init8 = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
θp = Dict(:D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7)

## 1. the gradient of the value-stripped filter likelihood
f(lb) = run_filter_path(θp, lb, flu.obs, 128)
lb0 = fill(log(3.0), 59)
Random.seed!(1)
g = ForwardDiff.gradient(f, lb0)
println("=== stripped: gradient wrt the 59 log β ===")
println("  f  = ", round(f(lb0), digits = 3))
println("  max |grad| = ", maximum(abs, g))

## 2. what happens without the strip: build the SSM straight from the Duals
function f_nostrip(lb)
    ssm = StateSpaceModel(
        SEIT4LInitial(init8),
        SEIT4LPathDynamics(Dict{Symbol, Float64}(θp), lb),
        PoissonObservation(θp[:ρ]),
    )
    _, ll = GeneralisedFilters.filter(Random.default_rng(), ssm, BF(128), flu.obs)
    return ll
end
println("\n=== unstripped: same gradient, Duals allowed through ===")
try
    ForwardDiff.gradient(f_nostrip, lb0)
    println("  no error (unexpected)")
catch e
    println("  ", typeof(e))
    msg = sprint(showerror, e)
    println("  ", first(split(msg, "\n"), 4))
end

## 3. a finite difference, to show the surface is not merely flat
println("\n=== finite differences on one coordinate ===")
for h in (1e-6, 1e-3, 1e-1)
    lp, lm = copy(lb0), copy(lb0)
    lp[30] += h
    lm[30] -= h
    Random.seed!(7)
    fp = mean(run_filter_path(θp, lp, flu.obs, 128) for _ in 1:20)
    Random.seed!(7)
    fm = mean(run_filter_path(θp, lm, flu.obs, 128) for _ in 1:20)
    println("  h=", h, "  (f(+h)-f(-h))/2h = ", round((fp - fm) / (2h), digits = 3))
end
