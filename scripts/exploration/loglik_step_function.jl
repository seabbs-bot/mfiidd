## Is the filter log-likelihood a step function of one log β, at a fixed seed?
using MFIIDD, CSV, DataFrames, DrWatson, Random, Statistics
flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
θp = Dict(:D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7)
lb0 = fill(log(3.0), 59)

hs = 10.0 .^ range(-8, 0, length = 17)
println("perturb logβ[30] by h, same seed each time")
Random.seed!(3)
base = run_filter_path(θp, lb0, flu.obs, 32; threaded = false)
for h in hs
    lb = copy(lb0)
    lb[30] += h
    Random.seed!(3)
    v = run_filter_path(θp, lb, flu.obs, 32; threaded = false)
    println("  h=", rpad(round(h, sigdigits = 2), 10), " Δll=", round(v - base, digits = 10))
end

## how many distinct values does it take over a small interval?
vals = Float64[]
for h in range(0.0, 0.05, length = 200)
    lb = copy(lb0)
    lb[30] += h
    Random.seed!(3)
    push!(vals, run_filter_path(θp, lb, flu.obs, 32; threaded = false))
end
println("\nover logβ[30] ∈ [log 3, log 3 + 0.05], 200 points at one seed:")
println("  distinct log-likelihood values: ", length(unique(round.(vals, digits = 9))))
println("  range: ", round(minimum(vals), digits = 2), " to ", round(maximum(vals), digits = 2))
