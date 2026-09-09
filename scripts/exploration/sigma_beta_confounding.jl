## Why does the joint posterior want σ ≈ 0.27 when the profile at the constant-rate
## posterior means wants σ = 0?
using CSV, DataFrames, DrWatson, Statistics, Distributions, Random, MFIIDD

pilot = CSV.read(joinpath(@__DIR__, "armA_pilot_chain.csv"), DataFrame)
const_ = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)

pars = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ]
println("posterior means")
println(rpad("", 8), rpad("constant", 10), "random walk")
for p in pars
    println(rpad(p, 8), rpad(round(mean(const_[!, p]), digits = 3), 10),
            round(mean(pilot[!, p]), digits = 3))
end
println(rpad("σ", 8), rpad("-", 10), round(mean(pilot.σ), digits = 3))

println("\nprior for σ: mean ", round(mean(truncated(Normal(0, 0.2), lower = 0)), digits = 3),
        " sd ", round(std(truncated(Normal(0, 0.2), lower = 0)), digits = 3),
        "  95% ", round.(quantile.(truncated(Normal(0, 0.2), lower = 0), [0.025, 0.975]), digits = 3))
println("posterior for σ: mean ", round(mean(pilot.σ), digits = 3),
        " sd ", round(std(pilot.σ), digits = 3),
        "  95% ", round.(quantile(pilot.σ, [0.025, 0.975]), digits = 3))

println("\ncorrelation of σ with the others, in the random-walk posterior")
for p in pars
    println("  ", rpad(p, 8), round(cor(pilot.σ, pilot[!, p]), digits = 3))
end

## implied mean transmission: E[β_t] = β_0 exp(σ² t / 2)
println("\nthe walk is multiplicative, so E[β_t] = β_0 exp(σ² t / 2)")
for (nm, r0, di, σ) in (("constant posterior", mean(const_.R_0), mean(const_.D_inf), 0.0),
                        ("walk posterior", mean(pilot.R_0), mean(pilot.D_inf), mean(pilot.σ)))
    β0 = r0 / di
    println("  ", rpad(nm, 20), " β₀=", rpad(round(β0, digits = 3), 7),
            " E[β₃₀]=", rpad(round(β0 * exp(σ^2 * 30 / 2), digits = 3), 7),
            " E[β₅₉]=", round(β0 * exp(σ^2 * 59 / 2), digits = 3))
end

## profile again, but at the random-walk posterior means rather than the
## constant-rate ones
flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
b2 = Dict(:D_lat => mean(pilot.D_lat), :D_inf => mean(pilot.D_inf),
          :α => mean(pilot.α), :D_imm => mean(pilot.D_imm), :ρ => mean(pilot.ρ),
          :logβ0 => log(mean(pilot.R_0) / mean(pilot.D_inf)))
println("\nσ profile at the RANDOM-WALK posterior means (512 particles, mean of 12)")
for σ in (0.0, 0.05, 0.1, 0.2, 0.27, 0.35, 0.5)
    Random.seed!(11)
    ls = [run_filter_rw(merge(b2, Dict(:σ => σ)), flu.obs, 512) for _ in 1:12]
    println("  σ=", rpad(σ, 6), " mean ll=", rpad(round(mean(ls), digits = 2), 9),
            " se=", round(std(ls) / sqrt(12), digits = 2))
end
