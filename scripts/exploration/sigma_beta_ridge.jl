## The (σ, β₀) likelihood surface, holding everything else at the constant-rate
## posterior means.
using CSV, DataFrames, DrWatson, Statistics, Random, MFIIDD, Printf
flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
const_ = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)

base = Dict(:D_lat => mean(const_.D_lat), :D_inf => mean(const_.D_inf),
            :α => mean(const_.α), :D_imm => mean(const_.D_imm), :ρ => mean(const_.ρ))
β_const = mean(const_.R_0) / mean(const_.D_inf)
println("holding D_lat=", round(base[:D_lat], digits = 2),
        " D_inf=", round(base[:D_inf], digits = 2),
        " α=", round(base[:α], digits = 2),
        " D_imm=", round(base[:D_imm], digits = 2),
        " ρ=", round(base[:ρ], digits = 2))
println("constant-rate posterior mean β = ", round(β_const, digits = 3))

σs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
β0s = round.(β_const .* [0.35, 0.5, 0.7, 0.85, 1.0, 1.2], digits = 3)

M = fill(NaN, length(β0s), length(σs))
for (i, β0) in enumerate(β0s), (j, σ) in enumerate(σs)
    θ = merge(base, Dict(:σ => σ, :logβ0 => log(β0)))
    Random.seed!(21)
    M[i, j] = mean(run_filter_rw(θ, flu.obs, 512) for _ in 1:10)
end

println("\nmean log-likelihood; rows β₀, columns σ")
print(rpad("β₀ \\ σ", 9))
for σ in σs
    print(rpad(σ, 9))
end
println()
for (i, β0) in enumerate(β0s)
    print(rpad(β0, 9))
    for j in eachindex(σs)
        @printf("%-9.1f", M[i, j])
    end
    println()
end

best = argmax(M)
println("\nbest cell: β₀=", β0s[best[1]], " σ=", σs[best[2]],
        " ll=", round(M[best], digits = 1))
println("cells within 2 log units of the best:")
for i in eachindex(β0s), j in eachindex(σs)
    if M[i, j] > M[best] - 2
        println("  β₀=", rpad(β0s[i], 7), " σ=", rpad(σs[j], 6),
                " ll=", round(M[i, j], digits = 1))
    end
end
CSV.write(joinpath(@__DIR__, "ridge.csv"),
          DataFrame(vcat([[β0s[i] σs[j] M[i, j]] for i in eachindex(β0s), j in eachindex(σs)]...),
                    [:β0, :σ, :ll]))
