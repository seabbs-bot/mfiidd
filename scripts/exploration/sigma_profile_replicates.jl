## Where does the σ profile peak, over replicate datasets from a known σ?
using MFIIDD, CSV, DataFrames, DrWatson, Random, Statistics
base = Dict(:D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7,
            :logβ0 => log(6.0 / 2.0))
σs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

for σ_true in (0.0, 0.15)
    peaks, gains = Float64[], Float64[]
    for rep in 1:10
        Random.seed!(1000 + rep)
        obs, _, _ = simulate_rw_data(Random.default_rng(),
                                     merge(base, Dict(:σ => σ_true)), 59)
        ll = map(σs) do σ
            θ = merge(base, Dict(:σ => σ))
            Random.seed!(2000 + rep)
            mean(run_filter_rw(θ, obs, 256) for _ in 1:8)
        end
        push!(peaks, σs[argmax(ll)])
        push!(gains, maximum(ll) - ll[1])
    end
    println("σ_true=", σ_true, " over 10 datasets")
    println("  peak σ: ", peaks)
    println("  gain over σ=0 (log units): ", round.(gains, digits = 1))
    println("  median peak=", median(peaks), " median gain=",
            round(median(gains), digits = 1))
end
