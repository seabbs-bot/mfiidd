## Is σ identified? Profile the filter likelihood over σ, on real and simulated data.
using MFIIDD, CSV, DataFrames, DrWatson, Random, Statistics, Distributions

flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)

base = Dict(:D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7,
            :logβ0 => log(6.0 / 2.0))

function profile(obs, σs; nrep = 12, np = 512, label = "")
    println("\n--- σ profile, ", label, " (", np, " particles, mean of ", nrep, ") ---")
    out = Float64[]
    for σ in σs
        θ = merge(base, Dict(:σ => σ))
        Random.seed!(11)
        lls = [run_filter_rw(θ, obs, np) for _ in 1:nrep]
        push!(out, mean(lls))
        println("  σ=", rpad(σ, 6), " mean ll=", rpad(round(mean(lls), digits = 2), 9),
                " sd=", round(std(lls), digits = 2))
    end
    return out
end

σs = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
profile(flu.obs, σs; label = "Tristan da Cunha")

## simulated data at a known σ, everything else at `base`
for σ_true in (0.0, 0.15, 0.4)
    Random.seed!(99)
    obs, lβ, _ = simulate_rw_data(Random.default_rng(),
                                  merge(base, Dict(:σ => σ_true)), 59)
    println("\nsimulated at σ=", σ_true, ": total cases=", sum(obs),
            ", logβ range=", round.(extrema(lβ), digits = 2))
    profile(obs, σs; label = "simulated, σ_true=$σ_true")
end
