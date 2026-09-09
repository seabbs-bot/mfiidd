## Is the rise-and-fall in the smoothed β(t) on the real data extreme, against
## datasets generated with a constant rate? A posterior predictive check whose
## test statistic is the shape of the smoothed transmission trajectory.
using Random, Statistics, DataFrames, CSV, DrWatson, MFIIDD
flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
n_obs = length(flu.obs)
state_df = CSV.read(joinpath(@__DIR__, "armA_pilot_chain.csv"), DataFrame)
cn = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)

function smooth_median(obs; n = 100, seed = 7)
    Random.seed!(seed)
    paths = map(1:n) do _
        row = state_df[rand(1:nrow(state_df)), :]
        θ = Dict(:D_lat => row.D_lat, :D_inf => row.D_inf, :α => row.α,
                 :D_imm => row.D_imm, :ρ => row.ρ, :σ => row.σ,
                 :logβ0 => log(row.R_0 / row.D_inf))
        first(filtered_beta(θ, obs, 128))
    end
    M = exp.(reduce(hcat, paths))
    return [median(M[t, :]) for t in 1:n_obs]
end

## test statistic: how far the smoothed β travels, peak to trough, in log units
spread(m) = log(maximum(m) / minimum(m))

obs_stat = spread(smooth_median(flu.obs))
println("real data: log range of the smoothed median β = ", round(obs_stat, digits = 2))

## null datasets: constant rate, parameters drawn from the constant-rate posterior
null = Float64[]
for rep in 1:20
    Random.seed!(5000 + rep)
    row = cn[rand(1:nrow(cn)), :]
    θ = Dict(:D_lat => row.D_lat, :D_inf => row.D_inf, :α => row.α,
             :D_imm => row.D_imm, :ρ => row.ρ, :σ => 0.0,
             :logβ0 => log(row.R_0 / row.D_inf))
    obs, _, _ = simulate_rw_data(Random.default_rng(), θ, n_obs)
    push!(null, spread(smooth_median(obs; seed = 7)))
end
println("null (σ = 0) datasets: ", round.(sort(null), digits = 2))
println("median null = ", round(median(null), digits = 2))
println("p = ", round(mean(null .>= obs_stat), digits = 3),
        "  (fraction of constant-rate datasets at least as extreme)")
