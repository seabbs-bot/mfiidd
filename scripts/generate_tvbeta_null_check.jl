# A posterior predictive check for the time-varying transmission session.
#
# The smoothed transmission trajectory on the Tristan da Cunha data rises
# through the first wave and falls away afterwards. A smoother is conditioned
# on the observations, so it moves to track them whether or not the rate
# varied, and the shape on its own says nothing. This script asks how much
# movement a constant rate produces, by simulating from the constant-rate
# posterior and smoothing those datasets the same way.
#
# The test statistic is the distance the median smoothed β travels, from its
# lowest point to its highest, in log units.
#
# Depends on data/pmmh_tvbeta_state.csv, so run it after
# scripts/generate_tvbeta_chains.jl. Takes a couple of minutes.
#
# Run with `julia --project=. --threads=auto scripts/generate_tvbeta_null_check.jl`.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using DataFrames
using CSV
using DrWatson

using MFIIDD

const N_PATHS = 100      # smoothed trajectories per dataset
const N_NULL = 40        # datasets simulated with a constant rate
const N_PARTICLES = 128

flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
n_obs = length(flu.obs)

## the fit whose posterior supplies the parameters the smoother runs at, so the
## real and simulated datasets are smoothed under identical assumptions
state_df = CSV.read(datadir("pmmh_tvbeta_state.csv"), DataFrame)
## the constant-rate fit the null datasets are drawn from
const_df = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)

"""
    smoothed_median_beta(obs; seed)

Median of `N_PATHS` smoothed transmission trajectories for `obs`, each drawn at
a fresh posterior draw so that the paths are independent of one another.
"""
function smoothed_median_beta(obs; seed = 7)
    Random.seed!(seed)
    paths = map(1:N_PATHS) do _
        row = state_df[rand(1:nrow(state_df)), :]
        θ = Dict(
            :D_lat => row.D_lat,
            :D_inf => row.D_inf,
            :α => row.α,
            :D_imm => row.D_imm,
            :ρ => row.ρ,
            :σ => row.σ,
            :logβ0 => log(row.R_0 / row.D_inf),
        )
        first(filtered_beta(θ, obs, N_PARTICLES))
    end
    M = exp.(reduce(hcat, paths))
    return [median(M[t, :]) for t in 1:n_obs]
end

travel(m) = log(maximum(m) / minimum(m))

rows = DataFrame(source = String[], statistic = Float64[])
push!(rows, ("observed", travel(smoothed_median_beta(flu.obs))))
println("observed: ", round(last(rows.statistic), digits = 3))

for rep in 1:N_NULL
    Random.seed!(5000 + rep)
    row = const_df[rand(1:nrow(const_df)), :]
    θ = Dict(
        :D_lat => row.D_lat,
        :D_inf => row.D_inf,
        :α => row.α,
        :D_imm => row.D_imm,
        :ρ => row.ρ,
        :σ => 0.0,
        :logβ0 => log(row.R_0 / row.D_inf),
    )
    obs, _, _ = simulate_rw_data(Random.default_rng(), θ, n_obs)
    push!(rows, ("simulated", travel(smoothed_median_beta(obs; seed = 7))))
    println("  null $rep: ", round(last(rows.statistic), digits = 3))
end

null = rows.statistic[rows.source .== "simulated"]
observed = only(rows.statistic[rows.source .== "observed"])
println("median null ", round(median(null), digits = 3),
        ", p = ", round(mean(null .>= observed), digits = 3))

CSV.write(datadir("tvbeta_null_check.csv"), rows)
println("wrote $(nrow(rows)) rows to $(datadir("tvbeta_null_check.csv"))")
