## Arm A pilot: PMMH with log β in the latent state.
using MFIIDD, CSV, DataFrames, DrWatson, Random, Statistics
using Turing, Distributions, MCMCChains, AdvancedMH, ForwardDiff
using SSMProblems, GeneralisedFilters

flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
init8 = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

@model function pmmh_rw(obs, n_particles)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)
    σ ~ truncated(Normal(0.0, 0.2), lower = 0.0)

    v = ForwardDiff.value
    θ = Dict(:D_lat => v(D_lat), :D_inf => v(D_inf), :α => v(α),
             :D_imm => v(D_imm), :ρ => v(ρ), :σ => v(σ),
             :logβ0 => log(v(R_0) / v(D_inf)))
    Turing.@addlogprob! run_filter_rw(θ, obs, n_particles)
end

n_warm = parse(Int, get(ENV, "NWARM", "10000"))
n_keep = parse(Int, get(ENV, "NKEEP", "10000"))
np = parse(Int, get(ENV, "NPART", "128"))

model = pmmh_rw(flu.obs, np)
Random.seed!(20260909)
t = @elapsed chain = sample(
    model, externalsampler(AdvancedMH.RobustAdaptiveMetropolis()), n_keep;
    num_warmup = n_warm, check_model = false, progress = false,
)
println("threads=", Threads.nthreads(), " particles=", np,
        " warmup=", n_warm, " kept=", n_keep)
println("wall clock: ", round(t / 60, digits = 2), " min")

pars = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ, :σ]
df = DataFrame(chain)
acc = mean(df.R_0[2:end] .!= df.R_0[1:(end - 1)])
println("acceptance: ", round(100 * acc, digits = 1), "%")
mc = Chains(Matrix(df[:, pars]), pars)
show(stdout, MIME("text/plain"), summarystats(mc))
println()
e = ess(mc)
for (i, p) in enumerate(pars)
    println("  ", rpad(p, 7), " ESS=", round(e[:, :ess][i], digits = 1),
            "  ESS/s=", round(e[:, :ess][i] / t, digits = 4))
end
CSV.write(joinpath(@__DIR__, "armA_pilot.csv"), df)
