## Shared setup for the sampler comparison.
##
## The priors are the ones in `scripts/pmmh_setup.jl`, so every sampler measured
## here targets the same posterior as the committed chain in
## `data/pmcmc_seit4l_chain.csv`.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Distributions
using DataFrames
using CSV
using MCMCChains
using Statistics

using MFIIDD

const PARAMETERS = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ]

## SEIT4L initial state: S = 279, I = 2 and the three returning islanders who
## are already past their infectious period, placed in the first T box.
const INIT_STATE = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

const PRIORS = (
    R_0 = truncated(Normal(3.0, 2.0), lower = 1.0),
    D_lat = truncated(Normal(2.0, 1.0), lower = 0.5),
    D_inf = truncated(Normal(3.0, 2.0), lower = 0.5),
    α = Beta(2, 2),
    D_imm = truncated(Normal(15.0, 10.0), lower = 1.0),
    ρ = Beta(2, 2),
)

flu_observations() =
    CSV.read(joinpath(@__DIR__, "..", "data", "flu_tdc_1971.csv"), DataFrame).obs

"""
    log_prior(θ)

Sum of the six prior log densities at `θ`, a `NamedTuple` or `Dict`.
"""
function log_prior(θ)
    return logpdf(PRIORS.R_0, θ[:R_0]) +
           logpdf(PRIORS.D_lat, θ[:D_lat]) +
           logpdf(PRIORS.D_inf, θ[:D_inf]) +
           logpdf(PRIORS.α, θ[:α]) +
           logpdf(PRIORS.D_imm, θ[:D_imm]) +
           logpdf(PRIORS.ρ, θ[:ρ])
end

θ_dict(v::AbstractVector) = Dict{Symbol, Float64}(PARAMETERS .=> v)
θ_vec(θ) = [Float64(θ[p]) for p in PARAMETERS]

"""
    committed_chain()

The saved PMMH SEIT4L chain, as a matrix with one column per parameter.
"""
function committed_chain()
    df = CSV.read(joinpath(@__DIR__, "..", "data", "pmcmc_seit4l_chain.csv"), DataFrame)
    return Matrix(df[:, PARAMETERS])
end

"""
    ess_report(draws, label, seconds)

Effective sample size per second on every parameter, and the time each sampler
would need to reach ESS 400 on its worst parameter. `draws` has one row per
retained iteration and one column per parameter.
"""
function ess_report(draws::AbstractMatrix, label, seconds)
    chn = Chains(reshape(draws, size(draws, 1), size(draws, 2), 1), PARAMETERS)
    e = DataFrame(ess(chn))
    e.ess_per_sec = e.ess ./ seconds
    worst = argmin(e.ess)

    println("\n", "="^72)
    println(label)
    println("  iterations kept: $(size(draws, 1))")
    println("  wall clock: $(round(seconds, digits = 1)) s")
    println("  seconds per iteration: $(round(seconds / size(draws, 1), digits = 5))")
    println("-"^72)
    for r in eachrow(e)
        println(
            rpad(string(r.parameters), 8),
            " ESS ",
            lpad(round(r.ess, digits = 1), 9),
            "   ESS/s ",
            lpad(round(r.ess_per_sec, digits = 4), 9),
        )
    end
    println("-"^72)
    w = e[worst, :]
    println("  worst parameter: $(w.parameters)")
    println("  ESS/s (worst): $(round(w.ess_per_sec, digits = 4))")
    println("  iterations per effective draw: $(round(size(draws, 1) / w.ess, digits = 1))")
    println("  time to ESS 400: $(round(400 / w.ess_per_sec / 60, digits = 1)) minutes")
    println("="^72)
    return e
end

"""
    posterior_table(draws, label)

Posterior mean and 95% interval for each parameter, for checking a sampler
against the committed chain.
"""
function posterior_table(draws::AbstractMatrix, label)
    println("\n$label posterior:")
    for (j, p) in enumerate(PARAMETERS)
        x = draws[:, j]
        println(
            rpad(string(p), 8),
            " mean ",
            lpad(round(mean(x), digits = 3), 8),
            "   sd ",
            lpad(round(std(x), digits = 3), 7),
            "   95% [",
            round(quantile(x, 0.025), digits = 3),
            ", ",
            round(quantile(x, 0.975), digits = 3),
            "]",
        )
    end
end
