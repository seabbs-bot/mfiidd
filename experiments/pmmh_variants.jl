## Machinery for the PMMH proposal ablation.
##
## One sampling loop with three things switchable, so that a variant differs
## from another in exactly one respect: the shape of the proposal covariance,
## how that covariance is adapted, and the scale the proposal is made on.
##
## `RobustAdaptiveMetropolis` is reimplemented here rather than driven through
## AdvancedMH, so that it runs in the same loop against the same filter with the
## same seed. It follows AdvancedMH 0.8.10: target acceptance 0.234, decay
## exponent 0.6, and the same rank-one update and downdate of the Cholesky
## factor. Its default initial factor is the identity, which is also AdvancedMH's
## default, and `initial_S` exists so that choice can be tested rather than
## inherited.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))
include(joinpath(@__DIR__, "pg.jl"))

using Printf
using JLD2
using LinearAlgebra:
    LowerTriangular, Cholesky, Diagonal, diag, lowrankupdate, lowrankdowndate, norm
using MCMCChains: Chains, ess

say(args...) = (println(args...); flush(stdout))

const N_PARTICLES = 128
const N_WARMUP = 20_000
const N_KEPT = 30_000

## ------------------------------------------------------------ parameterisation
##
## `:transformed` proposes on the log and logit scale, with the Jacobian.
## `:constrained` proposes on the parameters themselves and relies on the priors
## to return `-Inf` outside their support, which is what a proposal that ignores
## the bounds would do.

function target_at(v, transform, obs, n_particles)
    θ, lj = if transform === :transformed
        to_constrained(v), log_jacobian(v)
    else
        θ_dict(v), 0.0
    end
    lp = log_prior(θ)
    isfinite(lp) || return -Inf, θ
    ll = run_particle_filter(
        θ,
        obs,
        n_particles;
        init_state = INIT_STATE,
        threaded = true,
    )
    return lp + lj + ll, θ
end

start_vector(transform, θ) =
    transform === :transformed ? to_unconstrained(θ) : θ_vec(θ)

## ------------------------------------------------------------------ RAM proposal

mutable struct RAM
    S::LowerTriangular{Float64, Matrix{Float64}}
    iteration::Int
    γ::Float64
    α_target::Float64
end

RAM(d::Int; initial_S = nothing) = RAM(
    isnothing(initial_S) ? LowerTriangular(Matrix(1.0 * I, d, d)) :
    LowerTriangular(Matrix(Diagonal(collect(Float64, initial_S)))),
    0,
    0.6,
    0.234,
)

function ram_adapt!(r::RAM, logα, U)
    Δα = exp(logα) - r.α_target
    η = r.iteration^(-r.γ)
    ΔS = sqrt(η * abs(Δα)) .* (r.S * U) ./ norm(U)
    chol = Cholesky(r.S.data, :L, 0)
    r.S = (Δα > 0 ? lowrankupdate(chol, ΔS) : lowrankdowndate(chol, ΔS)).L
    return r
end

## ---------------------------------------------------------------- the sampler

"""
    run_variant(obs; shape, transform, freeze_after_warmup)

One PMMH chain. `shape` is `:full` or `:diagonal` for the empirical-covariance
proposal with a Robbins-Monro scale, or `:ram` for the rank-one adaptation.
"""
function run_variant(
    obs;
    shape = :full,
    transform = :transformed,
    freeze_after_warmup = false,
    n_particles = N_PARTICLES,
    n_warmup = N_WARMUP,
    n_iter = N_KEPT,
    seed = 20260909,
    initial_S = nothing,
)
    Random.seed!(seed)
    rng = Random.default_rng()

    θ0 = Dict{Symbol, Float64}(
        :R_0 => 6.36,
        :D_lat => 1.33,
        :D_inf => 2.14,
        :α => 0.47,
        :D_imm => 11.77,
        :ρ => 0.69,
    )
    v = start_vector(transform, θ0)
    lp, θ = target_at(v, transform, obs, n_particles)

    haario = ThetaStep(6)
    ram = RAM(6; initial_S)

    draws = Matrix{Float64}(undef, n_iter, 6)
    t_start = 0.0
    moved = 0

    for iter in 1:(n_warmup + n_iter)
        iter == n_warmup + 1 && (t_start = time())
        adapting = !(freeze_after_warmup && iter > n_warmup)

        if shape === :ram
            U = randn(rng, 6)
            v_prop = v .+ ram.S * U
            lp_prop, θ_prop = target_at(v_prop, transform, obs, n_particles)
            logα = min(lp_prop - lp, 0.0)
            accepted = randexp(rng) > -logα
            if accepted
                v, θ, lp = v_prop, θ_prop, lp_prop
                iter > n_warmup && (moved += 1)
            end
            ram.iteration += 1
            adapting && isfinite(logα) && ram_adapt!(ram, logα, U)
        else
            L = proposal_chol(haario)
            shape === :diagonal && (L = LowerTriangular(Matrix(Diagonal(diag(L)))))
            v_prop = v .+ exp(haario.log_scale) .* (L * randn(rng, 6))
            lp_prop, θ_prop = target_at(v_prop, transform, obs, n_particles)
            accepted = log(rand(rng)) < lp_prop - lp
            if accepted
                v, θ, lp = v_prop, θ_prop, lp_prop
                haario.accepted += 1
                iter > n_warmup && (moved += 1)
            end
            haario.proposed += 1
            if adapting
                adapt_scale!(haario, accepted)
                observe!(haario, v)
            end
        end

        iter > n_warmup && (draws[iter - n_warmup, :] .= θ_vec(θ))
    end

    return (draws = draws, elapsed = time() - t_start, accept = moved / n_iter)
end

