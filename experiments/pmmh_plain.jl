## PMMH written as a plain loop, sharing the parameter-step code with particle
## Gibbs.
##
## The course runs PMMH through Turing's `externalsampler`, which adds its own
## per-iteration work on top of the filter. Particle Gibbs here is a bare loop.
## Comparing the two directly would credit particle Gibbs with whatever that
## overhead costs, which is a property of the harness rather than of the
## sampler. This runs PMMH through the same loop and the same adaptive
## random-walk step, so the difference between them is the sampler.

using Statistics

"""
    pmmh_plain(obs; ...)

Particle marginal Metropolis-Hastings with an adaptive random-walk proposal on
the unconstrained parameters.

The accepted log-likelihood estimate is carried forward rather than recomputed,
which is what makes this pseudo-marginal: the chain is exact for the posterior
even though every likelihood it sees is an estimate.
"""
function pmmh_plain(
    obs;
    n_particles = 128,
    n_iter = 100_000,
    n_warmup = 20_000,
    θ_init = Dict{Symbol, Float64}(
        :R_0 => 6.36,
        :D_lat => 1.33,
        :D_inf => 2.14,
        :α => 0.47,
        :D_imm => 11.77,
        :ρ => 0.69,
    ),
    seed = 20260909,
    freeze_after_warmup = false,
)
    rng = Random.default_rng()
    Random.seed!(seed)

    step = ThetaStep(6)
    u = to_unconstrained(θ_init)
    θ = to_constrained(u)

    loglik(θ) =
        run_particle_filter(θ, obs, n_particles; init_state = INIT_STATE, threaded = true)

    ll = loglik(θ)
    lp = log_prior(θ) + log_jacobian(u) + ll

    draws = Matrix{Float64}(undef, n_iter, 6)
    t_start = 0.0
    moved = 0

    for iter in 1:(n_warmup + n_iter)
        iter == n_warmup + 1 && (t_start = time())

        L = proposal_chol(step)
        u_prop = u .+ exp(step.log_scale) .* (L * randn(rng, 6))
        θ_prop = to_constrained(u_prop)
        lp_prop = log_prior(θ_prop) + log_jacobian(u_prop) + loglik(θ_prop)

        accepted = log(rand(rng)) < lp_prop - lp
        if accepted
            u, θ, lp = u_prop, θ_prop, lp_prop
            step.accepted += 1
            iter > n_warmup && (moved += 1)
        end
        step.proposed += 1
        if !(freeze_after_warmup && iter > n_warmup)
            adapt_scale!(step, accepted)
            observe!(step, u)
        end

        iter > n_warmup && (draws[iter - n_warmup, :] .= θ_vec(θ))
    end

    return (
        draws = draws,
        elapsed = time() - t_start,
        accept = moved / n_iter,
    )
end
