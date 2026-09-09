## Particle Gibbs for SEIT4L.
##
## The sampler alternates two updates. A conditional SMC sweep redraws the
## latent path with the current path held as the reference particle, which
## `GeneralisedFilters` supports through the `ref_state` keyword to its
## bootstrap filter, and which needs only to simulate the dynamics. The
## parameter update then conditions on that complete path, where the SEIT4L
## likelihood is closed form: the path's sufficient statistics reduce it to a
## six-parameter expression that costs a few dozen nanoseconds and carries no
## filter noise at all.
##
## Because that expression is so cheap, the parameter update runs many
## Metropolis steps per sweep. That is deliberate. It makes the parameter draw
## effectively exact given the path, so whatever mixing remains is the Gibbs
## structure itself rather than a poorly tuned parameter step, and the result is
## an upper bound on what this family of samplers can do here.

using GeneralisedFilters:
    GeneralisedFilters, BF, AbstractCallback, PostInitCallback, PostUpdateCallback
using SSMProblems: StateSpaceModel
using LinearAlgebra: Symmetric, cholesky, I
using Statistics

const OffsetVector = GeneralisedFilters.OffsetArrays.OffsetVector

## ------------------------------------------------------------ path recording

"""
    PathCallback()

Records the particles and their ancestor indices at every step, so that a
particle drawn from the final weights can be traced back to its whole path.

This is `DenseAncestorCallback` without the `deepcopy`. That copy is what makes
the stock callback expensive here: `deepcopy` on a plain `Vector{Float64}` still
goes through the generic machinery and its `IdDict`, and the filter calls it
once per particle per day. Copying is not needed, because a particle's state is
allocated fresh by the dynamics and never written to afterwards, so keeping a
reference to it is enough. The reference particle is the one state that is
shared between iterations, and it is not mutated either.
"""
mutable struct PathCallback <: AbstractCallback
    states::Vector{Vector{Vector{Float64}}}
    ancestors::Vector{Vector{Int}}
end

PathCallback() = PathCallback(Vector{Vector{Float64}}[], Vector{Int}[])

function (c::PathCallback)(model, filter, state, data, ::PostInitCallback; kwargs...)
    empty!(c.states)
    empty!(c.ancestors)
    push!(c.states, getfield.(state.particles, :state))
    return nothing
end

function (c::PathCallback)(model, filter, step, state, data, ::PostUpdateCallback;
    kwargs...)
    push!(c.states, getfield.(state.particles, :state))
    push!(c.ancestors, getfield.(state.particles, :ancestor))
    return nothing
end

"""
    lineage(c, k)

The path of particle `k`, indexed from 0 so it can be handed straight back as
the next sweep's reference.
"""
function lineage(c::PathCallback, k::Integer)
    T = length(c.ancestors)
    v = Vector{Vector{Float64}}(undef, T + 1)
    a = k
    for t in T:-1:1
        v[t + 1] = c.states[t + 1][a]
        a = c.ancestors[t][a]
    end
    v[1] = c.states[1][a]
    return OffsetVector(v, -1)
end

## --------------------------------------------------------------- conditional SMC

"""
    csmc_path(rng, θ, obs, n_particles, ref; threaded)

One conditional SMC sweep. Returns a path drawn from the final weights and the
log-evidence. With `ref === nothing` this is an ordinary bootstrap filter, which
is how the first path is drawn.

The reference path occupies particle 1: the conditional resampler pins its
ancestor index there, and the prediction step returns the reference state rather
than simulating for it.
"""
function csmc_path(rng, θ, obs, n_particles, ref; threaded = true)
    model = StateSpaceModel(
        SEIT4LStatInitial(INIT_STATE),
        SEIT4LStatDynamics(θ),
        PoissonStatObservation(θ[:ρ]),
    )
    algo = threaded ? ThreadedBF(BF(n_particles)) : BF(n_particles)
    cb = PathCallback()

    final, ll = GeneralisedFilters.filter(
        rng,
        model,
        algo,
        obs;
        ref_state = ref,
        callback = cb,
    )

    log_w = getfield.(final.particles, :log_w)
    w = exp.(log_w .- maximum(log_w))
    u = rand(rng) * sum(w)
    k = something(findfirst(≥(u), cumsum(w)), length(w))

    return lineage(cb, k), ll
end

## -------------------------------------------------------- parameter space

logistic(u) = u ≥ 0 ? 1 / (1 + exp(-u)) : (e = exp(u); e / (1 + e))
logit(p) = log(p) - log1p(-p)

## log of the derivative of `logistic`, written through `abs` so it stays
## accurate in both tails
log_dlogistic(u) = -abs(u) - 2 * log1p(exp(-abs(u)))

"""
    to_unconstrained(θ), to_constrained(u)

Map between the six parameters and an unconstrained vector, so the parameter
step proposes on the whole real line. The three durations and `R_0` are shifted
by their prior's lower bound before taking logs; `α` and `ρ` go through a logit.
"""
to_unconstrained(θ) = [
    log(θ[:R_0] - 1.0),
    log(θ[:D_lat] - 0.5),
    log(θ[:D_inf] - 0.5),
    logit(θ[:α]),
    log(θ[:D_imm] - 1.0),
    logit(θ[:ρ]),
]

to_constrained(u) = Dict{Symbol, Float64}(
    :R_0 => 1.0 + exp(u[1]),
    :D_lat => 0.5 + exp(u[2]),
    :D_inf => 0.5 + exp(u[3]),
    :α => logistic(u[4]),
    :D_imm => 1.0 + exp(u[5]),
    :ρ => logistic(u[6]),
)

log_jacobian(u) =
    u[1] + u[2] + u[3] + u[5] + log_dlogistic(u[4]) + log_dlogistic(u[6])

"""
    log_target(u, st)

The conditional density of the parameters given a path, on the unconstrained
scale: prior, Jacobian and the complete-data likelihood from the path's
sufficient statistics.
"""
function log_target(u, st::PathStats)
    θ = to_constrained(u)
    return log_prior(θ) + log_jacobian(u) + complete_loglik(θ, st)
end

## ------------------------------------------------- adaptive parameter step

"""
    ThetaStep(d)

Random-walk Metropolis on the unconstrained parameters.

The proposal takes its shape from the running covariance of the draws and its
size from a scale adapted towards 23.4% acceptance. Both parts are needed. The
running covariance is of the marginal posterior, but each step conditions on a
path, and the conditional is far tighter than the marginal: the path fixes the
event counts, so it pins the rates to within roughly one over the square root of
the number of events. Proposing at the marginal scale would be several times too
wide, acceptance would collapse, and the sampler would look worse than the Gibbs
structure actually makes it. Adapting the scale separately removes that
confound, and it leaves the shape, which the two covariances broadly share.
"""
mutable struct ThetaStep
    mean::Vector{Float64}
    scatter::Matrix{Float64}
    count::Int
    log_scale::Float64
    accepted::Int
    proposed::Int
end

ThetaStep(d::Int) =
    ThetaStep(zeros(d), zeros(d, d), 0, log(2.38^2 / d) / 2, 0, 0)

function observe!(s::ThetaStep, u)
    s.count += 1
    δ = u .- s.mean
    s.mean .+= δ ./ s.count
    s.scatter .+= δ * (u .- s.mean)'
    return s
end

"""
    adapt_scale!(s, accepted)

Robbins-Monro update of the proposal scale towards 23.4% acceptance. The gain
decays as `n^-0.6`, so the adaptation vanishes and the chain keeps its target.
"""
function adapt_scale!(s::ThetaStep, accepted::Bool)
    γ = min(0.5, (s.proposed + 1.0)^(-0.6))
    s.log_scale += γ * ((accepted ? 1.0 : 0.0) - 0.234)
    s.log_scale = clamp(s.log_scale, -12.0, 4.0)
    return s
end

function proposal_chol(s::ThetaStep)
    d = length(s.mean)
    Σ = if s.count < 200
        Matrix(1e-4 * I, d, d)
    else
        Symmetric(s.scatter ./ (s.count - 1) + 1e-12 * I)
    end
    return cholesky(Σ).L
end

"""
    theta_update!(rng, s, u, st, n_steps)

Run `n_steps` Metropolis steps on the parameters with the path held fixed, and
return the final position. Each step evaluates only `log_target`, which is
arithmetic on the path's sufficient statistics, so a few hundred steps cost far
less than the sweep that produced the path.
"""
function theta_update!(rng, s::ThetaStep, u, st::PathStats, n_steps::Int)
    u = copy(u)
    lp = log_target(u, st)
    L = proposal_chol(s)
    d = length(u)

    for _ in 1:n_steps
        ## the scale is applied here rather than folded into `L`, so that a
        ## step adapted part way through a sweep takes effect immediately
        ## without refactorising the covariance
        u_prop = u .+ exp(s.log_scale) .* (L * randn(rng, d))
        lp_prop = log_target(u_prop, st)
        accepted = log(rand(rng)) < lp_prop - lp
        if accepted
            u, lp = u_prop, lp_prop
            s.accepted += 1
        end
        s.proposed += 1
        adapt_scale!(s, accepted)
        observe!(s, u)
    end
    return u
end

## ------------------------------------------------------------- the sampler

"""
    particle_gibbs(obs; ...)

Run particle Gibbs and return the parameter draws, the wall clock of the sampling
phase, and diagnostics.

`coalescence` records, for each iteration, the first day on which the new path
departs from the reference. Particle Gibbs inherits the particle filter's
ancestry collapse, so the early part of the path can be identical from one
iteration to the next however many iterations are run. That number is the check
on whether it is.
"""
function particle_gibbs(
    obs;
    n_particles = 128,
    n_iter = 20_000,
    n_warmup = 2_000,
    n_theta_steps = 50,
    θ_init = Dict{Symbol, Float64}(
        :R_0 => 6.36,
        :D_lat => 1.33,
        :D_inf => 2.14,
        :α => 0.47,
        :D_imm => 11.77,
        :ρ => 0.69,
    ),
    threaded = true,
    seed = 20260909,
)
    rng = Random.default_rng()
    Random.seed!(seed)

    T = length(obs)
    step = ThetaStep(6)
    u = to_unconstrained(θ_init)
    θ = to_constrained(u)

    ## first path from an unconditional sweep
    ref, _ = csmc_path(rng, θ, obs, n_particles, nothing; threaded)

    draws = Matrix{Float64}(undef, n_iter, 6)
    coalescence = Vector{Int}(undef, n_iter)
    t_start = 0.0

    for iter in 1:(n_warmup + n_iter)
        iter == n_warmup + 1 && (t_start = time())

        path, _ = csmc_path(rng, θ, obs, n_particles, ref; threaded)

        ## how far along the path the new draw agrees with the old one
        first_diff = T + 1
        for t in 1:T
            if path[t] != ref[t]
                first_diff = t
                break
            end
        end

        ref = path
        st = path_stats(path, obs)
        u = theta_update!(rng, step, u, st, n_theta_steps)
        θ = to_constrained(u)

        if iter > n_warmup
            k = iter - n_warmup
            draws[k, :] .= θ_vec(θ)
            coalescence[k] = first_diff
        end
    end

    elapsed = time() - t_start
    return (
        draws = draws,
        elapsed = elapsed,
        coalescence = coalescence,
        theta_accept = step.accepted / step.proposed,
    )
end
