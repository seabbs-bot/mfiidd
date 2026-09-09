using Random
using Distributions
using SSMProblems
using DifferentialEquations

"""
    seit4l_tvbeta_ode!(du, u, p, t)

SEIT4L with the transmission rate supplied directly.

`seit4l_ode!` takes `R_0` and `D_inf` and forms `β = R_0 / D_inf` itself, which
is what a constant-rate model wants. Here `β` is the first parameter, because a
time-varying rate changes from day to day while `D_inf` does not. Everything
else is `seit4l_ode!`: the same nine states, the same flows, and `u[9]`
accumulating the `E → I` transitions that make up incidence.
"""
function seit4l_tvbeta_ode!(du, u, p, t)
    β, ϵ, ν, τ, α = p

    S, E, I, T1, T2, T3, T4, L, Inc = u
    N = S + E + I + T1 + T2 + T3 + T4 + L
    infection = β * S * I / N

    du[1] = -infection + (1 - α) * τ * T4
    du[2] = infection - ϵ * E
    du[3] = ϵ * E - ν * I
    du[4] = ν * I - τ * T1
    du[5] = τ * T1 - τ * T2
    du[6] = τ * T2 - τ * T3
    du[7] = τ * T3 - τ * T4
    du[8] = α * τ * T4
    du[9] = ϵ * E
    return nothing
end

"""
    SEIT4L_TVBETA_PROBLEM

A template `ODEProblem` for `seit4l_tvbeta_ode!`, built once when the package
loads. The day step calls `remake` on it rather than building a fresh problem
every day.
"""
const SEIT4L_TVBETA_PROBLEM = ODEProblem(seit4l_tvbeta_ode!, zeros(9), (0.0, 1.0), zeros(5))

"""
    seit4l_day_step(u, β, ϵ, ν, τ, α)

Advance the nine-element state `u` by one day at transmission rate `β`.

`u[9]` is expected to be zero on entry, so on exit it holds the incidence over
that day alone. The tolerances are tighter than the solver's defaults because
the two samplers in the time-varying transmission session have to agree to
better than Monte Carlo error, and the default tolerance is loose enough to
show up in that comparison.
"""
function seit4l_day_step(u, β, ϵ, ν, τ, α)
    prob = remake(SEIT4L_TVBETA_PROBLEM; u0 = u, p = [β, ϵ, ν, τ, α], tspan = (0.0, 1.0))
    sol = solve(prob, Tsit5(); save_everystep = false, abstol = 1e-8, reltol = 1e-8)
    return sol[:, end]
end

"""
    tvbeta_incidence(logβ, D_lat, D_inf, α, D_imm, init_state)

Daily incidence over `length(logβ)` days, given the whole transmission
trajectory.

The trajectory is piecewise constant: day `t` is integrated at `exp(logβ[t])`.
The returned vector has one entry per observed day, so `inc[i]` is the incidence
the `i`-th observation counts and the likelihood indexes it directly.

The element type follows `logβ` and the rate parameters, so automatic
differentiation propagates through the solve.
"""
function tvbeta_incidence(logβ, D_lat, D_inf, α, D_imm, init_state)
    ϵ = 1 / D_lat
    ν = 1 / D_inf
    τ = 4 / D_imm

    T = promote_type(
        eltype(logβ),
        typeof(D_lat),
        typeof(D_inf),
        typeof(α),
        typeof(D_imm),
        eltype(init_state),
    )
    u = T[init_state[1:8]..., zero(T)]

    inc = Vector{T}(undef, length(logβ))
    for t in eachindex(logβ)
        u = seit4l_day_step(u, exp(logβ[t]), ϵ, ν, τ, α)
        inc[t] = u[9]
        ## reset the incidence accumulator so the next day counts its own cases
        u = T[u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], zero(T)]
    end
    return inc
end

"""
SEIT4L latent dynamics with `log β` carried in the state.

State vector: `[S, E, I, T1, T2, T3, T4, L, log β, daily_inc]`.

Each particle takes its own Normal increment of scale `σ` and then integrates
its day at the resulting rate. The epidemic is deterministic given the rate, so
the walk is the only source of randomness in the latent process and the filter
integrates over transmission trajectories alone.

`θ` holds `:D_lat`, `:D_inf`, `:α`, `:D_imm` and `:σ`.
"""
struct SEIT4LTVBetaDynamics <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEIT4LTVBetaDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    θ = dyn.θ
    ϵ, ν, τ, α = 1 / θ[:D_lat], 1 / θ[:D_inf], 4 / θ[:D_imm], θ[:α]

    logβ = prev_state[9] + θ[:σ] * randn(rng)
    u = Float64[
        prev_state[1],
        prev_state[2],
        prev_state[3],
        prev_state[4],
        prev_state[5],
        prev_state[6],
        prev_state[7],
        prev_state[8],
        0.0,
    ]
    u = seit4l_day_step(u, exp(logβ), ϵ, ν, τ, α)

    return Float64[u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], logβ, u[9]]
end

"""
Initial state for the time-varying model: the compartments, `log β_0`, and a
zero for the incidence that has not happened yet.
"""
struct SEIT4LTVBetaInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
    logβ0::Float64
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEIT4LTVBetaInitial; kwargs...)
    return vcat(prior.init_state, prior.logβ0, 0.0)
end

## the transmission trajectory and the incidence are the last two components of
## the state, whichever way the state is stored
_tvbeta_model(θ, init_state) = StateSpaceModel(
    SEIT4LTVBetaInitial(collect(Float64.(init_state)), log(θ[:R_0] / θ[:D_inf])),
    SEIT4LTVBetaDynamics(θ),
    PoissonObservation(θ[:ρ]),
)

"""
    run_filter_tvbeta(θ, obs, n_particles; init_state, threaded, nchunks)

Log-likelihood of `obs` under SEIT4L with `log β` following a Gaussian random
walk, with the trajectory integrated out by a bootstrap filter.

`θ` needs `:R_0`, `:D_lat`, `:D_inf`, `:α`, `:D_imm`, `:ρ` and `:σ`. `:R_0`
fixes the start of the walk at `log(R_0 / D_inf)` and does nothing else.

`nchunks` defaults to the thread count, so a seeded run reproduces only on a
machine with the same number of threads. Pass it explicitly wherever the numbers
matter.
"""
function run_filter_tvbeta(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    threaded = false,
    nchunks = Threads.nthreads(),
)
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    model = _tvbeta_model(θ_f64, init_state)
    algo = threaded ? ThreadedBF(BF(n_particles); nchunks) : BF(n_particles)
    _, log_lik = GeneralisedFilters.filter(default_rng(), model, algo, obs)
    return log_lik
end

"""
    filtered_tvbeta(θ, obs, n_particles; init_state)

One draw from the smoothing distribution of the transmission trajectory, as
`(logβ, incidence)`.

Built the same way as `filtered_incidence`: record every particle and its
ancestor, draw one particle in proportion to its final weight, and follow it
back.
"""
function filtered_tvbeta(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
)
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    model = _tvbeta_model(θ_f64, init_state)

    callback = DenseAncestorCallback(nothing)
    final, _ =
        GeneralisedFilters.filter(default_rng(), model, BF(n_particles), obs; callback)

    log_w = getfield.(final.particles, :log_w)
    w = exp.(log_w .- maximum(log_w))
    u = rand() * sum(w)
    idx = findfirst(>=(u), cumsum(w))
    path = get_ancestry(callback.container, idx)

    return ([path[t][9] for t in 1:length(obs)], [path[t][10] for t in 1:length(obs)])
end
