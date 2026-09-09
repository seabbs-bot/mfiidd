using Random
using Distributions
using SSMProblems

"""
    seit4l_step_beta!(rng, s, β, ϵ, ν, τ, α, dt)

One day of SEIT4L with the transmission rate supplied directly rather than
derived from `R_0` and `D_inf`.

`gillespie_step!` takes a parameter dictionary and computes `β = R_0 / D_inf`
once per call, which is what a constant-rate model wants. A time-varying rate
changes every day, so the rate is the argument here and the rest of the stepper
is unchanged: the same eight rates, the same stoichiometry, the same counting of
`E → I` transitions as incidence.
"""
function seit4l_step_beta!(
    rng::AbstractRNG,
    s::AbstractVector{Float64},
    β::Float64,
    ϵ::Float64,
    ν::Float64,
    τ::Float64,
    α::Float64,
    dt::Float64 = 1.0,
)
    t, daily_inc = 0.0, 0
    while t < dt
        r = seit4l_rates(s, β, ϵ, ν, τ, α)
        total_rate = sum(r)
        total_rate ≤ 0 && break

        τ_wait = randexp(rng) / total_rate
        t + τ_wait > dt && break
        t += τ_wait

        cum, rnd, event = 0.0, rand(rng) * total_rate, 0
        for i in 1:8
            cum += r[i]
            if rnd ≤ cum
                event = i
                break
            end
        end

        for j in 1:8
            s[j] += SEIT4L_STOICH[event][j]
        end

        event == 2 && (daily_inc += 1)
    end

    return daily_inc
end

## the three rates that do not vary, unpacked once per filter run rather than
## once per particle-day
function _fixed_rates(θ)
    return (1.0 / θ[:D_lat], 1.0 / θ[:D_inf], 4.0 / θ[:D_imm], θ[:α])
end

"""
SEIT4L with `log β` carried in the latent state.

State vector: `[S, E, I, T1, T2, T3, T4, L, log β, daily_inc]`.

Each particle draws its own increment `log β_t = log β_{t-1} + σ ε_t` with
`ε_t ~ Normal(0, 1)` and then simulates its day at its own rate, so the filter
integrates over the transmission trajectory in the same sweep that it integrates
over the epidemic. `θ` holds `:D_lat`, `:D_inf`, `:α`, `:D_imm` and `:σ`; there
is no `:R_0`, because the transmission rate is now a state rather than a
parameter.
"""
struct SEIT4LRWDynamics <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEIT4LRWDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    ϵ, ν, τ, α = _fixed_rates(dyn.θ)
    σ = dyn.θ[:σ]

    ## one 10-element vector per particle per day, for the reason
    ## `SEIT4LDynamics` allocates one 9-element vector: the compartments are
    ## copied in, the stepper advances them in place, and the walk and the
    ## incidence take the last two slots
    state = Vector{Float64}(undef, 10)
    @inbounds for i in 1:8
        state[i] = prev_state[i]
    end
    @inbounds state[9] = prev_state[9] + σ * randn(rng)
    @inbounds state[10] = seit4l_step_beta!(rng, state, exp(state[9]), ϵ, ν, τ, α, 1.0)
    return state
end

"""
Initial state for the random-walk model: the compartments, `log β_0`, and a zero
for the incidence that has not happened yet.
"""
struct SEIT4LRWInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
    logβ0::Float64
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEIT4LRWInitial; kwargs...)
    return vcat(prior.init_state, prior.logβ0, 0.0)
end

"""
SEIT4L with the transmission trajectory supplied from outside.

State vector: `[S, E, I, T1, T2, T3, T4, L, daily_inc]`, as for the
constant-rate model. `logβ` is a vector with one entry per day, fixed for the
whole filter run, so every particle simulates day `t` at the same rate and the
filter integrates over the epidemic alone.

This is the same generative model as `SEIT4LRWDynamics`. What differs is who
owns the transmission trajectory: here it is the outer sampler's, so a filter
run is a likelihood conditional on a path rather than a likelihood with the path
integrated out.
"""
struct SEIT4LPathDynamics <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
    logβ::Vector{Float64}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEIT4LPathDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    ϵ, ν, τ, α = _fixed_rates(dyn.θ)
    β = exp(dyn.logβ[step])

    state = Vector{Float64}(undef, 9)
    @inbounds for i in 1:8
        state[i] = prev_state[i]
    end
    @inbounds state[9] = seit4l_step_beta!(rng, state, β, ϵ, ν, τ, α, 1.0)
    return state
end

"""
    run_filter_rw(θ, obs, n_particles; init_state, threaded, nchunks)

Log-likelihood of `obs` under SEIT4L with `log β` following a Gaussian random
walk in the latent state. `θ` needs `:D_lat`, `:D_inf`, `:α`, `:D_imm`, `:σ`,
`:ρ` and `:logβ0`.
"""
function run_filter_rw(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    threaded = true,
    nchunks = Threads.nthreads(),
)
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    model = StateSpaceModel(
        SEIT4LRWInitial(collect(Float64.(init_state)), θ_f64[:logβ0]),
        SEIT4LRWDynamics(θ_f64),
        PoissonObservation(θ_f64[:ρ]),
    )
    algo = threaded ? ThreadedBF(BF(n_particles); nchunks) : BF(n_particles)
    _, log_lik = GeneralisedFilters.filter(default_rng(), model, algo, obs)
    return log_lik
end

"""
    run_filter_path(θ, logβ, obs, n_particles; init_state, threaded, nchunks)

Log-likelihood of `obs` under SEIT4L with the transmission trajectory `logβ`
held fixed. `θ` needs `:D_lat`, `:D_inf`, `:α`, `:D_imm` and `:ρ`.
"""
function run_filter_path(
    θ,
    logβ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    threaded = true,
    nchunks = Threads.nthreads(),
)
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    logβ_f64 = Float64[value(x) for x in logβ]
    model = StateSpaceModel(
        SEIT4LInitial(collect(Float64.(init_state))),
        SEIT4LPathDynamics(θ_f64, logβ_f64),
        PoissonObservation(θ_f64[:ρ]),
    )
    algo = threaded ? ThreadedBF(BF(n_particles); nchunks) : BF(n_particles)
    _, log_lik = GeneralisedFilters.filter(default_rng(), model, algo, obs)
    return log_lik
end

"""
    filtered_beta(θ, obs, n_particles; init_state)

One draw from the smoothing distribution of the transmission trajectory, as
`(logβ, incidence)`. Built the same way as `filtered_incidence`: record every
particle and its ancestor, draw one particle in proportion to its final weight,
and follow it back.
"""
function filtered_beta(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
)
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    model = StateSpaceModel(
        SEIT4LRWInitial(collect(Float64.(init_state)), θ_f64[:logβ0]),
        SEIT4LRWDynamics(θ_f64),
        PoissonObservation(θ_f64[:ρ]),
    )

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

"""
    simulate_rw_data(rng, θ, n_days; init_state)

Simulate one dataset from the random-walk model, returning
`(obs, logβ, incidence)`. Used to check whether `σ` is recoverable from data
generated at a known value.
"""
function simulate_rw_data(
    rng::AbstractRNG,
    θ,
    n_days::Integer;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
)
    ϵ, ν, τ, α = _fixed_rates(θ)
    s = vcat(collect(Float64.(init_state)), 0.0)
    logβ = θ[:logβ0]
    lβ, inc, obs = Float64[], Int[], Int[]
    for _ in 1:n_days
        logβ += θ[:σ] * randn(rng)
        i = seit4l_step_beta!(rng, s, exp(logβ), ϵ, ν, τ, α, 1.0)
        push!(lβ, logβ)
        push!(inc, i)
        push!(obs, rand(rng, Poisson(max(θ[:ρ] * i, 1e-10))))
    end
    return (obs, lβ, inc)
end
