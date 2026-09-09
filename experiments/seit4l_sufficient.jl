## A SEIT4L simulator that also records the sufficient statistics of the
## complete-data likelihood, and the state-space plumbing that carries them
## through a particle filter.
##
## For a Markov jump process the complete-data likelihood factorises over the
## transition types. Each type contributes the number of times it fired times
## the log of its rate constant, minus that constant times the integral of its
## exposure over the time at risk. A Gillespie path knows every event and every
## waiting time, so both quantities are exact arithmetic and carry no filter
## noise. The particle only has to carry them, one day at a time, so that the
## statistics of a whole path are the sum along a lineage.

using Random
using Distributions
using SSMProblems

## State layout: the eight compartments, the day's incidence, the eight
## transition counts, and the four exposure integrals.
const IDX_INC = 9
const IDX_COUNTS = 10:17
const IDX_EXP_BETA = 18   ## ∫ S I / N dt, the exposure for β
const IDX_EXP_E = 19      ## ∫ E dt, for ϵ
const IDX_EXP_I = 20      ## ∫ I dt, for ν
const IDX_EXP_T = 21      ## ∫ (T1 + T2 + T3 + T4) dt, for τ
const STATE_LEN = 21

"""
    gillespie_step_stats!(rng, s, θ, dt = 1.0)

Advance the eight compartments in `s` by `dt` and write the day's incidence,
transition counts and exposure integrals into the rest of `s`.

The exposure integrals are accumulated over each inter-event interval and over
the final partial interval, so the day's contribution to `-rate * time at risk`
is exact rather than a daily-state approximation.
"""
function gillespie_step_stats!(
    rng::AbstractRNG,
    s::AbstractVector{Float64},
    θ,
    dt::Float64 = 1.0,
)
    β = θ[:R_0] / θ[:D_inf]
    ϵ = 1.0 / θ[:D_lat]
    ν = 1.0 / θ[:D_inf]
    τ = 4.0 / θ[:D_imm]
    α = θ[:α]

    @inbounds for j in IDX_INC:STATE_LEN
        s[j] = 0.0
    end

    t = 0.0
    daily_inc = 0

    @inbounds while true
        S, E, I, T1, T2, T3, T4, L = s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]
        N = S + E + I + T1 + T2 + T3 + T4 + L

        r = (
            β * S * I / N,
            ϵ * E,
            ν * I,
            τ * T1,
            τ * T2,
            τ * T3,
            (1 - α) * τ * T4,
            α * τ * T4,
        )
        total = sum(r)

        ## the interval this iteration accounts for: either the waiting time to
        ## the next event, or whatever is left of the day
        held = if total ≤ 0
            dt - t
        else
            w = randexp(rng) / total
            t + w > dt ? dt - t : w
        end

        s[IDX_EXP_BETA] += S * I / N * held
        s[IDX_EXP_E] += E * held
        s[IDX_EXP_I] += I * held
        s[IDX_EXP_T] += (T1 + T2 + T3 + T4) * held

        total ≤ 0 && break
        t + held ≥ dt && break
        t += held

        ## select and apply the event
        cum, rnd, event = 0.0, rand(rng) * total, 8
        for i in 1:8
            cum += r[i]
            if rnd ≤ cum
                event = i
                break
            end
        end

        for j in 1:8
            s[j] += MFIIDD.SEIT4L_STOICH[event][j]
        end
        s[IDX_COUNTS[event]] += 1.0
        event == 2 && (daily_inc += 1)
    end

    @inbounds s[IDX_INC] = daily_inc
    return daily_inc
end

"""
SEIT4L latent dynamics whose state carries the complete-data sufficient
statistics alongside the compartments.
"""
struct SEIT4LStatDynamics <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEIT4LStatDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    state = Vector{Float64}(undef, STATE_LEN)
    @inbounds for i in 1:8
        state[i] = prev_state[i]
    end
    gillespie_step_stats!(rng, state, dyn.θ)
    return state
end

"""
Poisson observation on the ninth element.

The plain interface reads `state[end]`, which is the last exposure integral
once the statistics are appended, so this reads the incidence slot by index.
"""
struct PoissonStatObservation <: SSMProblems.ObservationProcess
    ρ::Float64
end

function SSMProblems.distribution(
    obs::PoissonStatObservation,
    step::Integer,
    state;
    kwargs...,
)
    return Poisson(max(obs.ρ * state[IDX_INC], 1e-10))
end

struct SEIT4LStatInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEIT4LStatInitial; kwargs...)
    state = zeros(STATE_LEN)
    @inbounds for i in 1:8
        state[i] = prior.init_state[i]
    end
    return state
end

"""
    PathStats

The sufficient statistics of a complete SEIT4L path: how many times each
transition fired, and the integrated exposure for each rate constant.

`obs_cases` and `obs_inc` summarise the observation process the same way. With
the incidence path fixed, the Poisson log-likelihood is
`obs_cases * log(ρ) - ρ * obs_inc` plus terms that do not involve `ρ`, so those
two numbers are all a `ρ` update needs. Days where the path has zero incidence
are excluded from both, because the filter floors the Poisson mean at `1e-10`
there and the contribution stops depending on `ρ`.
"""
struct PathStats
    n::NTuple{8, Float64}
    exp_beta::Float64
    exp_E::Float64
    exp_I::Float64
    exp_T::Float64
    obs_cases::Float64
    obs_inc::Float64
    zero_days::Int
end

"""
    path_stats(path, obs)

Sum the per-day statistics along `path`, an ancestry indexed from 0, and pair
them with the observations.
"""
function path_stats(path, obs)
    counts = zeros(8)
    eb = ee = ei = et = 0.0
    cases = inc_total = 0.0
    zero_days = 0

    for t in 1:length(obs)
        s = path[t]
        @inbounds for k in 1:8
            counts[k] += s[IDX_COUNTS[k]]
        end
        eb += s[IDX_EXP_BETA]
        ee += s[IDX_EXP_E]
        ei += s[IDX_EXP_I]
        et += s[IDX_EXP_T]

        inc = s[IDX_INC]
        if inc > 0
            cases += obs[t]
            inc_total += inc
        else
            zero_days += 1
        end
    end

    return PathStats(
        (
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            counts[5],
            counts[6],
            counts[7],
            counts[8],
        ),
        eb,
        ee,
        ei,
        et,
        cases,
        inc_total,
        zero_days,
    )
end

"""
    complete_loglik(θ, st)

Complete-data log-likelihood at `θ` given the path statistics `st`, up to an
additive constant that depends on the path but not on `θ`.

Every rate is a constant times a state-dependent factor, and only the constant
involves `θ`, so the state-dependent factors at the event times drop out. The
two exits from `T4` share the rate `τ` and split it by `α`, so their exposure
terms combine into the single `τ * exp_T` below and `α` survives only in the
counts.
"""
function complete_loglik(θ, st::PathStats)
    β = θ[:R_0] / θ[:D_inf]
    ϵ = 1.0 / θ[:D_lat]
    ν = 1.0 / θ[:D_inf]
    τ = 4.0 / θ[:D_imm]
    α = θ[:α]
    ρ = θ[:ρ]

    n = st.n
    n_tau = n[4] + n[5] + n[6] + n[7] + n[8]

    return n[1] * log(β) - β * st.exp_beta + n[2] * log(ϵ) - ϵ * st.exp_E + n[3] * log(ν) -
           ν * st.exp_I + n_tau * log(τ) - τ * st.exp_T +
           n[7] * log1p(-α) +
           n[8] * log(α) +
           st.obs_cases * log(ρ) - ρ * st.obs_inc
end
