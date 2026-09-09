## Checks on the sufficient-statistic simulator, before it is used to sample.
##
## 1. The statistics recover the parameters. For a Markov jump process the
##    maximum-likelihood estimate of a rate constant is the number of events of
##    that type over the integrated exposure, so a long path simulated at a
##    known θ must return those values. This tests the exposure integrals and
##    the event counts together, and it fails loudly if either is wrong.
## 2. The incidence it produces has the same distribution as the simulator the
##    course uses, so the statistics are being recorded off the same model.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))

using Statistics
using Printf

θ = Dict{Symbol, Float64}(
    :R_0 => 5.7,
    :D_lat => 1.45,
    :D_inf => 1.74,
    :α => 0.45,
    :D_imm => 11.4,
    :ρ => 0.68,
)

## ---------------------------------------------------------------- 1. recovery
##
## The process is absorbing: once the outbreak burns out there are no more
## events, so a single long path stops informing the rates. The statistics are
## pooled over many independent outbreaks instead, all simulated at the same θ,
## which is what makes the estimates tight enough to be a test.
Random.seed!(11)
n_paths = 600
n_days = 120

counts = zeros(8)
eb = ee = ei = et = 0.0
for _ in 1:n_paths
    s = zeros(STATE_LEN)
    s[1:8] .= INIT_STATE
    for _ in 1:n_days
        gillespie_step_stats!(Random.default_rng(), s, θ)
        for k in 1:8
            counts[k] += s[IDX_COUNTS[k]]
        end
        global eb += s[IDX_EXP_BETA]
        global ee += s[IDX_EXP_E]
        global ei += s[IDX_EXP_I]
        global et += s[IDX_EXP_T]
    end
end

β_true = θ[:R_0] / θ[:D_inf]
ϵ_true = 1.0 / θ[:D_lat]
ν_true = 1.0 / θ[:D_inf]
τ_true = 4.0 / θ[:D_imm]

est = (
    β = counts[1] / eb,
    ϵ = counts[2] / ee,
    ν = counts[3] / ei,
    τ = sum(counts[4:8]) / et,
    α = counts[8] / (counts[7] + counts[8]),
)
truth = (β = β_true, ϵ = ϵ_true, ν = ν_true, τ = τ_true, α = θ[:α])

println("Rate-constant recovery pooled over $n_paths outbreaks of $n_days days")
println("  total events: $(Int(sum(counts)))")
ok = Ref(true)
for k in keys(truth)
    rel = abs(est[k] - truth[k]) / truth[k]
    ok[] &= rel < 0.02
    @printf("  %-3s estimate %8.5f  truth %8.5f  rel err %6.4f\n", k, est[k], truth[k], rel)
end
println(ok[] ? "  PASS: every rate recovered within 2%" : "  FAIL: a rate is off")

## The likelihood must peak at those estimates. Perturbing one parameter away
## from its maximum-likelihood value has to lower the complete-data likelihood,
## which checks the algebra in `complete_loglik` against the statistics.
st = PathStats(
    (counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7],
        counts[8]),
    eb,
    ee,
    ei,
    et,
    0.0,
    0.0,
    0,
)
θ_hat = Dict{Symbol, Float64}(
    :R_0 => est.β / est.ν,
    :D_lat => 1 / est.ϵ,
    :D_inf => 1 / est.ν,
    :α => est.α,
    :D_imm => 4 / est.τ,
    :ρ => 0.5,
)
base = complete_loglik(θ_hat, st)
println("\nComplete-data likelihood is maximised at the estimates")
peak_ok = Ref(true)
for p in (:R_0, :D_lat, :D_inf, :α, :D_imm)
    for factor in (0.97, 1.03)
        θ_p = copy(θ_hat)
        θ_p[p] *= factor
        Δ = complete_loglik(θ_p, st) - base
        peak_ok[] &= Δ < 0
        @printf("  %-6s × %.2f  Δloglik %10.2f\n", p, factor, Δ)
    end
end
println(peak_ok[] ? "  PASS: every perturbation lowers it" : "  FAIL: not at a maximum")

## ------------------------------------------------- 2. same model as the course
##
## Both simulators are run from the island's initial state over the observation
## window. If the statistics were recorded off a different process, the mean
## incidence curves would separate.
Random.seed!(22)
n_rep = 4000
T = 59
inc_stat = zeros(n_rep, T)
inc_course = zeros(n_rep, T)

for rep in 1:n_rep
    s1 = zeros(STATE_LEN)
    s1[1:8] .= INIT_STATE
    for t in 1:T
        inc_stat[rep, t] = gillespie_step_stats!(Random.default_rng(), s1, θ)
    end

    s2 = collect(INIT_STATE)
    for t in 1:T
        inc_course[rep, t] = gillespie_step!(Random.default_rng(), s2, θ)
    end
end

m1 = vec(mean(inc_stat, dims = 1))
m2 = vec(mean(inc_course, dims = 1))
se = sqrt.(vec(var(inc_stat, dims = 1)) ./ n_rep .+ vec(var(inc_course, dims = 1)) ./ n_rep)
z = (m1 .- m2) ./ max.(se, 1e-12)

println("\nDaily incidence against the course simulator over $n_rep replicates")
@printf("  total mean incidence: statistics %.2f, course %.2f\n", sum(m1), sum(m2))
@printf("  largest |z| across the %d days: %.2f\n", T, maximum(abs.(z)))
println(
    maximum(abs.(z)) < 4 ?
    "  PASS: the two simulators agree within Monte Carlo error" :
    "  FAIL: the incidence distributions differ",
)
