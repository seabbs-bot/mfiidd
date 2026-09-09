## The comparison that decides it: particle Gibbs at several particle counts,
## against PMMH run through the same loop and the same parameter step.
##
##     julia --project=. --threads=12 experiments/bench_all.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))
include(joinpath(@__DIR__, "pg.jl"))
include(joinpath(@__DIR__, "pmmh_plain.jl"))

using Printf
using JLD2
using MCMCChains: Chains, ess

const PG_ITER = 15_000
const PMMH_ITER = 50_000

say(args...) = (println(args...); flush(stdout))

obs = flu_observations()
say("threads: $(Threads.nthreads())")

## warm the code paths so the timed runs measure sampling
particle_gibbs(obs; n_particles = 32, n_iter = 20, n_warmup = 10, n_theta_steps = 10)
pmmh_plain(obs; n_particles = 32, n_iter = 20, n_warmup = 10)

summary_rows = []

function record!(label, draws, elapsed, extra = "")
    e = ess_report(draws, label, elapsed)
    worst = argmin(e.ess)
    push!(
        summary_rows,
        (
            label = label,
            seconds_per_iter = elapsed / size(draws, 1),
            worst = string(e[worst, :parameters]),
            ess_per_sec = e[worst, :ess_per_sec],
            minutes_to_400 = 400 / e[worst, :ess_per_sec] / 60,
        ),
    )
    isempty(extra) || say(extra)
    posterior_table(draws, label)
    flush(stdout)
    return e
end

## --------------------------------------- how noisy is the likelihood estimate
##
## PMMH is tuned by the spread of the log-likelihood estimate at a fixed θ.
## The usual advice is to pick the particle count that puts its standard
## deviation somewhere between 1 and 1.7: below that the extra particles buy
## less than they cost, above it the chain starts sticking on an overestimate.
## The course uses 128, and this says whether that is the right number.
say("\nspread of the log-likelihood estimate at the posterior mean")
θ_centre = Dict{Symbol, Float64}(
    :R_0 => 6.36,
    :D_lat => 1.33,
    :D_inf => 2.14,
    :α => 0.47,
    :D_imm => 11.77,
    :ρ => 0.69,
)
for N in (16, 32, 64, 128, 256)
    lls = [
        run_particle_filter(θ_centre, obs, N; init_state = INIT_STATE, threaded = true)
        for _ in 1:200
    ]
    t = time()
    for _ in 1:50
        run_particle_filter(θ_centre, obs, N; init_state = INIT_STATE, threaded = true)
    end
    @printf(
        "  %4d particles: mean %8.2f, sd %5.2f, %.5f s per run\n",
        N,
        mean(lls),
        std(lls),
        (time() - t) / 50
    )
end
flush(stdout)

## ------------------------------------------------------------------- PMMH
pmmh_results = Dict{Int, Any}()
for N in (32, 64, 128)
    say("\nrunning PMMH, $N particles, $PMMH_ITER iterations")
    r = pmmh_plain(obs; n_particles = N, n_iter = PMMH_ITER, n_warmup = 15_000)
    pmmh_results[N] = r
    record!(
        "PMMH, $N particles (plain loop, adaptive RWM)",
        r.draws,
        r.elapsed,
        @sprintf("  acceptance: %.1f%%", 100 * r.accept),
    )
end
jldsave(
    joinpath(@__DIR__, "pmmh_plain_draws.jld2");
    draws = Dict(N => pmmh_results[N].draws for N in keys(pmmh_results)),
    elapsed = Dict(N => pmmh_results[N].elapsed for N in keys(pmmh_results)),
)

## --------------------------------------------------------- particle Gibbs
pg_results = Dict{Int, Any}()
for N in (32, 64, 128, 256)
    say("\nrunning particle Gibbs, $N particles, $PG_ITER iterations")
    r = particle_gibbs(
        obs;
        n_particles = N,
        n_iter = PG_ITER,
        n_warmup = 1_500,
        n_theta_steps = 200,
    )
    pg_results[N] = r
    record!(
        "Particle Gibbs, $N particles",
        r.draws,
        r.elapsed,
        @sprintf(
            "  parameter-step acceptance: %.1f%%\n  path departs from reference at day: median %d, mean %.1f\n  sweeps that changed nothing: %.1f%%",
            100 * r.theta_accept,
            median(r.coalescence),
            mean(r.coalescence),
            100 * mean(r.coalescence .== length(obs) + 1)
        ),
    )
end

jldsave(
    joinpath(@__DIR__, "pg_draws.jld2");
    draws = Dict(N => pg_results[N].draws for N in keys(pg_results)),
    elapsed = Dict(N => pg_results[N].elapsed for N in keys(pg_results)),
    coalescence = Dict(N => pg_results[N].coalescence for N in keys(pg_results)),
)

posterior_table(committed_chain(), "committed PMMH chain")

## ------------------------------------------------------------------ summary
say("\n\n", "="^88)
say("SUMMARY (worst-mixing parameter)")
say("="^88)
say(
    rpad("sampler", 46),
    rpad("s/iter", 10),
    rpad("worst", 8),
    rpad("ESS/s", 10),
    "min to ESS 400",
)
for r in summary_rows
    @printf(
        "%-46s%-10.5f%-8s%-10.4f%.1f\n",
        r.label,
        r.seconds_per_iter,
        r.worst,
        r.ess_per_sec,
        r.minutes_to_400
    )
end
flush(stdout)
