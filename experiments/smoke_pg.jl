## Quick checks that the conditional SMC sweep behaves, before spending a chain
## on it.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "seit4l_sufficient.jl"))
include(joinpath(@__DIR__, "pg.jl"))

using Printf

obs = flu_observations()
θ = Dict{Symbol, Float64}(
    :R_0 => 6.36,
    :D_lat => 1.33,
    :D_inf => 2.14,
    :α => 0.47,
    :D_imm => 11.77,
    :ρ => 0.69,
)

Random.seed!(3)

## The statistics-carrying filter must give the same log-likelihood as the one
## the course uses, or the sampler is targeting a different model.
lls_stat = [csmc_path(Random.default_rng(), θ, obs, 128, nothing)[2] for _ in 1:40]
lls_course = [run_particle_filter(θ, obs, 128; init_state = INIT_STATE, threaded = true)
              for _ in 1:40]
@printf(
    "log-likelihood: statistics filter %.2f (sd %.2f), course filter %.2f (sd %.2f)\n",
    mean(lls_stat),
    std(lls_stat),
    mean(lls_course),
    std(lls_course)
)

## A path drawn from the filter should reproduce the observations through the
## reporting rate, and its statistics should be of the right size.
path, _ = csmc_path(Random.default_rng(), θ, obs, 128, nothing)
inc = [path[t][IDX_INC] for t in 1:length(obs)]
st = path_stats(path, obs)
@printf("\npath total incidence %.0f, observed cases %d, ρ × incidence %.0f\n",
    sum(inc), sum(obs), θ[:ρ] * sum(inc))
@printf("infections n1 = %.0f, E→I n2 = %.0f (must equal path incidence %.0f)\n",
    st.n[1], st.n[2], sum(inc))
@printf("exposures: β %.1f, E %.1f, I %.1f, T %.1f\n",
    st.exp_beta, st.exp_E, st.exp_I, st.exp_T)

## Timing of one sweep, against the plain filter the baseline uses.
csmc_path(Random.default_rng(), θ, obs, 128, path)
t = time()
for _ in 1:100
    global path
    path, _ = csmc_path(Random.default_rng(), θ, obs, 128, path)
end
t_csmc = (time() - t) / 100

run_particle_filter(θ, obs, 128; init_state = INIT_STATE, threaded = true)
t = time()
for _ in 1:100
    run_particle_filter(θ, obs, 128; init_state = INIT_STATE, threaded = true)
end
t_bf = (time() - t) / 100

@printf("\nseconds per sweep: conditional SMC %.5f, course bootstrap filter %.5f\n",
    t_csmc, t_bf)

## Cost of the parameter step, which is what makes many inner steps affordable.
step = ThetaStep(6)
u = to_unconstrained(θ)
theta_update!(Random.default_rng(), step, u, st, 10)
t = time()
for _ in 1:200
    theta_update!(Random.default_rng(), step, u, st, 50)
end
@printf("seconds per 50 parameter steps: %.6f\n", (time() - t) / 200)

## Does the path actually move? Ancestry collapse would show as a first
## difference that sits at the far end of the window.
first_diffs = Int[]
for _ in 1:100
    global path
    new_path, _ = csmc_path(Random.default_rng(), θ, obs, 128, path)
    fd = length(obs) + 1
    for t in 1:length(obs)
        if new_path[t] != path[t]
            fd = t
            break
        end
    end
    push!(first_diffs, fd)
    path = new_path
end
@printf(
    "\nfirst day the path departs from the reference: median %d, mean %.1f, max %d of %d\n",
    median(first_diffs),
    mean(first_diffs),
    maximum(first_diffs),
    length(obs)
)
println("fraction of sweeps that changed nothing: ",
    mean(first_diffs .== length(obs) + 1))
