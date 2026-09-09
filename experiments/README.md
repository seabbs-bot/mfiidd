# Sampler experiments for SEIT4L

A scouting branch, not a change to the course.
Nothing here is included in the site, and no session imports it.
The question was whether any sampler beats PMMH with a bootstrap particle filter on the SEIT4L model, measured in effective sample size per second.

Everything is run from the repository root with

```bash
julia --project=. --threads=12 experiments/<script>.jl
```

## State

Finished. The proposal isolation completed its last run after the decision to stop, so `ablation2.txt` carries all five of its variants.

## What is here

| File | What it does |
|---|---|
| `common.jl` | Priors, data, initial state, and the effective-sample-size and posterior-comparison reporting every script uses |
| `seit4l_sufficient.jl` | A SEIT4L simulator that also records the sufficient statistics of the complete-data likelihood, and the state-space plumbing that carries them through a filter |
| `validate_sufficient.jl` | Three checks on that simulator: rate recovery, that the likelihood peaks at the estimates, and that its incidence matches the course simulator |
| `pg.jl` | Particle Gibbs: a conditional SMC sweep through `ref_state`, a light ancestry callback, and an adaptive parameter step on the complete-data likelihood |
| `pmmh_plain.jl` | PMMH as a plain loop sharing the same parameter step, so a comparison is not measuring Turing's overhead |
| `pmmh_variants.jl` | One sampling loop with the proposal covariance shape, its adaptation, and the scale it is proposed on all switchable |
| `smoke_pg.jl` | Quick checks on a conditional SMC sweep before spending a chain on it |
| `bench_pmmh.jl` | The baseline: PMMH exactly as `scripts/pmmh_setup.jl` runs it |
| `bench_all.jl` | Particle Gibbs at four particle counts against PMMH at three |
| `bench_followup.jl` | Whether the parameter step limits particle Gibbs, and repeat seeds for PMMH |
| `ablation.jl` | Which part of the parameter step carries the difference |
| `ablation2.jl` | Repeat seeds for the best variant, and whether a starting scale rescues RAM |
| `gf05_check.jl` | Written but never run. See below. |

The `.txt` file beside each script is its output, and carries every number quoted in the findings.
They are named `.txt` rather than `.log` because the repository ignores `*.log`.
The saved draws are not committed; see `.gitignore`.

## Findings

Iterations per effective draw on the worst-mixing parameter is the primary measure.
Wall clock on the machine these ran on drifted by up to a factor of two between runs, while iterations per effective draw reproduced across seeds and chain lengths.
Cost per iteration is within 5% between particle Gibbs and PMMH at equal particle count, so the ratio carries over to effective sample size per second.

Every posterior below was checked against `data/pmcmc_seit4l_chain.csv`, and the two marked wrong are wrong.

| Sampler, 128 particles | iters/eff draw |
|---|---|
| PMMH, Turing + RAM, log/logit — what the course runs | 247 |
| Particle Gibbs | 287 |
| PMMH, RAM, log/logit, reimplemented in a plain loop | 245 |
| PMMH, RAM, log/logit, given a starting scale | 177 |
| PMMH, RAM, constrained scale, identity initial factor | 487, and wrong |
| PMMH, RAM, constrained scale, given a starting scale | 39 |
| PMMH, empirical covariance + Robbins-Monro scale, log/logit | 62-80 |
| PMMH, empirical covariance + Robbins-Monro scale, constrained scale | 35-36 |
| PMMH, empirical covariance, diagonal only, log/logit | 169 |
| PMMH, 32 particles, any proposal | 490, and wrong |

Particle Gibbs is correct and slower.
Its effective sample size per second is flat in the particle count, because more particles buys proportionally better path mixing and costs proportionally more.
Sweeping the number of parameter steps per sweep from 1 to 1000 moves it from 0.21 to 0.30 effective draws per second and saturates by 200, so an exact parameter draw given the path still leaves it well behind PMMH, and a gradient-based parameter step cannot recover the gap.

The scale the proposal is made on is what costs the course its mixing.
Turing's `externalsampler` defaults to `unconstrained=true`, which puts RAM on `log(x - lower)` and `logit` coordinates.
Three of the six parameters have posteriors sitting close to their prior's lower bound, so that transform stretches a left tail the proposal then has to cover.

Holding the sampler fixed and varying only the scale, RAM given a starting scale goes from 177 iterations per effective draw on log/logit to 39 on the constrained parameters, and the empirical-covariance proposal from about 70 to about 35.
Holding the scale fixed and varying only the adaptation, the two proposals differ by a factor of two and a half on log/logit and are within 10% of each other on the constrained scale.
So the transform carries most of the difference, the adaptation carries the rest, and the adaptation only matters because the transform made the geometry hard.

That points at a small change rather than a new sampler. Passing `unconstrained=false` to `externalsampler`, with a starting scale for RAM, keeps the session's Turing shape and its sampler and takes 247 iterations per effective draw to 39.
The starting scale used here was half the posterior standard deviations, which is information you do not have before sampling; whether prior standard deviations serve as well is the obvious next question and was not measured.

Two failure modes are worth keeping in view.
PMMH at 32 particles looks two and a half times faster than the baseline and returns posterior standard deviations two to three times too narrow, because the log-likelihood estimate has a standard deviation of 6.1 there and the chain sticks on overestimates.
RAM on the constrained scale with its default identity initial factor collapses to 0.1% acceptance, because a proposal of unit standard deviation is hopeless for parameters living in the unit interval.

## A note on the library

`DenseAncestorCallback` calls `deepcopy` on every particle's state at every step.
On a plain `Vector{Float64}` that still goes through the generic machinery and its `IdDict`, and it doubled the cost of a conditional SMC sweep here, from 0.028 s to 0.013 s once replaced.
`PathCallback` in `pg.jl` is the same callback without the copy, which is safe because a particle's state is allocated fresh by the dynamics and never written to afterwards.
Anyone using ancestry storage in GeneralisedFilters will meet this.

## GeneralisedFilters 0.5

`gf05_check.jl` was written and never run, and the pin it needs was never applied, so `Project.toml` and `Manifest.toml` are untouched and the branch still resolves 0.4.2.

Running it needs the compat bound in `Project.toml` relaxed to `"0.4, 0.5"` and

```julia
Pkg.add(url="https://github.com/TuringLang/SSMProblems.jl",
        subdir="GeneralisedFilters",
        rev="6ef3b0855759649984813757d7e0765404c6952a")
```

0.5 is unreleased, so nothing depending on it can reach the course yet.
Reading its source rather than running it: the signatures of every method `ThreadedBF` overrides or forwards are unchanged from 0.4.2, so it should port without edits, though that is unverified.
`ConditionalSMC(pf)`, `ConditionalSMC(pf, AncestorSampling())` and `BackwardSimulation()` are all there, and there is a `ParticleGibbs(csmc, NUTS(0.8))` that packages the composite sampler.

Ancestor sampling is expected to fail against these dynamics, and `gf05_check.jl` is written to confirm that concretely rather than assume it.
It computes ancestor weights from `SSMProblems.logdensity` of the latent dynamics, which for a Gillespie simulator would have to be the one-day transition density marginalised over every jump path, and that is not available.
Supplying the complete-data density instead would not rescue it: the reference's recorded event counts and exposure integrals belong to the state it started from, so evaluated against any other candidate ancestor they are inconsistent, and the weights degenerate rather than merely erroring.
