using MFIIDD, CSV, DataFrames, DrWatson, Random, Statistics
flu = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
θc = Dict(:R_0 => 6.0, :D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7)
base = Dict(:D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7,
            :logβ0 => log(3.0))
println("nchunks=4, 128 particles, min of 30 runs")
tc = minimum(@elapsed(run_particle_filter(θc, flu.obs, 128; threaded = true)) for _ in 1:30)
println("  constant     ", round(tc, digits = 4), " s")
for σ in (0.0, 0.1, 0.2, 0.3)
    θ = merge(base, Dict(:σ => σ))
    t = minimum(@elapsed(run_filter_rw(θ, flu.obs, 128; nchunks = 4)) for _ in 1:30)
    println("  walk σ=", rpad(σ, 5), " ", rpad(round(t, digits = 4), 8),
            " s   ratio ", round(t / tc, digits = 2))
end
## and the arm B path filter, which does the same epidemic work with no walk
lb = fill(log(3.0), 59)
θp = Dict(:D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7)
tp = minimum(@elapsed(run_filter_path(θp, lb, flu.obs, 128; nchunks = 4)) for _ in 1:30)
println("  fixed path   ", round(tp, digits = 4), " s   ratio ", round(tp / tc, digits = 2))
