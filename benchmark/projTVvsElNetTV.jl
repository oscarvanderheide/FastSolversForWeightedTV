using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Load image
y = Float32.(readdlm("./data/T1.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint sets
ε_rel = 0.5f0
μ = 1.5f0
ε_tv = ε_rel*normTV(y)
ε_elnet = ε_rel*normElNetTV(y, μ)
p0 = zeros(Float32, size(y)..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=3000, nesterov=true)
C_tv = tv_ball_2D(n, ε_tv; opt=opt, gpu=flag_gpu)
C_elnet = elnettv_ball_2D(n, μ, ε_elnet; opt=opt, gpu=flag_gpu)

# Projection
x_tv, fval_hist_tv, err_rel_tv = project_debug(y, C_tv)
x_elnet, fval_hist_elnet, err_rel_elnet = project_debug(y, C_elnet)

# Plot
figure()
loglog((fval_hist_tv[1:2000].-fval_hist_tv[end])./(fval_hist_tv[1].-fval_hist_tv[end]))
loglog((fval_hist_elnet[1:2000].-fval_hist_elnet[end])./(fval_hist_elnet[1].-fval_hist_elnet[end]))
legend(("TV", "Elastic net TV"))