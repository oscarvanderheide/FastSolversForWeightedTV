using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Load image
y = Float32.(readdlm("./data/T1.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint set
ε_rel = 0.5f0
ε = ε_rel*normTV(y)
C = tv_ball_2D(n, ε; gpu=flag_gpu)

# Projection
p0 = zeros(Float32, n..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=1000, nesterov=true)
@benchmark x, p = project(y, C; opt=opt, dual_est=true) |> cpu
y = y |> cpu