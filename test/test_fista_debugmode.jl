using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Load image
y = Float32.(readdlm("./data/T1.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint set
ε_rel = 0.1f0
ε = ε_rel*normTV(y)
p0 = zeros(Float32, size(y)..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=0.1f0/8f0, niter=100, nesterov=true)
C = tv_ball_2D(n, ε; opt=opt, gpu=flag_gpu)

# Projection
_, fval_hist1, err_rel1 = project_debug(y, C)
C.opt.nesterov = false
_, fval_hist2, err_rel2 = project_debug(y, C)

# Plot
figure()
plot(fval_hist1)
plot(fval_hist2)
legend(("Nesterov on", "Nesterov off"))
figure()
plot(err_rel1)
plot(err_rel2)
legend(("Nesterov on", "Nesterov off"))