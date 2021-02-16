using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Load image
y = Float32.(readdlm("./data/T1.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint set
ε_rel = 0.5f0
μ = 0.1f0
ε = ε_rel*normElNetTV(y, μ)
p0 = zeros(Float32, size(y)..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=1000, nesterov=true)
C = elnettv_ball_2D(n, μ, ε; opt=opt, gpu=flag_gpu)

# Projection
x1, fval_hist1, err_rel1 = project_debug(y, C)
C.opt.nesterov = false
x2, fval_hist2, err_rel2 = project_debug(y, C)

# Plot
figure()
plot(fval_hist1)
plot(fval_hist2)
legend(("Nesterov on", "Nesterov off"))
figure()
plot(err_rel1)
plot(err_rel2)
legend(("Nesterov on", "Nesterov off"))