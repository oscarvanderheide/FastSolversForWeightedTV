using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Load image
y = Float32.(readdlm("./data/T1.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint set
ε_rel = 0.5f0
ε = ε_rel*normTV(y)
C = tv_ball_2D(n, ε; gpu=flag_gpu)

# Projection
p0 = zeros(Float32, size(y)..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=1000, nesterov=true)
x, p = project(y, C, opt; dual_est=true) |> cpu
y = y |> cpu

# Plot
vmin=min(y...)
vmax=max(y...)
close("all")
figure()
subplot(1,2,1)
imshow(y; cmap="gray", vmin=vmin, vmax=vmax)
title("Original")
subplot(1,2,2)
imshow(x; cmap="gray", vmin=vmin, vmax=vmax)
title(string("TV projection, ", L"niter = ", string(opt.niter), ", ", L"\varepsilon = ", string(ε_rel)))
savefig("./plots/TVproj.png", dpi=300, transparent=false, bbox_inches="tight")