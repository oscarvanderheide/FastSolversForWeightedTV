using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Load image
y = Float32.(readdlm("./data/T1.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Proximal operator setup
λ = 0.1f0
p0 = zeros(Float32, size(y)..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=3000, nesterov=true)
g = tv_norm_2D(Float32, n; opt=opt, gpu=flag_gpu)

# Projection
x = proxy(y, λ, g) |> cpu
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
title(string("TV proximal, ", L"niter = ", string(opt.niter), ", ", L"\lambda = ", string(λ)))
savefig("./plots/TVprox.png", dpi=300, transparent=false, bbox_inches="tight")