using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Create noise image
n = (256, 256)
include("../data/shepp_logan.jl")
y = Float32.(shepp_logan(n...))+0.1f0*randn(Float32, n); flag_gpu && (y = y |> gpu)

# Proximal operator setup
λ = 0.1f0
opt = opt_fista(; niter=10000, nesterov=true, tol_x=1f-5)
g = tv_norm_2D(Float32, n; opt=opt, gpu=flag_gpu)

# Proximal operator eval
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