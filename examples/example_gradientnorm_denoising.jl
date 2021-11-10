using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, CUDA, PyPlot
CUDA.allowscalar(false)
using Random; Random.seed!(123)

# Random data
# dim = 2
dim = 3
n = tuple(256*ones(Int64, dim)...)
y_orig = Float32.(TestImages.shepp_logan(n[1:2]...))
dim == 3 && (y_orig = repeat(y_orig; outer=(1,1,n[3])))
y_orig = y_orig |> gpu
y_orig = y_orig/norm(y_orig, Inf)

# Weight for structural norm
# η = 0.1f0*structural_mean(y_orig)
# P = structural_weight(y_orig; η=η)
P = nothing

# Gradient norm
N = (2,1); method = "TV"
# N = (2,2); method = "L2V"
# N = (2,Inf); method = "LInfV"
~isnothing(P) && (method = string(method, "w"))
method = string(method, "_", dim, "D")
h = tuple(ones(Float32, dim)...)
g = gradient_norm(N..., n, h; weight=P) |> gpu
ε_orig = g(y_orig)
dim == 2 && (steplength = 1f0/8f0)
dim == 3 && (steplength = 1f0/12f0)
opt = opt_fista(steplength; niter=1000, tol_x=nothing, nesterov=true)

# Artificial noise
y = y_orig+gpu(0.1f0*randn(Float32, n))

# Proxy
λ = 0.5f0*norm(y)^2/g(y)
xproxy = proxy(y, λ, g, opt)
err_proxy = norm(xproxy-y_orig)/norm(y_orig)
println("Proxy rel err: ", err_proxy)

# Projection test
ε = 0.5f0*ε_orig
xproj = project(y, ε, g, opt)
err_proj = norm(xproj-y_orig)/norm(y_orig)
println("Proj rel err: ", err_proj)

# Plot proxy
vmin = 0f0
vmax = 1f0
close("all")
figure()
subplot(1,2,1)
dim == 2 && imshow(cpu(y); cmap="gray", vmin=vmin, vmax=vmax)
dim == 3 && imshow(cpu(y[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(1,2,2)
dim == 2 && imshow(cpu(xproxy); cmap="gray", vmin=vmin, vmax=vmax)
dim == 3 && imshow(cpu(xproxy[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title(string(method, " proxy, ", L"$\mathrm{err}_{\mathrm{rel}}$ = ", string(err_proxy)))
axis("off")
savefig(string("./plots/", method, "_prox.png"), dpi=300, transparent=false, bbox_inches="tight")

# Plot proxy
figure()
subplot(1,2,1)
dim == 2 && imshow(cpu(y); cmap="gray", vmin=vmin, vmax=vmax)
dim == 3 && imshow(cpu(y[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(1,2,2)
dim == 2 && imshow(cpu(xproj); cmap="gray", vmin=vmin, vmax=vmax)
dim == 3 && imshow(cpu(xproj[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title(string(method, " proj., ", L"$\mathrm{err}_{\mathrm{rel}}$ = ", string(err_proj)))
axis("off")
savefig(string("./plots/", method, "_proj.png"), dpi=300, transparent=false, bbox_inches="tight")