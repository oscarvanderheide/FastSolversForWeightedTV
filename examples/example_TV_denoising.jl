using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, CUDA, PyPlot
CUDA.allowscalar(false)
using Random; Random.seed!(123)

# Random data
n = (256, 256)
y_orig = Float32.(TestImages.shepp_logan(n...)) |> gpu
y_orig = y_orig/norm(y_orig, Inf)

# TV norm
g = TV_norm_2D() |> gpu
ε_orig = g(y_orig)

# Artificial noise
y = y_orig+gpu(0.1f0*randn(Float32, n))

# Optimization options
opt = opt_fista(; steplength=1f0/8f0, niter=2000, tol_x=nothing, nesterov=true)

# Proxy
λ = 0.5f0*norm(y)^2/g(y)
xproxy = proxy(y, λ, g, opt)
err_proxy = norm(xproxy-y_orig)/norm(y_orig)
println("Proxy rel err: ", err_proxy)

# Projection test
ε = ε_orig
xproj = project(y, ε, g, opt)
err_proj = norm(xproj-y_orig)/norm(y_orig)
println("Proj rel err: ", err_proj)

# Plot proxy
vmin = 0f0
vmax = 1f0
close("all")
figure()
subplot(1,2,1)
imshow(cpu(y); cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(1,2,2)
imshow(cpu(xproxy); cmap="gray", vmin=vmin, vmax=vmax)
title(string("TV proxy denoising, ", L"$\mathrm{err}_{\mathrm{rel}}$ = ", string(err_proxy)))
axis("off")
savefig("./plots/TVprox.png", dpi=300, transparent=false, bbox_inches="tight")

# Plot proxy
figure()
subplot(1,2,1)
imshow(cpu(y); cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(1,2,2)
imshow(cpu(xproj); cmap="gray", vmin=vmin, vmax=vmax)
title(string("TV projection denoising, ", L"$\mathrm{err}_{\mathrm{rel}}$ = ", string(err_proj)))
axis("off")
savefig("./plots/TVproj.png", dpi=300, transparent=false, bbox_inches="tight")