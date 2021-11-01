using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, PyPlot
using Random; Random.seed!(123)

# Random data
n = (256, 256, 256)
y_orig = repeat(Float32.(TestImages.shepp_logan(n[1:2]...)); outer=(1,1,n[3])) |> gpu
y_orig[:,:,1:50] .= 0f0
y_orig[:,:,end-50:end] .= 0f0
y_orig = y_orig/norm(y_orig, Inf)

# TV norm
g = TV_norm_3D(n) |> gpu
ε_orig = g(y_orig)

# Artificial noise
y = y_orig+0.1f0*gpu(randn(Float32, n))

# Optimization options
opt = opt_fista(; steplength=1f0/12f0, niter=1000, tol_x=nothing, nesterov=true)

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
imshow(cpu(y[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom (3D)")
axis("off")
subplot(1,2,2)
imshow(cpu(xproxy[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title(string("TV proxy (3D), ", L"$\mathrm{err}_{\mathrm{rel}}$ = ", string(err_proxy)))
axis("off")
savefig("./plots/TV3Dprox.png", dpi=300, transparent=false, bbox_inches="tight")

# Plot projection
figure()
subplot(1,2,1)
imshow(cpu(y[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom (3D)")
axis("off")
subplot(1,2,2)
imshow(cpu(xproj[:,:,128]); cmap="gray", vmin=vmin, vmax=vmax)
title(string("TV proj. (3D), ", L"$\mathrm{err}_{\mathrm{rel}}$ = ", string(err_proj)))
axis("off")
savefig("./plots/TV3Dproj.png", dpi=300, transparent=false, bbox_inches="tight")