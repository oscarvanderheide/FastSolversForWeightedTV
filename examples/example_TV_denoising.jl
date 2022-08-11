using LinearAlgebra, FastSolversForWeightedTV, ConvexOptimizationUtils, Flux, TestImages, PyPlot
using Random; Random.seed!(123)

flag_gpu=true
# flag_gpu=false

# Random data
n = (256, 256, 256)
y_orig = Float32.(TestImages.shepp_logan(n[1:2]...))
y_orig = repeat(y_orig; outer=(1,1,n[3]))
y_orig = y_orig/norm(y_orig, Inf)

# Weight for structural norm
η = 0.1f0*structural_mean(y_orig)
P = structural_weight(y_orig; η=η)

# Gradient norms
h = (1f0, 1f0, 1f0)
g_sTV = gradient_norm(2, 1, n, h; weight=P)
g_TV  = gradient_norm(2, 1, n, h; weight=nothing)
opt = FISTA_optimizer(12f0; Nesterov=true, niter=100, reset_counter=10, verbose=false)

# Artificial noise
y = y_orig+0.1f0*randn(Float32, n)

# Moving arrays to gpu
if flag_gpu
    y = y |> gpu
    g_sTV = g_sTV |> gpu
    g_TV = g_TV |> gpu
end

# Proxy
λ_sTV = 0.5f0*norm(y)^2/g_sTV(y)
λ_TV  = 0.5f0*norm(y)^2/g_TV(y)
xproxy_sTV = proxy(y, λ_sTV, g_sTV, opt) |> cpu
xproxy_TV  = proxy(y, λ_TV, g_TV, opt) |> cpu

# Plot proxy
vmin = 0f0
vmax = 1f0
close("all")
figure()
subplot(1,3,1)
imshow(y[:,:,128]; cmap="gray", vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(1,3,2)
imshow(xproxy_TV[:,:,128]; cmap="gray", vmin=vmin, vmax=vmax)
title("TV denoising (z=128)")
axis("off")
subplot(1,3,3)
imshow(xproxy_sTV[:,:,128]; cmap="gray", vmin=vmin, vmax=vmax)
title("sTV denoising (z=128)")
axis("off")
savefig(string("./plots/sTVvsTV_proxy.png"), dpi=300, transparent=false, bbox_inches="tight")