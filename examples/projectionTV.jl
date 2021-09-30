using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, TestImages
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Create noise image
n = (256, 256)
y_clean = Float32.(TestImages.shepp_logan(n...))
y = y_clean+0.1f0*randn(Float32, n)
flag_gpu && (y = y |> gpu)

# # Projection operator setup
# opt = opt_fista(; niter=10000, nesterov=true, tol_x=1f-5)
# g = tv_norm_2D(Float32, n; opt=opt, gpu=false)
# ε = g(y_clean)
# C = tv_ball_2D(n, ε; gpu=flag_gpu, opt=opt)

# Projection operator setup
niter_all = 10000
niter_inner = 10
niter_outer = Int64(niter_all/niter_inner)
opt_proxy = opt_fista(; niter=niter_inner, nesterov=true)
opt = opt_adaptive_proxy(; niter=niter_outer, tol_x=1f-5, opt_proxy=opt_proxy)
g = tv_norm_2D(Float32, n; opt=opt_proxy, gpu=false)
ε = g(y_clean)
C = tv_ball_2D(n, ε; gpu=flag_gpu, opt=opt)

# Projection operator eval
x = project(y, C) |> cpu
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
title(string("TV projection, ", L"niter = ", string(opt.niter), ", ", L"\varepsilon = ", string(ε)))
savefig("./plots/TVproj.png", dpi=300, transparent=false, bbox_inches="tight")