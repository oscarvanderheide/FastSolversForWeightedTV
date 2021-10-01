using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, CUDA, PyPlot, Statistics
CUDA.allowscalar(false)
using Random; Random.seed!(123)

# Random data
n = (64, 64)
y_orig = Float32.(TestImages.shepp_logan(n...))
y_orig = y_orig/norm(y_orig, Inf)
nc = 3; nb = 4
y_orig = repeat(y_orig; outer=(1,1,nc,nb)) |> gpu

# Artificial noise
y = y_orig+gpu(0.1f0*randn(Float32, size(y_orig)))

# TV norm
g = TV_norm_batch_2D() |> gpu

# Optimization options
opt = opt_fista(; steplength=1f0/8f0, niter=100, tol_x=nothing, nesterov=true)

# Proxy
ny2 = mean(sum(y_orig.^2; dims=(1,2)))
λ = 0.5f0*ny2/mean(g(y))
xproxy = proxy(y, λ, g, opt) |> cpu

# Plot proxy
figure()
for i=1:nc, j =1:nb
    subplot(nc,nb,(j-1)*nc+i)
    imshow(xproxy[:,:,i,j]; cmap="gray", vmin=0, vmax=1)
    title(string("nc=", i, ", nb=", j))
    axis("off")
end
savefig("./plots/TVprox_batch.png", dpi=300, transparent=false, bbox_inches="tight")