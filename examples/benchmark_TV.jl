using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, BenchmarkTools
using Random; Random.seed!(123)

# Optimization options
opt = opt_fista(; steplength=1f0/8f0, niter=10000, tol_x=nothing, nesterov=true)

# TV norm
g = TV_norm_2D() |> gpu

# Numerical phantom
n = (256, 256)
y_orig = Float32.(TestImages.shepp_logan(n...)) |> gpu
y_orig = y_orig/norm(y_orig, Inf)
y = y_orig+gpu(0.1f0*randn(Float32, n))

# Proxy
λ = 0.5f0*norm(y)^2/g(y)
@btime proxy(y, λ, g, opt);

# Projection test
ε = 0.9f0*g(y)
@btime project(y, ε, g, opt);