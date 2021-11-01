using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, BenchmarkTools, PyPlot
using CUDA; CUDA.allowscalar(false)
using Random; Random.seed!(123)

flag_gpu = true
# flag_gpu = false

# Optimization options
opt = opt_fista(; steplength=1f0/12f0, niter=5, tol_x=nothing, nesterov=true)

# TV norm
n = (256,256,256)
g = TV_norm_3D(n); flag_gpu && (g = g |> gpu)

# Numerical phantom
y = repeat(Float32.(TestImages.shepp_logan(n[1:2]...)); outer=(1,1,n[3])); flag_gpu && (y = y |> gpu)
y = y/norm(y, Inf)
flag_gpu ? (y .+= gpu(0.1f0*randn(Float32, n))) : (y .+= 0.1f0*randn(Float32, n))

# Proxy
λ = 0.5f0*norm(y)^2/g(y)
@btime proxy(y, λ, g, opt);

# Projection test
ε = 0.9f0*g(y)
@btime project(y, ε, g, opt);