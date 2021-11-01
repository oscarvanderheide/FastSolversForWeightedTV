using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, BenchmarkTools
using CUDA; CUDA.allowscalar(false)
using Random; Random.seed!(123)

# Optimization options
flag_gpu = true
# flag_gpu = false
opt = opt_fista(; steplength=1f0/8f0, niter=100, tol_x=nothing, nesterov=true)

# TV norm
n = (256, 256)
g = TV_norm_2D(n); flag_gpu && (g = g |> gpu)

# Numerical phantom
y = Float32.(TestImages.shepp_logan(n...)); flag_gpu && (y = y |> gpu)
y = y/norm(y, Inf)
y_clean = deepcopy(y)
flag_gpu ? (y .+= gpu(0.1f0*randn(Float32, n))) : (y .+= 0.1f0*randn(Float32, n))

# Proxy
λ = 0.5f0*norm(y)^2/g(y)
@btime proxy(y, λ, g, opt);

# Projection test
ε = 0.9f0*g(y_clean)
@btime project(y, ε, g, opt);