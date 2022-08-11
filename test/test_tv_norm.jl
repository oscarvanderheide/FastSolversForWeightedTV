using LinearAlgebra, FastSolversForWeightedTV, ConvexOptimizationUtils, CUDA, Flux, Test, Random
Random.seed!(123)
CUDA.allowscalar(false)
include("./test_utils.jl")


## 1-D

# flag_gpu = true
flag_gpu = false

# Random data
T = Float64
CT = Complex{T}
step = T(1e-6)
rtol = T(1e-3)
n = (32,)
y = randn(CT, n); flag_gpu && (y = y |> gpu)
h = (T(1),)

g = gradient_norm(2, 1, n, h; complex=true); flag_gpu && (g = g |> gpu)

# Optimization options
niter = 1000
opt = FISTA_optimizer(T(4); Nesterov=true, niter=niter, reset_counter=20, verbose=false)

# Gradient test (proxy)
λ = T(0.5)*norm(y)^2/g(y)
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=step, rtol=rtol)

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=rtol

# Gradient test (projection)
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=step, rtol=rtol)


## 2-D

# Random data
T = Float64
CT = Complex{T}
n = (8,8)
y = randn(CT, n); flag_gpu && (y = y |> gpu)
h = (T(1),T(1))

g = gradient_norm(2, 1, n, h; complex=true); flag_gpu && (g = g |> gpu)

# Optimization options
niter = 2000
opt = FISTA_optimizer(T(8); Nesterov=true, niter=niter, reset_counter=20, verbose=false)

# Gradient test (proxy)
λ = T(0.5)*norm(y)^2/g(y)
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=step, rtol=rtol)

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=rtol

# Gradient test
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=step, rtol=rtol)


## 3-D

# Random data
n = (8,8,8)
h = (T(1),T(1),T(1))
y = randn(T, n[1:2]); flag_gpu && (y = y |> gpu)
y = CT.(repeat(y; outer=(1,1,8)))

g = gradient_norm(2, 1, n, h; complex=true); flag_gpu && (g = g |> gpu)

# Optimization options
niter = 4000
opt = FISTA_optimizer(T(12); Nesterov=true, niter=niter, reset_counter=10, verbose=false)

# Gradient test (proxy)
λ = T(0.5)*norm(y)^2/g(y)
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=step, rtol=rtol)

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=rtol

# Gradient test (projection)
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=step, rtol=rtol)