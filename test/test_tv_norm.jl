using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)
include("./test_utils.jl")


## 1-D

# flag_gpu = true
flag_gpu = false

# Random data
T = Float64
CT = Complex{T}
step = T(1e-5)
rtol = T(1e-4)
n = (32,)
y = randn(CT, n); flag_gpu && (y = y |> gpu)
h = (T(1),)

g = gradient_norm(2, 1, n, h; complex=true); flag_gpu && (g = g |> gpu)

# Optimization options
niter = 1000
opt = opt_fista(T(1/4); niter=niter, tol_x=nothing, Nesterov=true)

# Proxy
λ = T(0.5)*norm(y)^2/g(y)
x = proxy(y, λ, g, opt)

# Gradient test
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=step, rtol=rtol)

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=rtol

# Gradient test
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=step, rtol=rtol)


## 2-D

# Random data
T = Float64
CT = Complex{T}
step = T(1e-5)
rtol = T(1e-4)
n = (8,8)
y = randn(CT, n); flag_gpu && (y = y |> gpu)
h = (T(1),T(1))

g = gradient_norm(2, 1, n, h; T=CT); flag_gpu && (g = g |> gpu)

# Optimization options
niter = 1000
opt = opt_fista(T(1/8); niter=niter, tol_x=nothing, Nesterov=true)

# Proxy
λ = T(0.5)*norm(y)^2/g(y)
x = proxy(y, λ, g, opt)

# Gradient test
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
niter = 1000
opt = opt_fista(T(1/12); niter=niter, tol_x=nothing, Nesterov=true)

# Proxy
λ = T(0.5)*norm(y)^2/g(y)
x = proxy(y, λ, g, opt)

# Gradient test
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=step, rtol=rtol)

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=rtol

# Gradient test
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=step, rtol=rtol)