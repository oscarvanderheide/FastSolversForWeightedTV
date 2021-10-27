using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, TestImages
CUDA.allowscalar(false)
include("./test_utils.jl")


## 2-D

# flag_gpu = true
flag_gpu = false

# Random data
T = Float64
# T = Float32
n = (8,8)
y = T.(TestImages.shepp_logan(n...)); flag_gpu && (y = y |> gpu)

g = TV_norm_2D(; T=T); flag_gpu && (g = g |> gpu)

# Optimization options
opt = opt_fista(; steplength=T(1/8), niter=100000, tol_x=nothing, nesterov=true)

# Proxy
λ = T(2)*T(0.5)*norm(y)^2/g(y)
x = proxy(y, λ, g, opt)

# Gradient test
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=T(1e-5), rtol=T(1e-3))

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=T(1e-4)

# Gradient test
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=T(1e-5), rtol=T(1e-4))


## 3-D

# flag_gpu = true
flag_gpu = false

# Random data
T = Float64
# T = Float32
n = (8,8,8)
y = T.(TestImages.shepp_logan(n[1:2]...)); flag_gpu && (y = y |> gpu)
y = repeat(y; outer=(1,1,8))

g = TV_norm_3D(; T=T); flag_gpu && (g = g |> gpu)

# Optimization options
opt = opt_fista(; steplength=T(1/24), niter=100000, tol_x=nothing, nesterov=true)

# Proxy
λ = T(2)*T(0.5)*norm(y)^2/g(y)
x = proxy(y, λ, g, opt)

# Gradient test
fun = proxy_objfun(λ, g, opt)
test_grad(fun, y; step=T(1e-4), rtol=T(1e-3))

# Projection test
ε = T(0.1)*g(y)
x = project(y, ε, g, opt)
@test g(x) ≈ ε rtol=T(1e-4)

# Gradient test
fun = proj_objfun(ε, g, opt)
test_grad(fun, y; step=T(1e-5), rtol=T(1e-4))