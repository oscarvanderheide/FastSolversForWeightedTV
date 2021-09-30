using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, BenchmarkTools
CUDA.allowscalar(false)
include("./test_utils.jl")

# flag_gpu = true
flag_gpu = false

# Random data
T = Float64
n = (256, 256)
y = randn(T, n..., 2); flag_gpu && (y = y |> gpu)

# Norm
for g = [norm_2D(2,2; T=T), norm_2D(2,1; T=T), norm_2D(2,Inf; T=T)]

    # Proxy
    λ = norm(y)^2/g(y)
    x = proxy(y, λ, g)

    # Gradient test
    fun = proxy_objfun(λ, g)
    test_grad(fun, y; step=T(1e-4), rtol=T(1e-5))

    # Projection test
    ε = T(0.5)*g(y)
    x = project(y, ε, g)
    @test g(x) ≈ ε rtol=T(1e-5)

    ## Gradient test
    fun = proj_objfun(ε, g)
    test_grad(fun, y; step=T(1e-4), rtol=T(1e-5))

end