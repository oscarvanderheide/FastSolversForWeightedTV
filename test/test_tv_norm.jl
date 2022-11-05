using LinearAlgebra, FastSolversForWeightedTV, ConvexOptimizationUtils, CUDA, Flux, Test, Random
Random.seed!(123)
CUDA.allowscalar(false)

T = Float64
n1d = 16
nc = 3
step = T(1e-5)
rtol = T(1e-2)

for dim = 1:3, flag_gpu = [false], is_complex = [true, false]
    
    # FISTA optimizer
    niter = 1000
    opt = FISTA_optimizer(T(4*dim); Nesterov=true, niter=niter, reset_counter=20, verbose=false)

    # TV norm
    CT = is_complex ? Complex{T} : T
    n = tuple(repeat([n1d,]; outer=dim)...)
    h = tuple(ones(T, dim)...)
    y = randn(CT, n); flag_gpu && (y = CT.(y |> gpu))
    g = gradient_norm(2, 1, n, h; complex=is_complex, optimizer=opt); flag_gpu && (g = g |> gpu)

    # Constraint
    C = zero_set(CT, randn(T, size(y)).>0)

    # Gradient test (proxy)
    λ = T(0.5)*norm(y)^2/g(y)
    fun = proxy_objfun(g+indicator(C), λ)
    @test test_grad(fun, y; step=step, rtol=rtol)

    # Projection test
    ε = T(0.1)*g(y)
    x = project(y, ε, g+indicator(C))
    @test (g(x) ≤ ε) || isapprox(g(x), ε; rtol=rtol)
    @test x ∈ C

    # Gradient test (projection)
    fun = proj_objfun(g+indicator(C), ε)
    @test test_grad(fun, y; step=step, rtol=rtol)

end