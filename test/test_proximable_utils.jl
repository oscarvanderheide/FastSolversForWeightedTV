using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Random data
n = (3,3)
h = (abs(randn(Float32)), abs(randn(Float32)))
x1 = randn(Float32, n)
x2 = randn(Float32, n)
y = cat(x1, x2; dims=3); flag_gpu && (y = y |> gpu)

# || ||_{2,1} norm
gfun = ell_norm(Float32, 2, 1)

# Proxy || ||_{2,1} norm
λ = 0.1f0
gval, x = proxy(y, λ, gfun)
@test gval ≈ norm21(y) rtol=1f-3
ptnorm_y = ptnorm2(y)
@test x ≈ (ptnorm_y.-λ).*y./ptnorm_y.*(ptnorm_y .>= λ) rtol=1f-3

# Multiplication with scalar
gval1, x1 = proxy(y, 3f0*λ, gfun)
gval2, x2 = proxy(y, λ,     3f0*gfun)
@test gval1 ≈ gval2 rtol=1f-3
@test x1 ≈ x2 rtol=1f-3