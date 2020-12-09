using LinearAlgebra, VectorFields, DifferentialOperatorsForTV, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Random data
n = (3,3)
h = (abs(randn(Float32)), abs(randn(Float32)))
x1 = toField(randn(Float32, n))
x2 = toField(randn(Float32, n))
y = [x1; x2]; flag_gpu && (y = y |> gpu)

# || ||_{2,1} norm
gfun = VectorFieldNorm{Float32,2,1}()

# Proxy || ||_{2,1} norm
λ = 0.1f0
gval, x = proxy(λ, gfun, y)
@test gval ≈ norm(y; p1=2, p2=1) rtol=1f-3
ptnorm_y = ptnorm(y; p=2)
@test x ≈ (ptnorm_y-λ)*y/ptnorm_y*(ptnorm_y >= λ) rtol=1f-3

# Multiplication with scalar
gval1, x1 = proxy(3f0*λ, gfun,     y)
gval2, x2 = proxy(λ,     3f0*gfun, y)
@test gval1 ≈ gval2 rtol=1f-3
@test x1 ≈ x2 rtol=1f-3

# Convex conjugation
gval1, x1 = proxy(λ,     conjugate(gfun), y)
gval2, p2 = proxy(1f0/λ, gfun,            y/λ); x2 = y-λ*p2
@test x1 ≈ x2 rtol=1f-3