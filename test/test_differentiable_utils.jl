using LinearAlgebra, FastSolversForWeightedTV, AbstractLinearOperators, Test

flag_gpu = true
# flag_gpu = false

# Random data
n = (1001, 2001)
h = (abs(randn(Float32)), abs(randn(Float32)))
x1 = randn(Float32, n)
x2 = randn(Float32, n)
y = cat(x1, x2; dims=3); flag_gpu && (y = y |> gpu)

# Linear operator
h = (1f0,1f0)
A = gradient_2D(n; h=h, gpu=flag_gpu)

# Least-square objective
fun = leastsquares_misfit(A, y)

# Gradient test
x = randn(Float32, n); flag_gpu && (x = x |> gpu)
g = similar(x)
fval = grad!(fun, x, g)
@test fval ≈ 0.5f0*norm(A*x-y)^2 rtol=1f-3
@test g ≈ adjoint(A)*(A*x-y) rtol=1f-3