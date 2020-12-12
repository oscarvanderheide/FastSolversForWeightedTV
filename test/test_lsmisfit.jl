using LinearAlgebra, VectorFields, DifferentialOperatorsForTV, FastSolversForWeightedTV, Flux, Test

flag_gpu = true
# flag_gpu = false

# Random data
n = (1001, 2001)
h = (abs(randn(Float32)), abs(randn(Float32)))
x1 = to_field(randn(Float32, n))
x2 = to_field(randn(Float32, n))
y = [x1; x2]; flag_gpu && (y = y |> gpu)

# Linear operator
h = (1f0,1f0)
A = gradient_op(n; h=h, flag_gpu=flag_gpu)

# Least-square objective
fun = LeastSquaresMisfit(A, y)

# Gradient test
x = to_field(randn(Float32, n)); flag_gpu && (x = x |> gpu)
f, g = grad(fun, x)
@test f ≈ 0.5f0*norm(A*x-y)^2 rtol=1f-3
@test g ≈ adjoint(A)*(A*x-y) rtol=1f-3