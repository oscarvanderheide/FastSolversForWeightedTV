using TotalVariationRegularization, LinearAlgebra, Flux, Test

# Random input
n = (1001, 2001)
h = (abs(randn(Float32)), abs(randn(Float32)))

# Operators
flag_gpu = true
# flag_gpu = false
Dx, Dy = derivative_ops(n; h=h, flag_gpu=flag_gpu)
Δ = laplacian_op(n; h=h, flag_gpu=flag_gpu)
∇ = gradient_op(n; h=h, flag_gpu=flag_gpu)
div = divergence_op(n; h=h, flag_gpu=flag_gpu)

# Coherency test
u = randn(Float32, n) |> gpu
a = Δ*u
b = -(adjoint(Dx)*(Dx*u)+adjoint(Dy)*(Dy*u))
@test isapprox(a, b; rtol = Float32(1e-3))

u = randn(Float32, n) |> gpu
a = ∇*u
b1 = Dx*u; b2 = Dy*u
@test isapprox(a.x, b1; rtol = 1f-3)
@test isapprox(a.y, b2; rtol = 1f-3)

u = VectorField2D(randn(Float32, n), randn(Float32, n)) |> gpu
a = div*u
b = -adjoint(∇)*u
@test isapprox(a, b; rtol = 1f-3)

u = randn(Float32, n) |> gpu
a = ∇*u
b = -adjoint(div)*u
@test isapprox(a.x, b.x; rtol = 1f-3)
@test isapprox(a.y, b.y; rtol = 1f-3)

# Adjoint test (Dx)
u = randn(Float32, n) |> gpu
v = randn(Float32, n) |> gpu
a = dot(Dx*u, v)
b = dot(u, adjoint(Dx)*v)
@test isapprox(a, b; rtol = 1f-3)

# Adjoint test (Dy)
u = randn(Float32, n) |> gpu
v = randn(Float32, n) |> gpu
a = dot(Dy*u, v)
b = dot(u, adjoint(Dy)*v)
@test isapprox(a, b; rtol = 1f-3)

# Adjoint test (Δ)
u = randn(Float32, n) |> gpu
v = randn(Float32, n) |> gpu
a = dot(Δ*u, v)
b = dot(u, adjoint(Δ)*v)
@test isapprox(a, b; rtol = 1f-3)

# Adjoint test (∇)
u = randn(Float32, n) |> gpu
v = VectorField2D(randn(Float32, n), randn(Float32, n)) |> gpu
a = dot(∇*u, v)
b = dot(u, adjoint(∇)*v)
@test isapprox(a, b; rtol = 1f-3)

# Adjoint test (div)
u = VectorField2D(randn(Float32, n), randn(Float32, n)) |> gpu
v = randn(Float32, n) |> gpu
a = dot(div*u, v)
b = dot(u, adjoint(div)*v)
@test isapprox(a, b; rtol = 1f-3)