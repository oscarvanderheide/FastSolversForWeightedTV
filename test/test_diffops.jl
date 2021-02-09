using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test
CUDA.allowscalar(false)

# Geometry
n = (1024, 2048)
h = (abs(randn(Float32)), abs(randn(Float32)))

# Operators
flag_gpu = true
# flag_gpu = false
T = Float32
Dx = horz_derivative_2D(n, h[1]; gpu=flag_gpu)
Dy = vert_derivative_2D(n, h[2]; gpu=flag_gpu)
∇ = gradient_2D(n, h; gpu=flag_gpu)

# Coherency test
u = randn(T, n); flag_gpu && (u = u |> gpu)
∇u = ∇*u
∇u_ = cat(Dx*u, Dy*u; dims=3)
@test ∇u ≈ ∇u_ rtol=1f-3

# Adjoint test (Dx)
u = randn(T, n); flag_gpu && (u = u |> gpu)
v = randn(T, n); flag_gpu && (v = v |> gpu)
a = dot(Dx*u, v)
b = dot(u, adjoint(Dx)*v)
@test a ≈ b rtol = 1f-3

# Adjoint test (Dy)
u = randn(T, n); flag_gpu && (u = u |> gpu)
v = randn(T, n); flag_gpu && (v = v |> gpu)
a = dot(Dy*u, v)
b = dot(u, adjoint(Dy)*v)
@test a ≈ b rtol = 1f-3

# Adjoint test (∇)
u = randn(T, n); flag_gpu && (u = u |> gpu)
v = randn(T, n..., 2); flag_gpu && (v = v |> gpu)
a = dot(∇*u, v)
b = dot(u, adjoint(∇)*v)
@test a ≈ b rtol = 1f-3