using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test
CUDA.allowscalar(false)

# Geometry
n = (1024, 2048)
h = (abs(randn(Float32)), abs(randn(Float32)))

# Operators
flag_gpu = true
# flag_gpu = false
T = Float32
∇ = gradient_2D(n; h=h, gpu=flag_gpu)

# Adjoint test (∇)
u = randn(T, n); flag_gpu && (u = u |> gpu)
v = randn(T, n..., 2); flag_gpu && (v = v |> gpu)
a = dot(∇*u, v)
b = dot(u, adjoint(∇)*v)
@test a ≈ b rtol = 1f-3