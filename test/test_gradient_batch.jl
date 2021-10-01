using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test
CUDA.allowscalar(false)

# Geometry
T = Float32
n = (1024, 2048)
h = (abs(randn(T)), abs(randn(T)))

# Operators
flag_gpu = true
# flag_gpu = false
∇ = gradient_batch_2D(; T=T); flag_gpu && (∇ = ∇ |> gpu)

# Adjoint test (∇)
nc = 3; nb = 5;
u = randn(T, n..., nc, nb); flag_gpu && (u = u |> gpu)
v = randn(T, n[1]-1, n[2]-1, 2*nc, nb); flag_gpu && (v = v |> gpu)
a = dot(∇*u, v)
b = dot(u, adjoint(∇)*v)
@test a ≈ b rtol = T(1e-3)