using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test
CUDA.allowscalar(false)

# Geometry
n = (1024, 2048)

# Operators
flag_gpu = true
# flag_gpu = false
T = Float32
w = randn(T, n..., 2); flag_gpu && (w = w |> gpu)
P = projvectorfield_2D(w; η=T(0.0))

# Adjoint test
u = randn(T, n..., 2); flag_gpu && (u = u |> gpu)
v = randn(T, n..., 2); flag_gpu && (v = v |> gpu)
a = dot(P*u, v)
b = dot(u, adjoint(P)*v)
@test a ≈ b rtol = 1f-3

# Projection test
@test (P*P)*u ≈ P*u rtol = 1f-3