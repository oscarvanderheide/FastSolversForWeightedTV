using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, TestImages, Test, Statistics
CUDA.allowscalar(false)

# Geometry
flag_gpu = true
# flag_gpu = false
T = Float32
# T = Float64
n = (256,256)
w = T.(TestImages.shepp_logan(n...))

# Operators
η = mean(ptnorm2_2D(gradient_2D(; T=T)*w))
P = structural_weight(w, η); flag_gpu && (P = P |> gpu)

# Adjoint test
u = randn(T, n[1]-1, n[2]-1, 2); flag_gpu && (u = u |> gpu)
v = randn(T, n[1]-1, n[2]-1, 2); flag_gpu && (v = v |> gpu)
a = dot(P*u, v)
b = dot(u, adjoint(P)*v)
@test a ≈ b rtol = T(1e-3)