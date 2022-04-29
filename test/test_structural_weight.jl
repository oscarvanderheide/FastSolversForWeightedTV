using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test
CUDA.allowscalar(false)

# Geometry
# flag_gpu = true
flag_gpu = false
T = Float64
n2d = (256,256)
n3d = (256,256,256)
u2d   = randn(T,          n2d); flag_gpu && (u2d   = u2d |> gpu)
u2d_c = randn(Complex{T}, n2d); flag_gpu && (u2d_c = u2d_c |> gpu)
u3d   = randn(T,          n3d); flag_gpu && (u3d   = u3d |> gpu)
u3d_c = randn(Complex{T}, n3d); flag_gpu && (u3d_c = u3d_c |> gpu)
rtol = T(1e-6)

# Operators
η2d = structural_mean(u2d)
P2d = structural_weight(u2d; η=η2d, γ=0.9); flag_gpu && (P2d = P2d |> gpu)
η2d_c = structural_mean(u2d_c)
P2d_c = structural_weight(u2d_c; η=η2d_c, γ=0.9); flag_gpu && (P2d_c = P2d_c |> gpu)
η3d = structural_mean(u3d)
P3d = structural_weight(u3d; η=η3d, γ=0.9); flag_gpu && (P3d = P3d |> gpu)
η3d_c = structural_mean(u3d_c)
P3d_c = structural_weight(u3d_c; η=η3d_c, γ=0.9); flag_gpu && (P3d_c = P3d_c |> gpu)

# Adjoint test (2d)
u = randn(T, n2d[1]-1, n2d[2]-1, 2); flag_gpu && (u = u |> gpu)
v = randn(T, n2d[1]-1, n2d[2]-1, 2); flag_gpu && (v = v |> gpu)
a = dot(P2d*u, v)
b = dot(u, adjoint(P2d)*v)
@test a ≈ b rtol = rtol

# Adjoint test (2d_c)
u = randn(Complex{T}, n2d[1]-1, n2d[2]-1, 2); flag_gpu && (u = u |> gpu)
v = randn(Complex{T}, n2d[1]-1, n2d[2]-1, 2); flag_gpu && (v = v |> gpu)
a = dot(P2d_c*u, v)
b = dot(u, adjoint(P2d_c)*v)
@test a ≈ b rtol = rtol

# Adjoint test (3d)
u = randn(T, n3d[1]-1, n3d[2]-1, n3d[3]-1, 3); flag_gpu && (u = u |> gpu)
v = randn(T, n3d[1]-1, n3d[2]-1, n3d[3]-1, 3); flag_gpu && (v = v |> gpu)
a = dot(P3d*u, v)
b = dot(u, adjoint(P3d)*v)
@test a ≈ b rtol = rtol

# Adjoint test (3d_c)
u = randn(Complex{T}, n3d[1]-1, n3d[2]-1, n3d[3]-1, 3); flag_gpu && (u = u |> gpu)
v = randn(Complex{T}, n3d[1]-1, n3d[2]-1, n3d[3]-1, 3); flag_gpu && (v = v |> gpu)
a = dot(P3d_c*u, v)
b = dot(u, adjoint(P3d_c)*v)
@test a ≈ b rtol = rtol