using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test, Random
Random.seed!(123)
CUDA.allowscalar(false)

# Geometry
T = Float64
n2d = (256, 256)
h2d = (abs(randn(T)), abs(randn(T)))
n3d = (256, 256, 256)
h3d = (abs(randn(T)), abs(randn(T)), abs(randn(T)))
nc = 3
nb = 16

# Operators
# flag_gpu = true
flag_gpu = false
∇2d    = gradient_operator(n2d, h2d; complex=false);               flag_gpu && (∇2d    = ∇2d    |> gpu)
∇2d_c  = gradient_operator(n2d, h2d; complex=true);                flag_gpu && (∇2d_c  = ∇2d_c  |> gpu)
∇2db   = gradient_operator_batch(n2d, nc, nb, h2d; complex=false); flag_gpu && (∇2db   = ∇2db   |> gpu)
∇2db_c = gradient_operator_batch(n2d, nc, nb, h2d; complex=true);  flag_gpu && (∇2db_c = ∇2db_c |> gpu)
∇3d    = gradient_operator(n3d, h3d; complex=false);               flag_gpu && (∇3d    = ∇3d    |> gpu)
∇3d_c  = gradient_operator(n3d, h3d; complex=true);                flag_gpu && (∇3d_c  = ∇3d_c  |> gpu)

# Adjoint test (∇2d)
rtol = T(1e-6)
u = randn(T, n2d);                   flag_gpu && (u = u |> gpu)
v = randn(T, n2d[1]-1, n2d[2]-1, 2); flag_gpu && (v = v |> gpu)
@test dot(∇2d*u, v) ≈ dot(u, adjoint(∇2d)*v) rtol=rtol

# Adjoint test (∇2d_c)
u = randn(Complex{T}, n2d);                   flag_gpu && (u = u |> gpu)
v = randn(Complex{T}, n2d[1]-1, n2d[2]-1, 2); flag_gpu && (v = v |> gpu)
@test dot(∇2d_c*u, v) ≈ dot(u, adjoint(∇2d_c)*v) rtol=rtol

# Adjoint test (∇2db)
u = randn(T, n2d..., nc, nb);               flag_gpu && (u = u |> gpu)
v = randn(T, n2d[1]-1, n2d[2]-1, 2*nc, nb); flag_gpu && (v = v |> gpu)
@test dot(∇2db*u, v) ≈ dot(u, adjoint(∇2db)*v) rtol=rtol

# Adjoint test (∇2db_c)
u = randn(Complex{T}, n2d..., nc, nb);               flag_gpu && (u = u |> gpu)
v = randn(Complex{T}, n2d[1]-1, n2d[2]-1, 2*nc, nb); flag_gpu && (v = v |> gpu)
@test dot(∇2db_c*u, v) ≈ dot(u, adjoint(∇2db_c)*v) rtol=rtol

# Adjoint test (∇3d)
u = randn(T, n3d);                             flag_gpu && (u = u |> gpu)
v = randn(T, n3d[1]-1, n3d[2]-1, n3d[3]-1, 3); flag_gpu && (v = v |> gpu)
@test dot(∇3d*u, v) ≈ dot(u, ∇3d'*v) rtol=rtol

# Adjoint test (∇3d_c)
u = randn(Complex{T}, n3d);                             flag_gpu && (u = u |> gpu)
v = randn(Complex{T}, n3d[1]-1, n3d[2]-1, n3d[3]-1, 3); flag_gpu && (v = v |> gpu)
@test dot(∇3d_c*u, v) ≈ dot(u, ∇3d_c'*v) rtol=rtol