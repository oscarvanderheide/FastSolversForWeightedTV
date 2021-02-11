using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Random data
n = (3,3)
h = (abs(randn(Float32)), abs(randn(Float32)))
x1 = randn(Float32, n)
x2 = randn(Float32, n)
y = cat(x1, x2; dims=3); flag_gpu && (y = y |> gpu)

# No constraints
C = no_constraints(Float32,2)
x1_ = project(x1, C)
@test x1_ ≈ x1 rtol=1f-3
C = no_constraints(Float32,3)
y_ = project(y, C)
@test y_ ≈ y rtol=1f-3

# Positive values
C = lowerlim_constraints_2D(0f0)
x1_ = project(x1, C)
x1__ = x1.*(x1.>=0f0)
@test x1_ ≈ x1__ rtol=1f-3

# Projection on l2Inf ball
C = ell_ball(2,Inf,1f0)
y_ = project(y, C)
ptn_y = ptnorm2(y)
y__ = y.*(ptn_y .<= 1f0)+y./ptn_y.*(ptn_y .>= 1f0)
@test y_ ≈ y__ rtol=1f-3

# Projection on l21 ball
p = randn(Float32, 1024, 2048, 2); flag_gpu && (p = p |> gpu)
C = ell_ball(2,1,0.01f0)
q = project(p, C)
@test norm21(q) ≈ C.ε rtol=1f-3
p2 = randn(Float32, 1024, 2048, 2); flag_gpu && (p2 = p2 |> gpu)
q2 = project(p2, C)
@test norm22(q-p)<=norm22(q2-p)
@test dot(p-q, q2-q)<=0f0