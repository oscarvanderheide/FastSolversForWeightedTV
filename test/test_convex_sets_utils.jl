using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

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
C = lowerlim_constraints(0f0)
x1_ = project(x1, C)
x1__ = x1.*(x1.>=0f0)
@test x1_ ≈ x1__ rtol=1f-3

# Unitary ball
C = ell_ball(2,Inf,1f0)
y_ = project(y, C)
ptn_y = ptnorm2(y;eps=0f0)
y__ = y.*(ptn_y .<= 1f0)+y./ptn_y.*(ptn_y .>= 1f0)
@test y_ ≈ y__ rtol=1f-3