using TotalVariationRegularization, LinearAlgebra, Test

# Initialization
n = (3, 2)
vx = randn(Float32, n)
vy = randn(Float32, n)
v = VectorField2D(vx, vy)

# Check field call
v.x
v.y

# Base
size(v)
v[1]
v[1] = 0f0

# Operations
v_ = VectorField2D(randn(Float32, n), randn(Float32, n))
v+v_
v-v_
-v
randn(Float32)*v
v*randn(Float32)
v/randn(Float32)

# Linear algebra
ptdot(v, v_)
ptnorm(v; p=1)
ptnorm(v; p=2)
ptnorm(v; p=Inf)
norm(v; p1=1,   p2=1)
norm(v; p1=1,   p2=2)
norm(v; p1=1,   p2=Inf)
norm(v; p1=2,   p2=1)
norm(v; p1=2,   p2=2)
norm(v; p1=2,   p2=Inf)
norm(v; p1=Inf, p2=1)
norm(v; p1=Inf, p2=2)
norm(v; p1=Inf, p2=Inf)