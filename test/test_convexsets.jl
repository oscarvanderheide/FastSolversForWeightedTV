using LinearAlgebra, VectorFields, DifferentialOperatorsForTV, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Random data
n = (3,3)
h = (abs(randn(Float32)), abs(randn(Float32)))
x1 = to_scalar_field(randn(Float32, n))
x2 = to_scalar_field(randn(Float32, n))
y = [x1; x2]; flag_gpu && (y = y |> gpu)

# No constraints
C = no_constraints(ScalarField2D{Float32})
x1_ = project(x1, C)
@test x1_ ≈ x1 rtol=1f-3
C = no_constraints(VectorField2D{Float32})
y_ = project(y, C)
@test y_ ≈ y rtol=1f-3

# Positive values
C = positive_values(Float32)
x1_ = project(x1, C)
x1__ = x1*(x1>=0f0)
@test x1_ ≈ x1__ rtol=1f-3

# Unitary ball
C = ell_ball(Float32,2,Inf)
y_ = project(y, C)
pt_norm2_y = sqrt.(y.array[:,:,1:1,:].^2+y.array[:,:,2:2,:].^2)
y__ = to_vector_field(y.array.*(pt_norm2_y .<= 1f0)+y.array./pt_norm2_y.*(pt_norm2_y .>= 1f0))
@test y_ ≈ y__ rtol=1f-3