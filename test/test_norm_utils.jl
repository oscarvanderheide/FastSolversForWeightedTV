using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Random data
n = (3,3)
h = (abs(randn(Float32)), abs(randn(Float32)))
x = randn(Float32, n); flag_gpu && (x = x |> gpu)

# Gradient operator
A = gradient_2D(n; h=h, gpu=flag_gpu)

# Consistency check
@test normA21(x, A) ≈ normTV(x; h=h) rtol=1f-3