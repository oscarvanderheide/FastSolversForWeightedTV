using BenchmarkTools, CUDA, Flux, FastSolversForWeightedTV
CUDA.allowscalar(false)

# Geometry
T = Float32
n = (1024, 2048)
h = (abs(randn(T)), abs(randn(T)))

# Operator
∇ = gradient_2D(n, h; gpu=false)
∇_gpu = gradient_2D(n, h; gpu=true)

# Inputs
u_cpu = randn(T, n)
u_gpu = randn(T, n) |> gpu

# Timings
@benchmark ∇*u_cpu
@benchmark ∇_gpu*u_gpu