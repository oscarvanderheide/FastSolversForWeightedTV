using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, Images, TestImages, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Load image
y = Float32.(testimage("mandril_gray")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint set
ε_rel = 0.5f0
ε = ε_rel*normTV(y)
opt = optFISTA(; steplength=1f0/8f0, niter=1000, nesterov=true)
C = TVball_2D(n, ε, opt; gpu=flag_gpu)

# Projection
x = project(y, C) |> cpu
y = y |> cpu