using LinearAlgebra, VectorFields, DifferentialOperatorsForTV, FastSolversForWeightedTV, CUDA, Flux, Test
using PyPlot, Images, TestImages
CUDA.allowscalar(false)

flag_gpu = true
# flag_gpu = false

# Test input
h = (1f0, 1f0)
y = eltype(h).(testimage("mandril_gray")); flag_gpu && (y = eltype(h).(y |> gpu))
# y = Float32.(testimage("cameraman")); flag_gpu && (y = y |> gpu)
n = size(y)
y = to_scalar_field(y)

# Problem specification
λ = 1f0
# λ = 0.07f0
L = gradient_op(n; h=h, flag_gpu=flag_gpu)

# Optimization options
opt = optProxyL21(; steplength=1f0/(8f0*λ), niter=200, tol=1f-3, nesterov=true, log=true, verbose=true)

# Solver: 0.5*||x-y||^2+||L*x||_{2,1}
@elapsed fval, x = solverProxyL21(y, λ, L; opt=opt)
x = x |> cpu