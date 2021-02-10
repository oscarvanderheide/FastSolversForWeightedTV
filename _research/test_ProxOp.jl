using LinearAlgebra, VectorFields, DifferentialOperatorsForTV, FastSolversForWeightedTV, CUDA, Flux, Test
using AbstractOperators, StructuredOptimization
using BenchmarkTools
using PyPlot, Images, TestImages
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Test input
y_ = Float64.(testimage("cameraman"))
n = size(y_)

# Proximal op implementation
L_ = Variation(size(y_))
λ = 0.07
u = Variable(size(L_,1)...)
@elapsed @minimize ls(L_'*u-y_) + conj(λ*norm(u,2,1,2)) with ForwardBackward(tol = 1e-3, gamma = 1.0/8.0, fast=true, verbose=true, maxit=100)
x_ = y_-L_'*(~u)

# Problem specification
y = to_scalar_field(y_); flag_gpu && (y = y |> gpu)
h = (1e0, 1e0)
L = gradient_op(n; h=h, flag_gpu=flag_gpu)

# Optimization options
opt = optProxyL21(; steplength=1e0/(8e0*λ), niter=200, tol=1e-3, nesterov=true, log=true, verbose=true)

# Solver: 0.5*||x-y||^2+||L*x||_{2,1}
@elapsed fval, x = solverProxyL21(y, λ, L; opt=opt)
x = x |> cpu