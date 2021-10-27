using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, BenchmarkTools, AbstractLinearOperators
using CUDA; CUDA.allowscalar(false)
using Random; Random.seed!(123)

T = Float32
# flag_gpu = true
flag_gpu = false
g = norm_3D(2,2,1; T=T); flag_gpu && (g = g |> gpu)

n = (256, 256, 256)
# n = (128,128,128)
p = randn(T, n..., 3); flag_gpu && (p = p |> gpu)

λ = 0.5f0*norm(p)^2/g(p)
# @btime proxy(p, λ, g);

A = identity_operator(typeof(p), n)
gA = g∘A
opt = opt_fista(; steplength=T(1), niter=10)

gT = conjugate(gA.g)
f = leastsquares_misfit(λ*adjoint(gA.A), p)+λ*gT

p0 = randn(T, n..., 3); flag_gpu && (p0 = p0 |> gpu)
# q = minimize(f, p0, opt)

# ptmp = similar(p)
# @btime grad!(f.f, p, ptmp);

# @btime proxy!(p, opt.steplength, gT, ptmp);

@btime q = minimize(f, p0, opt);