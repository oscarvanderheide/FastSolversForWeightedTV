using LinearAlgebra, FastSolversForWeightedTV, Flux, TestImages, BenchmarkTools
using CUDA; CUDA.allowscalar(false)
using Random; Random.seed!(123)

T = Float32
g = norm_3D(2,2,1; T=Complex{T}) |> gpu

n = (256, 256, 256)
p = randn(Complex{T}, n..., 3) |> gpu

λ = 0.5f0*norm(p)^2/g(p)
@btime proxy(p, complex(λ), g);