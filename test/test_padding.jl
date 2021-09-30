using LinearAlgebra, CUDA, Flux, FastSolversForWeightedTV, Test
CUDA.allowscalar(false)

# Size
n = (1024, 2048)

# Padding
p = (1, 2, 3, 4)
n_ext = n.+(p[1]+p[2],p[3]+p[4])

# Adjoint test
P = FastSolversForWeightedTV.padding("", Float32, p...)
u = randn(Float32, n) |> gpu
v = randn(Float32, n) |> gpu
@test dot(pad(u, P), v) ≈ dot(u, restrict(v, P)) rtol=1f-3

P = FastSolversForWeightedTV.padding("zero", Float32, p...)
u = randn(Float32, n) |> gpu
v = randn(Float32, n_ext) |> gpu
@test dot(pad(u, P), v) ≈ dot(u, restrict(v, P)) rtol=1f-3

P = FastSolversForWeightedTV.padding("copy", Float32, p...)
u = randn(Float32, n) |> gpu
v = randn(Float32, n_ext) |> gpu
@test dot(pad(u, P), v) ≈ dot(u, restrict(v, P)) rtol=1f-3

P = FastSolversForWeightedTV.padding("periodic", Float32, p...)
u = randn(Float32, n) |> gpu
v = randn(Float32, n_ext) |> gpu
@test dot(pad(u, P), v) ≈ dot(u, restrict(v, P)) rtol=1f-3