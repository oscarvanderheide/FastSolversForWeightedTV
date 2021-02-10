using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Projection on l2Inf ball
y = randn(Float32, 512, 512, 2)
C = ell_ball(2,1,0.01f0)
@benchmark project(y, C)


# # Projection on l21 ball
# p = randn(Float32, 1024, 2048, 2); flag_gpu && (p = p |> gpu)
# C = ell_ball(2,1,0.01f0)
# q = project(p, C)
# @test norm21(q) ≈ C.ε rtol=1f-3
# p2 = randn(Float32, 1024, 2048, 2); flag_gpu && (p2 = p2 |> gpu)
# q2 = project(p2, C)
# @test norm22(q-p)<norm22(q2-p)
# @test dot(p-q, q2-q)<0f0