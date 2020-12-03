using DrWatson
@quickactivate "TotalVariationRegularization"

using TotalVariationRegularization, LinearAlgebra, Test, PyPlot, Images, TestImages
using Flux
using CUDA; CUDA.allowscalar(false)

# Test input
h = (1f0, 1f0)
b = Float32.(testimage("mandril_gray")) |> gpu
n = size(b)

# Options
opt = optTVFGP(; niter=200, s=nothing, D=nothing, Proj=x -> max.(x, 0f0), verbose=false, log=false)

# Solving proximal operator
α = 1f0
u = solverTVFastGradientProjection(b, α; h=h, opt=opt)

# Cpu
u = u |> cpu
b = b |> cpu

# Plotting
fig = figure()
subplot(1, 2, 1)
imshow(b)
subplot(1, 2, 2)
imshow(u)
savefig("./plots/test_proxop.png")