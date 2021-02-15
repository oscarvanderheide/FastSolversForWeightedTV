using LinearAlgebra, FastSolversForWeightedTV, CUDA, Flux, Test, PyPlot, DelimitedFiles, BenchmarkTools
CUDA.allowscalar(false)

# flag_gpu = true
flag_gpu = false

# Load image
v = Float32.(readdlm("./data/T1.txt")); flag_gpu && (v = v |> gpu); v = circshift(v, (30,0))
y = Float32.(readdlm("./data/T2.txt")); flag_gpu && (y = y |> gpu)
n = size(y)

# Constraint set
ε_rel = 0.1f0
η = 0.01f0
p0 = zeros(Float32, n..., 2); flag_gpu && (p0 = p0 |> gpu)
opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=1000, nesterov=true)
ε = ε_rel*normTsqV(y, v, η)
C = tsqv_ball_2D(v, η, ε; opt=opt)

# Projection
x = project(y, C) |> cpu
y = y |> cpu

# Plot
close("all")
figure()
subplot(1,3,1)
imshow(v; cmap="gray")
title("Structural guide")
subplot(1,3,2)
imshow(y; cmap="gray")
title("Original")
subplot(1,3,3)
imshow(x; cmap="gray")
title(string("Struct.-TV proj., ", L"niter = ", string(opt.niter), ", ", L"\varepsilon = ", string(ε_rel)))
savefig("./plots/structTsqVproj.png", dpi=300, transparent=false, bbox_inches="tight")