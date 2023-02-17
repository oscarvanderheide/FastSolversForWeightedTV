# Examples

We briefly describe how to use the tools provided by this package. We focus, here, on a 3D TV-denoising example with GPU acceleration.

For starters, let's make sure to install all the needed packages! Type `]` and
```julia
@(v1.8) pkg> add CUDA, TestImages, PyPlot, https://github.com/grizzuti/AbstractLinearOperators.git, https://github.com/grizzuti/AbstractProximableFunctions.git, https://github.com/grizzuti/FastSolversForWeightedTV.git
```
The packages `AbstractLinearOperators`, `AbstractProximableFunctions` provide some general utilities that are combined with `FastSolversForWeightedTV` to specify the solvers needed for the computation of the proximal operator. In this tutorial, we use `PyPlot` for image visualization, but many other packages may fit the bill.

To load the relevant modules, type in the Julia REPL:
```julia
# Package load
using LinearAlgebra, CUDA, TestImages, PyPlot
using AbstractProximableFunctions, FastSolversForWeightedTV
```

Let's load the 2D Shepp-Logan phantom and make it 3D:
```julia
# Prepare data
n = (256, 256, 256)
y_clean = Float32.(TestImages.shepp_logan(n[1:2]...)) # 2D Shepp-Logan of size 256x256
y_clean = repeat(y_clean; outer=(1,1,n[3]))           # 3D "augmentation"
y_clean = y_clean/norm(y_clean, Inf)                  # Normalization
y_noisy = y_clean+0.1f0*randn(Float32, n)             # Adding noise
y_noisy = CuArray(noisy)                              # Move data to GPU
```

Now that we prepared the noisy data, we define the regularization functional based on TV.