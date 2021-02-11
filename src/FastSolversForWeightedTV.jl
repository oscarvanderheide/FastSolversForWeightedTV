module FastSolversForWeightedTV

# Modules
using LinearAlgebra, NNlib, CUDA, Flux, AbstractLinearOperators, Roots

# Gradient operator
include("./padding.jl")
include("./gradient.jl")

# Differentiable functions
include("./differentiable_type.jl")
include("./differentiable_utils.jl")

# Proximable functions
include("./norm_utils.jl")
include("./proximable_type.jl")
include("./proximable_utils.jl")

# Optimizable functions
include("./optim_type.jl")

# FISTA solver
include("./fista_solver.jl")

# Convex set
include("./convex_sets.jl")
include("./convex_sets_utils.jl")

end