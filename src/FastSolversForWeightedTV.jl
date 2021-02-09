module FastSolversForWeightedTV

# Modules
using LinearAlgebra, NNlib, CUDA, Flux, AbstractLinearOperators

# Gradient operator
include("./padding.jl")
include("./gradient.jl")

# Differentiable functions
include("./differentiable_type.jl")
include("./differentiable_utils.jl")

# Proximable functions
include("./norm_utils.jl")
include("./proximable_type.jl")
include("./convex_sets.jl")
include("./proximable_utils.jl")
include("./convex_sets_utils.jl")

# FISTA solver
include("./fista_solver.jl")

# # L21 solvers (general)
# include("./proxyl21_solver.jl")

end