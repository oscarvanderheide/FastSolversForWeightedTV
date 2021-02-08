module FastSolversForWeightedTV

# Modules
using LinearAlgebra, NNlib, CUDA, Flux, AbstractLinearOperators

# Gradient operator
include("./padding.jl")
include("./gradient.jl")

# Differentiable functions
include("./differentiable_type.jl")
include("./differentiable_utils.jl")

# # Proximable functions
# include("./proximable_type.jl")

# # Convex set types and projections
# include("./convex_sets.jl")

# # FISTA
# include("./fista_solver.jl")

# # L21 solvers (general)
# include("./proxyl21_solver.jl")

end