module FastSolversForWeightedTV

# Modules
using LinearAlgebra, NNlib, CUDA, Flux, AbstractLinearOperators, Roots

# Gradient operator
include("./padding.jl")
include("./norm_utils.jl")
include("./gradient_utils.jl")
include("./gradient.jl")

# Functional types
include("./differentiable_type.jl")
include("./proximable_type.jl")
include("./optim_type.jl")

# FISTA solver
include("./fista_solver.jl")

# Functional utils
include("./differentiable_utils.jl")
include("./proximable_utils.jl")

# Convex set
include("./convex_sets.jl")
include("./convex_sets_utils.jl")

end