module FastSolversForWeightedTV

# Modules
using LinearAlgebra, NNlib, CUDA, Flux, AbstractLinearOperators, Roots

# Types and utils
include("./abstract_functional_types.jl")
include("./abstract_optimization_types.jl")
include("./type_utils.jl")

# Optimization solvers
include("./optimization_solvers.jl")

# Differentiable function examples
include("./differentiable_examples.jl")

# Proximable function examples
include("./proximable_examples.jl")

# # Set types
# include("./projectionable_sets.jl")

# # Gradient operator
# include("./padding.jl")
# include("./norm_utils.jl")
# include("./gradient_utils.jl")
# include("./gradient.jl")

# # Functional utils
# include("./differentiable_utils.jl")
# include("./proximable_utils.jl")

# # Convex set
# include("./convex_sets_utils.jl")

end