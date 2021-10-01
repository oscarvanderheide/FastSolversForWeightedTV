module FastSolversForWeightedTV

# Modules
using LinearAlgebra, AbstractLinearOperators, CUDA, NNlib, Flux, Roots

# Types and utils
include("./abstract_functional_types.jl")
include("./type_utils.jl")

# Optimization solvers
include("./optimization_solvers.jl")

# Differentiable function examples
include("./differentiable_examples.jl")

# Proximable function examples
include("./proximable_examples.jl")

# # Gradient operator
include("./gradient_operator.jl")

end