module FastSolversForWeightedTV

# Modules
using LinearAlgebra, AbstractLinearOperators, CUDA, NNlib, Flux, Roots

# Constant utils
const RealOrComplex{T<:Real} = Union{T,Complex{T}}

# Types and utils
include("./abstract_functional_types.jl")
include("./type_utils.jl")

# Optimization solvers
include("./optimization_solvers.jl")

# Differentiable function examples
include("./differentiable_examples.jl")

# Gradient operator
include("./gradient_operator.jl")

# Vector-field projection operator
include("./weighting_operator.jl")

# Proximable function examples
include("./proximable_examples.jl")

end