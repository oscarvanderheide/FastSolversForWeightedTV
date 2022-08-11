module FastSolversForWeightedTV

using LinearAlgebra, AbstractLinearOperators, ConvexOptimizationUtils, Flux, CUDA

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./weighting_operator.jl")
include("./gradient_operator.jl")
include("./gradient_norm.jl")

end