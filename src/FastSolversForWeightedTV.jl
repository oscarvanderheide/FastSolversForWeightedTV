module FastSolversForWeightedTV

using LinearAlgebra, AbstractLinearOperators, AbstractProximableFunctions, NNlib, CUDA

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./weighting_operator.jl")
include("./gradient_operator.jl")
include("./gradient_norm.jl")

end