using LinearAlgebra, FastSolversForWeightedTV, AbstractLinearOperators, Test

# Differentiable function concrete type
struct LSnorm{T,N}<:DifferentiableFunction{T,N}
    y::AbstractArray{T,N}
end
function FastSolversForWeightedTV.grad!(f::LSnorm{T,N}, x::AbstractArray{T,N}, g::AbstractArray{T,N}) where {T,N}
    g .= x-f.y
    fval = T(0.5)*norm(g)^2
    return fval
end

y = randn(Float32, 64, 128)
f = LSnorm(y)
x = randn(Float32, 64, 128)
grad(f, x)