#: Utilities for norm functions

export GradientNorm, gradient_norm


# Gradient mixed norm

struct GradientNorm{T,D1,D2,N1,N2}<:AbstractWeightedProximableFunction{T,D1,D2}
    weighted_mixed_norm::AbstractWeightedProximableFunction{T,D1}
end

function gradient_norm(N1::Number, N2::Number, n::NTuple{D,Int64}, h::NTuple{D,T}; weight::Union{Nothing,AbstractLinearOperator}=nothing, pareto_tol::Union{Nothing,Real}=nothing, complex::Bool=false, optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {D,T<:Real}
    ∇ = gradient_operator(n, h; complex=complex)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    complex ? (CT = Complex{T}) : (CT = T)
    return GradientNorm{CT,D,D+1,N1,N2}(weighted_prox(mixed_norm(CT, D, N1, N2; pareto_tol=pareto_tol), A∇; optimizer=optimizer))
end

ConvexOptimizationUtils.fun_eval(g::GradientNorm{T,D1,D2,N1,N2}, x::AbstractArray{T,D1}) where {T,D1,D2,N1,N2} = g.weighted_mixed_norm(x)

ConvexOptimizationUtils.get_optimizer(g::GradientNorm) = get_optimizer(g.weighted_mixed_norm)
ConvexOptimizationUtils.get_linear_operator(g::GradientNorm) = get_linear_operator(g.weighted_mixed_norm)
ConvexOptimizationUtils.get_prox(g::GradientNorm) = get_prox(g.weighted_mixed_norm)

ConvexOptimizationUtils.proxy!(p::AbstractArray{CT,D1}, λ::T, g::GradientNorm{CT,D1,D2,N1,N2}, q::AbstractArray{CT,D1}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,D1,D2,N1,N2,CT<:RealOrComplex{T}} = proxy!(p, λ, g.weighted_mixed_norm, q; optimizer=optimizer)

ConvexOptimizationUtils.project!(p::AbstractArray{CT,D1}, ε::T, g::GradientNorm{CT,D1,D2,N1,N2}, q::AbstractArray{CT,D1}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real,D1,D2,N1,N2,CT<:RealOrComplex{T}} = project!(p, ε, g.weighted_mixed_norm, q; optimizer=optimizer)

Flux.gpu(g::GradientNorm{T,D1,D2,N1,N2}) where {T,D1,D2,N1,N2} = GradientNorm{T,D1,D2,N1,N2}(weighted_prox(get_prox(g.weighted_mixed_norm), gpu(get_linear_operator(g.weighted_mixed_norm)); optimizer=get_optimizer(g.weighted_mixed_norm)))

Flux.cpu(g::GradientNorm{T,D1,D2,N1,N2}) where {T,D1,D2,N1,N2} = GradientNorm{T,D1,D2,N1,N2}(weighted_prox(get_prox(g.weighted_mixed_norm), cpu(get_linear_operator(g.weighted_mixed_norm)); optimizer=get_optimizer(g.weighted_mixed_norm)))