#: Utilities for norm functions

export GradientNorm, gradient_norm


# Gradient mixed norm

struct GradientNorm{T,D,N1,N2}<:AbstractProximableFunction{T,D}
    weighted_mixed_norm::WeightedProximableFunction{T,D}
end

function gradient_norm(N1::Number, N2::Number, n::NTuple{D,Int64}, h::NTuple{D,T}; weight::Union{Nothing,AbstractLinearOperator}=nothing, pareto_tol::Union{Nothing,Real}=nothing, complex::Bool=false, optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {D,T<:Real}
    ∇ = gradient_operator(n, h; complex=complex)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    complex ? (CT = Complex{T}) : (CT = T)
    return GradientNorm{CT,D,N1,N2}(weighted_prox(mixed_norm(CT, D, N1, N2; pareto_tol=pareto_tol), A∇; optimizer=optimizer))
end

ConvexOptimizationUtils.fun_eval(g::GradientNorm{T,D,N1,N2}, x::AbstractArray{T,D}) where {T,D,N1,N2} = g.weighted_mixed_norm(x)

ConvexOptimizationUtils.get_optimizer(g::GradientNorm) = get_optimizer(g.weighted_mixed_norm)

ConvexOptimizationUtils.proxy!(p::AbstractArray{CT,D}, λ::T, g::GradientNorm{CT,D,N1,N2}, q::AbstractArray{CT,D}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real, D, N1, N2, CT<:RealOrComplex{T}} = proxy!(p, λ, g.weighted_mixed_norm, q; optimizer=optimizer)

ConvexOptimizationUtils.project!(p::AbstractArray{CT,D}, ε::T, g::GradientNorm{CT,D,N1,N2}, q::AbstractArray{CT,D}; optimizer::Union{Nothing,AbstractConvexOptimizer}=nothing) where {T<:Real, D, N1, N2, CT<:RealOrComplex{T}} = project!(p, ε, g.weighted_mixed_norm, q; optimizer=optimizer)

Flux.gpu(g::GradientNorm{T,D,N1,N2}) where {T,D,N1,N2} = GradientNorm{T,D,N1,N2}(weighted_prox(g.weighted_mixed_norm.prox, gpu(g.weighted_mixed_norm.linear_operator); optimizer=g.weighted_mixed_norm.optimizer))

Flux.cpu(g::GradientNorm{T,D,N1,N2}) where {T,D,N1,N2} = GradientNorm{T,D,N1,N2}(weighted_prox(g.weighted_mixed_norm.prox, cpu(g.weighted_mixed_norm.linear_operator); optimizer=g.weighted_mixed_norm.optimizer))