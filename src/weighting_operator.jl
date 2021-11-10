export structural_weight, structural_mean


# Projection on vector field

struct ProjVectorField{T,N}<:AbstractLinearOperator{T,N,N}
    ξ::AbstractArray{T,N}
end

AbstractLinearOperators.domain_size(P::ProjVectorField) = size(P.ξ)
AbstractLinearOperators.range_size(P::ProjVectorField) = size(P.ξ)
AbstractLinearOperators.matvecprod(P::ProjVectorField{T,N}, u::AbstractArray{T,N}) where {T,N} = u-P.ξ.*ptdot(u,P.ξ)
AbstractLinearOperators.matvecprod_adj(P::ProjVectorField{T,N}, u::AbstractArray{T,N}) where {T,N} = P*u

function structural_weight(u::AbstractArray{T,N}; ∇::Union{Nothing,AbstractLinearOperator}=nothing, h::NTuple{N,S}=tuple(ones(Float32,N)...), η::U=0f0) where {T,N,S<:Real,U<:Real}
    ∇ === nothing && (∇ = gradient_operator(size(u), h; T=T); u isa CuArray && (∇ = gpu(∇)))
    ∇u = ∇*u
    return ProjVectorField{T,N+1}(∇u./ptnorm2(∇u; η=real(T)(η)))
end

Flux.gpu(P::ProjVectorField{T,N}) where {T,N} = ProjVectorField{T,N}(gpu(P.ξ))
Flux.cpu(P::ProjVectorField{T,N}) where {T,N} = ProjVectorField{T,N}(cpu(P.ξ))


# Utils

function structural_mean(u::AbstractArray{T,N}; ∇::Union{Nothing,AbstractLinearOperator}=nothing, h::NTuple{N,S}=tuple(ones(Float32,N)...)) where {T,N,S<:Real}
    ∇ === nothing && (∇ = gradient_operator(size(u), h; T=T); u isa CuArray && (∇ = gpu(∇)))
    return sum(ptnorm2(∇*u))[1]/prod(size(u))
end