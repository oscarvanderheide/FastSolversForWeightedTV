export structural_weight


# Projection on vector field (2-D)

struct ProjVectorField_2D{T}<:AbstractLinearOperator{AbstractArray{T,3},AbstractArray{T,3}}
    ∇ŵ::AbstractArray{T,3}
end

AbstractLinearOperators.domain_size(P::ProjVectorField_2D) = size(P.∇ŵ)
AbstractLinearOperators.range_size(P::ProjVectorField_2D) = size(P.∇ŵ)
AbstractLinearOperators.matvecprod(P::ProjVectorField_2D{T}, u::AbstractArray{T,3}) where {T<:Number} = u-P.∇ŵ.*ptdot(P.∇ŵ,u)
AbstractLinearOperators.matvecprod_adj(P::ProjVectorField_2D{T}, u::AbstractArray{T,3}) where {T<:Number} = P*u

structural_weight(P::ProjVectorField_2D) = P.∇ŵ

function structural_weight(u::AbstractArray{T,2}; ∇::Union{Nothing,AbstractGradient_2D{T}}=nothing, h::NTuple{2,S}=(1f0,1f0), η::U=0f0) where {T<:Number, S<:Real, U<:Real}
    ∇ === nothing && (∇ = gradient_2D(size(u); T=T, h=h); u isa CuArray && (∇ = gpu(∇)))
    ∇u = ∇*u
    return ProjVectorField_2D{T}(∇u./ptnorm2(∇u; η=T(η)))
end

Flux.gpu(P::ProjVectorField_2D{T}) where {T<:Real} = ProjVectorField_2D{T}(gpu(structural_weight(P)))
Flux.cpu(P::ProjVectorField_2D{T}) where {T<:Real} = ProjVectorField_2D{T}(cpu(structural_weight(P)))


# Projection on vector field (3-D)

struct ProjVectorField_3D{T}<:AbstractLinearOperator{AbstractArray{T,4},AbstractArray{T,4}}
    ∇ŵ::AbstractArray{T,4}
end

AbstractLinearOperators.domain_size(P::ProjVectorField_3D) = size(P.∇ŵ)
AbstractLinearOperators.range_size(P::ProjVectorField_3D) = size(P.∇ŵ)
AbstractLinearOperators.matvecprod(P::ProjVectorField_3D{T}, u::AbstractArray{T,4}) where {T<:Number} = u-P.∇ŵ.*ptdot(P.∇ŵ,u)
AbstractLinearOperators.matvecprod_adj(P::ProjVectorField_3D{T}, u::AbstractArray{T,4}) where {T<:Number} = P*u

structural_weight(P::ProjVectorField_3D) = P.∇ŵ

function structural_weight(u::AbstractArray{T,3}; ∇::Union{Nothing,AbstractGradient_3D{T}}=nothing, h::NTuple{3,S}=(1f0,1f0,1f0), η::U=0f0) where {T<:Number, S<:Real, U<:Real}
    ∇ === nothing && (∇ = gradient_3D(size(u); T=T, h=h); u isa CuArray && (∇ = gpu(∇)))
    ∇u = ∇*u
    return ProjVectorField_3D{T}(∇u./ptnorm2(∇u; η=T(η)))
end

Flux.gpu(P::ProjVectorField_3D{T}) where {T<:Real} = ProjVectorField_3D{T}(gpu(structural_weight(P)))
Flux.cpu(P::ProjVectorField_3D{T}) where {T<:Real} = ProjVectorField_3D{T}(cpu(structural_weight(P)))