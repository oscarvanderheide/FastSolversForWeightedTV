#: Utilities for weighted gradient projection

export ProjVectorField2D, normalize_vectorfield, projvectorfield_2D


# Type

struct ProjVectorField2D{T}<:AbstractLinearOperator{Array{T,3},Array{T,3}}
    vhat::Array{T,3}
end

AbstractLinearOperators.domain_size(P::ProjVectorField2D) = size(P.vhat)
AbstractLinearOperators.range_size(P::ProjVectorField2D) = size(P.vhat)
AbstractLinearOperators.matvecprod(P::ProjVectorField2D{T}, u::Array{T,3}) where T = u-ptdot(u,P.vhat).*P.vhat
AbstractLinearOperators.matvecprod_adj(P::ProjVectorField2D{T}, u::Array{T,3}) where T = u-ptdot(u,P.vhat).*P.vhat

struct CuProjVectorField2D{T}<:AbstractLinearOperator{CuArray{T,3},CuArray{T,3}}
    vhat::CuArray{T,3}
end

AbstractLinearOperators.domain_size(P::CuProjVectorField2D) = size(P.vhat)
AbstractLinearOperators.range_size(P::CuProjVectorField2D) = size(P.vhat)
AbstractLinearOperators.matvecprod(P::CuProjVectorField2D{T}, u::CuArray{T,3}) where T = u-ptdot(u,P.vhat).*P.vhat
AbstractLinearOperators.matvecprod_adj(P::CuProjVectorField2D{T}, u::CuArray{T,3}) where T = u-ptdot(u,P.vhat).*P.vhat


# Constructor

normalize_vectorfield(v::AbstractArray{T,3}; η::T=T(0)) where T = v./ptnorm2(v; η=η)
projvectorfield_2D(v::Array{T,3}; η::T=T(0)) where T = ProjVectorField2D{T}(normalize_vectorfield(v; η=η))
projvectorfield_2D(v::CuArray{T,3}; η::T=T(0)) where T = CuProjVectorField2D{T}(normalize_vectorfield(v; η=η))


# Utils

function projvectorfield_2D(u::Array{T,3}, v::Array{T,3}; η::T=T(0)) where T
    vhat = normalize_vectorfield(v; η=η)
    return u-ptdot(u,vhat).*vhat
end