#: Utilities for computing composite norms

export ptnorm1, ptnorm2, ptnormInf, norm21, norm22, norm2Inf, normA21, normTV


# Point-wise norms for vector fields

ptnorm1(p::AbstractArray{T,3}; eps::T=T(0)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+eps
ptnorm2(p::AbstractArray{T,3}; eps::T=T(0)) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2).+eps^2)
ptnormInf(p::AbstractArray{T,3}; eps::T=T(0)) where T = maximum(abs.(p).+eps; dims=3)[:,:,1]


# Norm for vector fields

function norm21(v::Array{T,3}; eps::T=T(0)) where T
    n = T(0)
    @inbounds for i = 1:size(v,1), j = 1:size(v,2)
        n += sqrt(v[i,j,1]^2+v[i,j,2]^2+eps^2)
    end
    return n
end
norm21(v::CuArray{T,3}; eps::T=T(0)) where T = sum(ptnorm2(v; eps=eps))

norm22(v::Array{T,3}; eps::T=T(0)) where T = norm(v,2)
norm22(v::CuArray{T,3}; eps::T=T(0)) where T = norm(v)

function norm2Inf(v::Array{T,3}; eps::T=T(0)) where T
    n = T(0)
    @inbounds for i = 1:size(v,1), j = 1:size(v,2)
        n = max(n, sqrt(v[i,j,1]^2+v[i,j,2]^2+eps^2))
    end
    return n
end
norm2Inf(v::CuArray{T,3}; eps::T=T(0)) where T = maximum(ptnorm2(v; eps=eps))


# TV norm

normA21(x::DT, A::AbstractLinearOperator{DT,RT}; eps::T=T(0)) where {T,DT<:AbstractArray{T,2},RT<:AbstractArray{T,3}} = norm21(A*x; eps=eps)
normTV(x::AbstractArray{T,2}; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1), eps::T=T(0)) where T = norm21(gradient_2D(x, p; h=h); eps=eps)