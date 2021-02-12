#: Utilities for computing composite norms

export ptdot, ptnorm1, ptnorm2, ptnorm2_reg, ptnormInf, norm21, norm22, norm2Inf, normA21, normTV, normTsqV, normBV


# Point-wise dot/norms for vector fields

ptdot(v1::AbstractArray{T,3}, v2::AbstractArray{T,3}) where T = sum(v1.*v2; dims=3)[:,:,1]
ptnorm1(p::AbstractArray{T,3}) where T = abs.(p[:,:,1])+abs.(p[:,:,2])
ptnorm2(p::AbstractArray{T,3}) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2))
ptnorm2_reg(p::AbstractArray{T,3}; η::T=T(0)) where T = sqrt.(p[:,:,1].^2+p[:,:,2].^2 .+η^2)
ptnormInf(p::AbstractArray{T,3}) where T = maximum(abs.(p); dims=3)[:,:,1]


# Norm for vector fields

function norm21(v::Array{T,3}) where T
    n = T(0)
    @inbounds for i = 1:size(v,1), j = 1:size(v,2)
        n += sqrt(v[i,j,1]^2+v[i,j,2]^2)
    end
    return n
end
norm21(v::CuArray{T,3}) where T = sum(ptnorm2(v))

norm22(v::Array{T,3}) where T = norm(v,2)
norm22(v::CuArray{T,3}) where T = norm(v)

function norm2Inf(v::Array{T,3}) where T
    n = T(0)
    @inbounds for i = 1:size(v,1), j = 1:size(v,2)
        n = max(n, sqrt(v[i,j,1]^2+v[i,j,2]^2))
    end
    return n
end
norm2Inf(v::CuArray{T,3}) where T = maximum(ptnorm2(v))


# TV-related norm

normA21(x::DT, A::AbstractLinearOperator{DT,RT}) where {T,DT<:AbstractArray{T,2},RT<:AbstractArray{T,3}} = norm21(A*x)

## 2,1

normTV(x::AbstractArray{T,2}; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where T = norm21(gradient_2D(x, p; h=h))
normTV(x::DT, y::DT; h::Tuple{T,T}=(T(1),T(1)), η::T=T(0), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where {T,DT<:AbstractArray{T,2}} = norm21(projvectorfield_2D(gradient_2D(x, p; h=h), gradient_2D(y, p; h=h); η=η))

## 2,2
normTsqV(x::AbstractArray{T,2}; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where T = norm22(gradient_2D(x, p; h=h))
normTsqV(x::DT, y::DT; h::Tuple{T,T}=(T(1),T(1)), η::T=T(0), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where {T,DT<:AbstractArray{T,2}} = norm22(projvectorfield_2D(gradient_2D(x, p; h=h), gradient_2D(y, p; h=h); η=η))

## 2,Inf

normBV(x::AbstractArray{T,2}; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where T = norm2Inf(gradient_2D(x, p; h=h))
normBV(x::DT, y::DT; h::Tuple{T,T}=(T(1),T(1)), η::T=T(0), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where {T,DT<:AbstractArray{T,2}} = norm2Inf(projvectorfield_2D(gradient_2D(x, p; h=h), gradient_2D(y, p; h=h); η=η))