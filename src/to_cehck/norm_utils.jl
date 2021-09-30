#: Utilities for computing composite norms

export ptdot, ptnorm1, ptnorm2, ptnormInf, norm21, norm22, norm2Inf, normA21, normTV, normTsqV


# Point-wise dot/norms for vector fields

ptdot(v1::AbstractArray{T,3}, v2::AbstractArray{T,3}) where T = sum(v1.*v2; dims=3)[:,:,1]
ptnorm1(p::AbstractArray{T,3}; η::T=eps(T)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+η
ptnorm2(p::AbstractArray{T,3}; η::T=eps(T)) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2).+η^2)
ptnormInf(p::AbstractArray{T,3}; η::T=eps(T)) where T = maximum(abs.(p).+η; dims=3)[:,:,1]


# Norm for vector fields

norm21(v::AbstractArray{T,3}; η::T=eps(T)) where T = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{T,3}; η::T=eps(T)) where T = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{T,3}; η::T=eps(T)) where T = maximum(ptnorm2(v; η=η))


# TV-related norms

normA21(x::DT, A::AbstractLinearOperator{DT,RT}; η::T=eps(T)) where {T,DT<:AbstractArray{T,2},RT<:AbstractArray{T,3}} = norm21(A*x; η=η)

## 2,1

normTV(x::AbstractArray{T,2}; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where T = norm21(gradient_2D(x, p; h=h))
normTV(x::DT, y::DT, η::T; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where {T,DT<:AbstractArray{T,2}} = norm21(projvectorfield_2D(gradient_2D(x, p; h=h), gradient_2D(y, p; h=h); η=η))

## 2,2

normTsqV(x::AbstractArray{T,2}; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where T = norm22(gradient_2D(x, p; h=h))
normTsqV(x::DT, y::DT, η::T; h::Tuple{T,T}=(T(1),T(1)), p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1)) where {T,DT<:AbstractArray{T,2}} = norm22(projvectorfield_2D(gradient_2D(x, p; h=h), gradient_2D(y, p; h=h); η=η))