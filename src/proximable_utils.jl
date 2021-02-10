#: Proximable function utilities


export ℓnorm_2D, ell_norm


# ℓnorm_2D

struct ℓnorm_2D{T,p1,p2}<:ProximableFunction{T,3}
    eps::T
end

ell_norm(T::DataType, p1::Number, p2::Number; eps::Number=0) = ℓnorm_2D{T,p1,p2}(T(eps))

## 2, 1

function proxy!(p::Array{T,3}, λ::T, g::ℓnorm_2D{T,2,1}, q::Array{T,3}; ptn::Union{Array{T,2},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2(p; eps=g.eps))
    # nx, ny, _ = size(p)
    # @inbounds for i = 1:nx, j=1:ny, k=1:2
    #     q[i,j,k] = (T(1)-λ/ptn[i,j])*(ptn[i,j] >= λ)*p[i,j,k]
    # end
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return q
end

function proxy!(p::CuArray{T,3}, λ::T, g::ℓnorm_2D{T,2,1}, q::CuArray{T,3}; ptn::Union{CuArray{T,2},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2(p; eps=g.eps))
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return q
end

## 2, Inf

function proxy!(p::Array{T,3}, λ::T, g::ℓnorm_2D{T,2,Inf}, q::Array{T,3}) where T
    C = ell_ball(2, 1, T(1); eps=g.eps)
    project!(p/λ, C, q)
    # @inbounds for i=1:length(q)
    #     q[i] = p[i]-λ*q[i]
    # end
    q .= p-λ*q
    return q
end

function proxy!(p::CuArray{T,3}, λ::T, g::ℓnorm_2D{T,2,Inf}, q::CuArray{T,3}) where T
    C = ell_ball(2, 1, T(1); eps=g.eps)
    project!(p/λ, C, q)
    q .= p-λ*q
    return q
end