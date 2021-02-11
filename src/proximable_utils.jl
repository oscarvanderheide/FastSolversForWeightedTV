#: Proximable function utilities


export ℓnorm_2D, ell_norm


# ℓnorm_2D

struct ℓnorm_2D{T,p1,p2}<:ProximableFunction{T,3} end

ell_norm(T::DataType, p1::Number, p2::Number) = ℓnorm_2D{T,p1,p2}()

## 2, 1

function proxy!(p::Array{T,3}, λ::T, g::ℓnorm_2D{T,2,1}, q::Array{T,3}; ptn::Union{Array{T,2},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2(p))
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return q
end

function proxy!(p::CuArray{T,3}, λ::T, g::ℓnorm_2D{T,2,1}, q::CuArray{T,3}; ptn::Union{CuArray{T,2},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2(p))
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return q
end

## 2, Inf

function proxy!(p::Array{T,3}, λ::T, g::ℓnorm_2D{T,2,Inf}, q::Array{T,3}) where T
    C = ell_ball(2, 1, T(1))
    project!(p/λ, C, q)
    q .= p-λ*q
    return q
end

function proxy!(p::CuArray{T,3}, λ::T, g::ℓnorm_2D{T,2,Inf}, q::CuArray{T,3}) where T
    C = ell_ball(2, 1, T(1))
    project!(p/λ, C, q)
    q .= p-λ*q
    return q
end