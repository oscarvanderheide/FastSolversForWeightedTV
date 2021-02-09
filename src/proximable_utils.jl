#: Proximable function utilities


export ell_norm


# ℓnorm

struct ℓnorm{T,p1,p2}<:ProximableFunction{AbstractArray{T,3}} end

ell_norm(T::DataType, p1::Number, p2::Number) = ℓnorm{T,p1,p2}()

function proxy!(p::DT, λ::T, ::ℓnorm{T,2,1}, q::DT; eps::T=T(0)) where {T,DT<:AbstractArray{T,3}}
    ptn = ptnorm2(p; eps=eps)
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return sum(ptn)
end