#: Convex set utilities


export ell_ball, upperlim_constraints, lowerlim_constraints, box_constraints, indicator


# ℓball{p1,p2} ε-ball: C={x:||x||_{p1,p2}<=ε}

## 2,Inf

struct ℓball{T,p1,p2}<:ConvexSet{AbstractArray{T,3}}
    ε::T
end

ell_ball(p1::Number, p2::Number, ε::T) where T = ℓball{T,p1,p2}(ε)

function project!(p::DT, ::ℓball{T,2,Inf}, q::DT; eps::T=T(0)) where {T,DT<:AbstractArray{T,3}}
    ptn = ptnorm2(p; eps=eps)
    q .= p.*((ptn.>=T(1))./ptn+(ptn.<T(1)))
    return q
end

## TO DO: (1,2), (2,2), (Inf,2), ...


# Box constraints

abstract type AbstractValueConstraints{DT}<:ConvexSet{DT} end

struct LowerLimConstraints{DT}<:AbstractValueConstraints{DT}
    a::Number
end

function project!(x::DT, C::LowerLimConstraints{DT}, y::DT) where {T,DT<:AbstractArray{T,2}}
    idx = x.<C.a
    y[idx] .= C.a
    y[(!).(idx)] .= x[(!).(idx)]
    return y
end

struct UpperLimConstraints{DT}<:AbstractValueConstraints{DT}
    b::Number
end

function project!(x::DT, C::UpperLimConstraints{DT}, y::DT) where {T,DT<:AbstractArray{T,2}}
    idx = x.>C.b
    y[idx] .= C.b
    y[(!).(idx)] .= x[(!).(idx)]
    return y
end

struct BoxConstraints{DT}<:AbstractValueConstraints{DT}
    a::Number
    b::Number
end

function project!(x::DT, C::BoxConstraints{DT}, y::DT) where {T,DT<:AbstractArray{T,2}}
    idx_a = x.<C.a
    idx_b = x.>C.b
    y[idx_a] .= C.a
    y[idx_b] .= C.b
    y[(!).(idx_a) || (!).(idx_b)] .= x[(!).(idx_a) || (!).(idx_b)]
    return y
end

upperlim_constraints(b::T) where T = UpperLimConstraints{AbstractArray{T,2}}(b)
lowerlim_constraints(a::T) where T = LowerLimConstraints{AbstractArray{T,2}}(a)
box_constraints(a::T, b::T) where T = BoxConstraints{AbstractArray{T,2}}(a, b)