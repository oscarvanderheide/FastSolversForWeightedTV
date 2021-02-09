#: Convex set utilities


export ConvexSet, project, project!, no_constraints, ell_ball, box_constraints, indicator


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
    y[x.<C.a] .= C.a
    return y
end

struct UpperLimConstraints{DT}<:AbstractValueConstraints{DT}
    b::Number
end

function project!(x::DT, C::UpperLimConstraints{DT}, y::DT) where {T,DT<:AbstractArray{T,2}}
    y[x.>C.b] .= C.b
    return y
end

struct BoxConstraints{DT}<:AbstractValueConstraints{DT}
    a::Number
    b::Number
end

function project!(x::DT, C::BoxConstraints{DT}, y::DT) where {T,DT<:AbstractArray{T,2}}
    y[x.<a] .= C.a
    y[x.>b] .= C.b
    return y
end

function box_constraints(a::Union{Nothing,T}, b::Union{Nothing,T}) where T
    (a === nothing && b === nothing) && return NoConstraints{AbstractArray{T,2}}()
    (a === nothing && b !== nothing) && return UpperLimitConstraints{AbstractArray{T,2}}(b)
    (a !== nothing && b === nothing) && return LowerLimitConstraints{AbstractArray{T,2}}(a)
    (a !== nothing && b !== nothing) && return BoxConstraints{AbstractArray{T,2}}(a, b)
end