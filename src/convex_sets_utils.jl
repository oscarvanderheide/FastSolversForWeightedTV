#: Convex set utilities


export ℓball_2D, ell_ball, upperlim_constraints_2D, lowerlim_constraints_2D, box_constraints_2D, A21ball_2D, tv_ball_2D


# ℓball_2D{p1,p2} ε-ball: C={x:||x||_{p1,p2}<=ε}

## 2,Inf

struct ℓball_2D{T,p1,p2}<:ConvexSet{T,3}
    ε::T
    eps::T
end

ell_ball(p1::Number, p2::Number, ε::T; eps::T=T(0)) where T = ℓball_2D{T,p1,p2}(ε, eps)

function project!(p::Array{T,3}, C::ℓball_2D{T,2,Inf}, q::Array{T,3}) where T
    ptn = ptnorm2(p; eps=C.eps)
    # nx, ny, _ = size(p)
    # @inbounds for i=1:nx, j=1:ny, k=1:2
    #     q[i,j,k] = p[i,j,k]*((ptn[i,j]>=C.ε)/ptn[i,j]+(ptn[i,j]<C.ε))
    # end
    q .= p.*((ptn.>=C.ε)./ptn+(ptn.<C.ε))
    return q
end

function project!(p::CuArray{T,3}, C::ℓball_2D{T,2,Inf}, q::CuArray{T,3}) where T
    ptn = ptnorm2(p; eps=C.eps)
    q .= p.*((ptn.>=C.ε)./ptn+(ptn.<C.ε))
    return q
end

## 2,1

function project!(p::Array{T,3}, C::ℓball_2D{T,2,1}, q::Array{T,3}) where T
    # @inbounds for i = 1:length(q)
    #     q[i] = p[i]/C.ε
    # end
    q .= p./C.ε
    ptn = ptnorm2(q; eps=C.eps)
    λ = pareto_search_proj21(ptn)
    proxy!(q, λ, ell_norm(T,2,1;eps=C.eps), q; ptn=ptn)
    # @inbounds for i = 1:length(q)
    #     q[i] = C.ε*q[i]
    # end
    q .= C.ε*q
    return q
end

function project!(p::CuArray{T,3}, C::ℓball_2D{T,2,1}, q::CuArray{T,3}) where T
    q .= p./C.ε
    ptn = ptnorm2(q; eps=C.eps)
    λ = pareto_search_proj21(ptn)
    proxy!(q, λ, ell_norm(T,2,1;eps=C.eps), q; ptn=ptn)
    q .= C.ε*q
    return q
end

function pareto_search_proj21(ptn::AbstractArray{T,2}) where T
    obj_fun = δ->objfun_paretosearch_proj21(δ, ptn)
    return find_zero(obj_fun, (T(0), maximum(ptn)))
end

function objfun_paretosearch_proj21(δ::T, ptn::Array{T,2}) where T
    # f = T(1)
    # @inbounds for i=1:length(ptn)
    #     ptn[i]>=δ && (f -= ptn[i]-δ)
    # end
    # ptn_ = ptn[ptn.>=δ]
    # ~isempty(ptn_) ? (return T(1)-sum(ptn_.-δ)) : (return T(1))
    # return f
    ptn_ = ptn[ptn.>=δ]
    return T(1)-sum(ptn_)+length(ptn_)*δ
end

function objfun_paretosearch_proj21(δ::T, ptn::CuArray{T,2}) where T
    # ptn_ = ptn[ptn.>=δ]
    # ~isempty(ptn_) ? (return T(1)-sum(ptn_.-δ)) : (return T(1))
    ptn_ = ptn[ptn.>=δ]
    return T(1)-sum(ptn_)+length(ptn_)*δ
end


# A21-ball & TV

struct A21ball_2D{T}<:ConvexSet{T,2}
    A::AbstractLinearOperator
    ε::T
    eps::T
end

function project!(y::DT, C::A21ball_2D{T}, x::DT; opt::OptimOptions{T}=opt_fista(), dual_est::Bool=false) where {T,DT<:AbstractArray{T,2}}

    # Least-squares misfit
    f = leastsquares_misfit(adjoint(C.A), y)

    # 2-Inf norm
    g = ell_norm(T,2,Inf; eps=C.eps)

    # Minimization
    p = minimize(f+C.ε*g, opt)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p

    dual_est ? (return x, p) : (return x)

end

project(y::AbstractArray{T,2}, C::A21ball_2D{T}; opt::OptimOptions{T}=opt_fista(), dual_est::Bool=false) where T = project!(y, C, similar(y); opt=opt, dual_est=dual_est)

tv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), eps::T=T(0), gpu::Bool=false) where T = A21ball_2D{T}(gradient_2D(n; h=h, gpu=gpu), ε, eps)


# Box constraints

abstract type AbstractValueConstraints2D{T}<:ConvexSet{T,2} end

struct LowerLimConstraints2D{T}<:AbstractValueConstraints2D{T}
    a::T
end

function project!(x::Array{T,2}, C::LowerLimConstraints2D{T}, y::Array{T,2}) where T
    @inbounds for i = 1:length(x)
        x[i] < C.a ? (y[i] = C.a) : (y[i] = x[i])
    end
    return y
end

function project!(x::CuArray{T,2}, C::LowerLimConstraints2D{T}, y::CuArray{T,2}) where T
    idx = x.<C.a
    y[idx] .= C.a
    y[(!).(idx)] .= x[(!).(idx)]
    return y
end

struct UpperLimConstraints2D{T}<:AbstractValueConstraints2D{T}
    b::T
end

function project!(x::Array{T,2}, C::UpperLimConstraints2D{T}, y::Array{T,2}) where T
    @inbounds for i = 1:length(x)
        x[i] > C.b ? (y[i] = C.b) : (y[i] = x[i])
    end
    return y
end

function project!(x::CuArray{T,2}, C::UpperLimConstraints2D{T}, y::CuArray{T,2}) where T
    idx = x.>C.b
    y[idx] .= C.b
    y[(!).(idx)] .= x[(!).(idx)]
    return y
end

struct BoxConstraints2D{T}<:AbstractValueConstraints2D{T}
    a::T
    b::T
end

function project!(x::Array{T,2}, C::BoxConstraints2D{T}, y::Array{T,2}) where T
    @inbounds for i = 1:length(x)
        y[i] = x[i]
        x[i] > C.b && (y[i] = C.b)
        x[i] < C.a && (y[i] = C.a)
    end
    return y
end

function project!(x::CuArray{T,2}, C::BoxConstraints2D{T}, y::CuArray{T,2}) where T
    idx_a = x.<C.a
    idx_b = x.>C.b
    y[idx_a] .= C.a
    y[idx_b] .= C.b
    y[(!).(idx_a) || (!).(idx_b)] .= x[(!).(idx_a) || (!).(idx_b)]
    return y
end

upperlim_constraints_2D(b::T) where T = UpperLimConstraints2D{T}(b)
lowerlim_constraints_2D(a::T) where T = LowerLimConstraints2D{T}(a)
box_constraints_2D(a::T, b::T) where T = BoxConstraints2D{T}(a, b)