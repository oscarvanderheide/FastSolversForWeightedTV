#: Utilities for computing composite norms

export ptnorm1, ptnorm2, ptnormInf, norm21, norm22, norm2Inf


# Point-wise norms for vector fields

ptnorm1(p::AbstractArray{T,3}; eps::T=T(1e-20)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+eps
ptnorm2(p::AbstractArray{T,3}; eps::T=T(1e-20)) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2).+eps^2)
ptnormInf(p::AbstractArray{T,3}; eps::T=T(1e-20)) where T = maximum(abs.(p).+eps; dims=3)[:,:,1]


# Norm for vector fields

function norm21(v::Array{T,3}) where T
    n = T(0)
    @inbounds for i = 1:size(v,1), j = 1:size(v,2)
        n += sqrt(v[i,j,1]^2+v[i,j,2]^2)
    end
    return n
end
norm21(v::CuArray{T,3}) where T = sum(ptnorm2(v; eps=T(0)))

norm22(v::AbstractArray{T,3}) where T = norm(v,2)

function norm2Inf(v::Array{T,3}) where T
    n = T(0)
    @inbounds for i = 1:size(v,1), j = 1:size(v,2)
        n = max(n, sqrt(v[i,j,1]^2+v[i,j,2]^2))
    end
    return n
end
norm2Inf(v::CuArray{T,3}) where T = maximum(ptnorm2(v; eps=T(0)))