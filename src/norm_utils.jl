#: Utilities for computing composite norms

export ptnorm1, ptnorm2, ptnormInf


# Point-wise norms for vector fields

ptnorm1(p::AbstractArray{T,3}; eps::T=T(1e-20)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+eps
ptnorm2(p::AbstractArray{T,3}; eps::T=T(1e-20)) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2).+eps^2)
ptnormInf(p::AbstractArray{T,3}; eps::T=T(1e-20)) where T = maximum(abs.(p).+eps; dims=3)[:,:,1]