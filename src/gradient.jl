#: Differential operators

export Gradient2D, CuGradient2D, gradient_2D


# Gradient (linear operator)

struct Gradient2D{T}<:AbstractLinearOperator{Array{T,2},Array{T,3}}
    spacing::Tuple{T,T}
    padding::Padding2D{T}
    cdims::DenseConvDims
    stencil::Array{T,4}
end

AbstractLinearOperators.domain_size(D::Gradient2D) = D.cdims.I.-(1,1)
AbstractLinearOperators.range_size(D::Gradient2D) = (D.cdims.I.-(1,1)..., 2)
AbstractLinearOperators.matvecprod(D::Gradient2D{T}, u::Array{T,2}) where T = conv(reshape(pad(u, D.padding), size(u,1)+1,size(u,2)+1, 1, 1), D.stencil, D.cdims)[:,:,:,1]
AbstractLinearOperators.matvecprod_adj(D::Gradient2D{T}, v::Array{T,3}) where T = restrict(∇conv_data(reshape(v, size(v)..., 1), D.stencil, D.cdims)[:,:,1,1], D.padding)

struct CuGradient2D{T}<:AbstractLinearOperator{CuArray{T,2},CuArray{T,3}}
    spacing::Tuple{T,T}
    padding::Padding2D{T}
    cdims::DenseConvDims
    stencil::CuArray{T,4}
end

AbstractLinearOperators.domain_size(D::CuGradient2D) = D.cdims.I.-(1,1)
AbstractLinearOperators.range_size(D::CuGradient2D) = (D.cdims.I.-(1,1)..., 2)
AbstractLinearOperators.matvecprod(D::CuGradient2D{T}, u::CuArray{T,2}) where T = conv(reshape(pad(u, D.padding), size(u,1)+1,size(u,2)+1, 1, 1), D.stencil, D.cdims)[:,:,:,1]
AbstractLinearOperators.matvecprod_adj(D::CuGradient2D{T}, v::CuArray{T,3}) where T = restrict(∇conv_data(reshape(v, size(v)..., 1), D.stencil, D.cdims)[:,:,1,1], D.padding)

function gradient_2D(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), pad_type::String="periodic", gpu::Bool=false) where T

    # Convolution
    D = cat(reshape([T(0) T(1)/h[1]; T(0) T(-1)/h[1]], 2, 2, 1, 1), reshape([T(0) T(0); T(1)/h[2] T(-1)/h[2]], 2, 2, 1, 1); dims=4)
    gpu && (D = D |> Flux.gpu)
    cdims = DenseConvDims((n[1]+1,n[2]+1,1,1), (2,2,1,2); stride=(1,1), padding=(0,0))

    # Padding
    P = padding(pad_type, T, 0, 1, 0, 1)

    gpu ? (return CuGradient2D{T}(h, P, cdims, D)) : (return Gradient2D{T}(h, P, cdims, D))

end

Flux.gpu(D::Gradient2D{T}) where T = CuGradient2D{T}(D.spacing, D.padding, D.cdims, gpu(D.stencil))
Flux.gpu(D::CuGradient2D{T}) where T = D
Flux.cpu(D::Gradient2D{T}) where T = D
Flux.cpu(D::CuGradient2D{T}) where T = Gradient2D{T}(D.spacing, D.padding, D.cdims, cpu(D.stencil))


# Gradient (functional)

function gradient_2D(u::Array{T,2}, ::PadZero2D{T}; h::Tuple{T,T}=(T(1),T(1))) where T
    v = Array{T,3}(undef, size(u)..., 2)
    v[1:end-1,:,1] = (u[2:end,:]-u[1:end-1,:])/h[1]
    v[end,    :,1] = (-u[end,:])/h[1]
    v[:,1:end-1,2] = (u[:,2:end]-u[:,1:end-1])/h[2]
    v[:,end,    2] = (-u[:,end])/h[2]
    return v
end

function gradient_2D(u::CuArray{T,2}, ::PadZero2D{T}; h::Tuple{T,T}=(T(1),T(1))) where T
    v = CuArray{T,3}(undef, size(u)..., 2)
    v[1:end-1,:,1] = (u[2:end,:]-u[1:end-1,:])/h[1]
    v[end,    :,1] = (-u[end,:])/h[1]
    v[:,1:end-1,2] = (u[:,2:end]-u[:,1:end-1])/h[2]
    v[:,end,    2] = (-u[:,end])/h[2]
    return v
end

function gradient_2D(u::Array{T,2}, ::PadCopy2D{T}; h::Tuple{T,T}=(T(1),T(1))) where T
    v = Array{T,3}(undef, size(u)..., 2)
    v[1:end-1,:,1] = (u[2:end,:]-u[1:end-1,:])/h[1]
    v[end,    :,1] .= T(0)
    v[:,1:end-1,2] = (u[:,2:end]-u[:,1:end-1])/h[2]
    v[:,end,    2] .= T(0)
    return v
end

function gradient_2D(u::CuArray{T,2}, ::PadCopy2D{T}; h::Tuple{T,T}=(T(1),T(1))) where T
    v = CuArray{T,3}(undef, size(u)..., 2)
    v[1:end-1,:,1] = (u[2:end,:]-u[1:end-1,:])/h[1]
    v[end,    :,1] .= T(0)
    v[:,1:end-1,2] = (u[:,2:end]-u[:,1:end-1])/h[2]
    v[:,end,    2] .= T(0)
    return v
end

function gradient_2D(u::Array{T,2}, ::PadPeriodic2D{T}; h::Tuple{T,T}=(T(1),T(1))) where T
    v = Array{T,3}(undef, size(u)..., 2)
    v[1:end-1,:,1] = (u[2:end,:]-u[1:end-1,:])/h[1]
    v[end,    :,1] = (u[1,:]-u[end,:])/h[1]
    v[:,1:end-1,2] = (u[:,2:end]-u[:,1:end-1])/h[2]
    v[:,end,    2] = (u[:,1]-u[:,end])/h[2]
    return v
end

function gradient_2D(u::CuArray{T,2}, ::PadPeriodic2D{T}; h::Tuple{T,T}=(T(1),T(1))) where T
    v = CuArray{T,3}(undef, size(u)..., 2)
    v[1:end-1,:,1] = (u[2:end,:]-u[1:end-1,:])/h[1]
    v[end,    :,1] = (u[1,:]-u[end,:])/h[1]
    v[:,1:end-1,2] = (u[:,2:end]-u[:,1:end-1])/h[2]
    v[:,end,    2] = (u[:,1]-u[:,end])/h[2]
    return v
end

gradient_2D(u::AbstractArray{T,2}; p::Padding2D{T}=PadPeriodic2D{T}(0,1,0,1), h::Tuple{T,T}=(T(1),T(1))) where T = gradient_2D(u, p; h=h)