# Convolution utils


using LinearMaps, NNlib


## Padding type

struct Pad
    x::Tuple{Int64,Int64} # padding along x-dim
    y::Tuple{Int64,Int64} # padding along y-dim
end

function Pad(px1::Int64, px2::Int64, py1::Int64, py2::Int64)
    return Pad((px1, px2), (py1, py2))
end

function Pad()
    return Pad((0, 0), (0, 0))
end


## Padding utils

function pad0(u::AbstractArray{T,2}, p::Pad)
    o1 = cuzeros(u, (p.x[1], size(u, 2)))
    o2 = cuzeros(u, (p.x[2], size(u, 2)))
    pu = cat(o1, u, o2; dims=1)
    o1 = cuzeros(pu, (size(pu, 1), p.y[1]))
    o2 = cuzeros(pu, (size(pu, 1), p.y[2]))
    return cat(o1, pu, o2; dims=2)
end

function restrict(u::AbstractArray{T,2}, p::Pad)
    return u[p.x[1]+1:end-p.x[2], p.y[1]+1:end-p.y[2]]
end


## Convolutions

function ConvLinearMap(n::Tuple{Int64,Int64}, W::AbstractArray{T,2}; pad0_output::Pad=Pad())

    # Convolutional dimensions
    W = reshape(W, size(W)..., 1, 1)
    cdims = DenseConvDims((n...,1,1), size(W); stride=(1,1), padding=(0,0))

    # Fw/Adj evaluations
    f(u) = pad0(conv(reshape(u, size(u)..., 1, 1), W, cdims)[:,:,1,1], pad0_output)
    function fT(u_)
        u = restrict(u_, pad0_output)
        return ∇conv_data(reshape(u, size(u)..., 1, 1), W, cdims)[:,:,1,1]
    end

    return LinearMap{T}(f, fT, prod(n), prod(n))

end