#: Padding utils

export Padding2D, NoPad2D, PadZero2D, PadCopy2D, PadPeriodic2D, pad, restrict, padding


# Abstract type

"""Expected behavior:
- pad(u::AbstractArray{T,N}, p::Padding)::typeof(u)
- restrict(u::AbstractArray{T,N}, p::Padding)::typeof(u)"""
abstract type Padding2D{T} end


# Concrete type

## No-padding

struct NoPad2D{T}<:Padding2D{T} end

pad(u::AbstractArray{T,N}, ::NoPad2D{T}) where {T,N} = u

restrict(u::AbstractArray{T,N}, ::NoPad2D{T}) where {T,N} = u

## Zero-padding

struct PadZero2D{T}<:Padding2D{T}
    horz1::Int64
    horz2::Int64
    vert1::Int64
    vert2::Int64
end

function pad(u::Array{T,2}, p::PadZero2D{T}) where T
    nx, ny = size(u)
    pu = cat(zeros(T, p.horz1, ny), u, zeros(T, p.horz2, ny); dims=1)
    return cat(zeros(T, size(pu,1), p.vert1), pu, zeros(T, size(pu,1), p.vert2); dims=2)
end

function pad(u::CuArray{T,2}, p::PadZero2D{T}) where T
    nx, ny = size(u)
    pu = cat(CUDA.zeros(T, p.horz1, ny), u, CUDA.zeros(T, p.horz2, ny); dims=1)
    return cat(CUDA.zeros(T, size(pu,1), p.vert1), pu, CUDA.zeros(T, size(pu,1), p.vert2); dims=2)
end

restrict(u::AbstractArray{T,2}, p::PadZero2D{T}) where T = u[p.horz1+1:end-p.horz2, p.vert1+1:end-p.vert2, :, :]

## Copy-padding

struct PadCopy2D{T}<:Padding2D{T}
    horz1::Int64
    horz2::Int64
    vert1::Int64
    vert2::Int64
end

function pad(u::AbstractArray{T,2}, p::PadCopy2D{T}) where T
    pu = cat(repeat(u[1:1, :], outer=(p.horz1,1)), u, repeat(u[end:end, :], outer=(p.horz2,1)); dims=1)
    return cat(repeat(pu[:, 1:1],outer=(1,p.vert1)), pu, repeat(pu[:, end:end], outer=(1,p.vert2)); dims=2)
end

function restrict(u::AbstractArray{T,2}, p::PadCopy2D{T}) where T
    ru = cat(sum(u[1:p.horz1+1, :]; dims=1), u[p.horz1+2:end-p.horz2-1, :], sum(u[end-p.horz2:end, :]; dims=1); dims=1)
    return cat(sum(ru[:, 1:p.vert1+1]; dims=2), ru[:, p.vert1+2:end-p.vert2-1], sum(ru[:, end-p.vert2:end]; dims=2); dims=2)
end

## Periodic-padding

struct PadPeriodic2D{T}<:Padding2D{T}
    horz1::Int64
    horz2::Int64
    vert1::Int64
    vert2::Int64
end

function pad(u::AbstractArray{T,2}, p::PadPeriodic2D{T}) where T
    nx, ny = size(u)
    o1 = u[end-p.horz1+1:end,:]
    o2 = u[1:p.horz2,:]
    pu = cat(o1, u, o2; dims=1)
    o1 = pu[:,end-p.vert1+1:end]
    o2 = pu[:,1:p.vert2]
    return cat(o1, pu, o2; dims=2)
end

function restrict(u::AbstractArray{T,2}, p::PadPeriodic2D{T}) where T
    nx_ext, ny_ext = size(u)
    ru_ = u[:, p.vert1+1:end-p.vert2]
    ru_[:, 1:p.vert2] += u[:, end-p.vert2+1:end]
    ru_[:, end-p.vert1+1:end] += u[:, 1:p.vert1]
    ru = ru_[p.horz1+1:end-p.horz2, :]
    ru[1:p.horz2, :] += ru_[end-p.horz2+1:end, :]
    ru[end-p.horz1+1:end, :] += ru_[1:p.horz1, :]
    return ru
end


# Utils

function padding(pad_type::String, T::DataType, horz1::Int64, horz2::Int64, vert1::Int64, vert2::Int64)
    pad_type == ""         && (return NoPad2D{T}())
    pad_type == "zero"     && (return PadZero2D{T}(horz1, horz2, vert1, vert2))
    pad_type == "copy"     && (return PadCopy2D{T}(horz1, horz2, vert1, vert2))
    pad_type == "periodic" && (return PadPeriodic2D{T}(horz1, horz2, vert1, vert2))
end