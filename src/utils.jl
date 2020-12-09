#: Utils

export update!

update!(x_::DT, x::DT) where {DT<:AbstractField2D} = (x_.array .= x.array)