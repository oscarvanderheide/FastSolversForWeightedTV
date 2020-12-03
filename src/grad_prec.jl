# Gradient preconditioning for TV


export identityMap, identityMapVectorField2D


identityMap(n::Tuple{Int64,Int64}) = LinearMap{T}(identity, identity, prod(n), prod(n))
identityMapVectorField2D(n::Tuple{Int64,Int64}) = LinearMap{T}(identity, identity, 2*prod(n), 2*prod(n))