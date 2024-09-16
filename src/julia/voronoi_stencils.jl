module Stencils

using ManagedLoops: @unroll

macro gen(expr)
    esc(:(@inline Base.@propagate_inbounds @generated $expr))
end

macro inl(expr)
    esc(:(@inline Base.@propagate_inbounds $expr))
end

#======================= dot product ======================#

# the factor 1/2 for the Perot formula is incorporated into inv_area

@gen dot_product(::Val{N}, ij::Int, domain, radius) where {N} = quote
    inv_area = inv(2 * radius * radius * domain.Ai[ij])
    edges = @unroll (domain.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (domain.le_de[edges[e]] for e = 1:$N)
    return (; inv_area, edges, hodges)
end

const AbsMat = AbstractMatrix

@gen dot_product(::Val{N}, st::NamedTuple, k, ucov::M, vcov::M) where {N,M<:AbsMat} = quote
    inv_area, edges, hodges = st
    return inv_area *
           @unroll sum(hodges[e] * (ucov[k, edges[e]] * vcov[k, edges[e]]) for e = 1:$N)
end

#============ the following stencils do not use metric information
# and should be part of CFDomains ===============================#

#======================= centered flux ======================#

# le_de non-dimensional
# ucov in m^2/s
# mass in kg/m² (rho.dz)
# => this flux in kg/s
# rescaled to kg/s/m^2 by caller using metric factor
# => div(flux) in kg/m^2/s

# the factor 1/2 for the centered average is incorporated into le_de
@inl centered_flux(ij::Int, domain) =
    ij, domain.edge_left_right[1, ij], domain.edge_left_right[2, ij], domain.le_de[ij] / 2

@inl function centered_flux(st::Tuple, k, mass::M, ucov::M) where {M<:AbstractMatrix}
    ij, left, right, le_de = st
    return le_de * ucov[k, ij] * (mass[k, left] + mass[k, right])
end

#========================= divergence =======================#

# flux must be a contravariant vector density = 2-form in 3D space
# in X/s for the flux of X

@gen divergence(::Val{N}, ij::Int, domain) where {N} = quote
    inv_area = inv(domain.Ai[ij])
    edges = @unroll (domain.primal_edge[e, ij] for e = 1:$N)
    signs = @unroll (domain.primal_ne[e, ij] for e = 1:$N)
    return (; inv_area, edges, signs)
end

@gen divergence(::Val{N}, st::NamedTuple, k, flux::AbstractMatrix) where {N} = quote
    (; inv_area, edges, signs) = st
    return @unroll inv_area * sum(flux[k, edges[e]] * signs[e] for e = 1:$N)
end

end #===== module ====#
