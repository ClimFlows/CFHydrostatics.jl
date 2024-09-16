module Stencils

using ManagedLoops: @unroll

macro gen(expr)
    esc(:(@inline Base.@propagate_inbounds @generated $expr))
end

macro inl(expr)
    esc(:(@inline Base.@propagate_inbounds $expr))
end

#======================= dot product ======================#

@gen dot_product(::Val{degree}, ij::Int, domain, radius) where {degree} =
    quote
        inv_area = inv(radius * radius * domain.Ai[ij])
        edges = @unroll (domain.primal_edge[e, ij] for e = 1:$degree)
        hodges = @unroll (domain.le_de[edges[e]] for e = 1:$degree)
        return (; inv_area, edges, hodges)
    end

@gen dot_product(::Val{degree}, st::NamedTuple, k, ucov::M, vcov::M) where {degree,M<:AbstractMatrix} =
    quote
        inv_area, edges, hodges = st
        @unroll inv_area *
                sum(hodges[e] * (ucov[k, edges[e]] * vcov[k, edges[e]]) for e = 1:$degree)
    end

#======================= centered flux ======================#

# le_de non-dimensional
# ucov in m^2/s
# mass in kg/m² (rho.dz)
# => this flux in kg/s
# rescaled to kg/s/m^2 by caller using metric factor
# => div(flux) in kg/m^2/s

@inl centered_flux(ij::Int, domain) =
    ij, domain.edge_left_right[1, ij], domain.edge_left_right[2, ij], domain.le_de[ij]

@inl function centered_flux(st::Tuple, k, mass::M, ucov::M) where {M<:AbstractMatrix}
    ij, left, right, le_de = st
    return le_de * ucov[k, ij] * (mass[k, left] + mass[k, right])/2
end

#========================= divergence =======================#

# belongs to CFDomains
# flux must be a contravariant vector density = 2-form in 3D space
# in X/s for the flux of X

@gen divergence(::Val{degree}, ij::Int, domain) where {degree} = quote
    inv_area = inv(domain.Ai[ij])
    edges = @unroll (domain.primal_edge[e, ij] for e = 1:$degree)
    signs = @unroll (domain.primal_ne[e, ij] for e=1:$degree)
    return (; inv_area, edges, signs)
end

@gen divergence(::Val{degree}, st::NamedTuple, k, flux::AbstractMatrix) where degree = quote
    (; inv_area, edges, signs) = st
    return @unroll inv_area * sum(flux[k, edges[e]]*signs[e] for e = 1:$degree)
end

end #===== module ====#
