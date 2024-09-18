module Stencils

using ManagedLoops: @unroll

macro gen(expr)
    esc(:(Base.@propagate_inbounds @generated $expr))
end

macro inl(expr)
    esc(:(Base.@propagate_inbounds $expr))
end

struct Stencil{Fun,Coefs}
    fun::Fun # operator to call
    coefs::Coefs # local mesh information
end
@inl (st::Stencil)(args...) = st.fun(st.coefs..., args...)

const Mat = AbstractMatrix

#======================= dot product ======================#

# the factor 1/2 for the Perot formula is incorporated into inv_area
# inv_area is incorporated into hodges
@gen dot_product(domain, ij, v::Val{N}, radius) where {N} = quote
    inv_area = inv(2 * radius * radius * domain.Ai[ij])
    edges = @unroll (domain.primal_edge[e, ij] for e = 1:$N)
    hodges = @unroll (inv_area * domain.le_de[edges[e]] for e = 1:$N)
    return Stencil(get_dot_product, (v, edges, hodges))
end

@gen get_dot_product(::Val{N}, edges, hodges, ucov, vcov, k) where {N} = quote
    @unroll sum(hodges[e] * (ucov[k, edges[e]] * vcov[k, edges[e]]) for e = 1:$N)
end

#============ the following stencils do not use metric information
# and should be part of CFDomains ===============================#

#======================== averaging =======================#

# cell -> edge
@inl average_ie(domain, ij) =
    Stencil(get_average_ie, (domain.edge_left_right[1, ij], domain.edge_left_right[2, ij]))

@inl get_average_ie(left, right, mass, k) = (mass[k, left] + mass[k, right]) / 2

# cell -> vertex
@inl function average_iv(domain, ij::Int)
    cells = @unroll (domain.dual_vertex[e, ij] for e = 1:3)
    weights = @unroll (domain.Riv2[e, ij] for e = 1:3)
    return Stencil(get_average_iv, (cells, weights))
end
@inl get_average_iv(cells, weights, mass, k) =
    @unroll sum(weights[e] * mass[k, cells[e]] for e = 1:3)

# vertex -> edge
@inl average_ve(domain, ij::Int) =
    Stencil(get_average_ve, (domain.edge_down_up[1, ij], domain.edge_down_up[2]))

@inl get_average_ve(up, down, qv, k) = (qv[k, down] + qv[k, up]) / 2

#======================= centered flux ======================#

# le_de includes the factor 1/2 for the centered average
@inl function centered_flux(domain, ij::Int)
    left_right, le_de = domain.edge_left_right, domain.le_de
    Stencil(get_centered_flux, (ij, left_right[1, ij], left_right[2, ij], le_de[ij] / 2))
end

# Makes sense for a conformal metric.
# It is the job of the caller to multiply the covariant velocity
# `ucov` (which has units m^2/s), or the flux, by the
# contravariant metric factor (which has units m^-2) so that,
# if mass is in kg, the flux and its divergence are in kg/s.
@inl get_centered_flux(ij, left, right, le_de, mass, ucov, k) =
    le_de * ucov[k, ij] * (mass[k, left] + mass[k, right])

#========================= divergence =======================#

# flux must be a contravariant vector density = 2-form in 3D space
# in X/s for the flux of X

# signs include the inv_area factor
@gen divergence(domain, ij::Int, v::Val{N}) where {N} = quote
    inv_area = inv(domain.Ai[ij])
    edges = @unroll (domain.primal_edge[e, ij] for e = 1:$N)
    signs = @unroll (inv_area * domain.primal_ne[e, ij] for e = 1:$N)
    return Stencil(get_divergence, (v, edges, signs))
end

@gen get_divergence(::Val{N}, edges, signs, flux, k) where {N} = quote
    @unroll sum(flux[k, edges[e]] * signs[e] for e = 1:$N)
end

#========================= curl =====================#

@inl function curl(domain, ij::Int)
    F = eltype(domain.Riv2)
    edges = @unroll (domain.dual_edge[e, ij] for e = 1:3)
    signs = @unroll (F(domain.dual_ne[e, ij]) for e = 1:3)
    return Stencil(get_curl, (edges, signs))
end

@inl get_curl(edges, signs, ucov, k) = @unroll sum(ucov[k, edges[e]] * signs[e] for e = 1:3)

#========================= gradient =====================#

@inl gradient(domain, ij::Int) =
    Stencil(get_gradient, (domain.edge_left_right[1, ij], domain.edge_left_right[2, ij]))

@inl get_gradient(left, right, q, k) = q[k, right] - q[k, left]

#=========================== TRiSK ======================#

# weight inclues the factor 1/2 of the centered average of qe
@inl TRiSK(domain, ij::Int, edge::Int) =
    Stencil(get_TRiSK, (ij, domain.trisk[edge, ij], domain.wee[edge, ij] / 2))

@inl get_TRiSK(ij, edge, weight, du, U, qe, k) =
    muladd(weight * U[k, edge], qe[k, ij] + qe[k, edge], du[k, ij])

end #===== module ====#
