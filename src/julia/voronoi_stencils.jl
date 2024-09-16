module Stencils

macro gen(expr)
    code = :(
        @inline Base.@propagate_inbounds @generated $expr
    )
    return esc(code)
end

#======================= stencils ======================#

@gen dot_product(::Val{degree}, stencil, k, ucov::M, vcov::M) where {degree, M<:Matrix} = quote
    inv_area, edges, hodges = stencil
    @unroll inv_area * sum(hodges[e] * (ucov[k, edges[e]] * vcov[k, edges[e]]) for e = 1:$degree)
end

@gen stencil(::typeof(dot_product), ::Val{degree}, ij, radius, domain) where degree = quote
    inv_area = inv(radius * radius * domain.Ai[ij])
    edges = @unroll (domain.primal_edge[e, ij] for e = 1:$degree)
    hodges = @unroll (domain.le_de[edges[e]] for e = 1:$degree)
    return inv_area, edges, hodges
end

end

