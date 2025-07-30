# mesh objects
cells(sphere) = eachindex(sphere.xyz_i)
duals(sphere) = eachindex(sphere.xyz_v)
edges(sphere) = eachindex(sphere.xyz_e)

# generic
apply_primal(op, sphere, arg::Vector, more...) = [
    (@unroll N in 5:7 op(sphere, cell, Val(N))(arg, more...)) for
    (cell, N) in enumerate(sphere.primal_deg)
]
apply_primal(op, sphere, arg::Matrix, more...) = [
    (@unroll N in 5:7 op(sphere, cell, Val(N))(arg, more..., k)) for
    k in axes(arg,1), (cell, N) in enumerate(sphere.primal_deg)
]

function apply_primal!(out_, mgr, op, sphere, arg::AbstractMatrix, more...)
    out = similar!(out_, arg, (size(arg,1), length(cells(sphere))))
    @with mgr,
    let (krange, objs) = (axes(arg,1), cells(sphere))
        @inbounds for cell in objs
            N = sphere.primal_deg[cell]
            @unroll N in 5:7 begin
                op_cell = op(sphere, cell, Val(N))
                for k in krange
                    out[k, cell] = op_cell(arg, more..., k)
                end
            end
        end
    end 
    return out
end

apply_trisk(op, sphere, arg::Vector, more...) = [
    (@unroll N in 9:10 op(sphere, edge, Val(N))(arg, more...)) for
    (edge, N) in enumerate(sphere.trisk_deg)
]

apply_trisk(op, sphere, arg::Matrix, more...) = [
    (@unroll N in 9:10 op(sphere, edge, Val(N))(arg, more..., k)) for
    k in axes(arg,1), (edge, N) in enumerate(sphere.trisk_deg)
]

function apply_trisk!(out_, mgr, op, sphere, arg::AbstractMatrix, more...)
    out = similar!(out_, arg, (size(arg,1), length(edges(sphere))))
    @with mgr,
    let (krange, objs) = (axes(arg,1), edges(sphere))
        @inbounds for edge in objs
            N = sphere.trisk_deg[edge]
            @unroll N in 9:10 begin
                op_edge = op(sphere, edge, Val(N))
                @simd for k in krange
                    v = op_edge(arg, more..., k)
                    out[k, edge] = v
                end
            end
        end
    end 
    return out
end

apply(objects, op, sphere, arg::Vector, more...) = [op(sphere, obj)(arg, more...) for obj in objects(sphere)]

function apply!(out_, mgr, objects, op, sphere, arg::AbstractMatrix, more...)
    out = similar!(out_, arg, (size(arg,1), length(objects(sphere))))
    @with mgr,
    let (krange, objs) = (axes(arg,1), objects(sphere))
        @inbounds for obj in objs
            op_obj = op(sphere, obj)
            for k in krange
                v = op_obj(arg, more..., k)
                out[k, obj] = v
            end
        end
    end 
    return out
end

#=
function apply!(out_, mgr, objects, op, sphere, arg::AbstractMatrix, more...)
    out = similar!(out_, arg, (size(arg,1), length(objects(sphere))))
    @with mgr,
    let (k0range, objs) = (1:32, objects(sphere))
        @inbounds for obj in objs
            op_obj = op(sphere, obj)
            for k0 in k0range
                for k in k0:32:size(arg,1)
                    v = op_obj(arg, more..., k)
                    out[k, obj] = v
                end
            end
        end
    end 
    return out
end
=#

# operators
gradient!(out, mgr, sphere, qi) = apply!(out, mgr, edges, Stencils.gradient, sphere, qi)
gradperp!(out, mgr, sphere, qv) = apply!(out, mgr, edges, Stencils.gradperp, sphere, qv)
perp!(out, mgr, sphere, un) = apply!(out, mgr, edges, Stencils.perp, sphere, un)
curl!(out, mgr, sphere, ue) = apply!(out, mgr, duals, Stencils.curl, sphere, ue)
average_iv(out, mgr, sphere, qi) = apply!(out, mgr, duals, Stencils.average_iv, sphere, qi)
average_ie(out, mgr, sphere, qi) = apply!(out, mgr, edges, Stencils.average_ie, sphere, qi)
average_ve(out, mgr, sphere, qv) = apply!(out, mgr, edges, Stencils.average_ve, sphere, qv)
TRiSK!(out, mgr, sphere, args...) = apply_trisk!(out, mgr, Stencils.TRiSK, sphere, args...)
divergence!(out, mgr, sphere, U) = apply_primal!(out, mgr, Stencils.divergence, sphere, U)

gradient3d(sphere, U) = apply_primal(Stencils.gradient3d, sphere, U)

dot(a::NTuple{3,F}, b::NTuple{3,F}) where {F} = @unroll sum(a[i] * b[i] for i = 1:3)
