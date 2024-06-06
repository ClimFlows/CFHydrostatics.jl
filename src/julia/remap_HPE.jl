module RemapHPE

using MutatingOrNot: Void
using ManagedLoops: @loops, @vec

using CFDomains: VHLayout, HVLayout
using CFTransport: CFTransport, OneDimFV, minmod_simd

const FV{Rank, Dim, kind} = OneDimFV{kind, Dim, Rank} # to dispatch on FV{Rank}
const AA{Rank, T} = AbstractArray{T, Rank} # to dispatch on AA{Rank}

# convention:
#      fun(non-fields, output fields..., #==# scratch #==# input fields...)
# or   fun(non-fields, output fields..., #==# input fields...)

vanleer(kind, ::HVLayout) = CFTransport.VanLeerScheme(kind, minmod_simd, 2, 2)
vanleer(kind, ::VHLayout) = CFTransport.VanLeerScheme(kind, minmod_simd, 1, 2)
flatten(x::AbstractArray, ::HVLayout{1}) = x
flatten(x::AbstractArray, ::HVLayout{2}) = reshape(x, :, size(x, 3), size(x,4))
flatten(x::Union{Tuple, NamedTuple}, layout) = map(y->flatten(y, layout), x)

absmax(x) = minimum(abs, x), maximum(abs,x)

function remap_density!(mgr, vanleer, new_massq, #==# fluxq, dq, q, #==# massq, mass, flux)
    q = concentrations!(mgr, q, #==# massq, mass)
    dq = slopes!(mgr, vanleer, dq, #==# q)
    fluxq = fluxes!(mgr, vanleer, fluxq, #==# dq, q, mass, flux)
    new_massq = update_density!(mgr, vanleer, new_massq, #==# fluxq, massq)
    return new_massq
end

function remap_scalar!(mgr, vanleer, new_q, #==# fluxq, dq, #==# q, mass, flux)
    dq = slopes!(mgr, vanleer, dq, #==# q)
    fluxq = fluxes!(mgr, vanleer, fluxq, #==# dq, q, mass, flux)
    new_q = update_scalar!(mgr, vanleer, new_q, #==# fluxq, q, flux, mass)
    return new_q
end

#======== concentrations ========#

concentrations!(mgr, ::Void, massq, mass) = concentrations!(mgr, similar(massq), massq, mass)
function concentrations!(mgr, q, massq, mass)
    @assert axes(q) == axes(massq)
    CFTransport.concentrations!(mgr, q, massq, mass)
    return q
end

#========== slopes =========#

slopes!(mgr, vanleer, ::Void, q) = slopes!(mgr, vanleer, similar(q), q)

function slopes!(mgr, vanleer::FV{N}, dq::AA{N}, q::AA{N}) where N
    @assert axes(dq) == axes(q)
    CFTransport.slopes!(mgr, vanleer, dq, q)
    zero_bottom_top!(mgr, vanleer, dq)
    return dq
end

zero_bottom_top!(mgr, ::FV{2,1}, q::AA{2}) = zero_bottom_top_VH!(mgr, q)
zero_bottom_top!(mgr, ::FV{2,2}, q::AA{2}) = zero_bottom_top_HV!(mgr, q)

@loops function zero_bottom_top_HV!(_, q)
    let range = axes(q,1)
        @vec for ij in range
            q[ij, 1] = 0
            q[ij, end] = 0
        end
    end
end

@loops function zero_bottom_top_VH!(_, q)
    let range = axes(q,2)
        @vec for ij in range
            q[1, ij] = 0
            q[end, ij] = 0
        end
    end
end

#========== fluxes =========#

fluxes!(mgr, vanleer, ::Void, q, dq, mass, flux) = fluxes!(mgr, vanleer, similar(flux), q, dq, mass, flux)

function fluxes!(mgr, vanleer::FV{N}, fluxq::AA{N}, dq::AA{N}, q::AA{N}, mass::AA{N}, flux::AA{N}) where N
    @assert axes(fluxq) == axes(flux)
    CFTransport.fluxes!(mgr, vanleer, fluxq, dq, q, mass, flux)
    zero_bottom_top!(mgr, vanleer, fluxq)
    # @info "fluxes!" absmax(flux) absmax(fluxq)
    return fluxq
end

#============= update ===========#

update_density!(mgr, vanleer, ::Void, #==# fluxq, massq) =
    update_density!(mgr, vanleer, similar(massq), #==# fluxq, massq)

function update_density!(mgr, vanleer::FV{N}, massqnew::AA{N}, #==# fluxq::AA{N}, massqnow::AA{N}) where N
    @assert axes(massqnew) == axes(massqnow)
    CFTransport.FV_update!(mgr, vanleer, massqnew, massqnow, fluxq)
    return massqnew
end

update_scalar!(mgr, vanleer, ::Void, #==# fluxq, qnow, flux, mass) =
    update_scalar!(mgr, vanleer, similar(qnow), #==# fluxq, qnow, flux, mass)

function update_scalar!(mgr, vanleer::FV{N}, qnew::AA{N}, #==# fluxq::AA{N}, qnow::AA{N}, flux::AA{N}, mass::AA{N}) where N
    @assert axes(qnew) == axes(qnow)
    CFTransport.FV_update!(mgr, vanleer, qnew, qnow, fluxq, flux, mass) # correct!
    return qnew
end

update_mass!(mgr, ::Void, newmass) = update_mass!(mgr, similar(newmass), newmass)
function update_mass!(mgr, mass, newmass)
    @assert axes(mass) == axes(newmass)
    update_mass_(mgr, mass, newmass)
    return mass
end

@loops function update_mass_(_, mass, newmass)
    let range = eachindex(mass)
        @vec for i in range
            mass[i] = newmass[i]
        end
    end
end

end
