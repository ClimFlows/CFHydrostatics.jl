module RemapHPE

using MutatingOrNot: Void
using ManagedLoops: @loops, @vec

using CFDomains: VHLayout, HVLayout
using CFTransport: CFTransport, OneDimFV, minmod_simd

const FV{Rank, Dim, kind} = OneDimFV{kind, Dim, Rank} # to dispatch on FV{Rank}
const AA{Rank, T} = AbstractArray{T, Rank}            # to dispatch on AA{Rank}
const AAV{Rank, T} = Union{Void, AbstractArray{T, Rank}} # AA or Void (output arguments)

# x = ifvoid(x,y) replaces output argument `x::Void` by `similar(y)`
ifvoid(x, y) = x
ifvoid(::Void, y) = similar(y)

# convention:
#      fun(non-fields, output fields..., #==# scratch #==# input fields...)
# or   fun(non-fields, output fields..., #==# input fields...)

vanleer(kind, ::HVLayout) = CFTransport.VanLeerScheme(kind, minmod_simd, 2, 2)
vanleer(kind, ::VHLayout) = CFTransport.VanLeerScheme(kind, minmod_simd, 1, 2)

# merge horizontal dimensions when there are two
# belongs to CFDomains ?
const AbsA{N,T} = AbstractArray{T,N}

flatten(x::AbsA, ::HVLayout{1}) = x
flatten(x::AbsA, ::VHLayout{1}) = x
flatten(x::AbsA{3}, ::HVLayout{2}) = reshape(x, :, size(x, 3))
flatten(x::AbsA{4}, ::HVLayout{2}) = reshape(x, :, size(x, 3), size(x,4))
flatten(x::AbsA{3}, ::VHLayout{2}) = reshape(x, size(x, 1), :)
flatten(x::AbsA{4}, ::VHLayout{2}) = reshape(x, size(x, 1), :, size(x,4))

flatten(x::Union{Tuple, NamedTuple}, layout) = map(y->flatten(y, layout), x)
flatten(::HVLayout) = HVLayout{1}()
flatten(::VHLayout) = VHLayout{1}()


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

function concentrations!(mgr, q::AAV{N}, massq::AA{N}, mass::AA{N}) where N
    q = ifvoid(q, massq)
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

zero_bottom_top!(mgr, ::FV{2,2}, q::AA{2}) = zero_bottom_top_HV!(mgr, q)
zero_bottom_top!(mgr, ::FV{2,1}, q::AA{2}) = zero_bottom_top_VH!(mgr, q)

@loops function zero_bottom_top_HV!(_, q)
    let range = axes(q,1)
        @inbounds for ij in range
            q[ij, 1] = 0
            q[ij, end] = 0
        end
    end
end

@loops function zero_bottom_top_VH!(_, q)
    let range = axes(q,2)
        @inbounds for ij in range
            q[1, ij] = 0
            q[end, ij] = 0
        end
    end
end

#========== fluxes =========#

function fluxes!(mgr, vanleer::FV{N}, fluxq::AAV{N}, dq::AA{N}, q::AA{N}, mass::AA{N}, flux::AA{N}) where N
    fluxq = ifvoid(fluxq, flux)
    @assert axes(fluxq) == axes(flux)
    CFTransport.fluxes!(mgr, vanleer, fluxq, dq, q, mass, flux)
    zero_bottom_top!(mgr, vanleer, fluxq)
    # @info "fluxes!" absmax(flux) absmax(fluxq)
    return fluxq
end

#============= update ===========#

function update_density!(mgr, vanleer::FV{N}, massqnew::AAV{N}, #==# fluxq::AA{N}, massqnow::AA{N}) where N
    massqnew = ifvoid(massqnew, massqnow)
    @assert axes(massqnew) == axes(massqnow)
    CFTransport.FV_update!(mgr, vanleer, massqnew, massqnow, fluxq)
    return massqnew
end

function update_scalar!(mgr, vanleer::FV{N}, qnew::AAV{N}, #==# fluxq::AA{N}, qnow::AA{N}, flux::AA{N}, mass::AA{N}) where N
    qnew = ifvoid(qnew, qnow)
    @assert axes(qnew) == axes(qnow)
    CFTransport.FV_update!(mgr, vanleer, qnew, qnow, fluxq, flux, mass) # correct!
    return qnew
end

function update_mass!(mgr, mass::AAV{N}, newmass::AA{N}) where N
    mass = ifvoid(mass, newmass)
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
