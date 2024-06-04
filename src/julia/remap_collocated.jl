module RemapCollocated

using CFTransport: remap_fluxes!, flatten
using ..RemapHPE: vanleer, remap_density!, remap_scalar!, update_mass!

# convention:
#      fun(non-fields, output fields..., #==# scratch #==# input fields...)
# or   fun(non-fields, output fields..., #==# input fields...)

function remap!(mgr, vcoord, layout, #==# new, #==# scratch, #==# now)
    (; mass, massq, ux, uy) = map( x->flatten(x, layout), now)
    vanleer_mq = vanleer(:density, layout)
    vanleer_u = vanleer(:scalar, layout)
    # mass fluxes and new mass
    # @info "remap!" minimum(abs, mass)
    flux, new_mass = remap_fluxes!(mgr, vcoord, flatten(layout), scratch.flux, scratch.new_mass, #==# mass)
    # @info "remap!" maximum(abs, flux) minimum(abs, new_mass)
    # vertical transport of densities
    # @info "remap!" minimum(abs, mass)
    new_massq = remap_density!(mgr, vanleer_mq, new.massq, #==# scratch.fluxq, scratch.slope, scratch.q, #==# massq, mass, flux)
    # vertical transport of momentum
    # @info "remap!" minimum(abs, mass)
    new_ux = remap_scalar!(mgr, vanleer_u, new.ux, #==# scratch.fluxq, scratch.slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, vanleer_u, new.uy, #==# scratch.fluxq, scratch.slope, #==# uy, mass, flux)
    new_mass = update_mass!(mgr, new.mass, #==# new_mass)
    # @info "remap!" minimum(abs, mass)
    # new_mass = new.mass # FIXME
    return (mass=new_mass, massq=new_massq, ux=new_ux, uy=new_uy)
end

end # module
