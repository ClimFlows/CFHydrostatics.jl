module RemapCollocated

using CFTransport: remap_fluxes!
using CFDomains: mass_coordinate
using ..RemapHPE: vanleer, remap_density!, remap_scalar!, update_mass!, flatten

# convention:
#      fun(non-fields, output fields..., #==# scratch #==# input fields...)
# or   fun(non-fields, output fields..., #==# input fields...)

function remap!(mgr, vcoord, layout, #==# new, #==# scratch, #==# now, schemes=(scalar=vanleer, momentum=vanleer))
    (; mass, massq, ux, uy) = map( x->flatten(x, layout), now)
    scheme_mq = schemes.scalar(:density, layout)
    scheme_u = schemes.momentum(:scalar, layout)
    # mass fluxes and new mass
    mcoord = mass_coordinate(vcoord, one(eltype(mass)))
    flux, new_mass = remap_fluxes!(mgr, mcoord, flatten(layout), scratch.flux, scratch.new_mass, #==# mass)
    # vertical transport of densities
    new_massq = remap_density!(mgr, scheme_mq, new.massq, #==# scratch.fluxq, scratch.slope, scratch.q, #==# massq, mass, flux)
    # vertical transport of momentum
    new_ux = remap_scalar!(mgr, scheme_u, new.ux, #==# scratch.fluxq, scratch.slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, scheme_u, new.uy, #==# scratch.fluxq, scratch.slope, #==# uy, mass, flux)
    new_mass = update_mass!(mgr, new.mass, #==# new_mass)
    return (mass=new_mass, massq=new_massq, ux=new_ux, uy=new_uy)
end

end # module
