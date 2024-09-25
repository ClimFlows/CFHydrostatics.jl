module RemapVoronoi

using MutatingOrNot: void, Void
using ManagedLoops: @with, @vec
using CFDomains: Stencils, VoronoiSphere, mass_coordinate
using CFTransport: remap_fluxes!
using ..RemapHPE: vanleer, remap_density!, remap_scalar!, update_mass!

# similar!(x,y) allocates only if x::Void
similar!(::Void, y...) = similar(y...)
similar!(x, y...) = x

remap!(new, scratch, model, state, schemes = (scalar = vanleer, momentum = vanleer)) =
    remap_staggered!(new, scratch, model, state, schemes)

function remap_staggered!(new, scratch, model, state, schemes)
    (; mass_consvar, ucov) = state
    (; mgr, vcoord, domain) = model
    vsphere, layout = domain.layer, domain.layout

    scheme_mq = schemes.scalar(:density, layout)
    scheme_u = schemes.momentum(:scalar, layout)

    metric_cov = model.planet.radius^2
    mcoord = mass_coordinate(model.vcoord, metric_cov) # pressure coordinate => covariant mass coordinate

    # Ensuring that new===state works requires that new_mass_air be a scratch array.
    # Scratch space is not needed for new_air_consvar and new_ucov.

    # mass fluxes and new mass
    mass_air = state.mass_air  # `mcoord` works with `mass_air` per unit surface on the unit sphere
    flux, new_mass_air =
        remap_fluxes!(mgr, mcoord, layout, scratch.flux, scratch.new_mass_air, mass_air)

    # vertical transport of densities
    new_mass_consvar, remap_consvar = remap_density!(
        mgr,
        scheme_mq,
        new.mass_consvar,
        scratch.remap_consvar,
        mass_consvar,
        mass_air,
        flux,
    )

    # vertical transport of momentum
    flux_e, mass_e = transfer_mass_flux!(
        mgr,
        scratch.flux_e,
        scratch.mass_e,
        ucov,
        mass_air,
        flux,
        vsphere,
    )
    new_ucov, remap_momentum =
        remap_scalar!(mgr, scheme_u, new.ucov, scratch.remap_momentum, ucov, mass_e, flux_e)

    scratch = (; new_mass_air, flux, flux_e, mass_e, remap_consvar, remap_momentum)
    new = (
        mass_air = (@. new.mass_air = new_mass_air),
        mass_consvar = new_mass_consvar,
        ucov = new_ucov,
    )
    return new, scratch
end

function transfer_mass_flux!(
    mgr,
    flux_e_,
    mass_e_,
    ucov,
    mass_air,
    flux,
    vsphere::VoronoiSphere,
)
    # interpolate mass and vertical mass flux to edges, where ucov lives
    mass_e = similar!(mass_e_, ucov)
    @with mgr, let (krange, ijrange) = axes(mass_e)
        for ij in ijrange
            avg = Stencils.average_ie(vsphere, ij)
            @vec for k in krange
                mass_e[k, ij] = avg(mass_air, k)
            end
        end
    end
    flux_e = similar!(flux_e_, ucov, size(flux, 1), size(ucov, 2))
    @with mgr, let (krange, ijrange) = axes(flux_e)
        for ij in ijrange
            avg = Stencils.average_ie(vsphere, ij)
            @vec for k in krange
                flux_e[k, ij] = avg(flux, k)
            end
        end
    end
    return flux_e, mass_e
end

end # module
