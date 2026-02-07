module RemapVoronoi

using MutatingOrNot: void, Void
using ManagedLoops: @with, @vec
using CFDomains: Stencils, VoronoiSphere, mass_coordinate
using CFTransport: remap_fluxes!
using ..RemapHPE: vanleer, remap_density!, remap_scalar!, update_mass!

# similar!(x,y) allocates only if x::Void
similar!(::Void, y...) = similar(y...)
similar!(x, y...) = x

remap!(new, tmp, model, state, schemes = (scalar = vanleer, momentum = vanleer)) =
    remap_staggered!(new, tmp, model, state, schemes)

function remap_staggered!(new, tmp, model, state, schemes)
    (; masscov_air, masscov_consvar, ucov) = state
    (; mgr, vcoord, domain) = model
    vsphere, layout = domain.layer, domain.layout

    scheme_mq = schemes.scalar(:density, layout)
    scheme_u = schemes.momentum(:scalar, layout)

    cell_area = @. tmp.cell_area = (model.planet.radius^2)*model.domain.layer.Ai # cell area in m²
    mcoord = mass_coordinate(vcoord, cell_area) # pressure coordinate => covariant mass coordinate

    # Ensuring that new===state works requires that new_masscov_air be a scratch array.
    # Scratch space is not needed for new_masscov_consvar and new_ucov.

    # mass fluxes and new mass
    flux, new_masscov_air =
        remap_fluxes!(mgr, mcoord, layout, tmp.flux, tmp.new_masscov_air, masscov_air)

    # vertical transport of densities
    new_masscov_consvar, remap_consvar = remap_density!(
        mgr,
        scheme_mq,
        new.masscov_consvar,
        tmp.remap_consvar,
        masscov_consvar,
        masscov_air,
        flux,
    )

    # vertical transport of momentum
    flux_e, mass_e = transfer_mass_flux!(
        mgr,
        tmp.flux_e,
        tmp.mass_e,
        ucov,
        masscov_air,
        flux,
        vsphere,
    )
    new_ucov, remap_momentum =
        remap_scalar!(mgr, scheme_u, new.ucov, tmp.remap_momentum, ucov, mass_e, flux_e)

    tmp = (; cell_area, new_masscov_air, flux, flux_e, mass_e, remap_consvar, remap_momentum)
    new = (
        masscov_air = (@. new.masscov_air = new_masscov_air),
        masscov_consvar = new_masscov_consvar,
        ucov = new_ucov,
    )
    return new, tmp
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
