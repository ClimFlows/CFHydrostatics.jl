module Dynamics

using MutatingOrNot: void, Void, similar

using ManagedLoops: @loops, @vec, no_simd
using SHTnsSpheres:
    analysis_scalar!,
    synthesis_scalar!,
    analysis_vector!,
    synthesis_vector!,
    synthesis_spheroidal!,
    divergence!,
    curl!,
    shtns_alloc

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, masses_spec, uv_spec) = (; mass_spec, masses_spec, uv_spec)

function tendencies!(dstate, scratch, model, state, t)
    # spectral fields are suffixed with _spec
    # vector, spectral = (spheroidal, toroidal)
    # vector, spatial = (ucolat, ulon)
    (; uv, flux, flux_spec, zeta, zeta_spec, qflux, qflux_spec) = scratch
    (; mass, p, geopot, consvar, B, exner, B_spec, exner_spec, grad_exner) = scratch
    (; masses, fluxes, fluxes_spec) = scratch
    (; mass_spec, masses_spec, uv_spec) = state
    dmass_spec, dmasses_spec, duv_spec =
        dstate.mass_spec, dstate.masses_spec, dstate.uv_spec
    mgr, sph, invrad2, fcov =
        model.mgr, model.domain.layer, model.planet.radius^-2, model.fcov
    mgr_spec = no_simd(mgr) # complex broadcasting + SIMD not supported

    # flux-form mass budget:
    #   ∂Φ/∂t = -∇(Φu, Φv)
    # uv is the momentum 1-form = a*(u,v)
    # gh is the 2-form a²Φ
    # divergence! is relative to the unit sphere
    #   => scale flux by radius^-2
    mass = synthesis_scalar!(mass, mass_spec, sph)
    uv = synthesis_vector!(uv, uv_spec, sph)

    flux = vector_spat(similar(mass, flux.ucolat), similar(mass, flux.ulon))
    mass_flux!(mgr, flux.ucolat, flux.ulon, -invrad2, uv.ucolat, uv.ulon, mass) # mutating only

    flux_spec = analysis_vector!(flux_spec, flux, sph)
    dmass_spec = divergence!(mgr_spec[dmass_spec], flux_spec, sph)

    # begin NEW
    masses = (
        air = synthesis_scalar!(masses.air, masses_spec.air, sph),
        consvar = synthesis_scalar!(masses.consvar, masses_spec.consvar, sph),
    )
    flx(f, m) = vector_spat(
        (@. mgr[f.ucolat] = (-invrad2) * uv.ucolat * m),
        (@. mgr[f.ulon] = (-invrad2) * uv.ulon * m),
    )
    fluxes = (
        air = vector_spat(
            (@. mgr[fluxes.air.ucolat] = (-invrad2) * uv.ucolat * masses.air),
            (@. mgr[fluxes.air.ulon] = (-invrad2) * uv.ulon * masses.air),
        ),
        consvar = vector_spat(
            (@. mgr[fluxes.consvar.ucolat] = (-invrad2) * uv.ucolat * masses.consvar),
            (@. mgr[fluxes.consvar.ulon] = (-invrad2) * uv.ulon * masses.consvar),
        ),
    )
    fluxes_spec = (
        air = analysis_vector!(fluxes_spec.air, fluxes.air, sph),
        consvar = analysis_vector!(fluxes_spec.consvar, fluxes.consvar, sph),
    )
    dmasses_spec = (
        air = divergence!(mgr_spec[dmasses_spec.air], fluxes_spec.air, sph),
        consvar = divergence!(mgr_spec[dmasses_spec.consvar], fluxes_spec.consvar, sph),
    )

    # curl-form momentum budget:
    #   ∂u/∂t = (f+ζ)v - θ∂π/∂x- ∂B/∂x
    #   ∂v/∂t = -(f+ζ)u - θ∂π/∂y - ∂B/∂y
    #   B = (u²+v²)2 + gz + (h - θπ)
    # uv is momentum = a*(u,v)
    # curl! is relative to the unit sphere
    # fcov, zeta and gh are the 2-forms a²f, a²ζ, a²Φ
    #   => scale B and qflux by radius^-2

    # p = hydrostatic_pressure!(p, model, mass)
    p = hydrostatic_pressure!(p, model, masses.air)

    # B, exner, consvar, geopot = Bernoulli!(B, exner, consvar, geopot, model, mass, p, uv)
    B, exner, consvar, geopot = Bernoulli!(B, exner, consvar, geopot, model, masses, p, uv)
    exner_spec = analysis_scalar!(exner_spec, exner, sph)
    grad_exner = synthesis_spheroidal!(grad_exner, exner_spec, sph)

    zeta_spec = curl!(mgr_spec[zeta_spec], uv_spec, sph)
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)
    qflux = vector_spat(
        (@. mgr[qflux.ucolat] =
            invrad2 * (zeta + fcov) * uv.ulon - consvar * grad_exner.ucolat),
        (@. mgr[qflux.ulon] =
            -invrad2 * (zeta + fcov) * uv.ucolat - consvar * grad_exner.ulon),
    )
    qflux_spec = analysis_vector!(qflux_spec, qflux, sph)

    B_spec = analysis_scalar!(B_spec, B, sph)
    duv_spec = vector_spec(
        (@. mgr_spec[duv_spec.spheroidal] = qflux_spec.spheroidal - B_spec),
        (@. mgr_spec[duv_spec.toroidal] = qflux_spec.toroidal),
    )
    scratch = (;
        uv,
        mass,
        flux,
        flux_spec,
        masses,
        fluxes,
        fluxes_spec,
        zeta,
        zeta_spec,
        qflux,
        qflux_spec,
        p,
        geopot,
        consvar,
        B,
        exner,
        B_spec,
        exner_spec,
        grad_exner,
    )

    return HPE_state(dmass_spec, dmasses_spec, duv_spec), scratch
end

@loops function mass_flux!(_, fx, fy, factor, ux, uy, mass)
    let (rx, ry, rz) = axes(ux)
        for y in ry, z in rz, q in axes(mass, 4)
            @vec for x in rx
                fx[x, y, z, q] = factor * mass[x, y, z, q] * ux[x, y, z]
                fy[x, y, z, q] = factor * mass[x, y, z, q] * uy[x, y, z]
            end
        end
    end
end

function hydrostatic_pressure!(p, model, mass::Array{Float64,4})
    air = @view mass[:,:,:,1]
    p = similar(air, p)
    compute_hydrostatic_pressure(model.mgr, p, model, air)
    return p
end

function hydrostatic_pressure!(p, model, air::Array{Float64,3})
    p = similar(air, p)
    compute_hydrostatic_pressure(model.mgr, p, model, air)
    return p
end

@loops function compute_hydrostatic_pressure(_, p, model, mass)
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ptop, nz = model.vcoord.ptop, size(p, 3)
        half_invrad2 = model.planet.radius^-2 / 2
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + half_invrad2 * mass[i, j, nz]
                for k = nz:-1:2
                    p[i, j, k-1] =
                        p[i, j, k] + half_invrad2 * (mass[i, j, k] + mass[i, j, k-1])
                end
            end
        end
    end
end

get_masses(masses::NamedTuple) = masses
get_masses(mass::Array) = @views (air=mass[:,:,:,1], consvar=mass[:,:,:,1])

function Bernoulli!(
    B,
    exner,
    consvar,
    Phi,
    model,
    mass,
    p::Array{Float64,3},
    uv,
)
    # similar(x,y) allocates only if y::Void
    B = similar(p, B)
    exner = similar(p, exner)
    consvar = similar(p, consvar)

    Phi = @. Phi = model.Phis
    compute_Bernoulli!(model.mgr, B, exner, consvar, Phi, get_masses(mass), p, uv, model)
    return B, exner, consvar, Phi
end

@loops function compute_Bernoulli!(_, B, exner, consvar, Phi, masses, p, uv, model)
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ux, uy = uv.ucolat, uv.ulon
        invrad2 = model.planet.radius^-2
        Exner = model.gas(:p, :consvar).exner_functions
        @inbounds for j in jrange
            for k in axes(p, 3)
                @vec for i in irange
                    ke = (invrad2 / 2) * (ux[i, j, k]^2 + uy[i, j, k]^2)
                    consvar_ijk = masses.consvar[i, j, k] / masses.air[i, j, k]
                    h, v, exner_ijk = Exner(p[i, j, k], consvar_ijk)
                    Phi_up = Phi[i, j] + invrad2 * masses.air[i, j, k] * v # geopotential at upper interface
                    B[i, j, k] =
                        ke + (Phi_up + Phi[i, j]) / 2 + (h - consvar_ijk * exner_ijk)
                    consvar[i, j, k] = consvar_ijk
                    exner[i, j, k] = exner_ijk
                    Phi[i, j] = Phi_up
                end
            end
        end
    end
end

end # module Dynamics
