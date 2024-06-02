module Dynamics

using MutatingOrNot: void, Void
using ManagedLoops: @loops, @vec
using SHTnsSpheres: analysis_scalar!, synthesis_scalar!, analysis_vector!, synthesis_vector!,
    synthesis_spheroidal!, divergence!, curl!, shtns_alloc

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, uv_spec) = (; mass_spec, uv_spec)

function mass_flux!(flux, model, mass, uv)
    invrad2 = model.planet.radius^-2
    flux = vector_spat(
        (@. flux.ucolat = -invrad2 * mass * uv.ucolat),
        (@. flux.ulon = -invrad2 * mass * uv.ulon),
    )
end

function tendencies!(dstate, model, state, scratch, t)
    (; dmass_spec, duv_spec) = tendencies_all(dstate, model, state, scratch, t)
    return HPE_state(dmass_spec, duv_spec)
end

function scratch_space(model, state)
    prec(::Type{Complex{F}}) where F = F
    sph, nz, F = model.domain.layer, size(state.mass_spec, 2), prec(eltype(state.mass_spec))

    scalar_spat(dims...) = shtns_alloc(F, Val(:scalar_spat), sph, dims...)
    scalar_spec(dims...) = shtns_alloc(F, Val(:scalar_spec), sph, dims...)
    vector_spat(dims...) = shtns_alloc(F, Val(:vector_spat), sph, dims...)
    vector_spec(dims...) = shtns_alloc(F, Val(:vector_spec), sph, dims...)

    # geopotential (2D buffer)
    geopot = scalar_spat()
    # velocity, mass, mass flux
    uv, mass = vector_spat(nz), scalar_spat(nz,2)
    flux, flux_spec = vector_spat(nz,2), vector_spec(nz, 2)
    # vorticity & its flux
    zeta, zeta_spec = scalar_spat(nz), scalar_spec(nz)
    qflux, qflux_spec = vector_spat(nz), vector_spec(nz)
    # pressure, consvar, Bernoulli, exner
    p, consvar = scalar_spat(nz), scalar_spat(nz)
    B, B_spec = scalar_spat(nz), scalar_spec(nz)
    exner, exner_spec, grad_exner = scalar_spat(nz), scalar_spec(nz), vector_spat(nz)

    # reuse some buffers
    exner = p
    zeta_spec = exner_spec
    qflux = grad_exner
    B_spec = zeta_spec

    return  (; uv, flux, flux_spec, zeta, zeta_spec, qflux, qflux_spec,
        mass, p, geopot, consvar, B, exner, B_spec, exner_spec, grad_exner)
end

function tendencies_all(dstate, model, state, scratch, t)
    # spectral fields are suffixed with _spec
    # vector, spectral = (spheroidal, toroidal)
    # vector, spatial = (ucolat, ulon)
    (; uv, flux, flux_spec, zeta, zeta_spec, qflux, qflux_spec) = scratch
    (; mass, p, geopot, consvar, B, exner, B_spec, exner_spec, grad_exner) = scratch
    (; mass_spec, uv_spec) = state
    dmass_spec, duv_spec = dstate.mass_spec, dstate.uv_spec
    sph, invrad2, fcov = model.domain.layer, model.planet.radius^-2, model.fcov

    # flux-form mass budget:
    #   ∂Φ/∂t = -∇(Φu, Φv)
    # uv is the momentum 1-form = a(u,v)
    # gh is the 2-form a²Φ
    # divergence! is relative to the unit sphere
    #   => scale flux by radius^-2
    mass = synthesis_scalar!(mass, mass_spec, sph)
    uv = synthesis_vector!(uv, uv_spec, sph)
    flux = vector_spat(
        (@. flux.ucolat = -invrad2 * mass * uv.ucolat),
        (@. flux.ulon = -invrad2 * mass * uv.ulon),
    )
    flux_spec = analysis_vector!(flux_spec, flux, sph)
    dmass_spec = divergence!(dmass_spec, flux_spec, sph)

    # curl-form momentum budget:
    #   ∂u/∂t = (f+ζ)v - θ∂π/∂x- ∂B/∂x
    #   ∂v/∂t = -(f+ζ)u - θ∂π/∂y - ∂B/∂y
    #   B = (u²+v²)2 + gz + (h - θπ)
    # uv is momentum = a*(u,v)
    # curl! is relative to the unit sphere
    # fcov, zeta and gh are the 2-forms a²f, a²ζ, a²Φ
    #   => scale B and qflux by radius^-2
    p = hydrostatic_pressure!(p, model, mass)
    B, exner, consvar = Bernoulli!(B, exner, consvar, geopot, model, mass, p, uv)
    exner_spec = analysis_scalar!(exner_spec, exner, sph)
    grad_exner = synthesis_spheroidal!(grad_exner, exner_spec, sph)

    zeta_spec = curl!(zeta_spec, uv_spec, sph)
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)
    qflux = vector_spat(
        (@. qflux.ucolat = invrad2 * (zeta + fcov) * uv.ulon - consvar * grad_exner.ucolat),
        (@. qflux.ulon  = -invrad2 * (zeta + fcov) * uv.ucolat - consvar * grad_exner.ulon),
    )
    qflux_spec = analysis_vector!(qflux_spec, qflux, sph)

    B_spec = analysis_scalar!(B_spec, B, sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )
    return (; dmass_spec, duv_spec, zeta, p, B, exner, consvar)
end

hydrostatic_pressure!(::Void, model, mass) = hydrostatic_pressure!(similar(@view mass[:,:,:,1]), model, mass)

function Bernoulli!(::Void, ::Void, ::Void, ::Void, model, mass::Array{Float64,4}, p::Array{Float64,3}, uv)
    B() = similar(p)
    Phi = similar(@view p[:,:,1])
    return Bernoulli!(B(), B(), B(), Phi, model, mass, p, uv)
end

function hydrostatic_pressure!(p::Array{Float64,3}, model, mass::Array{Float64,4})
    @assert size(mass,3) == size(p,3)
    radius, ptop, nz = model.planet.radius, model.vcoord.ptop, size(p,3)
    half_invrad2 = radius^-2 /2
    for i in axes(p,1), j in axes(p,2)
        p[i,j,nz] = ptop + half_invrad2*mass[i,j,nz,1]
    end
    for i in axes(p,1), j in axes(p,2), k in nz:-1:2
        p[i,j,k-1] = p[i,j,k] + half_invrad2*(mass[i,j,k,1]+mass[i,j,k-1,1])
    end
    return p
end

function Bernoulli!(B, exner, consvar, Phi, model, mass::Array{Float64,4}, p::Array{Float64,3}, uv)
    @assert size(mass,3) == size(p,3)
    @assert size(mass,4) == 2 # simple fluid
    Phi = @. Phi = model.Phis
    compute_Bernoulli!(model.mgr, B, exner, consvar, Phi, mass, p, uv, model)
    return B, exner, consvar
end

@loops function compute_Bernoulli!(_, B, exner, consvar, Phi, mass, p, uv, model)
    let (irange, jrange) = (axes(p,1), axes(p,2))
        ux, uy = uv.ucolat, uv.ulon
        invrad2 = model.planet.radius^-2
        Exner = model.gas(:p, :consvar).exner_functions
        @inbounds for j in jrange, k in axes(p,3)
            @vec for i in irange
                ke = (invrad2/2)*(ux[i,j,k]^2 + uy[i,j,k]^2)
                consvar_ijk = mass[i,j,k,2]/mass[i,j,k,1]
                h, v, exner_ijk = Exner(p[i,j,k], consvar_ijk)
                Phi_up = Phi[i,j] + invrad2*mass[i,j,k,1]*v # geopotential at upper interface
                B[i,j,k] = ke + (Phi_up+Phi[i,j])/2 + (h-consvar_ijk*exner_ijk)
                consvar[i,j,k] = consvar_ijk
                exner[i,j,k] = exner_ijk
                Phi[i,j] = Phi_up
            end
        end
    end
end

end # module Dynamics
