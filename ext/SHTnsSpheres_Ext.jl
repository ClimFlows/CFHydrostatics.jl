module SHTnsSpheres_Ext

using MutatingOrNot: void, Void
using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!, synthesis_scalar!
using CFPlanets: ShallowTradPlanet, coriolis
using CFDomains: shell
using CFHydrostatics: CFHydrostatics
using CFHydrostatics.RemapCollocated: remap!

import CFHydrostatics: HPE, initial_HPE_HV, HPE_diagnostics, HPE_tendencies!, HPE_remap!

# CUDA SHTns plans have a fixed batch size. Surface and interface
# diagnostics use plans with a different batch size from the dynamics.
with_auxiliary_sphere(f, sph, batch) = with_auxiliary_sphere(f, sph, batch, sph.lon)
with_auxiliary_sphere(f, sph, _, ::Array) = f(sph)
function with_auxiliary_sphere(f, sph, batch, grid)
    aux = SHTnsSphere(sph.nlat; backend=:cuda, precision=eltype(grid), batch)
    try
        return f(aux)
    finally
        close(aux)
    end
end

host_grid(a::Array) = a
host_grid(a) = Array(a)

to_grid(::Array, field) = field
to_grid(grid, field) = copyto!(similar(grid, size(field)), field)

function HPE(params, mgr, sph::SHTnsSphere, vcoord, geopotential, gas)
    (; radius, Omega), (; lon, lat) = params, sph
    planet = ShallowTradPlanet(radius, Omega)
    fcov = coriolis.(Ref(planet), lon, lat)
    Phis = to_grid(lon, geopotential.(host_grid(lon), host_grid(lat)))
    Phis = with_auxiliary_sphere(sph, 1) do surface
        synthesis_scalar!(void, analysis_scalar!(void, Phis, surface), surface)
    end
    return HPE(mgr, vcoord, planet, shell(params.nz, sph), gas, fcov, Phis)
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_air_spec, mass_consvar_spec, uv_spec) = (; mass_air_spec, mass_consvar_spec, uv_spec)

function initial_HPE_HV(model, nz, sph::SHTnsSphere, case)
    masses, ulon, ulat = CFHydrostatics.initial_HPE_HV_collocated(
        model,
        nz,
        host_grid(sph.lon),
        host_grid(sph.lat),
        model.gas,
        case,
    )
    mass_air_spec = analysis_scalar!(void, to_grid(sph.lon, masses.air), sph)
    mass_consvar_spec = analysis_scalar!(void, to_grid(sph.lon, masses.consvar), sph)
    uv = vector_spat(to_grid(sph.lon, -ulat), to_grid(sph.lon, ulon))
    uv_spec = analysis_vector!(void, uv, sph)
    HPE_state(mass_air_spec, mass_consvar_spec, uv_spec)
end

include("dynamics.jl")

HPE_tendencies!(dstate, scratch, model, _::SHTnsSphere, state, t) =
    Dynamics.tendencies!(dstate, scratch, model, state, t)
HPE_tendencies!(slow, fast, scratch, model, _::SHTnsSphere, state, t, tau) =
    Dynamics.tendencies!(slow, fast, scratch, model, state, t, tau)

include("diagnostics.jl")

HPE_remap!(mgr, model, ::SHTnsSphere, new, scratch, now) = #==#
    remap!(mgr, model.vcoord, model.domain.layout, new, scratch, now) #==#

HPE_diagnostics(_, ::SHTnsSphere) = Diagnostics.diagnostics()

end
