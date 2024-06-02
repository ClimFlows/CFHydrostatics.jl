module SHTnsSpheres_Ext

using MutatingOrNot: void, Void
using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!
using CFPlanets: ShallowTradPlanet, coriolis
using CFDomains: shell
using CFHydrostatics: CFHydrostatics
import CFHydrostatics: HPE, initial_HPE_HV, HPE_diagnostics, HPE_dstate, HPE_scratch, HPE_tendencies!

function HPE(params, mgr, sph::SHTnsSphere, vcoord, geopotential, gas)
    (; radius, Omega), (; lon, lat) = params, sph
    planet = ShallowTradPlanet(radius, Omega)
    fcov = coriolis.(Ref(planet), lon, lat)
    Phis = geopotential.(lon, lat)
    return HPE(mgr, vcoord, planet, shell(params.nz, sph), gas, fcov, Phis)
end

## these "constructors" seem to help with type stability
vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_spec, uv_spec) = (; mass_spec, uv_spec)

function initial_HPE_HV(model, nz, sph::SHTnsSphere, case)
    mass, ulon, ulat = CFHydrostatics.initial_HPE_HV_collocated(model, nz, sph.lon, sph.lat, model.gas, case)
    mass_spec = analysis_scalar!(void, mass, sph)
    uv_spec = analysis_vector!(void, vector_spat(-ulat, ulon), sph)
    HPE_state(mass_spec, uv_spec)
end

include("dynamics.jl")

HPE_tendencies!(dstate, model, _::SHTnsSphere, state, scratch, t) =
    Dynamics.tendencies!(dstate, model, state, scratch, t)

HPE_scratch(model, ::SHTnsSphere, state) = Dynamics.scratch_space(model, state)

function HPE_dstate(_, ::SHTnsSphere, state)
    sim(x) = similar(x)
    sim(x::NamedTuple) = map(sim, x)
    sim(state)
end

include("diagnostics.jl")

HPE_diagnostics(_, ::SHTnsSphere) = Diagnostics.diagnostics()

end
