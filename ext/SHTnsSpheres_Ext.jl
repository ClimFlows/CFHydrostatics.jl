module SHTnsSpheres_Ext

using MutatingOrNot: void, Void
using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!
using CFPlanets: ShallowTradPlanet, coriolis
using CFDomains: shell
using CFHydrostatics: CFHydrostatics
using CFHydrostatics.RemapCollocated: remap!

import CFHydrostatics: HPE, initial_HPE_HV, HPE_diagnostics, HPE_tendencies!, HPE_remap!

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

HPE_tendencies!(dstate, scratch, model, _::SHTnsSphere, state, t) =
    Dynamics.tendencies!(dstate, scratch, model, state, t)

include("diagnostics.jl")

HPE_remap!(mgr, model, ::SHTnsSphere, new, #==# scratch, #==# now) =
    remap!(mgr, model.vcoord, model.domain.layout, new, #==# scratch, #==# now)

HPE_diagnostics(_, ::SHTnsSphere) = Diagnostics.diagnostics()

end
