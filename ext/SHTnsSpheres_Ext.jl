module SHTnsSpheres_Ext

using MutatingOrNot: void, Void
using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!
using CFPlanets: ShallowTradPlanet, coriolis
using CFDomains: shell
using CFHydrostatics: CFHydrostatics
import CFHydrostatics: initial_HPE_HV, HPE

function HPE(params, sph::SHTnsSphere, vcoord, geopotential, gas)
    (; radius, Omega), (; lon, lat) = params, sph
    planet = ShallowTradPlanet(radius, Omega)
    fcov = coriolis.(Ref(planet), lon, lat)
    Phis = geopotential.(lon, lat)
    return HPE(vcoord, planet, shell(params.nz, sph), gas, fcov, Phis)
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

end
