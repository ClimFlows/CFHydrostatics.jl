module CFHydrostatics

using CFPlanets: ShallowTradPlanet
using CFDomains: CFDomains
using ClimFluids: AbstractFluid, SimpleFluid
import CFTimeSchemes

include("julia/vertical_coordinate.jl")

struct HPE{F, Manager, Coord, Domain, Fluid, TwoDimScalar}
    mgr::Manager
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    Phis::TwoDimScalar # surface geopotential
end

include("julia/initialize.jl")

"""
    diags = diagnostics(model::HPE)

Return `diags::CookBook` containing recipes to compute standard diagnostics.
This object is to be used as follows:

    session = open(diags; model, state)
    temp = session.temperature
    ...

with `state` the current state, obtained for instance from `initial_HPE`.
"""
diagnostics(model::HPE) = HPE_diagnostics(model, model.domain.layer)

CFTimeSchemes.tendencies!(dstate, model::HPE, state, scratch, t) =
    HPE_tendencies!(dstate, model, model.domain.layer, state, scratch, t)

CFTimeSchemes.scratch_space(model::HPE, state) = HPE_scratch(model, model.domain.layer, state)

CFTimeSchemes.model_dstate(model::HPE, state) = HPE_dstate(model, model.domain.layer, state)

# implemented in SHTnsSpheres_Ext
function HPE end
function HPE_diagnostics end
function HPE_tendencies! end
function HPE_scratch end
function HPE_dstate end

end
