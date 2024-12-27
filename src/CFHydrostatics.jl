module CFHydrostatics

using ManagedLoops: @loops, @vec
using CFPlanets: ShallowTradPlanet
using CFDomains: CFDomains, VHLayout, HVLayout, Shell, pressure_level
using ClimFluids: AbstractFluid, SimpleFluid
import CFTimeSchemes

struct HPE{F, Manager, Coord, Domain, Fluid, TwoDimScalar}
    mgr::Manager
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    Phis::TwoDimScalar # surface geopotential
end

# override this function and set one of these flags to zero to debug the computation of ducov/dt
debug_flags() = (ke=1, Phi=1, gradB=1, CgradExner=1, qU=1)

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

CFTimeSchemes.tendencies!(dstate, scratch, model::HPE, state, t) =
    HPE_tendencies!(dstate, scratch, model, model.domain.layer, state, t)

CFTimeSchemes.tendencies!(slow, fast, scratch, model::HPE, state, t, tau) =
    HPE_tendencies!(slow, fast, scratch, model, model.domain.layer, state, t, tau)

vertical_remap!(backend, model, new, scratch, now) =
    HPE_remap!(backend, model, model.domain.layer, new, scratch, now)

# implemented in SHTnsSpheres_Ext
function HPE end
function HPE_diagnostics end
function HPE_tendencies! end
function HPE_scratch end
function HPE_dstate end
function HPE_remap! end

include("julia/voronoi.jl")

include("julia/remap_HPE.jl")
include("julia/remap_collocated.jl")
include("julia/remap_voronoi.jl")

# include("julia/ext/SHTnsSpheres_Ext.jl") # only when developing (also Project.toml)

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end
