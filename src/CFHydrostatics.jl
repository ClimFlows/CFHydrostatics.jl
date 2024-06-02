module CFHydrostatics

using CFPlanets: ShallowTradPlanet
using CFDomains: CFDomains
using ClimFluids: AbstractFluid, SimpleFluid

include("julia/vertical_coordinate.jl")

struct HPE{F, Coord, Domain, Fluid, TwoDimScalar}
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    Phis::TwoDimScalar # surface geopotential
end

include("julia/initialize.jl")

end
