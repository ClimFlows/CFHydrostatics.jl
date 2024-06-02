module CFHydrostatics

import CFDomains
using ClimFluids: SimpleFluid

include("julia/vertical_coordinate.jl")

include("julia/initialize.jl")

end
