# CFHydrostatics

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/CFHydrostatics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/CFHydrostatics.jl/dev/)
[![Build Status](https://github.com/ClimFlows/CFHydrostatics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/CFHydrostatics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/CFHydrostatics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/CFHydrostatics.jl)

`CFHydrostatics` aims at implementing the prognostic equations for various quasi-hydrostatic systems. For now, it supports the following features:
- fluid: single-component, compressible fluid
- geometry: traditional, shallow-atmosphere
- spatial discretization: spectral or Voronoi
- temporal discretization: explicit or IMEX

## Spectral models on CUDA

Use SHTnsSpheres 0.2.19 or later, a CUDA `SHTnsSphere`, and a CUDA loop
manager. Use the same precision for the sphere and the model parameters.
Initial conditions are sampled on the CPU and transferred to the CUDA grid.

Dynamics and diagnostics share the model's sphere. Its SHTns plans for
surface, layer, and interface fields are reused across diagnostic sessions.
See the [SpectralHPE example](https://github.com/ClimFlows/ClimFlowsExamples/tree/main/SpectralHPE.jl)
for a complete model setup.

## Voronoi adjoints

When Enzyme is loaded, an optional extension supplies a custom reverse rule for
the Voronoi mass budget. It composes the custom adjoints of VoronoiSpheres
operators, including the returned tracer flux, and supports CPU and CUDA arrays.
Full GPU reverse differentiation of the HPE tendencies remains incomplete; the
hydrostatic column kernel is the next KernelAbstractions–Enzyme limitation.
