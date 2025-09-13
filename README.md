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
