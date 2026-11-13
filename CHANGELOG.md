# Changelog

## 0.5.5

### Added

- Add an Enzyme custom reverse rule for the Voronoi mass budget on CPU and GPU,
  composed from VoronoiSpheres operator adjoints. Preserve and differentiate the
  returned tracer flux through the new `CenteredTracerFlux` operator.

## 0.5.4

### Fixed

- Restore the slow and fast tendencies for Voronoi IMEX schemes using
  covariant mass fields.
- Allocate the Voronoi wind diagnostic using the current state layout.

- Allocate dual-cell potential vorticity scratch with the state element type.
  This allows forward-mode differentiation of the Voronoi dynamics with fixed
  mesh and model parameters.

### Tests

- Check Enzyme reverse derivatives against ForwardDiff for Voronoi tendencies,
  RK4 steps and vertical remapping.

## 0.5.3

### Changed

- Compute the spectral model's grid-space mass and momentum flux components in
  fused managed loops. The model's loop manager selects the execution backend.

## 0.5.2

### Added

- Support for spectral hydrostatic models on CUDA, including Float32
  dynamics and diagnostics with SHTnsSpheres 0.2.19 or later.

### Changed

- Spectral diagnostics use the model's sphere and reuse its CUDA transform
  plans for surface, layer, and interface fields across diagnostic sessions.
