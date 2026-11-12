# Changelog

## 0.5.2

### Added

- Support for spectral hydrostatic models on CUDA, including Float32
  dynamics and diagnostics with SHTnsSpheres 0.2.19 or later.

### Changed

- Spectral diagnostics use the model's sphere and reuse its CUDA transform
  plans for surface, layer, and interface fields across diagnostic sessions.
