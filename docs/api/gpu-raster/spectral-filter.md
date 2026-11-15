# Spectral raster filtering

## `GPURasterSpectralFilter`

Exported by `@luma-contributors/analysis/raster-analysis`. Filters a complete resident raster by multiplying its Fourier spectrum by a real response and applying the inverse transform. Both transforms are the published `GPUFFT2D` from `@luma.gl/graph/gpu-core`; no FFT implementation is carried here. NumPy `fft2` / `ifft2` is the independent numerical reference (`test/oracles/raster-spectral-filter.py` under `modules/analysis`).

`grid` is a published `GPURasterMetadata`, and `input` is a published `GPURasterBand` (buffer or texture). Each dimension must be a power of two from 2 through `MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION` (2048). Complex fields must also fit the device storage binding limit; the contributor checks that before creating nodes. `estimatedCommandNodeCount` bounds the two published transforms, band canonicalization and the five filtering/counting nodes. There is no CPU twin: this is raster work with parallel transforms.

The boundary is periodic. There is no padding, apodization, inferred nodata filling, or independent tile filtering. Opposite image edges touch; discontinuities can ring, especially for ideal pass bands. Supply a complete domain with the boundary treatment your analysis requires. Frequencies are in **cycles per pixel** along the column/row axes; the grid affine, CRS and ground cell sizes do not affect this pixel-frequency operation.

`filter` selects the compile-time response. With signed FFT frequencies `f = (columnFrequency, rowFrequency)` and `r = length(f)`:

| Filter | Response |
| --- | --- |
| `low-pass` | 1 for `r <= cutoff`, otherwise 0 |
| `high-pass` | 1 for `r > cutoff`, otherwise 0 (complement of low-pass) |
| `band-pass` | 1 for `lowerCutoff <= r <= upperCutoff`, otherwise 0 |
| `gaussian` | `exp(-0.5 * r² / sigma²)`; sigma is in frequency space |
| `notch` | 0 inside or on either disk centered at `notchCenter` and its negative, otherwise 1 |

Notch distances wrap on the frequency torus (period 1 per axis), including Nyquist aliases. Conjugate bins share one discrete frequency representative so they receive identical responses even at rounding boundaries. Nyquist-axis distances use `0.5 - abs(center)` directly, avoiding cancellation in wrapped subtraction. Removing conjugate disks preserves a real result. DC has frequency zero. A zero low-pass cutoff keeps only the mean; a zero high-pass cutoff removes only the mean. A zero notch radius removes exact matching bins. A maximum radial cutoff includes the Nyquist corner.

`parameters` is a packed float32 view with at least `GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH` (8) rows. Use `getGPURasterSpectralFilterParameterValues(settings)` and a caller-owned `GPUParameterBuffer`. The layout is `[cutoff, lowerCutoff, upperCutoff, sigma, notchColumn, notchRow, notchRadius, 0]`. Defaults are `[0.25, 0, 0.25, 0.1, 0.25, 0, 0.05, 0]`. Radial settings lie in `[0, sqrt(0.5)]`, lower cutoff must not exceed upper cutoff, notch centers lie in `[-0.5, 0.5]`, and sigma lies in `[MINIMUM_RASTER_SPECTRAL_SIGMA, sqrt(0.5)]` (minimum approximately `1e-6`). The packer validates all settings, even those unused by the selected filter. Invalid settings written directly to the view produce invalid outputs on the GPU.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPURasterSpectralFilter,
  getGPURasterSpectralFilterParameterValues
} from '@luma-contributors/analysis/raster-analysis';

const parameters = new GPUParameterBuffer(device, {
  id: 'filter-settings', format: 'float32', length: 8,
  values: getGPURasterSpectralFilterParameterValues({cutoff: 0.125})
});
graph.add(new GPURasterSpectralFilter({
  grid, input, filter: 'low-pass', parameters: parameters.importToGraph(graph),
  output: {values, validity, status: {invalidCount}}
}));
// Between encodings of the same compiled graph:
parameters.write(getGPURasterSpectralFilterParameterValues({cutoff: 0.25}));
// Destroy the compiled graph before parameters.destroy().
```

Storage, calibration and nodata metadata are validated at construction. Buffer values must be packed scalars matching `input.format`, and validity must be packed uint32. Texture storage must preserve the scalar format (`r32float`/`rgba32float`, `r32uint`/`rgba32uint`, or `r32sint`/`rgba32sint`) and select one 2D mip, one array layer and one sample. Channel zero is the default and the only accepted channel for scalar textures; RGBA textures accept channels 0–3. A selected mip or layer may have a nonzero base index. Scale and offset must fit finite float32; float nodata must fit finite float32 or be NaN. Integer nodata must be an integer in the raw storage range: `[0, 4294967295]` for uint32 or `[-2147483648, 2147483647]` for sint32. Band scale/offset, sentinel nodata, validity, and nonfinite observations are canonicalized through published raster primitives. Because all Fourier coefficients depend on the whole image, **one missing input invalidates every output**. Invalid output values are NaN and `output.validity` is zero. Otherwise validity is one for finite results. Optional `output.status.invalidCount` is a standard `GPUInvalidStatusPort` scalar counting missing input cells, before global propagation; it does not count invalid parameter settings or floating-point overflows. Output values and validity cover at least `width * height` rows. Input and output padding rows are ignored; outputs cannot overlap inputs or one another.

{/* lens:GPURasterSpectralFilter */}

::lens{scenario=gpu-raster.raster-spectral-filter rung=graph}

The synthetic 4×4 image is mean five plus a Nyquist checkerboard of amplitude two. A low-pass
cutoff of zero keeps only DC, so every filtered cell is five, every validity is one, and the
missing-input count is zero. The graph exposes the published forward and inverse transforms.

Compile-time properties are dimensions, band storage/calibration, filter and view layouts. Observations, validity and frequency parameters are per-frame. Float32 FFT arithmetic has the usual rounding and dynamic-range limits; a nonfinite final result is invalid. This is a frequency mask, not general FFT convolution.

{/* parameters: GPURasterSpectralFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterSpectralFilter` parameters

- Packer: `getGPURasterSpectralFilterParameterValues(settings?, target?)`, writes `GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH` (8) `float32` values.
  - `target` (`unknown`)

Settings of `getGPURasterSpectralFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `cutoff` (optional) | `number` | Low-pass includes this radius; high-pass excludes it. Default 0.25 cycles/pixel. |
| `lowerCutoff` (optional) | `number` | - |
| `upperCutoff` (optional) | `number` | - |
| `sigma` (optional) | `number` | Gaussian standard deviation in frequency space, default 0.1 cycles/pixel. |
| `notchCenter` (optional) | `readonly [number, number]` | Notch center (column,row) frequency in [-0.5,0.5]; its conjugate is also removed. |
| `notchRadius` (optional) | `number` | - |

{/* /parameters */}
