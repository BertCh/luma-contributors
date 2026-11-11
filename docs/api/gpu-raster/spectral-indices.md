# Calibrated spectral indices

`GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI`, and `GPURasterSAVI` compose
`GPURasterBandMath` into graph-native recipes. They accept co-registered buffer bands,
caller-owned float32 outputs, and a separate uint32 validity output. They do not upload,
submit, read back, resample, choose sensor bands, or decode cloud/quality bits.
`GPURasterNDVI` remains the existing vegetation normalized-difference contributor.

## Formula and band contracts

All symbols below denote calibrated, dimensionless reflectance. Outputs are dimensionless
ratios, without the integer scaling used by some delivered satellite products.

| Contributor | Required named bands | Formula |
| --- | --- | --- |
| `GPURasterNDWI`, `method: 'mcfeeters'` | `green`, `nearInfrared` | `(green - nearInfrared) / (green + nearInfrared)` |
| `GPURasterNDWI`, `method: 'gao'` | `nearInfrared`, `shortwaveInfrared` | `(nearInfrared - shortwaveInfrared) / (nearInfrared + shortwaveInfrared)` |
| `GPURasterNBR` | `nearInfrared`, `shortwaveInfrared2` | `(nearInfrared - shortwaveInfrared2) / (nearInfrared + shortwaveInfrared2)` |
| `GPURasterEVI` | `nearInfrared`, `red`, `blue` | `2.5 * (nearInfrared - red) / (nearInfrared + 6 * red - 7.5 * blue + 1)` |
| `GPURasterSAVI` | `nearInfrared`, `red` | `(1 + soilAdjustment) * (nearInfrared - red) / (nearInfrared + red + soilAdjustment)` |

NDWI requires an explicit method. OTB calls the Gao recipe `Water:NDWI` and the
McFeeters open-water recipe `Water:NDWI2`; those names describe different band choices.
See [OTB RadiometricIndices 10.0](https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html).
The original Gao recipe uses channels near 0.86 and 1.24 micrometers. Substituting a
1.6-micrometer sensor band is an adaptation, and does not establish equivalent spectral
response. See [Gao's original paper](https://doi.org/10.1016/S0034-4257(96)00067-3).

NBR specifically uses SWIR2, around 2.2 micrometers; EVI fixes gain 2.5, red coefficient 6,
blue coefficient 7.5, and background constant 1. SAVI exposes `soilAdjustment` in `[0, 1]`,
with default 0.5. These follow the published
[USGS NBR](https://www.usgs.gov/landsat-missions/landsat-normalized-burn-ratio),
[EVI](https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index), and
[SAVI](https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index) formulas.
SAVI with zero adjustment has the NDVI formula; it does not replace the existing NDVI API.

For a concrete sensor mapping, the Landsat 8/9 OLI bands are:

| Input | OLI band | Wavelength, micrometers |
| --- | --- | --- |
| `blue` | 2 | 0.45–0.51 |
| `green` | 3 | 0.53–0.59 |
| `red` | 4 | 0.64–0.67 |
| `nearInfrared` | 5 | 0.85–0.88 |
| SWIR1, a possible adapted Gao input | 6 | 1.57–1.65 |
| `shortwaveInfrared2` | 7 | 2.11–2.29 |

These are the [USGS sensor band ranges](https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research),
not universal wavelength bounds or automatic sensor selection. The recipe cannot infer a
buffer's actual spectral response or physical units.

## Calibration, masks, and denominators

Each `GPURasterBufferBand` supplies `scale` and `offset`, applying
`reflectance = raw * scale + offset`. Defaults are one and zero for already calibrated
inputs. Raw integer nodata is compared in its native format before it can be lost to
float32 conversion. Each source's `validity` accepts any nonzero flag. The output
intersects every required source domain and rejects nonfinite source, calibrated,
intermediate, and result values. Invalid output samples are canonical quiet NaNs,
with an output validity flag of zero; accepted flags are one.

Calibration must express reflectance as a fraction, not percent, radiance, or unchanged
scaled digital numbers. EVI and SAVI contain additive reflectance constants and therefore
are not invariant to arbitrary input scaling. Even normalized differences change when
an additive calibration offset is omitted. Landsat Collection 2 surface reflectance,
for example, uses scale `0.0000275`, offset `-0.2`, and raw fill value `0`.
See the [USGS scaling guidance](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products).

`epsilon` defaults to zero. A pixel is valid only when the absolute value of its final
formula denominator is **strictly greater** than `epsilon`; equality is rejected.
EVI tests `nearInfrared + 6 * red - 7.5 * blue + 1`, and SAVI tests
`nearInfrared + red + soilAdjustment`. Intermediate sums are not independently subjected
to that threshold. Negative denominators are allowed. No source reflectance or output
index is implicitly clipped, so valid results can lie outside `[-1, 1]`. An optional
`clamp: [minimum, maximum]` explicitly clips finite output results after validity checks.

## Usage

```typescript
import {GPURasterEVI} from '@luma-contributors/analysis/raster-analysis';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';

function addVegetationIndex(
  graph: GPUCommandGraph,
  width: number,
  height: number,
  nearInfrared: GPURasterBufferBand,
  red: GPURasterBufferBand,
  blue: GPURasterBufferBand,
  output: GraphDataView<'float32'>,
  outputValidity: GraphDataView<'uint32'>
): void {
  new GPURasterEVI({
    width,
    height,
    nearInfrared,
    red,
    blue,
    output,
    outputValidity,
    epsilon: 0.000001
  }).addToGraph(graph);
}
```

Inputs and outputs must belong to the same graph and contain one packed sample per pixel.
Outputs must use separate buffers from all input samples, masks, and each other. Shared
read-only source buffers and masks are allowed. Supply distinct `id` values for multiple
recipes in one graph. Re-encoding a compiled graph consumes current input values and masks;
format, dimensions, calibration, methods, and scalar options are fixed at construction.
All imported buffers remain caller-owned. Intermediate buffers belong to the graph.

## Composition and verification

NDWI and NBR each add one band-math pass; SAVI adds three. EVI adds six, including explicit
red and blue calibration stages before weighting. Every pass uses at most six storage
bindings, including all masks. Operations retain the primitive's float32 arithmetic;
this composition does not claim a fused-kernel performance advantage. Dimensions and
buffer sizes remain subject to the public raster primitive and device limits.

The checked-in `raster-spectral-indices.py` oracle calls `spyndex.computeIndex` from
`spyndex==0.12.0` with `numpy==2.2.6` and its bundled, offline formula catalog. Its thirteen
cases include supplied calibration, independent masks, input updates, exact denominator
boundaries, signed denominators, explicit clamping, and SAVI adjustment endpoints.
Calibration and validity are explicit adapters around those independent formula calls.
The Gao case uses the algebraically identical library NDMI formula on the caller-selected
shortwave samples; this is formula-composition evidence, not direct OTB execution or
proof of sensor-response equivalence. See the [Spyndex API](https://spyndex.readthedocs.io/en/latest/reference.html).

GPU tests compare against these independent float64 values with absolute and relative
tolerances of `1e-5`, accounting for float32 calibration and intermediate rounding.
The fixtures keep accepted stress denominators away from zero; the epsilon policy is
not an error-bound certificate for arbitrary ill-conditioned inputs. Additional tests
exercise exact uint32 nodata, nonfinite samples, nonzero buffer offsets, canonical NaNs,
re-encoding, and borrowed-buffer ownership. Node tests inspect all composed shader
sources and validation boundaries. Numerical GPU execution is a separate verification
gate from those node construction tests.

## Executable examples

These examples use explicit synthetic inputs and expose real GPU graph captures.

### GPURasterNDWI

{/* lens:GPURasterNDWI */}

::lens{scenario=gpu-raster.raster-ndwi rung=graph}

### GPURasterNBR

{/* lens:GPURasterNBR */}

::lens{scenario=gpu-raster.raster-nbr rung=graph}

### GPURasterEVI

{/* lens:GPURasterEVI */}

::lens{scenario=gpu-raster.raster-evi rung=graph}

### GPURasterSAVI

{/* lens:GPURasterSAVI */}

::lens{scenario=gpu-raster.raster-savi rung=graph}
