# Calibrated spectral indices

`GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI`, and `GPURasterSAVI` compose
`GPURasterBandMath` into graph-native recipes. They accept co-registered buffer or texture bands,
caller-owned float32 outputs, and a separate uint32 validity output. They do not upload,
submit, read back, resample, choose sensor bands, or decode cloud/quality bits.
`GPURasterNDVI` is the upstream vegetation normalized-difference contributor, imported and
scheduled differently from the four analysis recipes:

```ts
import {GPURasterNDVI, type GPURasterNDVIProps} from '@luma.gl/graph/gpu-raster';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

export function addNDVI(graph: GPUCommandGraph, props: GPURasterNDVIProps): void {
  new GPURasterNDVI(props).addToGraph(graph);
}
```

NDVI and `GPURasterBandMath` accept buffer bands. For textures, explicitly gather each
selected channel and its validity before scheduling NDVI. `applyCalibration: true` applies each
texture band's scale/offset once; the gathered buffer bands below are already calibrated:

```ts
import {createTransientView, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPURasterNDVI, GPURasterTextureToBuffer, type GPURasterTextureBand} from '@luma.gl/graph/gpu-raster';

export function addTextureNDVI(
  graph: GPUCommandGraph, width: number, height: number,
  nearInfrared: GPURasterTextureBand, red: GPURasterTextureBand,
  output: GraphDataView<'float32'>, outputValidity: GraphDataView<'uint32'>
): void {
  const nearInfraredValues = createTransientView(graph, 'texture-ndvi-nir', 'float32', width * height);
  const nearInfraredValidity = createTransientView(graph, 'texture-ndvi-nir-validity', 'uint32', width * height);
  const redValues = createTransientView(graph, 'texture-ndvi-red', 'float32', width * height);
  const redValidity = createTransientView(graph, 'texture-ndvi-red-validity', 'uint32', width * height);
  new GPURasterTextureToBuffer({
    id: 'texture-ndvi-gather-nir', input: nearInfrared, applyCalibration: true,
    output: nearInfraredValues, outputValidity: nearInfraredValidity
  }).addToGraph(graph);
  new GPURasterTextureToBuffer({
    id: 'texture-ndvi-gather-red', input: red, applyCalibration: true,
    output: redValues, outputValidity: redValidity
  }).addToGraph(graph);
  new GPURasterNDVI({
    id: 'texture-ndvi', width, height,
    nearInfrared: {id: 'nir', format: 'float32', validity: nearInfraredValidity,
      storage: {kind: 'buffer', values: nearInfraredValues}},
    red: {id: 'red', format: 'float32', validity: redValidity,
      storage: {kind: 'buffer', values: redValues}},
    output, outputValidity
  }).addToGraph(graph);
}
```

The two texture dimensions must equal `width` and `height`. Each gather rejects raw nodata,
source-mask zeros and nonfinite values; its output validity reaches the matching NDVI buffer
band. NDVI intersects those two validity domains and rejects unusable denominators. The graph
owns these four transient views; output buffers remain caller-owned. Use distinct ID prefixes
when adding more than one copy of this example to a graph.

For a texture-capable recipe without explicit gather wiring, `GPURasterSAVI` with
`soilAdjustment: 0` computes the NDVI formula and internally gathers textures; it retains the
SAVI API and `graph.add` scheduling contract. See the
[NDVI-to-temporal-mean handoff](./remote-sensing.md#ndvi-to-temporal-mean) for per-date outputs,
QA and stack layout.

Add these producers with `graph.add(new GPURasterNBR(props))` (likewise NDWI, EVI and SAVI).
`getCommandNodes(graph)` returns the four analysis recipes' ordered passes without scheduling them.
For those four recipes use `graph.add`; upstream NDVI uses `.addToGraph(graph)`. EVI returns six band-math passes, SAVI three, and NDWI/NBR one each.
Texture inputs add public `GPURasterTextureToBuffer` gather passes for the selected channel,
resolving raw nodata, validity and calibration before band math. They read texture contents on every
submission. Buffer inputs retain their existing band-math route.
Repeated inspection on the same graph reuses those passes. It is safe to inspect with
`getCommandNodes(graph)` and then schedule once with `graph.add(producer)`.
The returned list is frozen and readonly; attempts to remove or reorder inspected passes cannot
change what a subsequent add schedules.

Each producer is implemented in its own file under `raster-analysis/spectral-indices`, sharing the
band-math adapter. The established `gpu-raster-spectral-indices` source module forwards the same
four classes and props types; package imports, pass IDs and pass order are preserved.

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

Each `GPURasterBand` supplies `scale` and `offset`, applying
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
  graph.add(new GPURasterEVI({
    width,
    height,
    nearInfrared,
    red,
    blue,
    output,
    outputValidity,
    epsilon: 0.000001
  }));
}
```

Inputs and outputs must belong to the same graph and contain one packed sample per pixel.
Outputs must use separate buffers from all input samples, masks, and each other. Shared
read-only source buffers and masks are allowed. Separation uses the underlying physical buffer,
including separate graph imports: reimporting the same bytes cannot make an output independent
of a source. Supply distinct `id` values for multiple
recipes in one graph. Re-encoding a compiled graph consumes current input values and masks;
format, dimensions, calibration, methods, and scalar options are fixed at construction.
Each producer captures those fields, output views and nested source-band/storage descriptors once,
including inherited getters. Later caller-map mutation cannot replace a validated output with a
physical source alias or change the declared band bindings. Input buffer contents remain live per run.
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

Construct `GPURasterNDWI` using the [Formula and band contracts](#formula-and-band-contracts). See [Usage](#usage) and [Calibration, masks, and denominators](#calibration-masks-and-denominators).

{/* lens:GPURasterNDWI */}

::lens{scenario=gpu-raster.raster-ndwi rung=graph}

### GPURasterNBR

Construct `GPURasterNBR` using the [Formula and band contracts](#formula-and-band-contracts). See [Usage](#usage) and [Calibration, masks, and denominators](#calibration-masks-and-denominators).

{/* lens:GPURasterNBR */}

::lens{scenario=gpu-raster.raster-nbr rung=graph}

### GPURasterEVI

Construct `GPURasterEVI` using the [Formula and band contracts](#formula-and-band-contracts). See [Usage](#usage) and [Calibration, masks, and denominators](#calibration-masks-and-denominators).

{/* lens:GPURasterEVI */}

::lens{scenario=gpu-raster.raster-evi rung=graph}

### GPURasterSAVI

Construct `GPURasterSAVI` using the [Formula and band contracts](#formula-and-band-contracts). See [Usage](#usage) and [Calibration, masks, and denominators](#calibration-masks-and-denominators).

{/* lens:GPURasterSAVI */}

::lens{scenario=gpu-raster.raster-savi rung=graph}
