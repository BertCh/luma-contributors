// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainOutputSelection} from './terrain-output-selection';

import {
  getTerrainHorizonSectorOutput,
  getTerrainHorizonAnisotropicWeightWGSL
} from './terrain-horizon-output';
export {
  getTerrainHorizonSectorOutput,
  type TerrainHorizonSectorOutputProps
} from './terrain-horizon-output';
import {
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH
} from './terrain-horizon-parameters';
export {
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT,
  getGPUTerrainHorizonParameterValues,
  type GPUTerrainHorizonSettings
} from './terrain-horizon-parameters';
import {validateTerrainBindings} from '../terrain-processing/terrain-binding-preflight';
import {clamp} from '@math.gl/core';
import {getPackedUint32Values} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/graph/gpu-core';
import {
  GPURasterBufferToTexture,
  type GPURasterBand,
  type GPURasterHaloStage,
  type GPURasterMetadata
} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterGridOrthogonal,
  validateRasterParameters,
  validateRasterOutputTexture
} from '../../raster-analysis/raster-band-utils';
import {
  getTerrainHorizonReadWGSL,
  getTerrainHorizonStorageLength,
  TERRAIN_HORIZON_UNORM16_SCALE,
  TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainHorizonFormat,
  validateTerrainHorizonView,
  validateTerrainIlluminationBindingSize,
  validateTerrainIlluminationView,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';
import {
  RASTER_PIXEL_DIRECTION_WGSL,
  type GPURasterCellSizeMode,
  type GPURasterRowDirection,
  validateRasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {resolveTerrainRowDirection} from '../terrain-curvature/terrain-geomorphometry-utils';
import {
  createTerrainHorizonSweepNode,
  getTerrainHorizonGroundCellSizeWGSL,
  createTerrainSweepSectorTable,
  TERRAIN_SWEEP_MAXIMUM_EXTENT,
  TERRAIN_SWEEP_SECTOR_WGSL
} from './terrain-horizon-sweep';

export type {GPUTerrainHorizonFormat} from './terrain-illumination-utils';

/**
 * How {@link GPUTerrainHorizon} finds horizons:
 * - `'march'`: bounded ray march with bilinear samples on the baked step schedule, O(steps) per
 *   pixel and sector. Each ray is a straight pixel line along the sector's ground azimuth at the
 *   pixel's own row, and sample distances are ground meters at that row's cell size. The cell
 *   size is not re-evaluated along the ray: at 60 N and 30 km on a 45 degree sector the samples
 *   lie at a true bearing of 44.75 degrees and 0.14% closer than assumed; re-scaling per row would
 *   only reach 44.83 degrees (a rhumb line, which is not the geodesic either).
 * - `'sweep'`: discrete horizons on digital lines through pixel centres with upper-hull
 *   pointers (Stewart 1998 style), amortised O(1) per pixel and sector. The search is exact (it
 *   equals brute force over the same samples); the sampling is nearest-pixel: each sample takes
 *   the elevation of a pixel centre on the digital line and the distance `k * hypot(1, slope)`
 *   cells along the ideal line, not the distance to that centre. The lateral offset from the ray
 *   through the pixel's own centre is below one pixel (up to 0.86 px at line slope 0.3), so
 *   values differ from the march by sampling, not by precision. `stepGrowth` must be 1. One digital line
 *   family has one pixel slope, so on anisotropic cells it follows each sector's ground azimuth
 *   at the reference row (`height / 2` by default, or the shared `sweepReferenceCellSize`);
 *   other rows of a `'geographic'` tile see the azimuth
 *   rotated by the change of cos(latitude) across the tile, worst at 45 degree sectors: at 60 N,
 *   0.11 degrees at the edge rows of a 0.25 degree tall tile, 0.44 degrees for 1 degree, 2.3
 *   degrees for 5 degrees (about `tan(lat) * halfSpan / 2` radians). The march converts the azimuth
 *   at each pixel's own row and has no such term.
 */
export type GPUTerrainHorizonAlgorithm = 'march' | 'sweep';

// The extra march corner must remain a safe-integer halo accepted by the published assembler.
const TERRAIN_HORIZON_MAXIMUM_MARCH_RADIUS = Number.MAX_SAFE_INTEGER - 1;

/** Degrees per code step of the `'unorm16'` horizon format: `180 / 65534`. */
export const GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES = TERRAIN_HORIZON_UNORM16_STEP_DEGREES;

/**
 * Encodes a horizon angle as a 16-bit code: 0 means invalid (NaN or non-finite); otherwise
 * `1 + floor(clamp(angle, -90, 90) * 65534 / 180 + 32767.5)`, so codes 1 to 65535 cover
 * `[-90, 90]` degrees in steps of {@link GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES}.
 *
 * The GPU computes the same expression in float32 with a float32 scale, so GPU and CPU codes
 * may differ by one at rounding ties.
 */
export function encodeGPUTerrainHorizonUnorm16(angleDegrees: number): number {
  if (!Number.isFinite(angleDegrees)) {
    return 0;
  }
  const clamped = clamp(angleDegrees, -90, 90);
  return 1 + Math.floor(clamped * TERRAIN_HORIZON_UNORM16_SCALE + 32767 + 0.5);
}

/** Decodes a 16-bit horizon code to degrees; code 0 decodes to NaN. */
export function decodeGPUTerrainHorizonUnorm16(code: number): number {
  return code === 0 ? NaN : (code - 1) * TERRAIN_HORIZON_UNORM16_STEP_DEGREES - 90;
}

/**
 * Unpacks `elementCount` angles in degrees from packed `'unorm16'` words (element `e` lives in
 * word `floor(e / 2)`, low half for even `e`). Invalid elements are NaN. Numeric arrays are words;
 * `Uint8Array`, `ArrayBuffer`, and `{bytes}` readbacks are little-endian bytes, respecting the
 * supplied view's byte offset and length. Requires a nonnegative integral element count and at
 * least `ceil(elementCount / 2)` whole words.
 */
export function unpackGPUTerrainHorizonUnorm16(
  input: ArrayLike<number> | ArrayBuffer | {bytes: Uint8Array | ArrayBuffer},
  elementCount: number
): Float64Array {
  const reader = 'unpackGPUTerrainHorizonUnorm16';
  validateRequirement(
    reader,
    Number.isSafeInteger(elementCount) && elementCount >= 0,
    `elementCount must be a nonnegative safe integer; received ${elementCount}`
  );
  const requiredWords = Math.ceil(elementCount / 2);
  let words: Uint32Array;
  try {
    words = getPackedUint32Values('bytes' in input ? input.bytes : input, requiredWords);
  } catch (error) {
    validateRequirement(
      reader,
      false,
      `input requires at least ${requiredWords} whole words: ${String(error)}`
    );
  }
  const angles = new Float64Array(elementCount);
  for (let element = 0; element < elementCount; element++) {
    const word = words[Math.floor(element / 2)];
    angles[element] = decodeGPUTerrainHorizonUnorm16((word >>> ((element & 1) * 16)) & 0xffff);
  }
  return angles;
}

/** Smallest supported `GPUTerrainHorizonProps.directionCount`. */
export const GPU_TERRAIN_HORIZON_MINIMUM_DIRECTION_COUNT = 4;

/** Largest supported `GPUTerrainHorizonProps.directionCount`. */
export const GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT = 64;

/**
 * Returns the ray-march sample distances in pixels used by {@link GPUTerrainHorizon}.
 *
 * `d[0] = 1` and `d[k + 1] = max(d[k] + 1, d[k] * stepGrowth)` while not above `maximumRadius`;
 * `maximumRadius` itself is appended when the sequence stops short of it. `stepGrowth = 1` gives
 * one-pixel steps; larger values give geometric steps that reach far horizons with few samples.
 * Values are rounded to float32 so CPU oracles match the baked WGSL constants.
 * `maximumSampleDistance` optionally bounds materialization to the useful grid distance. It
 * filters the original schedule; it never adds a sample at the cutoff. Contributors use the
 * ceiling of the pixel-centre diagonal (at least one), retaining a conservative rounding margin.
 *
 * @throws If `maximumRadius` is not a positive safe integer, or `stepGrowth` or
 * `maximumSampleDistance` is not finite and `>= 1`.
 */
export function getGPUTerrainHorizonStepDistances(
  maximumRadius: number,
  stepGrowth: number = 1,
  maximumSampleDistance: number = maximumRadius
): Float32Array {
  validateTerrainHorizonSchedule('Terrain horizon', maximumRadius, stepGrowth);
  validateRequirement(
    'Terrain horizon',
    Number.isFinite(maximumSampleDistance) && maximumSampleDistance >= 1,
    'maximumSampleDistance must be finite and >= 1'
  );
  const maximumDistance = Math.min(maximumRadius, maximumSampleDistance);
  const distances: number[] = [];
  for (let distance = 1; distance <= maximumDistance; ) {
    distances.push(distance);
    distance = Math.max(distance + 1, distance * stepGrowth);
  }
  if (maximumRadius <= maximumSampleDistance && distances[distances.length - 1] < maximumRadius) {
    distances.push(maximumRadius);
  }
  return Float32Array.from(distances);
}

/** Validates the schedule without allocating it; sweep construction uses only the radius. */
function validateTerrainHorizonSchedule(
  id: string,
  maximumRadius: number,
  stepGrowth: number
): void {
  validateCapacity(id, 'maximumRadius', maximumRadius, 1, Number.MAX_SAFE_INTEGER);
  validateRequirement(
    id,
    Number.isFinite(stepGrowth) && stepGrowth >= 1,
    'stepGrowth must be finite and >= 1'
  );
}

/**
 * Returns the unit ground-space direction `(east, row-ward)` of one horizon sector, which is also
 * the pixel-space march direction on square ground cells.
 *
 * Sector `d` of `directionCount` has azimuth `d * 360 / directionCount` degrees clockwise from
 * north. Columns increase east; rows increase south for `rowDirection: 'south'`. Components are
 * rounded to float32, and round-off below 1e-9 snaps to 0. The march kernels turn it into a pixel
 * step per row with the row's ground cell size (`getRasterPixelDirection`), so sector azimuths are
 * ground azimuths on anisotropic (for example geographic) grids too.
 */
export function getGPUTerrainHorizonDirection(
  sector: number,
  directionCount: number,
  rowDirection: GPURasterRowDirection = 'south'
): [number, number] {
  const azimuth = (2 * Math.PI * sector) / directionCount;
  const northRowSign = rowDirection === 'south' ? -1 : 1;
  // Snap round-off (sin(PI) = 1.2e-16) to exact zero so axis rays stay on their pixel column/row.
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : Math.fround(value));
  return [snap(Math.sin(azimuth)), snap(northRowSign * Math.cos(azimuth))];
}

/**
 * Caller-owned outputs of {@link GPUTerrainHorizon}; at least one is required.
 */
export type GPUTerrainHorizonOutput = TerrainOutputSelection<{
  /**
   * Horizon angles in degrees, pixel-major: element `pixel * directionCount + sector`.
   * `float32` format: `width * height * directionCount` values. `unorm16` format: a `uint32` view of
   * `ceil(width * height * directionCount / 2)` words.
   */
  horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /**
   * Sky-view factor per pixel in `[0, 1]`, the solid-angle definition of Zaksek et al. (2011):
   * `1 - mean(sin(max(h, 0)))`, the fraction of the hemisphere's solid angle that is visible. Use
   * it for relief visualization. It is not the fraction of isotropic diffuse irradiance: see
   * `irradianceSkyViewFactor`.
   */
  skyViewFactor?: GraphDataView<'float32'>;
  /**
   * Cosine-weighted (irradiance) sky-view factor per pixel in `[0, 1]`, the horizontal-receiver
   * form of Dozier and Frew (1990): `mean(cos^2(max(h, 0)))`, written as
   * `1 - mean(sin^2(max(h, 0)))`. It is the fraction of the isotropic diffuse irradiance of an
   * open horizontal surface that the pixel receives, and the factor `GPUSolarIrradiance` expects
   * as its `skyViewFactor`. A uniform 30 degree horizon gives 0.75 here and 0.5 in
   * `skyViewFactor`; on an infinite plane of slope `s` it is exactly `(1 + cos(s)) / 2`. Computed
   * from the stored horizon map: with `horizonFormat: 'unorm16'` and a bound `output.horizon` it
   * uses the quantised angles (error below 3e-5); without `output.horizon` the contributor keeps
   * a transient float32 map of `width * height * directionCount` values.
   */
  irradianceSkyViewFactor?: GraphDataView<'float32'>;
  /** Positive openness per pixel in degrees. */
  positiveOpenness?: GraphDataView<'float32'>;
  /**
   * Negative openness per pixel in degrees: `90 - mean(h_nadir)`, where `h_nadir` is the
   * horizon of the inverted terrain (z factor negated; curvature still drops samples).
   */
  negativeOpenness?: GraphDataView<'float32'>;
  /**
   * Anisotropic sky-view factor per pixel (Kokalj and Somrak 2019, RVT `asvf`): with the
   * sector's RVT direction angle `a = (360 - t) mod 360` for compass azimuth `t`, main direction
   * `A` (RVT convention) and `w = (1 - wMin) * cos((a - A) / 2)^level + wMin` (signed power, as in
   * RVT), it is `1 - sum(w * sin(max(h, 0))) / sum(w)`. Requires 12 parameter values.
   */
  anisotropicSkyViewFactor?: GraphDataView<'float32'>;
  /** Per-pixel 1 where the center elevation is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving the sky-view factor. */
  skyViewFactorTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
}>;

/**
 * Properties for {@link GPUTerrainHorizon}.
 *
 * Topology: grid size, elevation format and calibration, `directionCount`, `maximumRadius`,
 * `stepGrowth`, `cellSizeMode`, `rowDirection`, `horizonFormat`, and which outputs exist.
 * Per-frame: `parameters` (grid cell size and latitude band, z factor, curvature, maximum
 * distance, anisotropy) and elevation contents.
 *
 * Cell size model: the grid's cell size is projected meters (`'uniform'`), equatorial Web Mercator
 * meters or degrees with a latitude-dependent spacing per row (`'web-mercator'`, `'geographic'`).
 */
export type GPUTerrainHorizonProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-horizon'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /**
   * Per-frame parameters with at least 8 float32 values (12 when `output.anisotropicSkyViewFactor`
   * is bound; 24 with `parameterLayout: 'sweep'`), see {@link getGPUTerrainHorizonParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainHorizonOutput;
  /** Number of azimuth sectors, an integer in `[4, 64]`. Defaults to 16. */
  directionCount?: number;
  /** Search radius in pixels, a positive safe integer. March accepts at most `Number.MAX_SAFE_INTEGER - 1` and needs a `maximumRadius + 1` halo. */
  maximumRadius: number;
  /** Geometric step growth, see {@link getGPUTerrainHorizonStepDistances}. Defaults to 1. */
  stepGrowth?: number;
  /**
   * Horizon search, see {@link GPUTerrainHorizonAlgorithm}. Defaults to `'march'`. `'sweep'` needs
   * `stepGrowth` 1 and grid extents up to 32767. The sweep is not universally faster: it has a
   * large fixed cost per pixel and sector, so it wins at large radii and loses at small ones.
   * Measured with 16 sectors on an Apple-silicon laptop: 1024^2 at the full radius 1023 sweep 41 ms
   * against march about 1300 ms, at radius 256 about 100 ms against 360 ms; 512^2 at radius 128
   * sweep 60 ms (125 ms with every output) against march about 20 ms. At 512^2 with every output
   * the sweep still loses at radius 256 (218 against 58 ms) and wins at the full radius 511 (36
   * against 109 ms). Prefer `'march'` for small radii and `'sweep'` near the full tile radius.
   */
  algorithm?: GPUTerrainHorizonAlgorithm;
  /**
   * Compile-time interpretation of parameters. Defaults to `'standard'`: base settings and
   * optional anisotropy; spare view capacity has no meaning. `'sweep'` requires 24 values packed
   * with `sweepReferenceCellSize`, `sweepSourceGrid` and per-frame `sweepOrigin`. It anchors sweep
   * phase and both algorithms' source-row metrics. Sweep halos, and any latitude-dependent halo,
   * require this explicit layout. Placement changes reuse the same compiled shader.
   */
  parameterLayout?: 'standard' | 'sweep';
  /**
   * Upper bound on `pixels * steps * sectors` one `'march'` dispatch may cost; the sectors are
   * split over as many dispatches as needed (at least one sector each) so a dispatch stays short
   * on large tiles. Results do not depend on it. Defaults to 2^30.
   */
  maximumStepsPerDispatch?: number;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Direction in which the row index increases. Defaults to the direction the grid states: the
   * sign of the row term `grid.affine[4]` (negative: `'south'`, a north-up raster; positive:
   * `'north'`). An explicit value that contradicts the grid throws at construction.
   */
  rowDirection?: GPURasterRowDirection;
  /**
   * Storage of `output.horizon`. `'float32'` (default) stores degrees; `'unorm16'` packs two
   * 16-bit codes per `uint32` word (see {@link encodeGPUTerrainHorizonUnorm16}), a half-size map
   * with 0.0027 degree steps. Sky-view factor and openness sums always use the unquantised angle.
   */
  horizonFormat?: GPUTerrainHorizonFormat;
};

/**
 * Computes horizon-angle maps for `directionCount` azimuth sectors by bounded ray marching, plus
 * sky-view factor, openness and anisotropic sky-view factor from the same rays.
 *
 * For every valid pixel and sector, the ray walks the {@link getGPUTerrainHorizonStepDistances}
 * schedule, samples elevation bilinearly (a sample is skipped when any of its four corners is
 * invalid; the ray stops when it leaves the grid or `maximumDistance`), and keeps the largest
 * elevation angle `atan((zFactor * (z - z0) - c * d^2) / d)` in degrees. A ray with no valid sample
 * reports 0 (flat horizon). Ground distance uses the center row's cell size.
 * The final scheduled sample at `maximumRadius` is included. For GRASS `r.horizon` cardinal
 * comparisons on square cells, use `distance=1`, curvature `1 / (2 * 6371000)`, compare
 * `max(horizon, 0)`, and set its `maxdistance` to `(maximumRadius - 0.5) * cellSize`:
 * GRASS tests its cap after evaluating the sample. A nodata computational-region halo also
 * keeps GRASS's initial southwest-cell sentinel outside the terrain. These adapters do not
 * imply parity for arbitrary azimuths, where the sampling paths differ.
 *
 * - Sky-view factor (Zaksek et al. 2011): `1 - mean(sin(max(h, 0)))`, the visible fraction of
 *   the hemisphere's solid angle. This is the relief-visualization definition.
 * - Irradiance sky-view factor (Dozier and Frew 1990, horizontal receiver):
 *   `mean(cos^2(max(h, 0)))`, the fraction of isotropic diffuse irradiance received. This is the
 *   definition `GPUSolarIrradiance` needs; the two differ (0.5 against 0.75 under a uniform 30
 *   degree horizon).
 * - Positive openness (Yokoyama et al. 2002): `mean(90 - h)` degrees.
 * - Negative openness: `90 - mean(h_nadir)` with `h_nadir` the horizon of the inverted terrain
 *   (z factor negated, curvature still drops), tracked in the same march loop.
 * - Anisotropic sky-view factor (Kokalj and Somrak 2019, RVT `asvf`), with RVT's weights and
 *   direction convention: sector weights `(1 - wMin) * cos((a - A) / 2)^level + wMin`, where `a` is
 *   the sector's RVT direction angle (counterclockwise from north, `(360 - t) mod 360` for compass
 *   azimuth `t`), `A` is `rvtDirectionDegrees` exactly as RVT's `asvf_dir`, and the power is
 *   signed with an integer level. The heaviest sector is at compass azimuth `(360 - A) mod 360`.
 *   The direction convention and the three reductions (sky-view factor, openness, anisotropic
 *   factor) are checked against rvt-py 2.2.1. RVT's numbers are reproduced only where it reads the
 *   same samples: RVT takes whole pixels (1 to the radius on the axes, `(k, k)` on the diagonals,
 *   a rounded staircase in between) and reflect-pads the array by the radius. There the march
 *   with 4 sectors and the sweep with 4 or 8 agree with RVT to 3e-7 (3e-5 degrees of openness).
 *   Elsewhere the difference is sampling: on a plane of gradient 0.5 with 16 sectors the march
 *   gives the closed form, while RVT reads a sky-view factor 0.022 lower and an openness 3.3
 *   degrees lower (the sweep 0.019 and 2.6 degrees lower). Pixels nearer than `maximumRadius` to
 *   the edge differ from RVT because rays stop at the edge here and see mirrored terrain there.
 *
 * With `algorithm: 'sweep'` the per-sector march is replaced by an exact upper-hull sweep over
 * digital lines through pixel centres (one invocation per line, amortised O(1) per pixel and
 * sector, and a second nadir sweep when `negativeOpenness` is bound); every output keeps its
 * meaning. Measured on 1024^2 with 16 sectors: 98.6 ms against 411 ms at radius 256 and 40.6 ms
 * against 1474 ms at radius 1023 on an Apple-silicon laptop.
 *
 * The march runs all sectors of a pixel in one invocation, in chunks of sectors sized so one dispatch
 * stays under about 2^30 sample steps; sums accumulate in fixed sector order, so results are
 * deterministic and identical to one dispatch per sector. Invalid centers receive NaN and validity 0. Pixels closer
 * than `maximumRadius` to the tile edge see a truncated horizon. For march, pass a tile with a
 * `maximumRadius + 1` halo: bilinear sampling reads the next corner even at zero weight, and
 * that corner's validity matters. Obtain the published `GPURasterHaloStage` via `getHaloStage()`.
 * The explicit `parameterLayout: 'sweep'` reads source origin, reference cells and a common source
 * row georeference from 24 per-frame parameters. It preserves sweep phase, curvature decisions,
 * row spacing and distance cutoffs across windows; `getHaloStage()` returns `maximumRadius` for
 * this sweep. Any latitude-dependent halo also requires this layout, including march. Standard
 * sweep parameters retain the local model and refuse halo assembly, regardless of view padding.
 * CPU sweep oracles accept the same origin, packed reference cell and source georeference.
 * March splits displacement into integer/fractional parts before adding the integer pixel origin,
 * so cropping cannot change the validity corners through float32 coordinate addition.
 */
export class GPUTerrainHorizon implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainHorizonProps;
  /** Number of azimuth sectors. */
  readonly directionCount: number;
  /** Grid-bounded ray-march distances in pixels; empty for sweep, which has no march schedule. */
  readonly stepDistances: Float32Array;
  /** Storage format of the optional `output.horizon`. */
  readonly horizonFormat: GPUTerrainHorizonFormat;
  /** Horizon search algorithm. */
  readonly algorithm: GPUTerrainHorizonAlgorithm;
  /** Explicit parameter interpretation; spare capacity never selects a layout. */
  readonly parameterLayout: 'standard' | 'sweep';
  /** Validated radius captured independently of the caller's props object. */
  readonly maximumRadius: number;
  /** Row direction the kernels use: the prop, or the direction the grid affine states. */
  readonly rowDirection: GPURasterRowDirection;

  constructor(props: GPUTerrainHorizonProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'elevation',
      'parameters',
      'output',
      'directionCount',
      'maximumRadius',
      'stepGrowth',
      'algorithm',
      'parameterLayout',
      'maximumStepsPerDispatch',
      'cellSizeMode',
      'rowDirection',
      'horizonFormat'
    ]);
    this.id = props.id ?? 'terrain-horizon';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    const directionCount = props.directionCount ?? 16;
    if (
      !Number.isSafeInteger(directionCount) ||
      directionCount < GPU_TERRAIN_HORIZON_MINIMUM_DIRECTION_COUNT ||
      directionCount > GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT
    ) {
      throw new Error(`${id} directionCount must be an integer in [4, 64]`);
    }
    this.directionCount = directionCount;
    validateTerrainHorizonSchedule(id, props.maximumRadius, props.stepGrowth ?? 1);
    this.maximumRadius = props.maximumRadius;
    this.parameterLayout = props.parameterLayout ?? 'standard';
    validateRequirement(
      id,
      this.parameterLayout === 'standard' || this.parameterLayout === 'sweep',
      'parameterLayout must be standard or sweep'
    );
    if ((props.algorithm ?? 'march') === 'march') {
      validateCapacity(
        id,
        'maximumRadius',
        this.maximumRadius,
        1,
        TERRAIN_HORIZON_MAXIMUM_MARCH_RADIUS
      );
    }
    if (
      !output.horizon &&
      !output.skyViewFactor &&
      !output.irradianceSkyViewFactor &&
      !output.positiveOpenness &&
      !output.negativeOpenness &&
      !output.anisotropicSkyViewFactor &&
      !output.outputValidity &&
      !output.skyViewFactorTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainHorizonFormat(id, props.horizonFormat ?? 'float32');
    this.horizonFormat = props.horizonFormat ?? 'float32';
    this.algorithm = props.algorithm ?? 'march';
    if (this.algorithm !== 'march' && this.algorithm !== 'sweep') {
      throw new Error(`${id} algorithm must be march or sweep`);
    }
    if (this.algorithm === 'sweep') {
      if ((props.stepGrowth ?? 1) !== 1) {
        throw new Error(`${id} stepGrowth applies to the march algorithm only`);
      }
      if (
        props.grid.width > TERRAIN_SWEEP_MAXIMUM_EXTENT ||
        props.grid.height > TERRAIN_SWEEP_MAXIMUM_EXTENT
      ) {
        throw new Error(`${id} sweep supports extents up to ${TERRAIN_SWEEP_MAXIMUM_EXTENT}`);
      }
    }
    this.stepDistances =
      this.algorithm === 'march'
        ? getGPUTerrainHorizonStepDistances(
            props.maximumRadius,
            props.stepGrowth ?? 1,
            Math.max(1, Math.ceil(Math.hypot(props.grid.width - 1, props.grid.height - 1)))
          )
        : new Float32Array(0);
    if (
      props.maximumStepsPerDispatch !== undefined &&
      !(Number.isFinite(props.maximumStepsPerDispatch) && props.maximumStepsPerDispatch > 0)
    ) {
      throw new Error(`${id} maximumStepsPerDispatch must be positive`);
    }
    validateTerrainHorizonView(
      id,
      'output.horizon',
      output.horizon,
      this.horizonFormat,
      pixelCount,
      directionCount
    );
    validateTerrainIlluminationView(
      id,
      'output.skyViewFactor',
      output.skyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.irradianceSkyViewFactor',
      output.irradianceSkyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.positiveOpenness',
      output.positiveOpenness,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.negativeOpenness',
      output.negativeOpenness,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.anisotropicSkyViewFactor',
      output.anisotropicSkyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.outputValidity',
      output.outputValidity,
      'uint32',
      pixelCount
    );
    validateRasterParameters(
      id,
      props.parameters,
      this.parameterLayout === 'sweep'
        ? GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH
        : output.anisotropicSkyViewFactor
          ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
          : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
    validateRasterOutputTexture(
      id,
      'output.skyViewFactorTexture',
      output.skyViewFactorTexture,
      ['r32float', 'rgba32float'],
      props.grid
    );
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    // Distances and gradients use the two pixel-axis lengths as if perpendicular.
    validateRasterGridOrthogonal(id, props.grid);
    this.rowDirection = resolveTerrainRowDirection(id, props.grid, props.rowDirection);
    validateRasterBuffersDistinct(
      id,
      [
        output.horizon,
        output.skyViewFactor,
        output.irradianceSkyViewFactor,
        output.positiveOpenness,
        output.negativeOpenness,
        output.anisotropicSkyViewFactor,
        output.outputValidity
      ],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns a complete stage for march, or an anchored sweep with a shared ground reference. */
  getHaloStage(): GPURasterHaloStage {
    validateRequirement(
      this.id,
      (this.props.cellSizeMode ?? 'uniform') === 'uniform' || this.parameterLayout === 'sweep',
      'latitude-dependent halo needs parameterLayout sweep with a shared sweepSourceGrid'
    );
    if (this.algorithm === 'march') {
      return {requiredHalo: this.maximumRadius + 1};
    }
    validateRequirement(
      this.id,
      this.parameterLayout === 'sweep',
      'algorithm sweep needs parameterLayout sweep for a seamless halo bound'
    );
    return {requiredHalo: this.maximumRadius};
  }

  /** Returns elevation canonicalization, the fused march (or one sweep node per sector), finalize, and texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTerrainBindings(graph, this.id, this.props, {
      'canonical and hull planes': 4 * this.props.grid.width * this.props.grid.height,
      'horizon map':
        !this.props.output.horizon && this.props.output.irradianceSkyViewFactor
          ? 4 * this.props.grid.width * this.props.grid.height * this.directionCount
          : 0,
      'sector table':
        this.algorithm === 'sweep' ||
        (this.horizonFormat === 'unorm16' && this.directionCount % 2 === 1)
          ? 256 * this.directionCount
          : 256 *
            Math.ceil(
              this.directionCount /
                Math.max(
                  1,
                  Math.min(
                    this.directionCount,
                    Math.floor(
                      (this.props.maximumStepsPerDispatch ?? FUSED_MARCH_STEP_BUDGET) /
                        (this.props.grid.width * this.props.grid.height * this.stepDistances.length)
                    )
                  )
                )
            )
    });
    const {id, props, directionCount} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, [output.skyViewFactorTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.horizon,
      output.skyViewFactor,
      output.irradianceSkyViewFactor,
      output.positiveOpenness,
      output.negativeOpenness,
      output.anisotropicSkyViewFactor,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    // The irradiance sky-view factor is reduced from the horizon map, so it keeps one (float32)
    // when the caller binds none.
    const horizonFormat: GPUTerrainHorizonFormat = output.horizon ? this.horizonFormat : 'float32';
    const horizonTarget: GraphDataView<'float32'> | GraphDataView<'uint32'> | undefined =
      output.horizon ??
      (output.irradianceSkyViewFactor
        ? createTransientView(
            graph,
            `${id}-horizon-map`,
            'float32',
            getTerrainHorizonStorageLength('float32', pixelCount, directionCount)
          )
        : undefined);
    if (horizonTarget) {
      validateTerrainIlluminationBindingSize(
        id,
        'horizon',
        getTerrainHorizonStorageLength(horizonFormat, pixelCount, directionCount) * 4,
        graph.device.limits.maxStorageBufferBindingSize
      );
    }
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const elevationValues = source.band.storage.values as GraphDataView<'float32'>;
    const elevationValidity = source.band.validity as GraphDataView<'uint32'>;
    const skyViewTarget =
      output.skyViewFactor ??
      (output.skyViewFactorTexture
        ? createTransientView(graph, `${id}-sky-view-factor`, 'float32', pixelCount)
        : undefined);
    const needsSums = Boolean(skyViewTarget || output.positiveOpenness);
    const sineSum = needsSums
      ? createTransientView(graph, `${id}-sine-sum`, 'float32', pixelCount)
      : undefined;
    const angleSum = needsSums
      ? createTransientView(graph, `${id}-angle-sum`, 'float32', pixelCount)
      : undefined;
    const anisotropicSum = output.anisotropicSkyViewFactor
      ? createTransientView(graph, `${id}-anisotropic-sum`, 'float32', pixelCount)
      : undefined;
    const nadirSum = output.negativeOpenness
      ? createTransientView(graph, `${id}-nadir-sum`, 'float32', pixelCount)
      : undefined;
    const rowDirection = this.rowDirection;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const distances =
      this.algorithm === 'march'
        ? Array.from(this.stepDistances, getWGSLFloatLiteral).join(', ')
        : '';
    const hull =
      this.algorithm === 'sweep'
        ? createTransientView(graph, `${id}-sweep-hull`, 'uint32', pixelCount)
        : undefined;
    // The march fuses sectors into few kernels: sums stay in registers and are stored once per
    // chunk, the centre is loaded once, and every sector of a pixel is written by one thread.
    // An odd sector count with packed unorm16 codes shares words between neighbouring pixels, which
    // only the one-sector-per-dispatch path can update safely.
    const fusedMarch = !hull && !(this.horizonFormat === 'unorm16' && directionCount % 2 === 1);
    if (fusedMarch) {
      nodes.push(
        ...getFusedMarchNodes(graph, {
          id,
          width,
          height,
          directionCount,
          rowDirection,
          distances,
          stepCount: this.stepDistances.length,
          sourceGridIndex: this.parameterLayout === 'sweep' ? 16 : undefined,
          maximumStepsPerDispatch: props.maximumStepsPerDispatch ?? FUSED_MARCH_STEP_BUDGET,
          cellSizeMode,
          elevationValues,
          settings: props.parameters,
          horizonFormat,
          horizon: horizonTarget,
          sineSum,
          angleSum,
          anisotropicSum,
          nadirSum
        })
      );
    }
    const sectorRanges =
      !fusedMarch && !hull
        ? createHorizonSectorRanges<Parameters>(graph, {id, directionCount, sectorsPerRange: 1})
        : undefined;
    if (sectorRanges) {
      nodes.push(sectorRanges.writeNode);
    }
    const sweepSectors = hull
      ? createTerrainSweepSectorTable<Parameters>(graph, {
          id,
          width,
          height,
          directions: Array.from({length: directionCount}, (_, sector) =>
            getGPUTerrainHorizonDirection(sector, directionCount, rowDirection)
          )
        })
      : undefined;
    if (sweepSectors) {
      nodes.push(sweepSectors.writeNode);
    }
    for (let sector = 0; !fusedMarch && sector < directionCount; sector++) {
      if (hull) {
        nodes.push(
          ...getSweepSectorNodes(graph, {
            id: `${id}-horizon-${sector}`,
            width,
            height,
            sectorRecord: sweepSectors!.getRecord(sector),
            directionCount,
            maximumRadius: this.maximumRadius,
            originIndex: this.parameterLayout === 'sweep' ? 14 : undefined,
            sourceGridIndex: this.parameterLayout === 'sweep' ? 16 : undefined,
            referenceCellSizeIndex: this.parameterLayout === 'sweep' ? 12 : undefined,
            cellSizeMode,
            elevationValues,
            settings: props.parameters,
            hull,
            horizonFormat,
            horizon: horizonTarget,
            sineSum,
            angleSum,
            anisotropicSum,
            nadirSum
          })
        );
        continue;
      }
      nodes.push(
        getHorizonNode(graph, {
          id: `${id}-horizon-${sector}`,
          width,
          height,
          sectorRange: sectorRanges!.getRange(sector),
          directionCount,
          rowDirection,
          distances,
          stepCount: this.stepDistances.length,
          sourceGridIndex: this.parameterLayout === 'sweep' ? 16 : undefined,
          cellSizeMode,
          elevationValues,
          settings: props.parameters,
          horizonFormat,
          horizon: horizonTarget,
          sineSum,
          angleSum,
          anisotropicSum,
          nadirSum
        })
      );
    }
    if (skyViewTarget || output.positiveOpenness || output.outputValidity) {
      const bindings: WGSLKernelBinding[] = [
        {
          name: 'elevationValidity',
          view: elevationValidity,
          type: 'u32',
          access: 'read'
        }
      ];
      if (sineSum && angleSum) {
        bindings.push(
          {name: 'sineSum', view: sineSum, type: 'f32', access: 'read'},
          {name: 'angleSum', view: angleSum, type: 'f32', access: 'read'}
        );
      }
      if (skyViewTarget) {
        bindings.push({
          name: 'skyView',
          view: skyViewTarget,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (output.positiveOpenness) {
        bindings.push({
          name: 'openness',
          view: output.positiveOpenness,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (output.outputValidity) {
        bindings.push({
          name: 'validityValues',
          view: output.outputValidity,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sky-view`,
          operation: 'GPUTerrainHorizon',
          variant: 'sky-view',
          bindings,
          invocationCount: pixelCount,
          declarations: `const DIRECTION_COUNT: f32 = ${directionCount}.0;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}`,
          body: `let isValid = elevationValidity[elevationValidityOffset + index] != 0u${
            sineSum ? ' &&\n    isFiniteValue(angleSum[angleSumOffset + index])' : ''
          };
  let invalidValue = getQuietNaN(index);
  ${
    skyViewTarget
      ? `skyView[skyViewOffset + index] = select(invalidValue,
    1.0 - sineSum[sineSumOffset + index] / DIRECTION_COUNT, isValid);`
      : ''
  }
  ${
    output.positiveOpenness
      ? `openness[opennessOffset + index] = select(invalidValue,
    90.0 - angleSum[angleSumOffset + index] / DIRECTION_COUNT, isValid);`
      : ''
  }
  ${output.outputValidity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
        })
      );
    }
    if (output.irradianceSkyViewFactor && horizonTarget) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-irradiance-sky-view`,
          operation: 'GPUTerrainHorizon',
          variant: `irradiance-sky-view-${horizonFormat}`,
          bindings: [
            {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'},
            {
              name: 'horizon',
              view: horizonTarget,
              type: horizonFormat === 'unorm16' ? 'u32' : 'f32',
              access: 'read'
            },
            {
              name: 'irradianceSkyView',
              view: output.irradianceSkyViewFactor,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: pixelCount,
          declarations: `const DIRECTION_COUNT: u32 = ${directionCount}u;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainHorizonReadWGSL(horizonFormat, 'horizon')}`,
          // Dozier and Frew (1990), horizontal receiver under an isotropic sky: the irradiance from
          // the sky above a horizon at elevation h is proportional to cos^2(h) per unit azimuth.
          body: `var sineSquaredSum = 0.0;
  for (var sector = 0u; sector < DIRECTION_COUNT; sector++) {
    let sine = sin(max(readHorizonAngle(index * DIRECTION_COUNT + sector), 0.0) * DEGREES_TO_RADIANS);
    sineSquaredSum += sine * sine;
  }
  let isValid = elevationValidity[elevationValidityOffset + index] != 0u &&
    isFiniteValue(sineSquaredSum);
  irradianceSkyView[irradianceSkyViewOffset + index] = select(getQuietNaN(index),
    1.0 - sineSquaredSum / f32(DIRECTION_COUNT), isValid);`
        })
      );
    }
    if (
      (anisotropicSum && output.anisotropicSkyViewFactor) ||
      (nadirSum && output.negativeOpenness)
    ) {
      // Kept apart from the sky-view kernel so no kernel binds more than 8 storage buffers.
      const bindings: WGSLKernelBinding[] = [
        {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'}
      ];
      let validExpression = 'elevationValidity[elevationValidityOffset + index] != 0u';
      let statements = '';
      if (anisotropicSum && output.anisotropicSkyViewFactor) {
        bindings.push(
          {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'anisotropicSum', view: anisotropicSum, type: 'f32', access: 'read'},
          {
            name: 'anisotropicSkyView',
            view: output.anisotropicSkyViewFactor,
            type: 'f32',
            access: 'read_write'
          }
        );
        validExpression += ' &&\n    isFiniteValue(anisotropicSum[anisotropicSumOffset + index])';
        statements += `var weightSum = 0.0;
  for (var sector = 0u; sector < DIRECTION_COUNT_U32; sector++) {
    weightSum += getAnisotropicWeight(sector);
  }
  anisotropicSkyView[anisotropicSkyViewOffset + index] = select(invalidValue,
    1.0 - anisotropicSum[anisotropicSumOffset + index] / weightSum, isValid);
  `;
      }
      if (nadirSum && output.negativeOpenness) {
        bindings.push(
          {name: 'nadirSum', view: nadirSum, type: 'f32', access: 'read'},
          {
            name: 'negativeOpennessValues',
            view: output.negativeOpenness,
            type: 'f32',
            access: 'read_write'
          }
        );
        validExpression += ' &&\n    isFiniteValue(nadirSum[nadirSumOffset + index])';
        statements += `negativeOpennessValues[negativeOpennessValuesOffset + index] = select(invalidValue,
    90.0 - nadirSum[nadirSumOffset + index] / DIRECTION_COUNT, isValid);
  `;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extended`,
          operation: 'GPUTerrainHorizon',
          variant: 'extended',
          bindings,
          invocationCount: pixelCount,
          declarations: `const DIRECTION_COUNT: f32 = ${directionCount}.0;
const DIRECTION_COUNT_U32: u32 = ${directionCount}u;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${anisotropicSum ? getTerrainHorizonAnisotropicWeightWGSL(directionCount) : ''}`,
          body: `let isValid = ${validExpression};
  let invalidValue = getQuietNaN(index);
  ${statements}`
        })
      );
    }
    if (output.skyViewFactorTexture && skyViewTarget) {
      const texture = output.skyViewFactorTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-sky-view-texture`,
            input: {
              id: `${id}-sky-view-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: skyViewTarget}
            },
            output: texture,
            channel: 0
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

/** Builds the sweep kernels of one sector: zenith outputs, then the nadir pass when needed. */
function getSweepSectorNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    /** Sector record from the contributor's {@link createTerrainSweepSectorTable}. */
    sectorRecord: GraphDataView<'uint32'>;
    directionCount: number;
    maximumRadius: number;
    originIndex?: number;
    sourceGridIndex?: number;
    referenceCellSizeIndex?: number;
    cellSizeMode: GPURasterCellSizeMode;
    elevationValues: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    hull: GraphDataView<'uint32'>;
    horizonFormat: GPUTerrainHorizonFormat;
    horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    sineSum?: GraphDataView<'float32'>;
    angleSum?: GraphDataView<'float32'>;
    anisotropicSum?: GraphDataView<'float32'>;
    nadirSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const common = {
    width: props.width,
    height: props.height,
    geometry: {kind: 'table' as const, record: props.sectorRecord},
    maximumRadius: props.maximumRadius,
    originIndex: props.originIndex,
    sourceGridIndex: props.sourceGridIndex,
    referenceCellSizeIndex: props.referenceCellSizeIndex,
    cellSizeMode: props.cellSizeMode,
    elevationValues: props.elevationValues,
    settings: props.settings,
    hull: props.hull
  };
  const nodes: GPUCommandNode<Parameters>[] = [];
  // The sweep binds elevation values, settings, hull pointers, and the sector record itself.
  if (props.horizon || props.sineSum || props.angleSum || props.anisotropicSum) {
    const output = getTerrainHorizonSectorOutput({
      sector: 0,
      sectorWGSL: TERRAIN_SWEEP_SECTOR_WGSL,
      directionCount: props.directionCount,
      horizonFormat: props.horizonFormat,
      horizon: props.horizon,
      sineSum: props.sineSum,
      angleSum: props.angleSum,
      anisotropicSum: props.anisotropicSum,
      mode: 'zenith',
      reservedBindingCount: 4
    });
    nodes.push(
      createTerrainHorizonSweepNode(graph, {
        ...common,
        id: props.id,
        zFactorSign: 1,
        outputBindings: output.bindings,
        outputDeclarations: output.declarations,
        outputWGSL: output.wgsl
      })
    );
  }
  if (props.nadirSum) {
    const output = getTerrainHorizonSectorOutput({
      sector: 0,
      sectorWGSL: TERRAIN_SWEEP_SECTOR_WGSL,
      directionCount: props.directionCount,
      nadirSum: props.nadirSum,
      mode: 'nadir',
      reservedBindingCount: 4
    });
    nodes.push(
      createTerrainHorizonSweepNode(graph, {
        ...common,
        id: `${props.id}-nadir`,
        zFactorSign: -1,
        outputBindings: output.bindings,
        outputDeclarations: output.declarations,
        outputWGSL: output.wgsl
      })
    );
  }
  return nodes;
}

/** Elements between sector range words: 256 bytes, so every range binds at element offset 0. */
const SECTOR_RANGE_STRIDE = 64;

/**
 * Creates the graph-owned sector ranges of the march kernels and the one-thread node that writes
 * them: range `r` is `[r * sectorsPerRange, min(directionCount, (r + 1) * sectorsPerRange))` at
 * element `r * 64`. Every march kernel reads its sectors from its range word, so all of them share
 * one WGSL source (library rule 12).
 */
function createHorizonSectorRanges<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; directionCount: number; sectorsPerRange: number}
): {
  rangeCount: number;
  writeNode: GPUCommandNode<Parameters>;
  getRange: (range: number) => GraphDataView<'uint32'>;
} {
  const {directionCount, sectorsPerRange} = props;
  const rangeCount = Math.ceil(directionCount / sectorsPerRange);
  const table = createTransientView(
    graph,
    `${props.id}-sector-range-words`,
    'uint32',
    rangeCount * SECTOR_RANGE_STRIDE
  );
  const writeNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-sector-ranges`,
    operation: 'GPUTerrainHorizon',
    variant: 'sector-ranges',
    bindings: [{name: 'ranges', view: table, type: 'u32', access: 'read_write'}],
    invocationCount: rangeCount,
    body: `let first = index * ${sectorsPerRange}u;
  ranges[rangesOffset + index * ${SECTOR_RANGE_STRIDE}u] = first;
  ranges[rangesOffset + index * ${SECTOR_RANGE_STRIDE}u + 1u] = min(${directionCount}u, first + ${sectorsPerRange}u);`
  });
  const getRange = (range: number): GraphDataView<'uint32'> =>
    graph.createDataView(table.buffer, {
      format: 'uint32',
      length: 2,
      byteOffset: table.byteOffset + range * SECTOR_RANGE_STRIDE * 4
    });
  return {rangeCount, writeNode, getRange};
}

/** Pixel access and bilinear sampling shared by the per-sector and the fused march kernels. */
const HORIZON_MARCH_SAMPLE_WGSL = `// Canonical elevation is NaN exactly where a pixel is invalid, so the march kernels need no
// validity binding (its slot holds the sector range word).
fn isValidPixel(column: u32, row: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + row * WIDTH + column]);
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation relative to \`origin\` at an in-grid pixel-center position; .y is 0 when any
// corner is invalid. Corners are made origin-relative before interpolating, so the interpolation
// rounds at the scale of the local relief instead of the absolute elevation (at 4000 m the absolute
// form loses about 2.4e-4 m, 1.4e-3 degrees over one 10 m step).
fn sampleElevation(base: vec2<u32>, fraction: vec2<f32>, origin: f32) -> vec2<f32> {
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  // Canonical elevation is NaN exactly where a pixel is invalid: one bit test per corner replaces
  // four validity loads.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00 - origin, corner10 - origin, fraction.x);
  let bottom = mix(corner01 - origin, corner11 - origin, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`;

/** Sample steps (pixels x steps x sectors) one fused march dispatch may cost, to bound its duration. */
const FUSED_MARCH_STEP_BUDGET = 2 ** 30;

/**
 * Builds the fused ray-march kernels: each invocation marches every sector of its pixel in order,
 * `sectorsPerChunk` sectors per dispatch. Per sector the arithmetic is that of
 * {@link getHorizonNode}; the sums add in the same sector order, kept in registers and stored once
 * per chunk (read back only when a later chunk continues them), so results are bit-identical to
 * one dispatch per sector with far less memory traffic and no per-sector dispatch.
 */
function getFusedMarchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    directionCount: number;
    rowDirection: GPURasterRowDirection;
    distances: string;
    stepCount: number;
    maximumStepsPerDispatch: number;
    cellSizeMode: GPURasterCellSizeMode;
    sourceGridIndex?: number;
    elevationValues: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    horizonFormat: GPUTerrainHorizonFormat;
    horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    sineSum?: GraphDataView<'float32'>;
    angleSum?: GraphDataView<'float32'>;
    anisotropicSum?: GraphDataView<'float32'>;
    nadirSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {directionCount} = props;
  const hasZenith = Boolean(
    props.horizon || props.sineSum || props.angleSum || props.anisotropicSum
  );
  const zenithOutput = hasZenith
    ? getTerrainHorizonSectorOutput({
        sector: 0,
        sectorWGSL: 'sector',
        directionCount,
        horizonFormat: props.horizonFormat,
        horizon: props.horizon,
        sineSum: props.sineSum,
        angleSum: props.angleSum,
        anisotropicSum: props.anisotropicSum,
        mode: 'zenith',
        reservedBindingCount: 3 + (props.nadirSum ? 1 : 0)
      })
    : undefined;
  const nadirOutput = props.nadirSum
    ? getTerrainHorizonSectorOutput({
        sector: 0,
        sectorWGSL: 'sector',
        directionCount,
        nadirSum: props.nadirSum,
        mode: 'nadir',
        reservedBindingCount: 3 + (zenithOutput?.bindings.length ?? 0)
      })
    : undefined;
  const bindings: WGSLKernelBinding[] = [
    {name: 'elevationValues', view: props.elevationValues, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    ...(zenithOutput?.bindings ?? []),
    ...(nadirOutput?.bindings ?? [])
  ];
  const nadir = Boolean(nadirOutput);
  const pixelCount = props.width * props.height;
  const sectorsPerChunk = clamp(
    Math.floor(props.maximumStepsPerDispatch / (pixelCount * props.stepCount)),
    1,
    directionCount
  );
  const directions = Array.from({length: directionCount}, (_, sector) => {
    const [x, y] = getGPUTerrainHorizonDirection(sector, directionCount, props.rowDirection);
    return `vec2<f32>(${getWGSLFloatLiteral(x)}, ${getWGSLFloatLiteral(y)})`;
  }).join(', ');
  const format = props.horizonFormat;
  const statements: string[] = [];
  if (props.horizon) {
    statements.push(
      format === 'unorm16'
        ? `// Both sectors of a word belong to this pixel (even sector count), so no other thread shares it.
      let element = pixel * DIRECTION_COUNT + sector;
      let wordIndex = horizonOffset + (element >> 1u);
      let shift = (element & 1u) * 16u;
      horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
        (encodeHorizonUnorm16(horizonAngle) << shift);`
        : 'horizon[horizonOffset + pixel * DIRECTION_COUNT + sector] = horizonAngle;'
    );
  }
  if (props.sineSum || props.anisotropicSum) {
    statements.push(
      'let sineTerm = sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);',
      ...(props.sineSum
        ? ['sineAccumulator = select(sineAccumulator + sineTerm, sineTerm, sector == 0u);']
        : []),
      ...(props.anisotropicSum
        ? [
            'let anisotropicTerm = getAnisotropicWeight(sector) * sineTerm;',
            'anisotropicAccumulator = select(anisotropicAccumulator + anisotropicTerm, anisotropicTerm, sector == 0u);'
          ]
        : [])
    );
  }
  if (props.angleSum) {
    statements.push(
      'angleAccumulator = select(angleAccumulator + horizonAngle, horizonAngle, sector == 0u);'
    );
  }
  const loads = [
    props.sineSum
      ? 'var sineAccumulator = select(0.0, sineSum[sineSumOffset + pixel], firstSector > 0u);'
      : '',
    props.angleSum
      ? 'var angleAccumulator = select(0.0, angleSum[angleSumOffset + pixel], firstSector > 0u);'
      : '',
    props.anisotropicSum
      ? 'var anisotropicAccumulator = select(0.0, anisotropicSum[anisotropicSumOffset + pixel], firstSector > 0u);'
      : '',
    props.nadirSum
      ? 'var nadirAccumulator = select(0.0, nadirSum[nadirSumOffset + pixel], firstSector > 0u);'
      : ''
  ].join('\n  ');
  const stores = [
    props.sineSum ? 'sineSum[sineSumOffset + pixel] = sineAccumulator;' : '',
    props.angleSum ? 'angleSum[angleSumOffset + pixel] = angleAccumulator;' : '',
    props.anisotropicSum
      ? 'anisotropicSum[anisotropicSumOffset + pixel] = anisotropicAccumulator;'
      : '',
    props.nadirSum ? 'nadirSum[nadirSumOffset + pixel] = nadirAccumulator;' : ''
  ].join('\n  ');
  const ranges = createHorizonSectorRanges<Parameters>(graph, {
    id: props.id,
    directionCount,
    sectorsPerRange: sectorsPerChunk
  });
  const nodes: GPUCommandNode<Parameters>[] = [ranges.writeNode];
  for (let chunk = 0; chunk < ranges.rangeCount; chunk++) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-horizon-march-${chunk}`,
        operation: 'GPUTerrainHorizon',
        variant: `march-fused-${props.cellSizeMode}${nadir ? '-nadir' : ''}`,
        bindings: [
          ...bindings,
          {name: 'sectorRange', view: ranges.getRange(chunk), type: 'u32', access: 'read'}
        ],
        invocationCount: pixelCount,
        declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const DIRECTION_COUNT: u32 = ${directionCount}u;
const DIRECTIONS: array<vec2<f32>, ${directionCount}> = array<vec2<f32>, ${directionCount}>(${directions});
const STEP_COUNT: u32 = ${props.stepCount}u;
const STEP_DISTANCES: array<f32, ${props.stepCount}> = array<f32, ${props.stepCount}>(${props.distances});
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${zenithOutput?.declarations ?? ''}
${getTerrainHorizonGroundCellSizeWGSL(props.cellSizeMode, props.sourceGridIndex)}
${HORIZON_MARCH_SAMPLE_WGSL}
${RASTER_PIXEL_DIRECTION_WGSL}`,
        body: `let pixel = index;
  let firstSector = sectorRange[sectorRangeOffset];
  let endSector = sectorRange[sectorRangeOffset + 1u];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let centerValid = isValidPixel(column, row);
  let centerElevation = select(0.0, getElevation(column, row), centerValid);
  let limit = vec2<i32>(i32(WIDTH - 1u), i32(HEIGHT - 1u));
  let center = vec2<i32>(i32(column), i32(row));
  ${loads}
  for (var sector = firstSector; sector < endSector; sector++) {
    // DIRECTIONS are ground azimuths; the pixel step follows them on anisotropic cells. The ray
    // keeps this row's cell size (see GPUTerrainHorizonAlgorithm for the along-ray error).
    let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
    let groundStep = length(direction * groundCell);
    var horizonAngle = getQuietNaN(index);
    var nadirAngle = getQuietNaN(index);
    if (centerValid && groundStep > 0.0 && isFiniteValue(groundStep)) {
      var hasSample = false;
      var maximumTangent = 0.0;
      var maximumNadirTangent = 0.0;
      for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
        let stepDistance = STEP_DISTANCES[stepIndex];
        let displacement = direction * stepDistance;
        let integerDisplacement = floor(displacement);
        let fraction = displacement - integerDisplacement;
        let base = center + vec2<i32>(integerDisplacement);
        if (any(base < vec2<i32>(0)) || any(base > limit) ||
            (base.x == limit.x && fraction.x > 0.0) ||
            (base.y == limit.y && fraction.y > 0.0)) { break; }
        let groundDistance = stepDistance * groundStep;
        if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
        let elevationSample = sampleElevation(vec2<u32>(base), fraction, centerElevation);
        if (elevationSample.y == 0.0) { continue; }
        let rise = zFactor * elevationSample.x -
          curvature * groundDistance * groundDistance;
        let tangent = rise / groundDistance;
        if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
        ${
          nadir
            ? `// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
        let nadirRise = -zFactor * elevationSample.x -
          curvature * groundDistance * groundDistance;
        let nadirTangent = nadirRise / groundDistance;
        if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`
            : ''
        }
        hasSample = true;
      }
      horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
      ${nadir ? 'nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);' : ''}
    }
    ${statements.join('\n    ')}
    ${nadir ? 'nadirAccumulator = select(nadirAccumulator + nadirAngle, nadirAngle, sector == 0u);' : ''}
  }
  ${stores}`
      })
    );
  }
  return nodes;
}

/**
 * Builds the ray-march kernel for one horizon sector. The sector is read from `sectorRange`, so
 * every sector's kernel has the same WGSL (library rule 12).
 */
function getHorizonNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    /** Range word `[sector, sector + 1]`, see {@link createHorizonSectorRanges}. */
    sectorRange: GraphDataView<'uint32'>;
    directionCount: number;
    rowDirection: GPURasterRowDirection;
    distances: string;
    stepCount: number;
    cellSizeMode: GPURasterCellSizeMode;
    sourceGridIndex?: number;
    elevationValues: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    horizonFormat: GPUTerrainHorizonFormat;
    horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    sineSum?: GraphDataView<'float32'>;
    angleSum?: GraphDataView<'float32'>;
    anisotropicSum?: GraphDataView<'float32'>;
    nadirSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'elevationValues',
      view: props.elevationValues,
      type: 'f32',
      access: 'read'
    },
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'sectorRange', view: props.sectorRange, type: 'u32', access: 'read'}
  ];
  const zenithOutput =
    props.horizon || props.sineSum || props.angleSum || props.anisotropicSum
      ? getTerrainHorizonSectorOutput({
          sector: 0,
          sectorWGSL: 'sector',
          directionCount: props.directionCount,
          horizonFormat: props.horizonFormat,
          horizon: props.horizon,
          sineSum: props.sineSum,
          angleSum: props.angleSum,
          anisotropicSum: props.anisotropicSum,
          mode: 'zenith',
          reservedBindingCount: 3 + (props.nadirSum ? 1 : 0)
        })
      : undefined;
  const nadirOutput = props.nadirSum
    ? getTerrainHorizonSectorOutput({
        sector: 0,
        sectorWGSL: 'sector',
        directionCount: props.directionCount,
        nadirSum: props.nadirSum,
        mode: 'nadir',
        reservedBindingCount: 3 + (zenithOutput?.bindings.length ?? 0)
      })
    : undefined;
  bindings.push(...(zenithOutput?.bindings ?? []), ...(nadirOutput?.bindings ?? []));
  const nadir = Boolean(nadirOutput);
  const directions = Array.from({length: props.directionCount}, (_, sector) => {
    const [x, y] = getGPUTerrainHorizonDirection(sector, props.directionCount, props.rowDirection);
    return `vec2<f32>(${getWGSLFloatLiteral(x)}, ${getWGSLFloatLiteral(y)})`;
  }).join(', ');
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainHorizon',
    variant: `march-${props.cellSizeMode}${nadir ? '-nadir' : ''}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const DIRECTION_COUNT: u32 = ${props.directionCount}u;
const DIRECTIONS: array<vec2<f32>, ${props.directionCount}> = array<vec2<f32>, ${props.directionCount}>(${directions});
const STEP_COUNT: u32 = ${props.stepCount}u;
const STEP_DISTANCES: array<f32, ${props.stepCount}> = array<f32, ${props.stepCount}>(${props.distances});
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${zenithOutput?.declarations ?? ''}
${getTerrainHorizonGroundCellSizeWGSL(props.cellSizeMode, props.sourceGridIndex)}
${HORIZON_MARCH_SAMPLE_WGSL}
${RASTER_PIXEL_DIRECTION_WGSL}`,
    body: `let pixel = index;
  let sector = sectorRange[sectorRangeOffset];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
  let groundStep = length(direction * groundCell);
  var horizonAngle = getQuietNaN(index);
  var nadirAngle = getQuietNaN(index);
  if (isValidPixel(column, row) && groundStep > 0.0 && isFiniteValue(groundStep)) {
    let center = vec2<i32>(i32(column), i32(row));
    let centerElevation = getElevation(column, row);
    let limit = vec2<i32>(i32(WIDTH - 1u), i32(HEIGHT - 1u));
    var hasSample = false;
    var maximumTangent = 0.0;
    var maximumNadirTangent = 0.0;
    for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
      let stepDistance = STEP_DISTANCES[stepIndex];
      let displacement = direction * stepDistance;
      let integerDisplacement = floor(displacement);
      let fraction = displacement - integerDisplacement;
      let base = center + vec2<i32>(integerDisplacement);
      if (any(base < vec2<i32>(0)) || any(base > limit) ||
          (base.x == limit.x && fraction.x > 0.0) ||
          (base.y == limit.y && fraction.y > 0.0)) { break; }
      let groundDistance = stepDistance * groundStep;
      if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
      let elevationSample = sampleElevation(vec2<u32>(base), fraction, centerElevation);
      if (elevationSample.y == 0.0) { continue; }
      let rise = zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let tangent = rise / groundDistance;
      if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
      ${
        nadir
          ? `// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
      let nadirRise = -zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let nadirTangent = nadirRise / groundDistance;
      if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`
          : ''
      }
      hasSample = true;
    }
    horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
    ${nadir ? 'nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);' : ''}
  }
  ${zenithOutput?.wgsl ?? ''}
  ${nadirOutput ? `{\n  let horizonAngle = nadirAngle;\n  ${nadirOutput.wgsl}\n  }` : ''}`
  });
}
