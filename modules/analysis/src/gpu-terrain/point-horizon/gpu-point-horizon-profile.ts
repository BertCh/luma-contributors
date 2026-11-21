// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainOutputSelection} from '../terrain-illumination/terrain-output-selection';

import {
  validateTerrainBindings,
  getTerrainPyramidParameterScratch
} from '../terrain-processing/terrain-binding-preflight';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCommandNodeBudget,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterBandViews,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterParameters
} from '../../raster-analysis/raster-band-utils';
import {
  createPointHorizonChunkStarts,
  createPointHorizonSource,
  getPointHorizonMarchWGSL,
  PRECISION_WGSL,
  resolvePointHorizonModel,
  validatePointHorizonRows,
  validatePointHorizonVerticalDatum,
  type GPUPointHorizonModelOptions,
  type ResolvedPointHorizonModel
} from './point-horizon-march';
import {
  GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH,
  POINT_HORIZON_PROFILE_PROJECTION_SLOT
} from './point-horizon-parameters';

/** Default number of rays per compute node of {@link GPUPointHorizonProfile}. */
export const GPU_POINT_HORIZON_RAYS_PER_DISPATCH = 65536;

/**
 * Command nodes besides the per-chunk march and the pyramid levels: elevation canonicalization
 * (up to two), the chunk start table and the derived skyline angle pass.
 */
const POINT_HORIZON_PROFILE_FIXED_NODE_ALLOWANCE = 4;

/**
 * Pyramid command nodes besides one per level: the fused level parameter upload (up to two).
 */
const POINT_HORIZON_PYRAMID_NODE_ALLOWANCE = 2;

/**
 * Properties for {@link GPUPointHorizonProfile}.
 *
 * Topology (baked into WGSL, see {@link GPUPointHorizonModelOptions}): grid size, projection,
 * row direction, height reference, traversal, azimuth division and the distance lattice. Per-frame:
 * `parameters` (curvature, distance cap, and the grid cell size or world size), `observers` and
 * the terrain contents.
 *
 * Projection and cell-size model: `'planar'` rasters are in projected meters with the cell size
 * of the grid; `'web-mercator'` rasters are windows of EPSG:3857 meters whose world size and top
 * edge the packer derives from the grid. Rays are great circles traced on a sphere of radius
 * 6371008.8 m; azimuths and distances are ground bearings and ground meters in the WGS84 metric
 * at the observer latitude, the row scale every raster-grid terrain contributor uses.
 * See {@link GPUPointHorizonProjection} and the settings type in `point-horizon-parameters`.
 *
 * Curvature convention: terrain drops by `c * d^2` with `c = (1 - k) / (2R)` (see
 * `getGPUTerrainCurvatureCoefficient`; mt-image default `k = 0.13`, GDAL `-cc 0.85714` is `k = 1/7`).
 */
export type GPUPointHorizonProfileProps = GPUPointHorizonModelOptions & {
  /** Prefix for node and transient IDs. Defaults to `'point-horizon-profile'`. */
  id?: string;
  /** Raster grid: its width and height (at least 2) size the kernels. */
  grid: GPURasterMetadata;
  /** Terrain band (elevation), buffer or texture. Calibration, no-data and validity are honored. */
  elevation: GPURasterBand;
  /**
   * Observer rows `[column, row, height, 0]` in pixel-center index space (fractional allowed).
   * With `heightReference: 'ground'` (default) the eye is the bilinear ground plus `height`, with
   * `'absolute'` it is `height`. An observer outside the grid, or on an invalid ground sample in
   * ground mode, yields NaN in every output.
   */
  observers: GraphDataView<'float32x4'>;
  /** Per-frame parameters, see `getGPUPointHorizonProfileParameterValues`. At least 8 float32. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs; at least one is required. */
  output: GPUPointHorizonProfileOutput;
  /** Rays per compute node. Defaults to 65536. */
  raysPerDispatch?: number;
};

/** Caller-owned outputs of {@link GPUPointHorizonProfile}, one value per ray. */
export type GPUPointHorizonProfileOutput = TerrainOutputSelection<
  {
    /**
     * Optional `observers.length * azimuthBinCount` tangents `tBest = max((h - h_eye) / d - c d)`;
     * `-3.4028234663852886e38` when no valid sample was seen, NaN for an invalid observer.
     */
    tangent?: GraphDataView<'float32'>;
    /** Optional skyline elevation angles in degrees, `atan(tBest)`; -90 when no sample, NaN invalid. */
    skylineAngle?: GraphDataView<'float32'>;
    /** Optional distance of the skyline sample in meters; 0 when no sample, NaN invalid. */
    distance?: GraphDataView<'float32'>;
    /**
     * Optional debug counter: the number of lattice samples whose bilinear height was evaluated per
     * ray (`'march'` evaluates every sample up to the first out-of-grid one, `'pyramid'` omits the
     * skipped ones). All four outputs are supported: when tangent and skylineAngle are both
     * selected, a separate per-ray pass derives the angle to keep the march within eight bindings.
     */
    samples?: GraphDataView<'uint32'>;
  },
  'tangent' | 'skylineAngle' | 'distance'
>;

/**
 * Computes the skyline (horizon) profile seen from many observers over one terrain window.
 *
 * Each ray marches an exact power-of-two distance lattice (see
 * {@link GPUPointHorizonDistanceLatticeOptions}) with bilinear samples and keeps the maximum
 * apparent tangent. The `'pyramid'` traversal (default) skips samples a min-max mip pyramid proves
 * cannot raise the maximum, and is bit-identical to the exhaustive `'march'` traversal (Tevs, Ihrke
 * and Seidel 2008 "Maximum mipmaps"; Dick et al. 2009; the exact-skip proof is in
 * `point-horizon-march.ts`).
 * The first sample is the first lattice distance at or above `minimumDistance`; the last is
 * the last at or below `maximumDistance`. Neither endpoint is inserted if it is off lattice.
 * GRASS `r.horizon` samples a different, cell-based path: even on cardinal rays its maximum
 * angle and winning distance can differ. For matched cardinal cell-centre samples on a square
 * grid with power-of-two cell size (at least one metre), set `minimumStep` to that size,
 * `cellSteps` to 1, `stepFactor` to `cellSize / maximumDistance`, and the distance limits to
 * cell multiples. General fractional cell sizes require
 * a reference evaluated on this contributor's lattice. Negative horizons are retained here;
 * GRASS clips them to zero. Exact tangent ties keep the nearest sample.
 *
 * Precision ladder (from mt-image's GPU horizon, f32 on the GPU against f64 on the CPU): the eye is
 * an integer pixel plus an f32 fraction so sample positions keep sub-milli-pixel precision near the
 * eye; terrain heights are interpolated relative to the eye as an f32 hi/lo pair guarded by an
 * opaque zero so Metal fast math cannot re-associate them; Web Mercator breakpoints are offsets
 * from cancellation-free difference terms (`sin(phi2) - sin(phi1)`, `1 - cos(D)`) with small-angle
 * series for `atan` and `atanh`; azimuth sine and cosine use exact integer octant reduction; the
 * final `atan(tBest)` uses a reduced-argument series. No builtin `sin`, `cos` or `atan` is used
 * where accuracy matters (their WGSL accuracy is 2^-11 absolute / about 4096 ULP).
 *
 * Output index: `observer * azimuthBinCount + a`, azimuth `(firstAzimuthIndex + a) * 360 / azimuthCount`
 * degrees clockwise from grid north.
 * One compute node per chunk of at most `raysPerDispatch` rays. Construction allows at most
 * 32,768 estimated command nodes per contributor in total, including locally built preparation,
 * pyramid, chunk-table and derived-angle work. This total is distinct from rays per dispatch;
 * increase the chunk size or partition observers to stay within it.
 */
export class GPUPointHorizonProfile implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointHorizonProfileProps;
  /** Resolved model: lattice, segments and pyramid layout. @internal */
  private readonly model: ResolvedPointHorizonModel;
  /** Validated chunk size captured before caller-owned props can change. */
  private readonly raysPerDispatch: number;
  /**
   * Upper bound on the command nodes `getCommandNodes` returns: one march node per chunk of
   * `raysPerDispatch` rays plus canonicalization, pyramid, chunk-table and skyline-angle nodes.
   * Construction rejects more than `GPU_MAXIMUM_COMMAND_NODE_COUNT`.
   */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUPointHorizonProfileProps) {
    this.id = props.id ?? 'point-horizon-profile';
    this.props = props;
    const {id} = this;
    this.model = resolvePointHorizonModel(id, props.grid, props);
    validatePointHorizonVerticalDatum(id, props, props.elevation, this.model.heightReference);
    const {output} = props;
    if (!output.tangent && !output.skylineAngle && !output.distance) {
      throw new Error(`${id} requires at least one output`);
    }
    validateRasterParameters(id, props.parameters, GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH);
    validatePointHorizonRows(id, 'observers', props.observers, 'float32x4');
    if (props.observers.length < 1) {
      throw new Error(`${id} requires at least one observer`);
    }
    const rayCount = props.observers.length * this.model.azimuthBinCount;
    for (const [name, view] of [
      ['tangent', output.tangent],
      ['skylineAngle', output.skylineAngle],
      ['distance', output.distance]
    ] as const) {
      if (view) {
        validatePointHorizonRows(id, `output.${name}`, view, 'float32');
        if (view.length !== rayCount) {
          throw new Error(`${id} output.${name} must contain observers * azimuthBinCount values`);
        }
      }
    }
    const raysPerDispatch = props.raysPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    if (!Number.isInteger(raysPerDispatch) || raysPerDispatch < 1) {
      throw new Error(`${id} raysPerDispatch must be a positive integer`);
    }
    this.raysPerDispatch = raysPerDispatch;
    // Same chunking as `createPointHorizonChunkStarts`.
    const chunkSize = Math.min(raysPerDispatch, Math.max(rayCount, 1));
    const chunkCount = Math.max(Math.ceil(rayCount / chunkSize), 1);
    const builtPyramid = this.model.layout && !this.model.sharedPyramid ? this.model.layout : null;
    this.estimatedCommandNodeCount =
      chunkCount +
      POINT_HORIZON_PROFILE_FIXED_NODE_ALLOWANCE +
      (builtPyramid ? builtPyramid.levels.length + POINT_HORIZON_PYRAMID_NODE_ALLOWANCE : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterBuffersDistinct(
      id,
      [output.tangent, output.skylineAngle, output.distance, output.samples],
      [...getRasterBandViews(props.elevation), props.parameters, props.observers]
    );
  }

  /** Returns canonical elevation, pyramid level and per-chunk march nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTerrainBindings(graph, this.id, this.props, {
      'canonical elevation': 4 * this.props.grid.width * this.props.grid.height,
      'pyramid fused parameters': getTerrainPyramidParameterScratch(
        this.props.pyramid ? undefined : this.model.layout
      ),
      pyramid: this.model.layout && !this.props.pyramid ? 8 * this.model.layout.length : 0,
      'chunk table':
        256 *
        Math.max(
          1,
          Math.ceil(
            (this.props.observers.length * this.model.azimuthBinCount) / this.raysPerDispatch
          )
        )
    });
    const {id, props, model} = this;
    const {output} = props;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.pyramid?.combined,
      props.parameters,
      props.observers,
      output.tangent,
      output.skylineAngle,
      output.distance,
      output.samples
    ]);
    const source = createPointHorizonSource(graph, id, model, props.elevation);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const bindings: WGSLKernelBinding[] = [
      ...source.bindings,
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'observers', view: props.observers, type: 'f32', access: 'read'}
    ];
    if (output.tangent) {
      bindings.push({
        name: 'tangentOutput',
        view: output.tangent,
        type: 'f32',
        access: 'read_write'
      });
    }
    // With both angle outputs the march writes only the tangent and a per-ray pass derives the
    // skyline angle from it with the same pinned `atanAccurate` (bit-identical), so the march stays
    // within 8 storage buffers with every output on the pyramid traversal (core limit).
    const deriveSkylineAngle = Boolean(output.tangent && output.skylineAngle);
    if (output.skylineAngle && !deriveSkylineAngle) {
      bindings.push({
        name: 'skylineAngleOutput',
        view: output.skylineAngle,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.distance) {
      bindings.push({
        name: 'distanceOutput',
        view: output.distance,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.samples) {
      bindings.push({
        name: 'samplesOutput',
        view: output.samples,
        type: 'u32',
        access: 'read_write'
      });
    }
    // The chunk start word is the eighth binding at most.
    if (bindings.length > 7) {
      throw new Error(`${id} needs more than 8 storage bindings`);
    }
    const march = getPointHorizonMarchWGSL(model, POINT_HORIZON_PROFILE_PROJECTION_SLOT);
    const rayCount = props.observers.length * model.azimuthBinCount;
    const raysPerDispatch = this.raysPerDispatch;
    const writes = (value: string) =>
      [
        output.tangent
          ? `tangentOutput[tangentOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(-BIG, result.t, result.has)'};`
          : '',
        output.skylineAngle && !deriveSkylineAngle
          ? `skylineAngleOutput[skylineAngleOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(-90.0, atanAccurate(result.t) * DEGREES, result.has)'};`
          : '',
        output.distance
          ? `distanceOutput[distanceOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(0.0, result.d, result.has)'};`
          : '',
        output.samples
          ? `samplesOutput[samplesOutputOffset + ray] = ${value === 'nan' ? '0u' : 'result.evaluated'};`
          : ''
      ].join('\n  ');
    const chunks = createPointHorizonChunkStarts<Parameters>(graph, {
      id,
      operation: 'GPUPointHorizonProfile',
      itemCount: rayCount,
      maximumChunkSize: raysPerDispatch
    });
    nodes.push(chunks.writeNode);
    for (let chunk = 0; chunk < chunks.chunkCount; chunk++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-march-${chunk}`,
          operation: 'GPUPointHorizonProfile',
          variant: model.traversal,
          bindings: [...bindings, chunks.getChunkStart(chunk)],
          invocationCount: chunks.chunkSize,
          workgroupSize: 64,
          declarations: `const RAY_COUNT: u32 = ${rayCount}u;
${march}`,
          body: `let ray = chunkStart[chunkStartOffset] + index;
  if (ray >= RAY_COUNT) { return; }
  let observerIndex = ray / AZIMUTH_SPAN;
  let azimuthOffset = ray - observerIndex * AZIMUTH_SPAN;
  let observerBase = observersOffset + observerIndex * 4u;
  let eye = makeEye(observers[observerBase], observers[observerBase + 1u], observers[observerBase + 2u]);
  if (!eye.valid) {
    ${writes('nan')}
    return;
  }
  let bearing = azimuthSinCos(FIRST_AZIMUTH + azimuthOffset, AZIMUTH_COUNT);
  ${
    model.projection === 'web-mercator'
      ? `// Azimuths are ground bearings on the WGS84 ellipsoid; the ray is traced on the sphere.
  let direction = getSphereAzimuth(eye, bearing.x, bearing.y);`
      : 'let direction = bearing;'
  }
  let result = marchRay(eye, direction.x, direction.y, 0.0, false);
  ${writes('result')}`
        })
      );
    }
    if (output.tangent && output.skylineAngle && deriveSkylineAngle) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-skyline-angle`,
          operation: 'GPUPointHorizonProfile',
          variant: 'skyline-angle',
          bindings: [
            {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'tangent', view: output.tangent, type: 'f32', access: 'read'},
            {
              name: 'skylineAngleOutput',
              view: output.skylineAngle,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: rayCount,
          declarations: `const RAY_COUNT: u32 = ${rayCount}u;
${PRECISION_WGSL}`,
          // The march wrote nanValue() (invalid eye, copied through), -BIG (no horizon) or the tangent.
          body: `if (index >= RAY_COUNT) { return; }
  let t = tangent[tangentOffset + index];
  var angle = t;
  if ((bitcast<u32>(t) & 0x7fffffffu) <= 0x7f800000u) {
    angle = select(-90.0, atanAccurate(t) * DEGREES, t != -BIG);
  }
  skylineAngleOutput[skylineAngleOutputOffset + index] = angle;`
        })
      );
    }
    return nodes;
  }
}
