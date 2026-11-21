// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTerrainBindings,
  getTerrainPyramidParameterScratch
} from '../terrain-processing/terrain-binding-preflight';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCommandNodeBudget,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_TERRAIN_VISIBILITY} from '../terrain-analysis/gpu-terrain-viewshed';
import {
  getRasterBandViews,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterParameters
} from '../../raster-analysis/raster-band-utils';
import {GPU_POINT_HORIZON_RAYS_PER_DISPATCH} from './gpu-point-horizon-profile';
import {
  createPointHorizonChunkStarts,
  createPointHorizonSource,
  getPointHorizonMarchWGSL,
  resolvePointHorizonModel,
  validatePointHorizonRows,
  validatePointHorizonVerticalDatum,
  type GPUPointHorizonModelOptions,
  type ResolvedPointHorizonModel
} from './point-horizon-march';
import {
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  POINT_HORIZON_VISIBILITY_PROJECTION_SLOT
} from './point-horizon-parameters';

/**
 * Properties for {@link GPUPointHorizonVisibility}.
 *
 * Model options (projection, cell-size model, curvature convention, lattice) are those of
 * {@link GPUPointHorizonModelOptions}; `azimuthCount`, `firstAzimuthIndex` and `azimuthBinCount`
 * are ignored because the ray direction is the exact bearing to each target.
 */
export type GPUPointHorizonVisibilityProps = GPUPointHorizonModelOptions & {
  /** Prefix for node and transient IDs. Defaults to `'point-horizon-visibility'`. */
  id?: string;
  /** Raster grid: its width and height (at least 2) size the kernels. */
  grid: GPURasterMetadata;
  /** Terrain band (elevation). */
  elevation: GPURasterBand;
  /** Observer rows `[column, row, height, 0]`, see `GPUPointHorizonProfileProps.observers`. */
  observers: GraphDataView<'float32x4'>;
  /**
   * Target rows `[column, row, height, observerIndex]` in pixel-center index space. The target
   * elevation is the bilinear ground plus `height`; `observerIndex` is the integer-valued index of
   * the observer row it is tested from. An invalid index, an observer or target outside the grid,
   * or an invalid ground sample yields `noData`.
   */
  targets: GraphDataView<'float32x4'>;
  /** Per-frame parameters, see `getGPUPointHorizonVisibilityParameterValues`. At least 12 float32. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUPointHorizonVisibilityOutput;
  /** Targets per compute node. Defaults to 65536. */
  targetsPerDispatch?: number;
};

/** Caller-owned outputs of {@link GPUPointHorizonVisibility}, one row per target. */
export type GPUPointHorizonVisibilityOutput = {
  /** One `GPU_TERRAIN_VISIBILITY` code per target (hidden, visible, outOfRange, noData, marginal). */
  visibility: GraphDataView<'uint32'>;
  /**
   * Optional rows `[targetElevationDegrees, occluderElevationDegrees, skylineElevationDegrees,
   * onSkyline ? 1 : 0]`; the occluder and skyline are -90 when no sample was seen and the row is NaN
   * for `noData`, `outOfRange` and coincident points.
   */
  details?: GraphDataView<'float32x4'>;
};

/**
 * Classifies targets (peaks, towers, points of interest) as hidden, marginal or visible from
 * observers, and flags those on the skyline, following mt-image's `classifyPeak`.
 *
 * Per target the contributor computes the distance and bearing from the observer without cancellation
 * (planar: metres offsets; Web Mercator: the spherical arc scaled to WGS84 ground metres at the
 * observer latitude, `sqrt((N sinA)^2 + (M cosA)^2) / R` per sphere metre, the row scale of the
 * raster-grid contributors, from a spherical inverse written in haversine form with
 * `Δm = -2πΔrow/worldPixelSize` (derived from the grid by the packer) and `Δλ = 2πΔcolumn/worldPixelSize`, series for small `sinh`,
 * `sin` and `asin`), marches the full ray along that bearing with the same march as
 * `GPUPointHorizonProfile`, and records the running maximum tangent `tQ` before
 * `q = dP - max(targetIgnoreDistance, targetIgnoreFraction * dP)` and the skyline `tSky`. With
 * `αP = atan(tP)`, `αOcc = atan(tQ)`, `αSky = atan(tSky)` (accurate series, degrees) and
 * `tol = toleranceDegrees + (sigmaZ / dP) * 180 / π`:
 * `hidden` if `αP < αOcc - tol`; `marginal` if `αOcc - tol <= αP < αOcc + tol`; otherwise
 * `visible`. The three classes are ordered with no gap: a target exactly `tol` below the occluder
 * is `marginal`, as in the tolerance band of `GPUTerrainLineOfSight`. `onSkyline`
 * is `(visible or marginal) and αP >= αSky - skylineToleranceDegrees`.
 *
 * Construction limits the total estimated command nodes per contributor to 32,768, including
 * at least one target chunk, its chunk-table writer, elevation preparation and any locally
 * built pyramid. Increase `targetsPerDispatch` or partition targets when the budget is exceeded.
 */
export class GPUPointHorizonVisibility implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointHorizonVisibilityProps;
  /** Resolved model: lattice, segments and pyramid layout. @internal */
  private readonly model: ResolvedPointHorizonModel;
  /** Validated chunk size captured before caller-owned props can change. */
  private readonly targetsPerDispatch: number;

  /** Upper bound on all command nodes returned by getCommandNodes; maximum 32,768. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUPointHorizonVisibilityProps) {
    this.id = props.id ?? 'point-horizon-visibility';
    this.props = props;
    const {id} = this;
    const {output} = props;
    this.model = resolvePointHorizonModel(id, props.grid, props, {});
    validatePointHorizonVerticalDatum(id, props, props.elevation, this.model.heightReference);
    validateRasterParameters(id, props.parameters, GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH);
    validatePointHorizonRows(id, 'observers', props.observers, 'float32x4');
    validatePointHorizonRows(id, 'targets', props.targets, 'float32x4');
    if (props.observers.length < 1) {
      throw new Error(`${id} requires at least one observer`);
    }
    validateSinglePackedView(output.visibility, ['uint32'], `${id} output.visibility`);
    if (output.visibility.length !== props.targets.length) {
      throw new Error(`${id} output.visibility must contain one value per target`);
    }
    if (output.details) {
      validatePointHorizonRows(id, 'output.details', output.details, 'float32x4');
      if (output.details.length !== props.targets.length) {
        throw new Error(`${id} output.details must contain one row per target`);
      }
    }
    const targetsPerDispatch = props.targetsPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    if (!Number.isInteger(targetsPerDispatch) || targetsPerDispatch < 1) {
      throw new Error(`${id} targetsPerDispatch must be a positive integer`);
    }
    this.targetsPerDispatch = targetsPerDispatch;
    // Match createPointHorizonChunkStarts, including its minimum-one-chunk rule.
    const chunkSize = Math.min(targetsPerDispatch, Math.max(props.targets.length, 1));
    const chunkCount = Math.max(Math.ceil(props.targets.length / chunkSize), 1);
    const builtPyramid = this.model.layout && !this.model.sharedPyramid ? this.model.layout : null;
    // Up to two elevation preparation nodes and one chunk-table writer. A built pyramid
    // contributes at most one node per level plus up to two parameter preparation nodes.
    this.estimatedCommandNodeCount =
      chunkCount + 3 + (builtPyramid ? builtPyramid.levels.length + 2 : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateRasterBuffersDistinct(
      id,
      [output.visibility, output.details],
      [...getRasterBandViews(props.elevation), props.parameters, props.observers, props.targets]
    );
  }

  /** Returns canonical elevation, pyramid level and per-chunk classification nodes. */
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
        256 * Math.max(1, Math.ceil(this.props.targets.length / this.targetsPerDispatch))
    });
    const {id, props, model} = this;
    const {output} = props;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.pyramid?.combined,
      props.parameters,
      props.observers,
      props.targets,
      output.visibility,
      output.details
    ]);
    const source = createPointHorizonSource(graph, id, model, props.elevation);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const bindings: WGSLKernelBinding[] = [
      ...source.bindings,
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'observers', view: props.observers, type: 'f32', access: 'read'},
      {name: 'targets', view: props.targets, type: 'f32', access: 'read'},
      {name: 'visibilityOutput', view: output.visibility, type: 'u32', access: 'read_write'}
    ];
    if (output.details) {
      bindings.push({
        name: 'detailsOutput',
        view: output.details,
        type: 'f32',
        access: 'read_write'
      });
    }
    // The chunk start word is the eighth binding at most.
    if (bindings.length > 7) {
      throw new Error(`${id} needs more than 8 storage bindings`);
    }
    const mercator = model.projection === 'web-mercator';
    const geometry = mercator
      ? `let worldPixelSize = settings[settingsOffset + 1u];
  let halfLambda = PI * deltaColumn / worldPixelSize;
  let halfM = -PI * deltaRow / worldPixelSize;
  let m2 = eye.m + 2.0 * halfM;
  let meanM = eye.m + halfM;
  let cosP2 = 1.0 / coshS(m2);
  let sinHalfLambda = sinAny(halfLambda);
  let cosHalfLambda = cosAny(halfLambda);
  let sinhHalfM = sinhS(halfM);
  let haversine = eye.cosP * cosP2 * (sinhHalfM * sinhHalfM + sinHalfLambda * sinHalfLambda);
  let sphereDistance = 2.0 * EARTH_RADIUS * asinS(sqrt(min(haversine, 1.0)));
  let sinDeltaPhi = 2.0 * coshS(meanM) * sinhHalfM * eye.cosP * cosP2;
  let north = sinDeltaPhi + eye.sinP * cosP2 * 2.0 * sinHalfLambda * sinHalfLambda;
  let east = 2.0 * sinHalfLambda * cosHalfLambda * cosP2;
  let rho = sqrt(north * north + east * east);
  let sinA = east / max(rho, 1.0e-30);
  let cosA = north / max(rho, 1.0e-30);
  // Ground meters per sphere meter along this ray: the WGS84 metric at the eye, the row scale
  // of the raster-grid contributors (see POINT_HORIZON_GROUND_SCALE_WGSL).
  let groundScale = getRayGroundScale(eye, sinA, cosA);
  let distance = sphereDistance * groundScale;
  // One Web Mercator pixel along the ray at the target's latitude, in ground meters.
  let targetPixelLength = 2.0 * PI * EARTH_RADIUS * cosP2 / worldPixelSize * groundScale;`
      : `let deltaX = deltaColumn * settings[settingsOffset + 1u];
  let deltaY = deltaRow * settings[settingsOffset + 2u];
  let distance = sqrt(deltaX * deltaX + deltaY * deltaY);
  let sinA = deltaX / max(distance, 1.0e-30);
  let cosA = ${model.rowDirection === 'south' ? '-' : ''}deltaY / max(distance, 1.0e-30);
  // Ground length of one pixel step along the ray (the march's pixels-per-meter is (sinA / cellX, cosA / cellY)).
  let targetPixelLength = 1.0 / length(vec2<f32>(sinA / settings[settingsOffset + 1u], cosA / settings[settingsOffset + 2u]));`;
    const march = getPointHorizonMarchWGSL(model, POINT_HORIZON_VISIBILITY_PROJECTION_SLOT);
    const {targets} = props;
    const chunks = createPointHorizonChunkStarts<Parameters>(graph, {
      id,
      operation: 'GPUPointHorizonVisibility',
      itemCount: targets.length,
      maximumChunkSize: this.targetsPerDispatch
    });
    nodes.push(chunks.writeNode);
    const detailsWrite = output.details
      ? `let detailsBase = detailsOutputOffset + (chunkStart[chunkStartOffset] + index) * 4u;
  detailsOutput[detailsBase] = result.details.x;
  detailsOutput[detailsBase + 1u] = result.details.y;
  detailsOutput[detailsBase + 2u] = result.details.z;
  detailsOutput[detailsBase + 3u] = result.details.w;`
      : '';
    for (let chunk = 0; chunk < chunks.chunkCount; chunk++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify-${chunk}`,
          operation: 'GPUPointHorizonVisibility',
          variant: model.traversal,
          bindings: [...bindings, chunks.getChunkStart(chunk)],
          invocationCount: chunks.chunkSize,
          workgroupSize: 64,
          declarations: `const TARGET_COUNT: u32 = ${targets.length}u;
const OBSERVER_COUNT: u32 = ${props.observers.length}u;
const HIDDEN: u32 = ${GPU_TERRAIN_VISIBILITY.hidden}u;
const VISIBLE: u32 = ${GPU_TERRAIN_VISIBILITY.visible}u;
const OUT_OF_RANGE: u32 = ${GPU_TERRAIN_VISIBILITY.outOfRange}u;
const NO_DATA: u32 = ${GPU_TERRAIN_VISIBILITY.noData}u;
const MARGINAL: u32 = ${GPU_TERRAIN_VISIBILITY.marginal}u;
${march}
fn sinAny(x: f32) -> f32 { if (abs(x) > 1.6) { return sin(x); } return sinSeries(x); }
fn cosAny(x: f32) -> f32 { if (abs(x) > 1.6) { return cos(x); } return cosSeries(x); }

struct Classification { code: u32, details: vec4<f32> };

fn classifyTarget(index: u32) -> Classification {
  let notANumber = vec4<f32>(nanValue());
  let base = targetsOffset + (chunkStart[chunkStartOffset] + index) * 4u;
  let targetColumn = targets[base];
  let targetRow = targets[base + 1u];
  let targetHeight = targets[base + 2u];
  let observerValue = targets[base + 3u];
  if (!(observerValue >= 0.0 && observerValue < f32(OBSERVER_COUNT))) {
    return Classification(NO_DATA, notANumber);
  }
  let observerBase = observersOffset + u32(observerValue) * 4u;
  let observerColumn = observers[observerBase];
  let observerRow = observers[observerBase + 1u];
  let eye = makeEye(observerColumn, observerRow, observers[observerBase + 2u]);
  if (!eye.valid || !isInsideGrid(targetColumn, targetRow) || !isFiniteValue(targetHeight)) {
    return Classification(NO_DATA, notANumber);
  }
  let targetSample = sampleRelative(targetColumn, targetRow, eye.hi, eye.lo);
  if (targetSample.y == 0.0) {
    return Classification(NO_DATA, notANumber);
  }
  let relativeHeight = targetSample.x + targetHeight;
  let deltaColumn = targetColumn - observerColumn;
  let deltaRow = targetRow - observerRow;
  ${geometry}
  let maximumFrame = settings[settingsOffset + 3u];
  let maximumDistance = select(MAXIMUM_DISTANCE, min(maximumFrame, MAXIMUM_DISTANCE), maximumFrame > 0.0);
  if (distance > maximumDistance${mercator ? ' || sphereDistance > MAXIMUM_DISTANCE' : ''}) {
    return Classification(OUT_OF_RANGE, notANumber);
  }
  if (!(distance > 0.0)) {
    return Classification(VISIBLE, notANumber);
  }
  let curvature = settings[settingsOffset];
  // A negative packed ignore distance is the derived default: one ground pixel at the target.
  let packedIgnoreDistance = settings[settingsOffset + 8u];
  let ignoreDistance = select(packedIgnoreDistance, targetPixelLength, packedIgnoreDistance < 0.0);
  let stop = distance - max(ignoreDistance, settings[settingsOffset + 9u] * distance);
  let ray = marchRay(eye, sinA, cosA, stop, stop > 0.0);
  let targetTangent = relativeHeight / distance - opaque(distance * curvature);
  let targetAngle = atanAccurate(targetTangent) * DEGREES;
  let occluderAngle = select(-90.0, atanAccurate(ray.tQ) * DEGREES, ray.tQ > -3.0e38);
  let skylineAngle = select(-90.0, atanAccurate(ray.t) * DEGREES, ray.has);
  let tolerance = settings[settingsOffset + 5u] + (settings[settingsOffset + 6u] / distance) * DEGREES;
  var code = VISIBLE;
  if (targetAngle < occluderAngle - tolerance) {
    code = HIDDEN;
  } else if (targetAngle - occluderAngle < tolerance) {
    // Not hidden, so the target is at or above the lower edge of the band: the band is
    // [occluder - tolerance, occluder + tolerance), with no gap at its lower edge.
    code = MARGINAL;
  }
  let onSkyline = code != HIDDEN && targetAngle >= skylineAngle - settings[settingsOffset + 7u];
  return Classification(code, vec4<f32>(targetAngle, occluderAngle, skylineAngle, select(0.0, 1.0, onSkyline)));
}`,
          body: `if (chunkStart[chunkStartOffset] + index >= TARGET_COUNT) { return; }
  let result = classifyTarget(index);
  visibilityOutput[visibilityOutputOffset + chunkStart[chunkStartOffset] + index] = result.code;
  ${detailsWrite}`
        })
      );
    }
    return nodes;
  }
}
