// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateTerrainBindings} from '../terrain-processing/terrain-binding-preflight';
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
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters,
  validateRasterOutputTexture
} from '../../raster-analysis/raster-band-utils';
import {GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES} from './gpu-solar-shadow-mask';
import {
  createTerrainHorizonSweepNode,
  getTerrainSweepLineGeometry,
  getTerrainSweepSourceParameterValues,
  TERRAIN_SWEEP_GRID_SLOTS,
  TERRAIN_SWEEP_MAXIMUM_EXTENT
} from './terrain-horizon-sweep';
import {
  validateTerrainIlluminationCellSize,
  validateTerrainIlluminationView
} from './terrain-illumination-utils';
import {
  type GPURasterCellSizeMode,
  type GPURasterRowDirection,
  validateRasterCellSizeMode,
  writeRasterGridParameterValues
} from '../../analysis-core/contracts/raster-grid';
import {resolveTerrainRowDirection} from '../terrain-curvature/terrain-geomorphometry-utils';

/** Number of float32 values read from `GPUTerrainCastShadowProps.parameters`. */
export const GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH = 16;

/** Explicit shared-source sweep layout: 16 solar values, 12 source/reference values, four reserved zeros. */
export const GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH = 32;

/**
 * CPU-side description packed by {@link getGPUTerrainCastShadowParameterValues}.
 *
 * Cell-size model: the cell size and row edges of the grid passed next to it are interpreted by the
 * contributor's `cellSizeMode` as projected metres (`uniform`), equatorial Web Mercator metres or
 * degrees with latitude-dependent spacing (`web-mercator`, `geographic`).
 */
export type GPUTerrainCastShadowSettings = {
  /** Direction of the sun, degrees clockwise from north. See `getSolarPosition`. */
  azimuthDegrees: number;
  /** Elevation of the sun center above the astronomical horizon in degrees. */
  altitudeDegrees: number;
  /**
   * Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to
   * {@link GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES}.
   */
  angularRadiusDegrees?: number;
  /** Elevation multiplier. Defaults to 1. */
  zFactor?: number;
  /** Earth curvature and refraction drop in 1/meters, see `GPUTerrainHorizonSettings`. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Ground search radius in meters; `<= 0` is unlimited. Defaults to 0. */
  maximumDistance?: number;
  /**
   * Shared positive ground metres per overview pixel `[x, y]` for line geometry and curvature.
   * Together with `sweepSourceGrid`, selects the 32-value sweep layout (this pair is at 16..17).
   * Use the same pair in every window of a mosaic and update it consistently per frame.
   */
  sweepReferenceCellSize?: readonly [number, number];
  /** Per-frame window origin in overview pixels relative to `sweepSourceGrid`. Defaults to [0, 0]. */
  sweepOrigin?: readonly [number, number];
  /**
   * Common full overview grid for row spacing and distance cutoffs. Required with sweep settings;
   * its height is at most 16777215, and every window must lie inside it. Use the same grid in
   * each window, including the whole-grid evaluation.
   */
  sweepSourceGrid?: GPURasterMetadata;
};

/**
 * Packs settings into the 16-float layout read by {@link GPUTerrainCastShadow}:
 * `[cellX, cellY, zFactor, curvature, northEdge, southEdge, maximumDistance, azimuthDegrees,
 * altitudeDegrees, angularRadiusDegrees, xMajor (0/1), slopeFixed, travelSign, stepPixelLength,
 * groundEast, groundSouth]`, with the cell size and the row-edge world y taken from `grid`.
 *
 * The sun's ground direction `(sin(azimuth), -cos(azimuth))` (east, south) is stored in the last two
 * slots, and its digital line family for square cells and south-advancing rows is computed here on
 * the CPU in float64, so changing the sun never recompiles the graph. The values do not depend on
 * the grid extent: the contributor mirrors the line for its `rowDirection: 'north'` and, on anisotropic
 * ground cells (non-square grid cells, `'web-mercator'`, `'geographic'`), converts the ground
 * direction to a pixel slope with its own `cellSizeMode` and the ground cell of row `height / 2`.
 * The contributor is the only source of both modes, so settings and contributor cannot disagree.
 *
 * With `sweepReferenceCellSize` and `sweepSourceGrid`, writes the explicit 32-value sweep layout:
 * shared ground cells at 16..17, origin modulo 65536 at 18..19, shared source cell sizes at 20..21,
 * source north/south edges at 22..23, source height and window source row at 24..25, reserved zeros
 * at 26..31. `sweepOrigin` is per-frame overview placement relative to the shared source grid,
 * default [0, 0]. The window must lie inside that source; source height is at most 16777215.
 * Ordinary settings keep the 16-value layout even in larger targets; padding never selects sweep.
 *
 * @throws If the grid cell size is not finite and positive, the angular radius is negative or not
 * finite, the azimuth is not finite, or `target` is too short.
 */
export function getGPUTerrainCastShadowParameterValues(
  settings: GPUTerrainCastShadowSettings,
  grid: GPURasterMetadata,
  target?: Float32Array
): Float32Array {
  const usesSharedSource =
    settings.sweepReferenceCellSize !== undefined ||
    settings.sweepOrigin !== undefined ||
    settings.sweepSourceGrid !== undefined;
  const minimumLength = usesSharedSource
    ? GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH
    : GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH;
  const values = target ?? new Float32Array(minimumLength);
  validateTerrainIlluminationCellSize('getGPUTerrainCastShadowParameterValues', grid);
  validateParameterTarget('getGPUTerrainCastShadowParameterValues', values, minimumLength);
  const angularRadius = settings.angularRadiusDegrees ?? GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES;
  if (!Number.isFinite(angularRadius) || angularRadius < 0) {
    throw new Error('Terrain cast shadow angular radius must be finite and non-negative');
  }
  if (!Number.isFinite(settings.azimuthDegrees) || !Number.isFinite(settings.altitudeDegrees)) {
    throw new Error('Terrain cast shadow azimuth and altitude must be finite');
  }
  const azimuth = (settings.azimuthDegrees * Math.PI) / 180;
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : value);
  // Canonical frame: rows advance south, as in getGPUTerrainHorizonDirection(..., 'south').
  const groundDirection: [number, number] = [snap(Math.sin(azimuth)), snap(-Math.cos(azimuth))];
  // The line geometry does not depend on the grid extent, only the line count does.
  const geometry = getTerrainSweepLineGeometry(groundDirection, 1, 1);
  values.set([
    0,
    0,
    settings.zFactor ?? 1,
    settings.curvatureCoefficient ?? 0,
    0,
    0,
    settings.maximumDistance ?? 0,
    settings.azimuthDegrees,
    settings.altitudeDegrees,
    angularRadius,
    geometry.xMajor ? 1 : 0,
    geometry.slopeFixed,
    geometry.travelSign,
    geometry.stepPixelLength,
    groundDirection[0],
    groundDirection[1]
  ]);
  writeRasterGridParameterValues(values, grid, TERRAIN_SWEEP_GRID_SLOTS);
  if (usesSharedSource) {
    // Keep the source placement validation and numerical packing identical to terrain horizon.
    values.set(
      getTerrainSweepSourceParameterValues(
        'getGPUTerrainCastShadowParameterValues',
        settings,
        grid
      ),
      16
    );
    values.fill(0, 28, GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH);
  }
  return values;
}

/** Caller-owned outputs of {@link GPUTerrainCastShadow}; at least one is required. */
export type GPUTerrainCastShadowOutput = {
  /** Fraction of the solar disk above the local horizon per pixel, in `[0, 1]`. */
  sunVisibility?: GraphDataView<'float32'>;
  /** Horizon angle in degrees along the sun azimuth. */
  horizonAngle?: GraphDataView<'float32'>;
  /** Per-pixel 1 where the center elevation is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving `sunVisibility`. */
  sunVisibilityTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Properties for {@link GPUTerrainCastShadow}.
 *
 * Topology: grid size, elevation format and calibration, `maximumRadius`, `parameterLayout`, `cellSizeMode`,
 * `rowDirection`, and which outputs exist. Per-frame: `parameters` (sun, disk radius, grid cell
 * size and latitude band, z factor, curvature, maximum distance, shared source origin/grid/reference)
 * and elevation contents.
 * Cell-size model: see {@link GPUTerrainCastShadowSettings}.
 */
export type GPUTerrainCastShadowProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-cast-shadow'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels, each at most 32767. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 16 float32 values, or 32 with `parameterLayout: 'sweep'`. */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time interpretation of parameters. Defaults to `'standard'`, the local 16-value model.
   * `'sweep'` selects 32 values packed with a shared source grid/reference cell and per-frame
   * source origin. Only this explicit layout can provide a seamless halo through `getHaloStage()`.
   * Spare view capacity never selects a layout; placement and sun updates reuse the same shader.
   */
  parameterLayout?: 'standard' | 'sweep';
  /** Caller-owned outputs. */
  output: GPUTerrainCastShadowOutput;
  /**
   * Search radius in pixels along the sun line (counted as `k * hypot(1, slope)`), a positive
   * safe integer. The supported shared-source sweep's tile halo is this radius. Defaults to the tile diagonal, `ceil(hypot(width,
   * height))`, which covers the whole tile for every sun azimuth.
   */
  maximumRadius?: number;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Direction in which the row index increases. Defaults to the direction the grid states: the
   * sign of the row term `grid.affine[4]` (negative: `'south'`, a north-up raster; positive:
   * `'north'`). An explicit value that contradicts the grid throws at construction.
   */
  rowDirection?: GPURasterRowDirection;
};

/**
 * Single-sun terrain cast shadow without a horizon map.
 *
 * One exact hull sweep (see `getTerrainHorizonSweepNode`) along the digital line family of the
 * per-frame sun azimuth yields the horizon angle toward the sun for every pixel, in amortised O(1)
 * per pixel and exact against brute force (up to the line discretisation: the ray follows a
 * Bresenham-style digital line rather than a bilinear ray). `sunVisibility` is the area fraction
 * of the solar disk above that horizon, the same formula as `GPUSolarShadowMask`: with
 * `u = clamp((altitude - horizon) / radius, -1, 1)` it is `1 - (acos(u) - u * sqrt(1 - u^2)) / PI`,
 * a hard step when the radius is 0, and 0 when the whole disk is below the astronomical horizon.
 *
 * Sun position, disk radius, cell size, z factor, curvature, and maximum distance are per-frame
 * parameters; the line family is packed by the CPU settings packer, so animating time of day never
 * recompiles. On anisotropic ground cells the kernel converts the packed ground direction with
 * this contributor's `cellSizeMode` (see {@link getGPUTerrainCastShadowParameterValues}), so the
 * sun follows its ground azimuth with no mode in the settings. The line count depends on the azimuth and is unknown at compile time, so
 * `width + height` invocations are dispatched and surplus ones exit.
 *
 * Softness comes from the physical solar disk radius. mt-image's propagated-occluder softness
 * (`smooth(-w, w, H + bias - O)`, a lateral-diffusion approximation) was not adopted: the hull
 * sweep is equally O(1) amortised and exact against brute force. Pixels with an invalid centre
 * elevation receive NaN. Pixels closer than `maximumRadius` to the tile edge see a truncated
 * horizon. For seamless results use `parameterLayout: 'sweep'` with the same `sweepSourceGrid`
 * and `sweepReferenceCellSize` across windows, and each window's per-frame `sweepOrigin`. This
 * anchors phase, reference geometry/curvature and source-row spacing/distance cutoffs. Then
 * `getHaloStage()` returns `{requiredHalo: maximumRadius}`. Standard settings retain local line
 * phase and middle-row reference geometry; more padding alone cannot make that model seamless,
 * and `getHaloStage()` refuses it, including when the parameter view has spare capacity.
 */
export class GPUTerrainCastShadow implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainCastShadowProps;
  /** Search radius in pixels. */
  readonly maximumRadius: number;
  /** Row direction the kernels use: the prop, or the direction the grid affine states. */
  readonly rowDirection: GPURasterRowDirection;
  /** Explicit interpretation of parameter words; independent of view capacity. */
  readonly parameterLayout: 'standard' | 'sweep';

  constructor(props: GPUTerrainCastShadowProps) {
    this.id = props.id ?? 'terrain-cast-shadow';
    this.props = Object.freeze({...props});
    const {id} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const pixelCount = validateRasterGrid(id, props.grid);
    if (width > TERRAIN_SWEEP_MAXIMUM_EXTENT || height > TERRAIN_SWEEP_MAXIMUM_EXTENT) {
      throw new Error(
        `${id} grid width and height must not exceed ${TERRAIN_SWEEP_MAXIMUM_EXTENT}`
      );
    }
    const maximumRadius = props.maximumRadius ?? Math.ceil(Math.hypot(width, height));
    validateCapacity(id, 'maximumRadius', maximumRadius, 1, Number.MAX_SAFE_INTEGER);
    this.maximumRadius = maximumRadius;
    this.parameterLayout = props.parameterLayout ?? 'standard';
    validateRequirement(
      id,
      this.parameterLayout === 'standard' || this.parameterLayout === 'sweep',
      'parameterLayout must be standard or sweep'
    );
    if (
      !output.sunVisibility &&
      !output.horizonAngle &&
      !output.outputValidity &&
      !output.sunVisibilityTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(
      id,
      'output.sunVisibility',
      output.sunVisibility,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.horizonAngle',
      output.horizonAngle,
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
        ? GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH
        : GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH
    );
    validateRasterOutputTexture(
      id,
      'output.sunVisibilityTexture',
      output.sunVisibilityTexture,
      ['r32float', 'rgba32float'],
      props.grid
    );
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    this.rowDirection = resolveTerrainRowDirection(id, props.grid, props.rowDirection);
    validateRasterBuffersDistinct(
      id,
      [output.sunVisibility, output.horizonAngle, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns a complete halo stage only for the explicit shared-source sweep model. */
  getHaloStage(): GPURasterHaloStage {
    validateRequirement(
      this.id,
      this.parameterLayout === 'sweep',
      'parameterLayout must be sweep with shared source placement/reference for a seamless halo'
    );
    return {requiredHalo: this.maximumRadius};
  }

  /** Returns elevation canonicalization, the sweep node, and the optional texture node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTerrainBindings(graph, this.id, this.props, {
      'canonical elevation': 4 * this.props.grid.width * this.props.grid.height
    });
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, [output.sunVisibilityTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.sunVisibility,
      output.horizonAngle,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const visibilityTarget =
      output.sunVisibility ??
      (output.sunVisibilityTexture
        ? createTransientView(graph, `${id}-sun-visibility`, 'float32', pixelCount)
        : undefined);
    const outputBindings: WGSLKernelBinding[] = [];
    if (visibilityTarget) {
      outputBindings.push({
        name: 'visibilityValues',
        view: visibilityTarget,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.horizonAngle) {
      outputBindings.push({
        name: 'horizonValues',
        view: output.horizonAngle,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.outputValidity) {
      outputBindings.push({
        name: 'validityValues',
        view: output.outputValidity,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createTerrainHorizonSweepNode<Parameters>(graph, {
        id: `${id}-sweep`,
        width,
        height,
        geometry: {
          kind: 'settings',
          index: 10,
          groundDirectionIndex: 14,
          rowDirection: this.rowDirection
        },
        maximumRadius: this.maximumRadius,
        referenceCellSizeIndex: this.parameterLayout === 'sweep' ? 16 : undefined,
        originIndex: this.parameterLayout === 'sweep' ? 18 : undefined,
        sourceGridIndex: this.parameterLayout === 'sweep' ? 20 : undefined,
        cellSizeMode: props.cellSizeMode ?? 'uniform',
        zFactorSign: 1,
        elevationValues: source.band.storage.values as GraphDataView<'float32'>,
        settings: props.parameters,
        hull: createTransientView(graph, `${id}-hull`, 'uint32', pixelCount),
        outputBindings,
        outputWGSL: `let isValid = isFiniteValue(horizonAngle);
      let invalidValue = getQuietNaN(pixel);
      let altitudeDegrees = settings[settingsOffset + 8u];
      let radiusDegrees = settings[settingsOffset + 9u];
      var visibility = 0.0;
      if (altitudeDegrees + radiusDegrees > 0.0) {
        if (radiusDegrees > 0.0) {
          let u = clamp((altitudeDegrees - horizonAngle) / radiusDegrees, -1.0, 1.0);
          visibility = clamp(1.0 - (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / PI, 0.0, 1.0);
        } else {
          visibility = select(0.0, 1.0, altitudeDegrees > horizonAngle);
        }
      }
      ${visibilityTarget ? 'visibilityValues[visibilityValuesOffset + pixel] = select(invalidValue, visibility, isValid);' : ''}
      ${output.horizonAngle ? 'horizonValues[horizonValuesOffset + pixel] = horizonAngle;' : ''}
      ${output.outputValidity ? 'validityValues[validityValuesOffset + pixel] = select(0u, 1u, isValid);' : ''}`
      })
    );
    if (output.sunVisibilityTexture && visibilityTarget) {
      const texture = output.sunVisibilityTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-sun-visibility-texture`,
            input: {
              id: `${id}-sun-visibility-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: visibilityTarget}
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
