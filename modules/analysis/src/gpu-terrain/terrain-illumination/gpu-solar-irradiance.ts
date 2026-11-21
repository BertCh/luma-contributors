// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainOutputSelection} from './terrain-output-selection';

import {validateTerrainBindings} from '../terrain-processing/terrain-binding-preflight';
import {toRadians} from '@math.gl/core';
import {
  createTransientView,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/graph/gpu-core';
import {GPURasterBufferToTexture, type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  captureGraphCommandNodes,
  validateFiniteNumber,
  validateRequirement,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateParameterTarget,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterParameters,
  validateRasterOutputTexture
} from '../../raster-analysis/raster-band-utils';
import {
  GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MINIMUM_DIRECTION_COUNT
} from './gpu-terrain-horizon';
import {GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES} from './gpu-solar-shadow-mask';
import {getSolarPosition, validateSolarPositionLocation} from './solar-position';
import {
  getTerrainSolarVisibilityWGSL,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainHorizonFormat,
  validateTerrainHorizonView,
  validateTerrainIlluminationView,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';

/** Number of float32 values read from `GPUSolarIrradianceProps.parameters`. */
export const GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH = 8;

/** Float32 values per sun table row: azimuth, altitude, duration, direct normal irradiance. */
export const GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE = 4;

/** Largest exact float32 row count whose final scalar lane fits uint32 at binding offset zero. */
export const MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT = 1073741824;

/** Solar constant used by the Meinel clear-sky model in W/m^2. */
const SOLAR_CONSTANT = 1353;

/**
 * Direct normal irradiance of the Meinel clear-sky model in W/m^2: `1353 * 0.7^(AM^0.678)` with
 * the Kasten-Young (1989) air mass `AM = 1 / (cos(z) + 0.50572 * (96.07995 - z)^-1.6364)`, where
 * `z` is the zenith angle in degrees of the (refracted) sun altitude.
 *
 * @returns 0 when `altitudeDegrees <= 0`.
 */
export function getGPUSolarDirectNormalIrradiance(altitudeDegrees: number): number {
  if (!(altitudeDegrees > 0)) {
    return 0;
  }
  const zenithDegrees = 90 - altitudeDegrees;
  const airMass =
    1 / (Math.cos(toRadians(zenithDegrees)) + 0.50572 * (96.07995 - zenithDegrees) ** -1.6364);
  return SOLAR_CONSTANT * 0.7 ** (airMass ** 0.678);
}

/** Options for {@link getGPUSolarIrradianceSunTable}. */
export type GPUSolarIrradianceSunTableOptions = {
  /** Degrees east. */
  longitude: number;
  /** Degrees north in `[-90, 90]`. */
  latitude: number;
  /** Start of the period: Unix epoch milliseconds (UTC) or a `Date`. */
  start: number | Date;
  /** End of the period, after `start`. */
  end: number | Date;
  /** Sampling step in minutes. Defaults to 5. */
  stepMinutes?: number;
  /**
   * Direct normal irradiance in W/m^2 while the sun is up: a constant, or `'meinel'` for the
   * clear-sky model of {@link getGPUSolarDirectNormalIrradiance}. Defaults to `'meinel'`.
   */
  directNormalIrradiance?: number | 'meinel';
  /** Apply atmospheric refraction to the sun altitude. Defaults to true. */
  refraction?: boolean;
  /** Optional destination with at least `sampleCount * 4` values. */
  target?: Float32Array;
};

/**
 * Samples the sun over a period into the table read by {@link GPUSolarIrradiance}.
 *
 * Each row covers one step of `stepMinutes` and is evaluated at the midpoint of the interval it
 * covers; the last partial step is shortened and its `durationHours` is its real length. Rows are
 * `[azimuthDegrees, altitudeDegrees, durationHours, directNormalIrradiance]` (compass azimuth
 * clockwise from north, see `getSolarPosition`). The CPU float64 solar position keeps sunrise and
 * sunset timing exact; the table is then plain float32 data that one buffer write replaces.
 *
 * @returns The packed rows and their number. `values` preserves a supplied target, including its
 * padding. `counts.sampleCount` is the active row count; `counts.sampleCapacity` is the number of
 * complete allocated four-float rows. `formats.sunTable` names the consumer's scalar view format:
 * import `sampleCapacity * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE` float32 values. Tails are unchanged.
 * @throws If a time is not finite, `end <= start`, the step is not positive, or `target` is short.
 */
export function getGPUSolarIrradianceSunTable(options: GPUSolarIrradianceSunTableOptions): {
  values: Float32Array;
  sampleCount: number;
  counts: {sampleCount: number; sampleCapacity: number};
  formats: {sunTable: 'float32'};
} {
  validateSolarPositionLocation(
    'getGPUSolarIrradianceSunTable',
    options.longitude,
    options.latitude
  );
  const startMilliseconds =
    typeof options.start === 'number' ? options.start : options.start.getTime();
  const endMilliseconds = typeof options.end === 'number' ? options.end : options.end.getTime();
  if (!Number.isFinite(startMilliseconds) || !Number.isFinite(endMilliseconds)) {
    throw new Error('Solar irradiance sun table times must be finite');
  }
  if (!(endMilliseconds > startMilliseconds)) {
    throw new Error('Solar irradiance sun table end must be after start');
  }
  const stepMinutes = options.stepMinutes ?? 5;
  if (!Number.isFinite(stepMinutes) || stepMinutes <= 0) {
    throw new Error('Solar irradiance sun table stepMinutes must be positive');
  }
  const stepMilliseconds = stepMinutes * 60000;
  const sampleCount = Math.ceil((endMilliseconds - startMilliseconds) / stepMilliseconds - 1e-9);
  validateSolarIrradianceSampleCount('getGPUSolarIrradianceSunTable', sampleCount);
  const stride = GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE;
  const directNormal = options.directNormalIrradiance ?? 'meinel';
  validateRequirement(
    'getGPUSolarIrradianceSunTable',
    directNormal === 'meinel' || typeof directNormal === 'number',
    `directNormalIrradiance must be meinel or a finite nonnegative number; received ${directNormal}`
  );
  if (directNormal !== 'meinel') {
    validateRequirement(
      'getGPUSolarIrradianceSunTable',
      directNormal >= 0,
      `directNormalIrradiance must be nonnegative; received ${directNormal}`
    );
    validateFiniteNumber(
      'getGPUSolarIrradianceSunTable',
      'directNormalIrradiance',
      directNormal,
      0
    );
  }
  const values = options.target ?? new Float32Array(sampleCount * stride);
  if (values.length < sampleCount * stride) {
    throw new Error(`Solar irradiance sun table target must hold ${sampleCount * stride} values`);
  }
  for (let sample = 0; sample < sampleCount; sample++) {
    const intervalStart = startMilliseconds + sample * stepMilliseconds;
    const intervalEnd = Math.min(intervalStart + stepMilliseconds, endMilliseconds);
    const position = getSolarPosition(
      (intervalStart + intervalEnd) / 2,
      options.longitude,
      options.latitude,
      {refraction: options.refraction}
    );
    const irradiance =
      position.altitudeDegrees > 0
        ? directNormal === 'meinel'
          ? getGPUSolarDirectNormalIrradiance(position.altitudeDegrees)
          : directNormal
        : 0;
    values.set(
      [
        position.azimuthDegrees,
        position.altitudeDegrees,
        (intervalEnd - intervalStart) / 3600000,
        irradiance
      ],
      sample * stride
    );
  }
  return {
    values,
    sampleCount,
    counts: {sampleCount, sampleCapacity: Math.floor(values.length / stride)},
    formats: {sunTable: 'float32'}
  };
}

/** CPU-side description packed by {@link getGPUSolarIrradianceParameterValues}. */
export type GPUSolarIrradianceSettings = {
  /** Active rows in `[0, sampleCapacity]`, exactly representable in float32 and at most 1073741824. */
  sampleCount: number;
  /**
   * Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to
   * {@link GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES}.
   */
  angularRadiusDegrees?: number;
  /**
   * Diffuse horizontal irradiance of an open site in W/m^2, added while the sun is up and scaled
   * by `GPUSolarIrradianceProps.skyViewFactor` when one is bound (the irradiance sky-view factor,
   * not the solid-angle one). Defaults to 0.
   */
  diffuseIrradiance?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUSolarIrradiance}:
 * `[sampleCount, angularRadiusDegrees, diffuseIrradiance, 0, 0, 0, 0, 0]`.
 *
 * @throws If `sampleCount` is not a non-negative integer, the radius or diffuse irradiance is
 * negative or not finite, or `target` is too short.
 */
export function getGPUSolarIrradianceParameterValues(
  settings: GPUSolarIrradianceSettings,
  target: Float32Array = new Float32Array(GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUSolarIrradianceParameterValues',
    target,
    GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH
  );
  validateSolarIrradianceSampleCount('getGPUSolarIrradianceParameterValues', settings.sampleCount);
  const angularRadius = settings.angularRadiusDegrees ?? GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES;
  validateRequirement(
    'getGPUSolarIrradianceParameterValues',
    angularRadius >= 0,
    `angularRadiusDegrees must be nonnegative; received ${angularRadius}`
  );
  validateFiniteNumber(
    'getGPUSolarIrradianceParameterValues',
    'angularRadiusDegrees',
    angularRadius,
    0
  );
  const diffuse = settings.diffuseIrradiance ?? 0;
  validateRequirement(
    'getGPUSolarIrradianceParameterValues',
    diffuse >= 0,
    `diffuseIrradiance must be nonnegative; received ${diffuse}`
  );
  validateFiniteNumber('getGPUSolarIrradianceParameterValues', 'diffuseIrradiance', diffuse, 0);
  target.fill(0, 0, GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH);
  target.set([settings.sampleCount, angularRadius, diffuse]);
  return target;
}

/** Exact float32 row counts whose four-lane table offsets fit the uint32 consumer. @internal */
function validateSolarIrradianceSampleCount(id: string, sampleCount: number): void {
  validateRequirement(
    id,
    Number.isSafeInteger(sampleCount) &&
      sampleCount >= 0 &&
      Math.fround(sampleCount) === sampleCount &&
      sampleCount <= MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT,
    `sampleCount must be a nonnegative exact float32 integer at most MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT = ${MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT}; received ${sampleCount}`
  );
}

/** Caller-owned outputs of {@link GPUSolarIrradiance}; at least one is required. */
export type GPUSolarIrradianceOutput = TerrainOutputSelection<{
  /** Hours the solar disk is visible, summed over the sun table. */
  sunHours?: GraphDataView<'float32'>;
  /** Energy in Wh/m^2 on the (inclined) surface summed over the sun table. */
  insolation?: GraphDataView<'float32'>;
  /** Per-pixel 1 where every output is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving `sunHours`. */
  sunHoursTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving `insolation`. */
  insolationTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
}>;

/**
 * Properties for {@link GPUSolarIrradiance}.
 *
 * Topology: grid size, `directionCount`, `horizonFormat`, `sampleCapacity`, `samplesPerNode`, and
 * which inputs and outputs exist. Per-frame: `parameters`, the contents of `sunTable`, and input
 * contents. The contributor works in pixel space and never reads cell sizes: all terrain geometry is
 * already in the horizon map, slope, and aspect.
 * The float32 parameter layout requires an exact active `sampleCount`, at most 1073741824 total
 * rows, so its four-float table prefix fits uint32 indexing; per-node chunking does not raise it.
 */
export type GPUSolarIrradianceProps = {
  /** Prefix for node and transient IDs. Defaults to `'solar-irradiance'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Number of horizon sectors, an integer in `[4, 64]`, matching the `GPUTerrainHorizon`. */
  directionCount: number;
  /** Storage format of `horizon`. Defaults to `'float32'`; see `GPUSolarShadowMask`. */
  horizonFormat?: GPUTerrainHorizonFormat;
  /**
   * Pixel-major horizon angles in degrees from `GPUTerrainHorizon`: `width * height *
   * directionCount` float32 values, or for `'unorm16'` a `uint32` view of
   * `ceil(width * height * directionCount / 2)` words. A pixel is invalid when its sector 0
   * horizon is NaN (the horizon contributor writes NaN in every sector of an invalid pixel).
   */
  horizon: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /**
   * Sun rows, `sampleCapacity * 4` float32 values, see {@link getGPUSolarIrradianceSunTable}.
   * Rewrite it per frame to change location, date range, or step without recompiling.
   */
  sunTable: GraphDataView<'float32'>;
  /** Maximum number of sun table rows, an integer of at least 1 (topology). */
  sampleCapacity: number;
  /**
   * Rows processed per node (topology). Defaults to `sampleCapacity`. Smaller values split the
   * table across several nodes that accumulate into the outputs, which keeps each dispatch short
   * on large grids (GPU watchdog safety). Results are identical to the unchunked path.
   */
  samplesPerNode?: number;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUSolarIrradianceParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Horn or compatible slope in degrees. Used with `aspect` for `insolation`. */
  slope?: GraphDataView<'float32'>;
  /** Optional aspect in degrees clockwise from north, -1 when flat. Required with `slope`. */
  aspect?: GraphDataView<'float32'>;
  /**
   * Optional factor in `[0, 1]` multiplying the diffuse term of `insolation`: the fraction of the
   * open-site diffuse horizontal irradiance the pixel receives. For an isotropic sky that is the
   * cosine-weighted factor `mean(cos^2(max(h, 0)))` (Dozier and Frew 1990), which
   * `GPUTerrainHorizon` writes to `output.irradianceSkyViewFactor`. Do not bind that
   * contributor's `output.skyViewFactor` here: it is the solid-angle factor
   * `1 - mean(sin(max(h, 0)))` of Zaksek et al. (2011), made for relief visualization, and it
   * understates diffuse energy (a uniform 30 degree horizon: 0.50 against the correct 0.75).
   */
  skyViewFactor?: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUSolarIrradianceOutput;
};

/**
 * Accumulates sun-hours and insolation over a day or date range from a precomputed horizon map.
 *
 * For every sun table row whose disk is at least partly above the astronomical horizon
 * (`altitude + radius > 0`) the kernel interpolates the horizon at the sun azimuth and takes the
 * solar-disk visibility, with code identical to `GPUSolarShadowMask`. Then
 *
 * ```
 * sunHours   = sum(visibility * duration)
 * insolation = sum(DNI * visibility * max(cosIncidence, 0) * duration)
 *            + sum(diffuse * svf * duration)           (rows with altitude > 0)
 * ```
 *
 * with `cosIncidence` from slope and aspect when they are bound (a flat aspect of -1 gives
 * `sin(altitude)`), else `sin(altitude)` for a horizontal surface; the diffuse term uses
 * `svf = 1` without a sky-view factor. Durations are hours and irradiances W/m^2, so `insolation` is in
 * Wh/m^2. Pixels with a NaN horizon, slope, aspect, or sky-view factor receive NaN and validity 0.
 *
 * Diffuse term. `diffuse` is the diffuse horizontal irradiance of an open site and `svf` is
 * whatever raster is bound as `skyViewFactor`; the kernel only multiplies. For an isotropic sky
 * the physically right factor is the irradiance sky-view factor `mean(cos^2(max(h, 0)))`
 * (`GPUTerrainHorizon` `output.irradianceSkyViewFactor`), not the solid-angle sky-view factor
 * `1 - mean(sin(max(h, 0)))` (`output.skyViewFactor`), which is 33% low under a uniform 30 degree
 * horizon. The diffuse term is that of a horizontal receiver: slope and aspect enter the direct
 * term only. No `(1 + cos(slope)) / 2` tilt factor is applied, because a horizon measured on the
 * same terrain already contains the receiver's own slope (on an infinite plane of slope `s` the
 * irradiance sky-view factor is exactly `(1 + cos(s)) / 2`); the slope-azimuth term of Dozier and
 * Frew and radiation reflected by terrain are not modelled. For a tilted panel on open ground
 * bind a raster of `(1 + cos(tilt)) / 2`.
 *
 * One invocation per pixel loops over its chunk of table rows, so a day at 5 minute steps is 288
 * iterations with two horizon reads each; `samplesPerNode` splits long tables across nodes.
 * The active total row count must be exactly representable in float32 and at most
 * `MAXIMUM_GPU_SOLAR_IRRADIANCE_SAMPLE_COUNT = 1073741824` for a zero-offset storage binding;
 * the final lane including the binding's scalar offset must fit uint32. Chunking does not increase
 * this layout limit. Device buffer limits can require smaller tables.
 */
export class GPUSolarIrradiance implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSolarIrradianceProps;
  /** Storage format of the horizon input. */
  readonly horizonFormat: GPUTerrainHorizonFormat;
  /** Number of sun table rows handled by each accumulation node. */
  readonly samplesPerNode: number;

  constructor(props: GPUSolarIrradianceProps) {
    this.id = props.id ?? 'solar-irradiance';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    const {directionCount, sampleCapacity} = props;
    if (
      !Number.isSafeInteger(directionCount) ||
      directionCount < GPU_TERRAIN_HORIZON_MINIMUM_DIRECTION_COUNT ||
      directionCount > GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT
    ) {
      throw new Error(`${id} directionCount must be an integer in [4, 64]`);
    }
    if (!Number.isSafeInteger(sampleCapacity) || sampleCapacity < 1) {
      throw new Error(`${id} sampleCapacity must be a positive integer`);
    }
    this.samplesPerNode = props.samplesPerNode ?? sampleCapacity;
    if (!Number.isSafeInteger(this.samplesPerNode) || this.samplesPerNode < 1) {
      throw new Error(`${id} samplesPerNode must be a positive integer`);
    }
    if (
      !output.sunHours &&
      !output.insolation &&
      !output.outputValidity &&
      !output.sunHoursTexture &&
      !output.insolationTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    if (Boolean(props.slope) !== Boolean(props.aspect)) {
      throw new Error(`${id} slope and aspect must be provided together`);
    }
    validateTerrainHorizonFormat(id, props.horizonFormat ?? 'float32');
    this.horizonFormat = props.horizonFormat ?? 'float32';
    validateTerrainHorizonView(
      id,
      'horizon',
      props.horizon,
      this.horizonFormat,
      pixelCount,
      directionCount
    );
    validateTerrainIlluminationView(
      id,
      'sunTable',
      props.sunTable,
      'float32',
      sampleCapacity * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE
    );
    const lastTableLane =
      getViewElementOffset(props.sunTable) +
      sampleCapacity * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE -
      1;
    validateRequirement(
      id,
      Number.isSafeInteger(lastTableLane) && lastTableLane <= 0xffffffff,
      `sunTable last scalar lane must fit uint32 indexing (at most 4294967295); received ${lastTableLane}`
    );
    for (const [name, view] of [
      ['slope', props.slope],
      ['aspect', props.aspect],
      ['skyViewFactor', props.skyViewFactor],
      ['output.sunHours', output.sunHours],
      ['output.insolation', output.insolation]
    ] as const) {
      validateTerrainIlluminationView(id, name, view, 'float32', pixelCount);
    }
    validateTerrainIlluminationView(
      id,
      'output.outputValidity',
      output.outputValidity,
      'uint32',
      pixelCount
    );
    validateRasterParameters(id, props.parameters, GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH);
    for (const [name, texture] of [
      ['output.sunHoursTexture', output.sunHoursTexture],
      ['output.insolationTexture', output.insolationTexture]
    ] as const) {
      validateRasterOutputTexture(id, name, texture, ['r32float', 'rgba32float'], props.grid);
    }
    validateRasterBuffersDistinct(
      id,
      [output.sunHours, output.insolation, output.outputValidity],
      [
        props.horizon,
        props.sunTable,
        props.parameters,
        props.slope,
        props.aspect,
        props.skyViewFactor
      ]
    );
  }

  /** Returns one accumulation node per sun table chunk, an optional validity node, and texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const needsSolarTable = Boolean(
      this.props.output.sunHours ||
        this.props.output.insolation ||
        this.props.output.sunHoursTexture ||
        this.props.output.insolationTexture
    );
    validateTerrainBindings(
      graph,
      this.id,
      {
        ...this.props,
        sunTable: needsSolarTable ? this.props.sunTable : undefined,
        parameters: needsSolarTable ? this.props.parameters : undefined
      },
      {
        'texture staging':
          (this.props.output.sunHoursTexture && !this.props.output.sunHours) ||
          (this.props.output.insolationTexture && !this.props.output.insolation)
            ? 4 * this.props.grid.width * this.props.grid.height
            : 0
      }
    );
    const {id, props} = this;
    const {output} = props;
    const {directionCount, sampleCapacity} = props;
    const pixelCount = props.grid.width * props.grid.height;
    validateGraphViewsBelongToGraph(id, graph, [
      props.horizon,
      props.sunTable,
      props.parameters,
      props.slope,
      props.aspect,
      props.skyViewFactor,
      output.sunHours,
      output.insolation,
      output.outputValidity
    ]);
    for (const texture of [output.sunHoursTexture, output.insolationTexture]) {
      if (texture && texture.texture.graph !== graph) {
        throw new Error(`${id} views must belong to the target graph`);
      }
    }
    const hoursTarget =
      output.sunHours ??
      (output.sunHoursTexture
        ? createTransientView(graph, `${id}-sun-hours`, 'float32', pixelCount)
        : undefined);
    const insolationTarget =
      output.insolation ??
      (output.insolationTexture
        ? createTransientView(graph, `${id}-insolation`, 'float32', pixelCount)
        : undefined);
    const horizonType = this.horizonFormat === 'unorm16' ? 'u32' : 'f32';
    const inputBindings = (): WGSLKernelBinding[] => {
      const bindings: WGSLKernelBinding[] = [
        {name: 'horizon', view: props.horizon, type: horizonType, access: 'read'}
      ];
      if (props.slope && props.aspect) {
        bindings.push(
          {name: 'slopeValues', view: props.slope, type: 'f32', access: 'read'},
          {name: 'aspectValues', view: props.aspect, type: 'f32', access: 'read'}
        );
      }
      if (props.skyViewFactor) {
        bindings.push({name: 'skyView', view: props.skyViewFactor, type: 'f32', access: 'read'});
      }
      return bindings;
    };
    const hasGeometry = Boolean(props.slope && props.aspect);
    // Sector 0 of an invalid pixel is NaN, as are slope, aspect, and svf.
    const validityWGSL = `var isValid = isFiniteValue(readHorizonAngle(index * DIRECTION_COUNT));
  ${
    hasGeometry
      ? `isValid = isValid && isFiniteValue(slopeValues[slopeValuesOffset + index]) &&
    isFiniteValue(aspectValues[aspectValuesOffset + index]);`
      : ''
  }
  ${props.skyViewFactor ? 'isValid = isValid && isFiniteValue(skyView[skyViewOffset + index]);' : ''}`;
    const visibilityDeclarations = `${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainSolarVisibilityWGSL(this.horizonFormat, 'horizon', directionCount)}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (hoursTarget || insolationTarget) {
      for (
        let chunkStart = 0, chunk = 0;
        chunkStart < sampleCapacity;
        chunkStart += this.samplesPerNode, chunk++
      ) {
        const chunkEnd = Math.min(chunkStart + this.samplesPerNode, sampleCapacity);
        const bindings: WGSLKernelBinding[] = [
          ...inputBindings(),
          {name: 'sunTable', view: props.sunTable, type: 'f32', access: 'read'},
          {name: 'settings', view: props.parameters, type: 'f32', access: 'read'}
        ];
        if (hoursTarget) {
          bindings.push({
            name: 'hoursValues',
            view: hoursTarget,
            type: 'f32',
            access: 'read_write'
          });
        }
        if (insolationTarget) {
          bindings.push({
            name: 'insolationValues',
            view: insolationTarget,
            type: 'f32',
            access: 'read_write'
          });
        }
        const first = chunk === 0;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-sun-${chunk}`,
            operation: 'GPUSolarIrradiance',
            variant: `${this.horizonFormat}${insolationTarget ? '-insolation' : ''}`,
            bindings,
            invocationCount: pixelCount,
            declarations: `const CHUNK_START: u32 = ${chunkStart}u;
const CHUNK_END: u32 = ${chunkEnd}u;
const TABLE_STRIDE: u32 = ${GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE}u;
${visibilityDeclarations}`,
            body: `${validityWGSL}
  let radiusDegrees = settings[settingsOffset + 1u];
  let diffuseIrradiance = settings[settingsOffset + 2u];
  let sampleEnd = min(CHUNK_END, u32(max(settings[settingsOffset], 0.0)));
  var hours = 0.0;
  var energy = 0.0;
  if (isValid) {
    ${
      hasGeometry
        ? `let slopeRadians = slopeValues[slopeValuesOffset + index] * DEGREES_TO_RADIANS;
    let aspectDegrees = aspectValues[aspectValuesOffset + index];
    let sinSlope = sin(slopeRadians);
    let cosSlope = cos(slopeRadians);`
        : ''
    }
    ${props.skyViewFactor ? 'let skyViewFactor = skyView[skyViewOffset + index];' : 'let skyViewFactor = 1.0;'}
    for (var sampleIndex = CHUNK_START; sampleIndex < sampleEnd; sampleIndex++) {
      let row = sunTableOffset + sampleIndex * TABLE_STRIDE;
      let azimuthDegrees = sunTable[row];
      let altitudeDegrees = sunTable[row + 1u];
      let duration = sunTable[row + 2u];
      if (altitudeDegrees + radiusDegrees <= 0.0) {
        continue;
      }
      let visibility = getSolarDiskVisibility(
        altitudeDegrees, radiusDegrees, getHorizonAtAzimuth(index, azimuthDegrees));
      hours += visibility * duration;
      ${
        insolationTarget
          ? `let altitude = altitudeDegrees * DEGREES_TO_RADIANS;
      var cosIncidence = sin(altitude);
      ${
        hasGeometry
          ? `if (aspectDegrees >= 0.0) {
        cosIncidence = cosSlope * sin(altitude) +
          sinSlope * cos(altitude) * cos(azimuthDegrees * DEGREES_TO_RADIANS - aspectDegrees * DEGREES_TO_RADIANS);
      }`
          : ''
      }
      energy += sunTable[row + 3u] * visibility * max(cosIncidence, 0.0) * duration;
      if (altitudeDegrees > 0.0) {
        energy += diffuseIrradiance * skyViewFactor * duration;
      }`
          : ''
      }
    }
  }
  ${
    hoursTarget
      ? `hoursValues[hoursValuesOffset + index] = select(getQuietNaN(index), ${first ? 'hours' : 'hoursValues[hoursValuesOffset + index] + hours'}, isValid);`
      : ''
  }
  ${
    insolationTarget
      ? `insolationValues[insolationValuesOffset + index] = select(getQuietNaN(index), ${first ? 'energy' : 'insolationValues[insolationValuesOffset + index] + energy'}, isValid);`
      : ''
  }`
          })
        );
      }
    }
    if (output.outputValidity) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-validity`,
          operation: 'GPUSolarIrradiance',
          variant: 'validity',
          bindings: [
            ...inputBindings(),
            {name: 'validityValues', view: output.outputValidity, type: 'u32', access: 'read_write'}
          ],
          invocationCount: pixelCount,
          declarations: visibilityDeclarations,
          body: `${validityWGSL}
  validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`
        })
      );
    }
    for (const [name, target, texture] of [
      ['sun-hours', hoursTarget, output.sunHoursTexture],
      ['insolation', insolationTarget, output.insolationTexture]
    ] as const) {
      if (texture && target) {
        nodes.push(
          ...captureGraphCommandNodes(graph, () =>
            new GPURasterBufferToTexture({
              id: `${id}-${name}-texture`,
              input: {
                id: `${id}-${name}-band`,
                format: 'float32',
                storage: {kind: 'buffer', values: target}
              },
              output: texture,
              channel: 0
            }).addToGraph(graph)
          )
        );
      }
    }
    return nodes;
  }
}
