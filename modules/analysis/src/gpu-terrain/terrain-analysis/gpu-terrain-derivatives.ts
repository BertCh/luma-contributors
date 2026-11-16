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
  GPURasterGradient,
  type GPURasterBand,
  type GPURasterBorderMode,
  type GPURasterHaloStage,
  type GPURasterMetadata
} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  RASTER_WGSL_HELPERS,
  validateParameterTarget,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters,
  validateRasterOutputTexture
} from '../../raster-analysis/raster-band-utils';
import {
  getRasterGroundCellSizeWGSL,
  getRasterRowNorthSignWGSL,
  type GPURasterCellSizeMode,
  type GPURasterRowDirection,
  RASTER_DEGREES_TO_RADIANS_WGSL,
  validateRasterCellSizeMode,
  writeRasterGridParameterValues
} from '../../analysis-core/raster-grid';
import {
  getTerrainOneSidedPaddedBandNodes,
  getTerrainPaddedIndexWGSL,
  resolveTerrainRowDirection
} from '../terrain-curvature/terrain-geomorphometry-utils';

/** Slope output unit. */
export type GPUTerrainSlopeUnits = 'degrees' | 'percent';

/** Number of float32 values read from `GPUTerrainDerivativesProps.parameters`. */
export const GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH = 8;

/** Cell size and row-edge slots of the derivative parameter layout. */
const TERRAIN_DERIVATIVES_GRID_SLOTS = {cellSizeIndex: 0, northEdgeIndex: 5} as const;

/**
 * CPU-side description packed by {@link getGPUTerrainDerivativesParameterValues}. Cell size and
 * latitude band come from the grid passed next to it.
 */
export type GPUTerrainDerivativesSettings = {
  /** Elevation multiplier applied to derivatives. Defaults to 1. */
  zFactor?: number;
  /** Direction the light comes from, degrees clockwise from north. Defaults to 315. */
  azimuthDegrees?: number;
  /** Light elevation above the horizon in degrees. Defaults to 45. */
  altitudeDegrees?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUTerrainDerivatives}:
 * `[cellSizeX, cellSizeY, zFactor, azimuthDegrees, altitudeDegrees, northEdge, southEdge, 0]`,
 * with the cell size and the row-edge world y taken from `grid`.
 */
export function getGPUTerrainDerivativesParameterValues(
  settings: GPUTerrainDerivativesSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUTerrainDerivativesParameterValues',
    target,
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  target[2] = settings.zFactor ?? 1;
  target[3] = settings.azimuthDegrees ?? 315;
  target[4] = settings.altitudeDegrees ?? 45;
  target[7] = 0;
  writeRasterGridParameterValues(target, grid, TERRAIN_DERIVATIVES_GRID_SLOTS);
  return target;
}

/** Caller-owned outputs of {@link GPUTerrainDerivatives}; at least one is required. */
export type GPUTerrainDerivativesOutput = {
  /** Slope per pixel in degrees `[0, 90)` or percent. */
  slope?: GraphDataView<'float32'>;
  /** Downslope direction per pixel, degrees clockwise from north; -1 for flat cells. */
  aspect?: GraphDataView<'float32'>;
  /** Hillshade per pixel in `[0, 1]`. */
  hillshade?: GraphDataView<'float32'>;
  /** Per-pixel 1 where every output is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving the hillshade. */
  hillshadeTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Properties for {@link GPUTerrainDerivatives}.
 *
 * Topology: grid size, elevation format and calibration, which outputs exist, `cellSizeMode`,
 * `slopeUnits`, `rowDirection`, and `borderMode`. Per-frame: `parameters` (including the grid
 * cell size and latitude band) and elevation contents.
 */
export type GPUTerrainDerivativesProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-derivatives'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUTerrainDerivativesParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainDerivativesOutput;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Slope unit. Defaults to `'degrees'`. */
  slopeUnits?: GPUTerrainSlopeUnits;
  /**
   * Direction in which the row index increases. Defaults to the direction the grid states: the
   * sign of the row term `grid.affine[4]` (negative: `'south'`, a north-up raster; positive:
   * `'north'`). An explicit value that contradicts the grid throws at construction.
   */
  rowDirection?: GPURasterRowDirection;
  /**
   * Border treatment of the Horn 3x3 stencil. Defaults to `'one-sided'`, which extends the
   * surface by one cell with the quadratic through the three nearest in-grid samples, so the
   * centred difference across the edge is the second-order one-sided stencil
   * `(-3 f0 + 4 f1 - f2) / (2 h)` of luma.gl's `GPUFiniteDifference2D`: slope and aspect are exact
   * on a plane up to the border. The other values are the border modes of the official
   * `GPURasterGradient`, forwarded unchanged; `'clamp'` repeats the edge sample and so halves the
   * gradient component across the edge (a plane of slope 11.31 degrees reads 5.71 degrees in its
   * border column).
   */
  borderMode?: 'one-sided' | GPURasterBorderMode;
};

/**
 * Computes slope, aspect, and hillshade for an elevation tile with Horn's 3x3 method.
 *
 * Composes two Sobel `GPURasterGradient` passes and one fused shade kernel. Cell size, z factor,
 * sun position, and latitude band are read from `parameters` every encoding. Columns increase east
 * and rows go where the grid affine says (`rowDirection`). Aspect and the hillshade azimuth are
 * degrees clockwise from north, which on a rotated `'uniform'` grid is the grid's up axis. Border
 * pixels use one-sided differences by default (`borderMode`). Invalid pixels (an invalid center or
 * 3x3 neighbor, or invalid settings) receive NaN and validity 0. For seamless tiles, pass a tile
 * with `requiredHalo` source pixels per side: two for one-sided extrapolation, one otherwise.
 * The contributor satisfies the `GPURasterHaloStage` contract.
 */
export class GPUTerrainDerivatives implements GPUCommandNodeProducer, GPURasterHaloStage {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainDerivativesProps;
  /** Source-pixel radius, including the two inward samples used by one-sided extrapolation. */
  readonly requiredHalo: number;
  /** Row direction the kernels use: the prop, or the direction the grid affine states. */
  readonly rowDirection: GPURasterRowDirection;

  constructor(props: GPUTerrainDerivativesProps) {
    this.id = props.id ?? 'terrain-derivatives';
    this.props = props;
    this.requiredHalo = (props.borderMode ?? 'one-sided') === 'one-sided' ? 2 : 1;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    if (
      !output.slope &&
      !output.aspect &&
      !output.hillshade &&
      !output.outputValidity &&
      !output.hillshadeTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view, format] of [
      ['slope', output.slope, 'float32'],
      ['aspect', output.aspect, 'float32'],
      ['hillshade', output.hillshade, 'float32'],
      ['outputValidity', output.outputValidity, 'uint32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} output.${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} output.${name} must contain one value per pixel`);
        }
      }
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
    validateRasterOutputTexture(
      id,
      'output.hillshadeTexture',
      output.hillshadeTexture,
      ['r32float', 'rgba32float'],
      props.grid
    );
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    if (!['degrees', 'percent'].includes(props.slopeUnits ?? 'degrees')) {
      throw new Error(`${id} slopeUnits must be degrees or percent`);
    }
    this.rowDirection = resolveTerrainRowDirection(id, props.grid, props.rowDirection);
    if (
      !['one-sided', 'clamp', 'reflect', 'constant', 'nodata'].includes(
        props.borderMode ?? 'one-sided'
      )
    ) {
      throw new Error(`${id} borderMode must be one-sided, clamp, reflect, constant, or nodata`);
    }
    validateRasterBuffersDistinct(
      id,
      [output.slope, output.aspect, output.hillshade, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns optional elevation gather, two Sobel passes, shade, and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTerrainBindings(graph, this.id, this.props, {
      'gradient scratch':
        4 *
        (this.props.grid.width + ((this.props.borderMode ?? 'one-sided') === 'one-sided' ? 2 : 0)) *
        (this.props.grid.height + ((this.props.borderMode ?? 'one-sided') === 'one-sided' ? 2 : 0))
    });
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, [output.hillshadeTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.slope,
      output.aspect,
      output.hillshade,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    const borderMode = props.borderMode ?? 'one-sided';
    const isOneSided = borderMode === 'one-sided';
    // The one-sided ring is extrapolated from calibrated values, so that path canonicalizes first
    // (a float32 band that already has validity and no calibration passes through).
    const source = getRasterBufferBandNodes(
      graph,
      id,
      props.elevation,
      props.grid,
      isOneSided ? 'reuse-canonical' : false
    );
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const padded = isOneSided
      ? getTerrainOneSidedPaddedBandNodes(graph, {
          id,
          operation: 'GPUTerrainDerivatives',
          width,
          height,
          band: source.band
        })
      : undefined;
    nodes.push(...(padded?.nodes ?? []));
    const gradientWidth = padded?.width ?? width;
    const gradientHeight = padded?.height ?? height;
    const gradients = (['x', 'y'] as const).map(direction => {
      const values = createTransientView(
        graph,
        `${id}-gradient-${direction}-values`,
        'float32',
        gradientWidth * gradientHeight
      );
      const validity = createTransientView(
        graph,
        `${id}-gradient-${direction}-validity`,
        'uint32',
        gradientWidth * gradientHeight
      );
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterGradient({
            id: `${id}-gradient-${direction}`,
            width: gradientWidth,
            height: gradientHeight,
            input: padded?.band ?? source.band,
            output: values,
            outputValidity: validity,
            operator: 'sobel',
            direction,
            // Only the ring of the padded grid meets this border; no original pixel reads it.
            borderMode: borderMode === 'one-sided' ? 'clamp' : borderMode,
            scale: 1
          }).addToGraph(graph)
        )
      );
      return {values, validity};
    });
    const hillshadeTarget =
      output.hillshade ??
      (output.hillshadeTexture
        ? createTransientView(graph, `${id}-hillshade`, 'float32', pixelCount)
        : undefined);
    nodes.push(
      getShadeNode(graph, {
        id: `${id}-shade`,
        width,
        height,
        gradientX: gradients[0].values,
        gradientY: gradients[1].values,
        gradientValidity: gradients[0].validity,
        settings: props.parameters,
        slope: output.slope,
        aspect: output.aspect,
        hillshade: hillshadeTarget,
        validity: output.outputValidity,
        cellSizeMode: props.cellSizeMode ?? 'uniform',
        slopeUnits: props.slopeUnits ?? 'degrees',
        rowDirection: this.rowDirection,
        isPadded: isOneSided
      })
    );
    if (output.hillshadeTexture && hillshadeTarget) {
      const texture = output.hillshadeTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-hillshade-texture`,
            input: {
              id: `${id}-hillshade-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: hillshadeTarget}
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

/** Builds the fused per-pixel slope, aspect, hillshade, and validity kernel. */
function getShadeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    gradientX: GraphDataView<'float32'>;
    gradientY: GraphDataView<'float32'>;
    gradientValidity: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
    slope?: GraphDataView<'float32'>;
    aspect?: GraphDataView<'float32'>;
    hillshade?: GraphDataView<'float32'>;
    validity?: GraphDataView<'uint32'>;
    cellSizeMode: GPURasterCellSizeMode;
    slopeUnits: GPUTerrainSlopeUnits;
    rowDirection: GPURasterRowDirection;
    /** The gradients live on the one-cell padded grid of the one-sided border. */
    isPadded: boolean;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'gradientX', view: props.gradientX, type: 'f32', access: 'read'},
    {name: 'gradientY', view: props.gradientY, type: 'f32', access: 'read'},
    {name: 'gradientValidity', view: props.gradientValidity, type: 'u32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  if (props.slope)
    bindings.push({name: 'slopeValues', view: props.slope, type: 'f32', access: 'read_write'});
  if (props.aspect) {
    bindings.push({name: 'aspectValues', view: props.aspect, type: 'f32', access: 'read_write'});
  }
  if (props.hillshade) {
    bindings.push({
      name: 'hillshadeValues',
      view: props.hillshade,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.validity) {
    bindings.push({
      name: 'validityValues',
      view: props.validity,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainDerivatives',
    variant: `shade-${props.cellSizeMode}${props.isPadded ? '-padded' : ''}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const PI: f32 = 3.141592653589793;
${RASTER_DEGREES_TO_RADIANS_WGSL}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${getRasterRowNorthSignWGSL(props.rowDirection)}
${RASTER_WGSL_HELPERS}
${props.aspect ? GUARDED_ATAN2_WGSL : ''}
${getRasterGroundCellSizeWGSL(props.cellSizeMode, TERRAIN_DERIVATIVES_GRID_SLOTS)}`,
    body: `let row = index / WIDTH;
  ${getTerrainPaddedIndexWGSL(props.isPadded)}
  let rawX = gradientX[gradientXOffset + gradientIndex];
  let rawY = gradientY[gradientYOffset + gradientIndex];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let larger = max(abs(eastGradient), abs(northGradient));
  let smaller = min(abs(eastGradient), abs(northGradient));
  var gradientMagnitude = 0.0;
  if (larger > 0.0) {
    let ratio = smaller / larger;
    gradientMagnitude = larger * sqrt(1.0 + ratio * ratio);
  }
  let isValid = gradientValidity[gradientValidityOffset + gradientIndex] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient) && isFiniteValue(gradientMagnitude);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${
    props.slope
      ? `let slope = ${props.slopeUnits === 'degrees' ? 'atan(gradientMagnitude) * RADIANS_TO_DEGREES' : 'gradientMagnitude * 100.0'};
  slopeValues[slopeValuesOffset + index] = select(invalidValue, slope, isValid);`
      : ''
  }
  ${
    props.aspect
      ? `var aspect = -1.0;
  if (gradientMagnitude > 0.0) {
    // -northGradient is -0.0 on a pure east or west slope with north row order (FX-17).
    aspect = guardedAtan2(-eastGradient, -northGradient) * RADIANS_TO_DEGREES;
    if (aspect < 0.0) { aspect += 360.0; }
    if (aspect >= 360.0) { aspect -= 360.0; }
  }
  aspectValues[aspectValuesOffset + index] = select(invalidValue, aspect, isValid);`
      : ''
  }
  ${
    props.hillshade
      ? `let azimuth = settings[settingsOffset + 3u] * DEGREES_TO_RADIANS;
  let altitude = settings[settingsOffset + 4u] * DEGREES_TO_RADIANS;
  let light = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let shade = max(dot(normal, light), 0.0);
  hillshadeValues[hillshadeValuesOffset + index] = select(invalidValue, shade, isValid);`
      : ''
  }
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
  });
}
