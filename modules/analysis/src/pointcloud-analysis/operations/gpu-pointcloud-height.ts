// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {GPUCommandGraph, GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_POINT_CLOUD_MAXIMUM_IDW_POWER,
  GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS
} from './pointcloud-limits';
import {
  planGraphChunkDispatch,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getPointCloudChunkKernelNodes,
  validatePointCloudChunkViews
} from './pointcloud-chunk-kernel-nodes';
import type {PointCloudKernelProps} from './pointcloud-kernel-nodes';
import {validatePointCloudGridParameters} from './pointcloud-grid-parameters';
import {
  getPointCloudCellSource,
  validatePointCloudGrid,
  type PointCloudRasterGrid
} from './gpu-pointcloud-surfaces';

// Exact coverage mask/support/loop reach and bitcast float32 radius squared/power divided by two.
export const GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH = 5;

export type GPUPointCloudHeightSettings = {
  coverageBit?: number;
  interpolation?: GPUPointCloudHeightProps['interpolation'];
};

/** Pack runtime sampling settings; interpolation method and grid stay compile-time. */
export function getGPUPointCloudHeightParameterValues(
  settings: GPUPointCloudHeightSettings,
  grid: PointCloudRasterGrid
): Uint32Array {
  const identifier = 'gpu-pointcloud-height';
  validatePointCloudGrid(grid, identifier);
  validateCapacity(identifier, 'coverageBit', settings.coverageBit ?? 0xffffffff);
  const interpolation = settings.interpolation;
  const values = new Uint32Array(GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH);
  values[0] = settings.coverageBit ?? 0xffffffff;
  if (interpolation?.method === 'idw') {
    validatePointCloudHeightInterpolation(identifier, interpolation, grid);
    new Float32Array(values.buffer).set(
      [interpolation.radius ** 2, (interpolation.power ?? 2) / 2],
      1
    );
    values[3] = interpolation.minimumSupport ?? 1;
    values[4] = Math.ceil(interpolation.radius / grid.cellSize) + 1;
  }
  return values;
}

function validatePointCloudHeightInterpolation(
  identifier: string,
  interpolation: Extract<NonNullable<GPUPointCloudHeightProps['interpolation']>, {method: 'idw'}>,
  grid: PointCloudRasterGrid
): void {
  validateRequirement(
    identifier,
    interpolation.radius > 0 &&
      Math.fround(interpolation.radius ** 2) > 0 &&
      Number.isFinite(Math.fround(interpolation.radius ** 2)),
    'interpolation.radius must be positive with a finite nonzero float32 square'
  );
  validateMaximumCount(
    identifier,
    'interpolation.radius in cells',
    interpolation.radius / grid.cellSize,
    GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS
  );
  validateRequirement(
    identifier,
    Math.fround((interpolation.power ?? 2) / 2) > 0 &&
      (interpolation.power ?? 2) <= GPU_POINT_CLOUD_MAXIMUM_IDW_POWER,
    `interpolation.power must be positive in float32 and at most GPU_POINT_CLOUD_MAXIMUM_IDW_POWER (${GPU_POINT_CLOUD_MAXIMUM_IDW_POWER})`
  );
  validateCapacity(identifier, 'interpolation.minimumSupport', interpolation.minimumSupport ?? 1);
}

export type GPUPointCloudHeightOutput = {
  heights: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * Per-point status code (a value, not a bit set): `GPU_POINT_CLOUD_HEIGHT_STATUS_VALID` (0)
   * valid sample, `..._INVALID_POINT` (1) the point is outside the grid or not finite,
   * `..._UNSUPPORTED_TERRAIN` (2) the terrain under the point is unsupported (or the interpolation
   * lacks support), `..._NONFINITE` (3) the elevation or height overflowed float32.
   */
  status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

export type GPUPointCloudHeightProps = {
  /** Runtime origin/cellSize, packed by getGPUPointCloudGridParameterValues; grid dimensions stay fixed. */
  gridParameters?: GraphDataView<'float32'>;
  /** Runtime coverageBit and IDW numeric settings, packed by getGPUPointCloudHeightParameterValues. */
  parameters?: GraphDataView<'uint32'>;
  id?: string;
  positions: GraphDataView<'float32x3'> | GraphVectorView<'float32x3'>;
  grid: PointCloudRasterGrid;
  terrain: GraphDataView<'float32'>;
  /** Nonzero cells are supported. For Surfaces coverage pass coverageBit: 2. */
  coverage: GraphDataView<'uint32'>;
  coverageBit?: number;
  /** Cell sampling by default; bilinear requires every positive-weight corner, IDW fills from finite supported centers inside radius. */
  interpolation?:
    | {method: 'bilinear'}
    | {method: 'idw'; radius: number; power?: number; minimumSupport?: number};
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudHeightOutput;
};

/** `GPUPointCloudHeight` status code: valid sample. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_VALID = 0;

/** `GPUPointCloudHeight` status code: the point is outside the grid or has a nonfinite coordinate. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT = 1;

/** `GPUPointCloudHeight` status code: unsupported terrain under the point, or too little interpolation support. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN = 2;

/** `GPUPointCloudHeight` status code: the elevation or height is not finite in float32. */
export const GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE = 3;

/**
 * Height above ground, preserving negative heights. Positions, heights and status accept identical
 * ordered chunk lengths; terrain/coverage stay resident. Per-frame: parameters, gridParameters
 * (origin/cellSize) and input contents. Compile-time: grid dimensions/row direction, interpolation
 * method and chunk topology. IDW reach remains bounded by
 * GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS when packing parameters.
 */
export class GPUPointCloudHeight implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  constructor(readonly props: GPUPointCloudHeightProps) {
    props = snapshotConstructionFields(props, [
      'gridParameters',
      'parameters',
      'id',
      'positions',
      'grid',
      'terrain',
      'coverage',
      'coverageBit',
      'interpolation',
      'output'
    ]);
    this.id = props.id ?? 'gpu-pointcloud-height';
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const grid = snapshotConstructionFields(props.grid, [
      'origin',
      'cellSize',
      'width',
      'height',
      'rowDirection'
    ]);
    const interpolation =
      props.interpolation &&
      snapshotConstructionFields(
        props.interpolation as NonNullable<GPUPointCloudHeightProps['interpolation']> & {
          radius?: number;
          power?: number;
          minimumSupport?: number;
        },
        ['method', 'radius', 'power', 'minimumSupport']
      );
    this.props = props = Object.freeze({
      ...props,
      grid: Object.freeze({
        ...grid,
        origin: Object.freeze([grid.origin[0], grid.origin[1]] as const)
      }),
      interpolation,
      output: snapshotConstructionFields(props.output, ['heights', 'status'])
    });
    validatePointCloudGrid(props.grid, this.id);
    validatePointCloudGridParameters(this.id, props.gridParameters);
    if (props.interpolation?.method === 'idw')
      validatePointCloudHeightInterpolation(this.id, props.interpolation, props.grid);
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['uint32'], `${this.id} parameters`);
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH
      );
    }
    validateSinglePackedView(props.terrain, ['float32'], `${this.id} terrain`);
    validateSinglePackedView(props.coverage, ['uint32'], `${this.id} coverage`);
    for (const [name, view, format] of [
      ['positions', props.positions, 'float32x3'],
      ['heights', props.output.heights, 'float32'],
      ['status', props.output.status, 'uint32']
    ] as const)
      for (const chunk of getGraphViewChunks(view))
        validateSinglePackedView(chunk, [format], `${this.id} ${name}`);
    this.chunkPlan = planGraphChunkDispatch(this.id, {
      positions: props.positions,
      heights: props.output.heights,
      status: props.output.status
    });
    validatePointCloudChunkViews(
      this.id,
      this.chunkPlan,
      ['positions'],
      ['heights', 'status'],
      {
        terrain: props.terrain,
        coverage: props.coverage,
        parameters: props.parameters,
        gridParameters: props.gridParameters
      },
      [
        {
          inputNames: ['positions'],
          outputNames: ['heights', 'status'],
          inputs: {
            terrain: props.terrain,
            coverage: props.coverage,
            parameters: props.parameters,
            gridParameters: props.gridParameters
          }
        }
      ]
    );
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    const cells = props.grid.width * props.grid.height;
    validateViewLength(this.id, 'terrain', props.terrain, cells);
    validateViewLength(this.id, 'coverage', props.coverage, cells);
    validateViewLength(this.id, 'heights', props.output.heights, props.positions.length);
    validateViewLength(this.id, 'status', props.output.status, props.positions.length);
    validateCapacity(this.id, 'coverageBit', props.coverageBit ?? 0xffffffff);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      grid,
      terrain,
      coverage,
      interpolation,
      output: {heights, status}
    } = this.props;
    if (!this.chunkPlan.nonemptyChunkCount) return [];
    validateGraphChunkDispatchBindings(graph, this.id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, this.id, {
      terrain,
      coverage,
      parameters: this.props.parameters,
      gridParameters: this.props.gridParameters
    });
    const coverageMask = this.props.parameters
      ? 'readParameters(0u)'
      : `${this.props.coverageBit ?? 0xffffffff}u`;
    const supported = `(readCoverage(sampleIndex)&${coverageMask})!=0u && abs(readTerrain(sampleIndex))<=3.402823e38`;
    let sample = `if ((readCoverage(cellIndex) & ${coverageMask}) == 0u || !(abs(readTerrain(cellIndex)) <= 3.402823e38)) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u); return; }
elevation = readTerrain(cellIndex);`;
    if (interpolation?.method === 'bilinear')
      sample = `
let centered=coordinate-vec2f(0.5);let lower=vec2i(floor(centered));let fraction=fract(centered);
var valid=true;
for(var vertical=0;vertical<2;vertical++){for(var horizontal=0;horizontal<2;horizontal++){
 let weight=select(1.0-fraction.x,fraction.x,horizontal==1)*select(1.0-fraction.y,fraction.y,vertical==1);
 if(weight==0.0){continue;}let location=lower+vec2i(horizontal,vertical);
 if(any(location<vec2i(0))||any(location>=vec2i(${grid.width},${grid.height}))){valid=false;continue;}
 let sampleIndex=u32(location.y)*${grid.width}u+u32(location.x);
 if(!(${supported})){valid=false;continue;}elevation+=weight*readTerrain(sampleIndex);
}}
if(!valid){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u);return;}`;
    if (interpolation?.method === 'idw') {
      const reach = this.props.gridParameters
        ? '(i32(ceil(getScaledQuotient(sqrt(radiusSquared), readGridParameters(3u), 1.0, 0))) + 1)'
        : this.props.parameters
          ? 'i32(readParameters(4u))'
          : Math.ceil(interpolation.radius / grid.cellSize) + 1;
      const radiusSquared = this.props.parameters
        ? 'bitcast<f32>(readParameters(1u))'
        : Math.fround(interpolation.radius ** 2);
      const halfPower = this.props.parameters
        ? 'bitcast<f32>(readParameters(2u))'
        : Math.fround((interpolation.power ?? 2) / 2);
      const minimumSupport = this.props.parameters
        ? 'readParameters(3u)'
        : `${interpolation.minimumSupport ?? 1}u`;
      sample = `${
        this.props.gridParameters
          ? `let radiusSquared = ${radiusSquared};
if (!(getScaledQuotient(sqrt(radiusSquared), readGridParameters(3u), 1.0, 0) <= ${GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS}.0)) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u); return; }`
          : ''
      }
let center=vec2i(floor(coordinate));var meanElevation=0.0;var totalWeight=0.0;var minimumSquaredDistance=3.402823e38;var support=0u;var exact=false;
for(var row=max(0,center.y-${reach});row<=min(${grid.height - 1},center.y+${reach});row++){
 for(var column=max(0,center.x-${reach});column<=min(${grid.width - 1},center.x+${reach});column++){
  let sampleIndex=u32(row)*${grid.width}u+u32(column);if(!(${supported})){continue;}
  let difference=(vec2f(f32(column),f32(row))+vec2f(0.5)-coordinate)*${this.props.gridParameters ? 'readGridParameters(3u)' : Math.fround(grid.cellSize)};
  let distanceSquared=dot(difference,difference);if(distanceSquared>${radiusSquared}){continue;}
  support++;if(distanceSquared==0.0){exact=true;elevation=readTerrain(sampleIndex);continue;}
  // Normalize to the closest sample so near-centre queries cannot overflow inverse powers.
  if(distanceSquared<minimumSquaredDistance){
   totalWeight*=pow(getScaledQuotient(distanceSquared,minimumSquaredDistance,1.0,0),${halfPower});
   minimumSquaredDistance=distanceSquared;
  }
  let weight=pow(getScaledQuotient(minimumSquaredDistance,distanceSquared,1.0,0),${halfPower});
  let combinedWeight=totalWeight+weight;
  meanElevation=meanElevation*(totalWeight/combinedWeight)+readTerrain(sampleIndex)*(weight/combinedWeight);
  totalWeight=combinedWeight;
 }}
let valid=exact||(support>=${minimumSupport}&&totalWeight>0.0);
if(!exact&&totalWeight>0.0){elevation=meanElevation;}if(!valid){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_UNSUPPORTED_TERRAIN}u);return;}`;
    }
    const kernelProps: PointCloudKernelProps = {
      id: this.id,
      length: positions.length,
      inputs: {
        positions: getGraphViewChunks(positions)[0],
        terrain,
        coverage,
        ...(this.props.parameters ? {parameters: this.props.parameters} : {}),
        ...(this.props.gridParameters ? {gridParameters: this.props.gridParameters} : {})
      },
      outputs: {heights: getGraphViewChunks(heights)[0], status: getGraphViewChunks(status)[0]},
      body: `let position = readPositions(index);
${getPointCloudCellSource(grid, 'position', Boolean(this.props.gridParameters))}
writeHeights(index, 0.0);
if (!inside || !all(abs(position) <= vec3f(3.402823e38))) { writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_INVALID_POINT}u); return; }
var elevation=0.0;
${sample}
let height=position.z-elevation;
// Finite observations can still overflow interpolation or the final signed difference.
if(any((bitcast<vec2u>(vec2f(elevation,height))&vec2u(0x7f800000u))==vec2u(0x7f800000u))){writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_NONFINITE}u);return;}
writeHeights(index,height);writeStatus(index, ${GPU_POINT_CLOUD_HEIGHT_STATUS_VALID}u);`
    };
    return getPointCloudChunkKernelNodes(graph, {...kernelProps, plan: this.chunkPlan});
  }
}
