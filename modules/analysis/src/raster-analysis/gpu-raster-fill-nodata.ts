// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from './raster-grid/raster-storage-preflight';
import {
  validateCapacity,
  validateRequirement,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateGPUStatusPort, type GPUInvalidStatusPort} from '../analysis-core/contracts/status';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  snapshotRasterBand,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph
} from './raster-band-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {createRasterCountNodes} from './remote-sensing/raster-kernel-utils';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  requireRasterLength,
  validateRasterExpansionPorts,
  validateRasterExpansionBand,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from './raster-expansion-utils';

/** Largest search radius in cells; the brute-force disk search costs O(radius squared) per cell. */
const MAXIMUM_SEARCH_DISTANCE = 256;

/** Caller-owned prepared surface and provenance. */
export type GPURasterFillNoDataOutput = {
  values: GraphDataView<'float32'>;
  validity: GraphDataView<'uint32'>;
  filled: GraphDataView<'uint32'>;
  /**
   * Required named status (an object; its entries are optional). `invalidCount` (one `uint32`
   * row): the number of cells of the result that are still invalid, that is, cells with
   * `output.validity` 0 because the bounded search found no valid source. This is the status
   * vocabulary's `invalidCount` read over the output cells (what the operation could not fill), not
   * over input rows rejected by validation: a missing input cell the search does fill is not
   * counted, and an input cell that is invalid and unreachable is.
   */
  status: GPUInvalidStatusPort;
};
/** Continuous interpolation of original valid samples; filled samples never become donors. */
export type GPURasterFillNoDataProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBand;
  /**
   * Euclidean distance in pixel coordinates; positive integer, at most 256. Every missing pixel
   * scans a (2 * maximumDistance + 1)^2 window in one dispatch, so on large grids with many missing
   * pixels a large distance can approach GPU watchdog limits; prefer the smallest sufficient distance.
   */
  maximumDistance: number;
  /** Original-donor nearest, all-donor IDW, or GDAL four-quadrant IDW (no smoothing). */
  method?: 'nearest' | 'inverse-distance' | 'gdal-four-quadrant';
  output: GPURasterFillNoDataOutput;
};

/**
 * Bounded, deterministic nodata interpolation on the GPU. Original valid calibrated values are
 * preserved exactly. The default all-donor IDW and nearest contracts remain distinct from
 * the explicit GDAL four-quadrant selection. All methods use zero smoothing iterations.
 */
export class GPURasterFillNoData implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFillNoDataProps;
  constructor(props: GPURasterFillNoDataProps) {
    props = {...props, input: snapshotRasterBand(props.input)};
    this.id = props.id ?? 'raster-fill-nodata';
    this.props = props;
    validateRasterTextureBand(this.id, props.input, props.grid);
    validateRasterExpansionBand(this.id, 'input', props.input);
    validateRasterExpansionPorts(
      this.id,
      props,
      {
        'output.values': ['float32'],
        'output.validity': ['uint32'],
        'output.filled': ['uint32'],
        'output.status.invalidCount': ['uint32']
      },
      ['output.status.invalidCount']
    );
    // Reject unsupported interpolation names instead of silently choosing all-donor IDW.
    validateRequirement(
      this.id,
      props.method === undefined ||
        ['nearest', 'inverse-distance', 'gdal-four-quadrant'].includes(props.method),
      "method must be 'nearest', 'inverse-distance' or 'gdal-four-quadrant'"
    );
    validateRequirement(
      this.id,
      typeof props.output.status === 'object' && props.output.status !== null,
      'output.status is required (an object holding the optional invalidCount view, the count of cells left invalid)'
    );
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      getRasterBandViews(props.input),
      getOutputViews(props.output)
    );
    requireRasterLength(props.output.values, count, this.id, 'output.values');
    requireRasterLength(props.output.validity, count, this.id, 'output.validity');
    requireRasterLength(props.output.filled, count, this.id, 'output.filled');
    validateGPUStatusPort(this.id, props.output.status);
    if (props.output.status.invalidCount) {
      requireRasterLength(
        props.output.status.invalidCount,
        1,
        this.id,
        'output.status.invalidCount'
      );
    }
    validateCapacity(this.id, 'maximumDistance', props.maximumDistance, 1, MAXIMUM_SEARCH_DISTANCE);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    const storageCellCount = storageProps.grid.width * storageProps.grid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...getRasterBandViews(storageProps.input),
        ...getRasterPreflightOutputViews(storageProps.output)
      ],
      {
        'canonical band planes': storageCellCount * 4
      }
    );

    const {props, id} = this;
    const {output, grid} = props;
    const count = grid.width * grid.height;
    validateRasterBandBelongsToGraph(id, graph, props.input, []);
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      ...getOutputViews(output)
    ]);
    const source = getRasterBufferBandNodes(graph, id, props.input, grid, true);
    return [
      ...source.nodes,
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPURasterFillNoData',
        invocationCount: count,
        bindings: [
          {name: 'source', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'sourceValidity', view: source.band.validity!, type: 'u32', access: 'read'},
          {name: 'values', view: output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          {name: 'filled', view: output.filled, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `var result = source[sourceOffset + index];
var valid = sourceValidity[sourceValidityOffset + index] != 0u;
let originalValid = valid;
if (!valid) {
  var total = 0.0; var weight = 0.0; var nearestDistance = 3.402823e38;
  let column = i32(index % ${grid.width}u); let row = i32(index / ${grid.width}u);
  ${
    props.method === 'gdal-four-quadrant'
      ? getFourQuadrantSearch(grid, props.maximumDistance)
      : `  for (var rowOffset = -${props.maximumDistance}; rowOffset <= ${props.maximumDistance}; rowOffset++) {
    for (var columnOffset = -${props.maximumDistance}; columnOffset <= ${props.maximumDistance}; columnOffset++) {
      let squaredDistance = columnOffset * columnOffset + rowOffset * rowOffset;
      let sampleColumn = column + columnOffset; let sampleRow = row + rowOffset;
      if (squaredDistance == 0 || squaredDistance > ${props.maximumDistance * props.maximumDistance} || sampleColumn < 0 || sampleRow < 0 || sampleColumn >= ${grid.width} || sampleRow >= ${grid.height}) { continue; }
      let sampleIndex = u32(sampleRow) * ${grid.width}u + u32(sampleColumn);
      if (sourceValidity[sourceValidityOffset + sampleIndex] == 0u) { continue; }
      let distance = sqrt(f32(squaredDistance));
      ${props.method === 'nearest' ? `if (distance < nearestDistance) { nearestDistance = distance; total = source[sourceOffset + sampleIndex]; weight = 1.0; }` : `let contribution = 1.0 / distance; total += source[sourceOffset + sampleIndex] * contribution; weight += contribution;`}
    }
  }`
  }

  valid = weight > 0.0; result = total / max(weight, 1.0e-30);
}
valid = valid && isFiniteValue(result);
values[valuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), result, valid);
validity[validityOffset + index] = u32(valid);
filled[filledOffset + index] = u32(valid && !originalValid);`
      }),
      ...(output.status.invalidCount
        ? createRasterCountNodes(graph, {
            id: `${id}-unresolved`,
            operation: 'GPURasterFillNoData',
            values: output.validity,
            count,
            condition: 'value == 0u',
            output: output.status.invalidCount
          })
        : [])
    ];
  }
}

function getOutputViews(output: GPURasterFillNoDataOutput): GraphDataView[] {
  return [
    output.values,
    output.validity,
    output.filled,
    ...(output.status.invalidCount ? [output.status.invalidCount] : [])
  ];
}

/** GDAL-compatible search order, quadrant axes, border clamping and strict distance ties. */
function getFourQuadrantSearch(grid: GPURasterMetadata, maximumDistance: number): string {
  return `var quadrantDistances = array<i32, 4>(${(maximumDistance + 1) ** 2}, ${(maximumDistance + 1) ** 2}, ${(maximumDistance + 1) ** 2}, ${(maximumDistance + 1) ** 2});
  var quadrantValues: array<f32, 4>;
  for (var step = 0; step <= ${maximumDistance}; step++) {
    for (var side = 0; side < 2; side++) {
      if (side == 1 && step == 0) { continue; }
      let sampleColumn = select(max(0, column - step), min(${grid.width - 1}, column + step), side == 1);
      for (var direction = 0; direction < 2; direction++) {
        let quadrant = u32(side * 2 + direction);
        for (var verticalOffset = direction; verticalOffset <= ${maximumDistance}; verticalOffset++) {
          let sampleRow = select(row - verticalOffset, row + verticalOffset, direction == 1);
          if (sampleRow < 0 || sampleRow >= ${grid.height}) { break; }
          let sampleIndex = u32(sampleRow) * ${grid.width}u + u32(sampleColumn);
          if (sourceValidity[sourceValidityOffset + sampleIndex] == 0u) { continue; }
          let horizontalOffset = sampleColumn - column;
          let squaredDistance = horizontalOffset * horizontalOffset + verticalOffset * verticalOffset;
          if (squaredDistance > 0 && squaredDistance < quadrantDistances[quadrant]) {
            quadrantDistances[quadrant] = squaredDistance;
            quadrantValues[quadrant] = source[sourceOffset + sampleIndex];
          }
          // Only the closest original donor in this column/direction can win.
          break;
        }
      }
    }
  }
  for (var quadrant = 0u; quadrant < 4u; quadrant++) {
    if (quadrantDistances[quadrant] <= ${maximumDistance ** 2}) {
      let contribution = inverseSqrt(f32(quadrantDistances[quadrant]));
      total += quadrantValues[quadrant] * contribution;
      weight += contribution;
    }
  }`;
}
