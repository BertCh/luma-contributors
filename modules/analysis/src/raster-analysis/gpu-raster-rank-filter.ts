// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber
} from '../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getRasterBandViews, getRasterBufferBandNodes} from './raster-band-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from './raster-expansion-utils';

/** Largest footprint radius; each cell insertion-sorts (2 radius + 1) squared values in private memory. */
const MAXIMUM_RADIUS = 8;

export type GPURasterRankFilterProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBufferBand;
  /** Square footprint radius in [0,8]. */
  radius: number;
  /** Quantile in [0,1], default 0.5. Selects floor(q * n), capped at n - 1 (SciPy rank convention). */
  quantile?: number;
  borderMode?: 'clamp' | 'nodata';
  noDataPolicy?: 'propagate' | 'ignore';
  output: {
    values: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    changed?: GraphDataView<'uint32'>;
  };
};

/** Raster-native rank selection. Invalid centers remain invalid; omitted neighbors never become zeros. */
export class GPURasterRankFilter implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterRankFilterProps;
  constructor(props: GPURasterRankFilterProps) {
    this.id = props.id ?? 'raster-rank-filter';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      getRasterBandViews(props.input),
      Object.values(props.output)
    );
    for (const [name, view] of Object.entries(props.output)) {
      requireRasterLength(view, count, this.id, `output.${name}`);
    }
    validateCapacity(this.id, 'radius', props.radius, 0, MAXIMUM_RADIUS);
    validateFiniteNumber(this.id, 'quantile', props.quantile ?? 0.5, 0, 1);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      ...Object.values(output)
    ]);
    const source = getRasterBufferBandNodes(graph, id, props.input, grid, true);
    return [
      ...source.nodes,
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPURasterRankFilter',
        invocationCount: grid.width * grid.height,
        bindings: [
          {name: 'source', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'sourceValidity', view: source.band.validity!, type: 'u32', access: 'read'},
          {name: 'values', view: output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          ...(output.changed
            ? [
                {
                  name: 'changed',
                  view: output.changed,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `var samples: array<f32, ${(2 * props.radius + 1) ** 2}>; var sampleCount = 0u;
var valid = sourceValidity[sourceValidityOffset + index] != 0u;
for (var rowOffset = -${props.radius}; rowOffset <= ${props.radius}; rowOffset++) {
 for (var columnOffset = -${props.radius}; columnOffset <= ${props.radius}; columnOffset++) {
  var column = i32(index % ${grid.width}u) + columnOffset; var row = i32(index / ${grid.width}u) + rowOffset;
  if (column < 0 || row < 0 || column >= ${grid.width} || row >= ${grid.height}) {
   ${props.borderMode === 'nodata' ? `${props.noDataPolicy === 'ignore' ? '' : 'valid = false;'} continue;` : `column = clamp(column, 0, ${grid.width - 1}); row = clamp(row, 0, ${grid.height - 1});`}
  }
  let sampleIndex = u32(row) * ${grid.width}u + u32(column);
  if (sourceValidity[sourceValidityOffset + sampleIndex] == 0u) { ${props.noDataPolicy === 'ignore' ? '' : 'valid = false;'} continue; }
  let value = source[sourceOffset + sampleIndex]; var insertion = sampleCount;
  loop { if (insertion == 0u || samples[insertion-1u] <= value) { break; } samples[insertion] = samples[insertion-1u]; insertion--; }
  samples[insertion] = value; sampleCount++;
 }
}
valid = valid && sampleCount > 0u;
var result = bitcast<f32>(0x7fc00000u | (index & 0u));
if (valid) { result = samples[min(u32(floor(${(props.quantile ?? 0.5).toFixed(9)} * f32(sampleCount))), sampleCount-1u)]; }
values[valuesOffset+index] = result; validity[validityOffset+index] = u32(valid);
${output.changed ? 'changed[changedOffset+index] = u32(valid && result != source[sourceOffset+index]);' : ''}`
      })
    ];
  }
}
