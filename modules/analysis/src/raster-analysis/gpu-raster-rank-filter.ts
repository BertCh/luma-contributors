// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from './raster-grid/raster-storage-preflight';
import {
  validateCapacity,
  validateFiniteNumber,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterHaloStage, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  snapshotRasterBand,
  snapshotRasterGrid,
  validateRasterOption,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph
} from './raster-band-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionPorts,
  validateRasterExpansionBand,
  validateRasterExpansionViews
} from './raster-expansion-utils';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';

/** Largest footprint radius; each cell insertion-sorts (2 radius + 1) squared values in private memory. */
const MAXIMUM_RADIUS = 8;

/** Caller-owned dense outputs of {@link GPURasterRankFilter}. */
export type GPURasterRankFilterOutput = {
  values: GraphDataView<'float32'>;
  validity: GraphDataView<'uint32'>;
  changed?: GraphDataView<'uint32'>;
};

export type GPURasterRankFilterProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBand;
  /** Square footprint radius in [0,8]. */
  radius: number;
  /** Quantile in [0,1], default 0.5. Selects floor(q * n), capped at n - 1 (SciPy rank convention). */
  quantile?: number;
  borderMode?: 'clamp' | 'nodata';
  /** Defaults to propagate; strict is an alias. ignore omits invalid neighbors, but
   * an invalid center always remains invalid.
   */
  noDataPolicy?: 'propagate' | 'strict' | 'ignore';
  output: GPURasterRankFilterOutput;
};

/**
 * Raster-native rank selection. Invalid centers remain invalid; omitted neighbors never become zeros.
 * `requiredHalo` is `radius`: execute on a halo-expanded source and keep only core outputs.
 */
export class GPURasterRankFilter implements GPUCommandNodeProducer, GPURasterHaloStage {
  readonly id: string;
  readonly props: GPURasterRankFilterProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract): the square footprint radius. */
  readonly requiredHalo: number;
  constructor(props: GPURasterRankFilterProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'input',
      'radius',
      'quantile',
      'borderMode',
      'noDataPolicy',
      'output'
    ]);
    props = {
      ...props,
      input: snapshotRasterBand(props.input),
      grid: snapshotRasterGrid(props.grid),
      output: snapshotConstructionFields(props.output, ['values', 'validity', 'changed'])
    };
    this.id = props.id ?? 'raster-rank-filter';
    validateRasterOption(this.id, 'borderMode', props.borderMode ?? 'clamp', ['clamp', 'nodata']);
    validateRasterOption(this.id, 'noDataPolicy', props.noDataPolicy ?? 'propagate', [
      'propagate',
      'strict',
      'ignore'
    ]);
    this.props = props;
    validateRasterTextureBand(this.id, props.input, props.grid);
    validateRasterExpansionBand(this.id, 'input', props.input);
    validateRasterExpansionPorts(
      this.id,
      props,
      {
        'output.values': ['float32'],
        'output.validity': ['uint32'],
        'output.changed': ['uint32']
      },
      ['output.changed']
    );
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
    this.requiredHalo = props.radius;
    validateFiniteNumber(this.id, 'quantile', props.quantile ?? 0.5, 0, 1);
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
      {'canonical band planes': storageCellCount * 4}
    );

    const {id, props} = this;
    const {grid, output} = props;
    validateRasterBandBelongsToGraph(id, graph, props.input, []);
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
