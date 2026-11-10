// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

const OPERATION = 'GPUUtilizationVolumeIsopleths';

/** Rows summed by one invocation of a prefix level; three levels cover 256^3 cells. */
const BLOCK_SIZE = 256;

/**
 * Largest accepted cell count: three prefix levels of {@link BLOCK_SIZE} rows each, the last one a
 * single invocation (16,777,216 cells).
 */
export const GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS = BLOCK_SIZE ** 3;

/** Number of float32 elements in a {@link GPUUtilizationVolumeIsopleths} parameter view. */
export const GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUUtilizationVolumeIsopleths}. */
export type GPUUtilizationVolumeIsoplethsSettings = {
  /** Isopleth level in `(0, 1]`, for example 0.95 for the 95% home range. */
  level: number;
};

/**
 * Packs the float parameters of {@link GPUUtilizationVolumeIsopleths}.
 *
 * Layout: `[level, 0, 0, 0]` as float32.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPUUtilizationVolumeIsoplethsParameterValues(
  settings: GPUUtilizationVolumeIsoplethsSettings,
  target: Float32Array = new Float32Array(GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} parameter target must hold ${GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH} elements`
    );
  }
  validateFiniteNumber(OPERATION, 'level', settings.level, Number.MIN_VALUE, 1);
  target.set([settings.level, 0, 0, 0]);
  return target;
}

/** Properties for {@link GPUUtilizationVolumeIsopleths}. */
export type GPUUtilizationVolumeIsoplethsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'utilization-volume-isopleths'`. */
  id?: string;
  /**
   * Utilization density per cell (`GPUBrownianBridgeUtilization` output, or any non-negative
   * surface). Negative and NaN cells count as 0. Only the cell values matter, so a constant cell
   * area does not change the result. Per-frame contents.
   */
  density: GraphDataView<'float32'>;
  /** Per-frame parameters, at least 4 float32 rows, from {@link getGPUUtilizationVolumeIsoplethsParameterValues}. */
  parameters: GraphDataView<'float32'>;
  output: {
    /**
     * Cumulative share of the grid's total mass held by the cells at least as dense as this one,
     * including this cell: the volume of `getVolumeUD` in `move`. 0 for a cell of zero density.
     * Cells of equal density are ordered by cell index (the sort is stable), so a tie spreads
     * across consecutive volumes.
     */
    volume: GraphDataView<'float32'>;
    /**
     * 1 for the cells of the smallest set of densest cells whose mass reaches `level`: a cell is
     * in when the share of mass in the cells before it (in density order) is below `level` and its
     * own density is positive. 0 otherwise.
     */
    inside?: GraphDataView<'uint32'>;
    /** Optional one row: the total mass (sum of the cell values). */
    totalMass?: GraphDataView<'float32'>;
  };
};

/**
 * Volume isopleths of a utilization distribution: the cumulative probability volume of every cell
 * when cells are taken from the densest down (`getVolumeUD` of `move`), and the mask of the
 * `level` isopleth (the home range at that level).
 *
 * Method: a stable radix sort of the cells by density (descending), then an exclusive prefix sum of
 * the sorted densities by three levels of 256-row block sums, each level one bounded invocation per
 * block. The sums are float32 in a fixed order, so results are reproducible; the relative rounding
 * error of a volume is about 1e-6 for 1e6 cells. Per-frame: `density` and `parameters`.
 * Compile-time: the cell count and which outputs exist.
 *
 * Reference: tests compare with a binary64 sort and cumulative sum.
 */
export class GPUUtilizationVolumeIsopleths implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of cells. */
  readonly cellCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPUUtilizationVolumeIsoplethsProps>;

  constructor(props: GPUUtilizationVolumeIsoplethsProps) {
    const id = props.id ?? 'utilization-volume-isopleths';
    validatePackedView(props.density, ['float32'], `${id} density`);
    this.cellCount = props.density.length;
    validateCapacity(id, 'cellCount', this.cellCount, 2);
    validateMaximumCount(
      id,
      'cellCount',
      this.cellCount,
      GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS
    );
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH,
      `parameters must contain ${GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH} float32 rows`
    );
    const {output} = props;
    validatePackedView(output.volume, ['float32'], `${id} output.volume`);
    validateViewLength(id, 'output.volume', output.volume, this.cellCount);
    if (output.inside) {
      validatePackedView(output.inside, ['uint32'], `${id} output.inside`);
      validateViewLength(id, 'output.inside', output.inside, this.cellCount);
    }
    if (output.totalMass) {
      validatePackedView(output.totalMass, ['float32'], `${id} output.totalMass`);
      validateRequirement(
        id,
        output.totalMass.length >= 1,
        'output.totalMass must contain one row'
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.volume, output.inside, output.totalMass],
      [props.density, props.parameters]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, cellCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.density,
      props.parameters,
      output.volume,
      output.inside,
      output.totalMass
    ]);
    const blockCount = Math.ceil(cellCount / BLOCK_SIZE);
    const groupCount = Math.ceil(blockCount / BLOCK_SIZE);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', cellCount);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', cellCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', cellCount);
    const sortedCells = createTransientView(graph, `${id}-sorted-cells`, 'uint32', cellCount);
    const sortedValues = createTransientView(graph, `${id}-sorted-values`, 'float32', cellCount);
    const localPrefix = createTransientView(graph, `${id}-local-prefix`, 'float32', cellCount);
    const blockTotals = createTransientView(graph, `${id}-block-totals`, 'float32', blockCount);
    const blockPrefix = createTransientView(graph, `${id}-block-prefix`, 'float32', blockCount);
    const groupTotals = createTransientView(graph, `${id}-group-totals`, 'float32', groupCount);
    const groupPrefix = createTransientView(graph, `${id}-group-prefix`, 'float32', groupCount);
    const total = output.totalMass ?? createTransientView(graph, `${id}-total`, 'float32', 1);
    const common = `const BLOCK: u32 = ${BLOCK_SIZE}u;`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'keys',
        invocationCount: cellCount,
        bindings: [
          {name: 'density', view: props.density, type: 'f32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'cells', view: cells, type: 'u32', access: 'read_write'}
        ],
        body: /* wgsl */ `
  let value = density[densityOffset + index];
  // Non-negative floats order like their bit patterns; negative and NaN cells count as 0.
  keys[keysOffset + index] = select(0u, bitcast<u32>(value), value > 0.0 && value < 3.4e38);
  cells[cellsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: cells,
        outputKeys: sortedKeys,
        outputValues: sortedCells,
        direction: 'descending',
        algorithm: 'radix'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-values`,
        operation: OPERATION,
        variant: 'values',
        invocationCount: cellCount,
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedValues', view: sortedValues, type: 'f32', access: 'read_write'}
        ],
        body: 'sortedValues[sortedValuesOffset + index] = bitcast<f32>(sortedKeys[sortedKeysOffset + index]);'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-blocks`,
        operation: OPERATION,
        variant: 'level-0',
        invocationCount: blockCount,
        bindings: [
          {name: 'sortedValues', view: sortedValues, type: 'f32', access: 'read'},
          {name: 'localPrefix', view: localPrefix, type: 'f32', access: 'read_write'},
          {name: 'blockTotals', view: blockTotals, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: /* wgsl */ `
  var sum = 0.0;
  let end = min((index + 1u) * BLOCK, ${cellCount}u);
  for (var row = index * BLOCK; row < end; row = row + 1u) {
    localPrefix[localPrefixOffset + row] = sum;
    sum = sum + sortedValues[sortedValuesOffset + row];
  }
  blockTotals[blockTotalsOffset + index] = sum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-groups`,
        operation: OPERATION,
        variant: 'level-1',
        invocationCount: groupCount,
        bindings: [
          {name: 'blockTotals', view: blockTotals, type: 'f32', access: 'read'},
          {name: 'blockPrefix', view: blockPrefix, type: 'f32', access: 'read_write'},
          {name: 'groupTotals', view: groupTotals, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: /* wgsl */ `
  var sum = 0.0;
  let end = min((index + 1u) * BLOCK, ${blockCount}u);
  for (var block = index * BLOCK; block < end; block = block + 1u) {
    blockPrefix[blockPrefixOffset + block] = sum;
    sum = sum + blockTotals[blockTotalsOffset + block];
  }
  groupTotals[groupTotalsOffset + index] = sum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'level-2',
        invocationCount: 1,
        bindings: [
          {name: 'groupTotals', view: groupTotals, type: 'f32', access: 'read'},
          {name: 'groupPrefix', view: groupPrefix, type: 'f32', access: 'read_write'},
          {name: 'total', view: total, type: 'f32', access: 'read_write'}
        ],
        body: /* wgsl */ `
  var sum = 0.0;
  for (var group = 0u; group < ${groupCount}u; group = group + 1u) {
    groupPrefix[groupPrefixOffset + group] = sum;
    sum = sum + groupTotals[groupTotalsOffset + group];
  }
  total[totalOffset] = sum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        invocationCount: cellCount,
        bindings: [
          {name: 'localPrefix', view: localPrefix, type: 'f32', access: 'read_write'},
          {name: 'blockPrefix', view: blockPrefix, type: 'f32', access: 'read'},
          {name: 'groupPrefix', view: groupPrefix, type: 'f32', access: 'read'}
        ],
        declarations: common,
        body: /* wgsl */ `
  let block = index / BLOCK;
  localPrefix[localPrefixOffset + index] = localPrefix[localPrefixOffset + index] +
    blockPrefix[blockPrefixOffset + block] + groupPrefix[groupPrefixOffset + block / BLOCK];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-volume`,
        operation: OPERATION,
        variant: 'scatter',
        invocationCount: cellCount,
        bindings: [
          {name: 'sortedCells', view: sortedCells, type: 'u32', access: 'read'},
          {name: 'sortedValues', view: sortedValues, type: 'f32', access: 'read'},
          {name: 'before', view: localPrefix, type: 'f32', access: 'read'},
          {name: 'total', view: total, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'volume', view: output.volume, type: 'f32', access: 'read_write'},
          ...(output.inside
            ? [{name: 'inside', view: output.inside, type: 'u32', access: 'read_write'} as const]
            : [])
        ],
        body: /* wgsl */ `
  let value = sortedValues[sortedValuesOffset + index];
  let preceding = before[beforeOffset + index];
  let mass = total[totalOffset];
  let cell = sortedCells[sortedCellsOffset + index];
  let hasMass = mass > 0.0;
  volume[volumeOffset + cell] = select(0.0, min((preceding + value) / mass, 1.0), hasMass && value > 0.0);
${
  output.inside
    ? `  inside[insideOffset + cell] = select(0u, 1u, hasMass && value > 0.0 && preceding / mass < parameters[parametersOffset]);`
    : '  _ = parameters[parametersOffset];'
}`
      })
    ];
  }
}
