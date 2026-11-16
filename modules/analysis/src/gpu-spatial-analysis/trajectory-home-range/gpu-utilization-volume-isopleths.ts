// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateTrajectoryBindings} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  GPUSort,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateTrajectoryWritableDispatches} from '../trajectory-analysis/trajectory-binding-validation';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createUtilizationOffsetNode,
  createUtilizationPrefixNode,
  getUtilizationPrefixLayout,
  UTILIZATION_PREFIX_BLOCK_SIZE,
  UTILIZATION_PREFIX_PARAMETER_BYTE_LENGTH
} from './gpu-utilization-volume-isopleths-kernels';

const OPERATION = 'GPUUtilizationVolumeIsopleths';

/** Largest row count accepted by the published stable GPUSort; device bindings also limit it. */
export const GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS = 0x80000000;

// Eight radix digits, each with histogram/scatter and at most four two-node scan levels.
const SORT_COMMAND_NODE_ALLOWANCE = 8 * (2 + 2 * 4);

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
     * including this cell, so the densest cell holds its own share and the least dense positive
     * cell holds 1. A cell of zero density (also negative or NaN) holds 1: it is outside every
     * isopleth below the full one, and `volume <= level` selects the cells of `inside` other than
     * the one whose own mass crosses the level. Cells of equal density are ordered by cell index (the sort
     * is stable), so a tie spreads across consecutive volumes. R `move::getVolumeUD` is the same
     * idea; its exact convention (whether a cell's own mass is counted) was not checked here.
     */
    volume: GraphDataView<'float32'>;
    /**
     * 1 for the cells of the smallest set of densest cells whose mass reaches `level`: a cell is
     * in when the share of mass in the cells before it (in density order) is below `level` and its
     * own density is positive. 0 otherwise.
     */
    inside?: GraphDataView<'uint32'>;
    /** Optional one row: the fixed-order float32 total mass, with rounding at each tree level. */
    totalMass?: GraphDataView<'float32'>;
  };
};

/**
 * Volume isopleths of a utilization distribution: the cumulative probability volume of every cell
 * when cells are taken from the densest down, each cell's own mass included, and the mask of the
 * `level` isopleth (the home range at that level). The volume raster of R `move` (`getVolumeUD`)
 * serves the same purpose; no parity with it is claimed.
 *
 * Method: a stable radix sort of the cells by density (descending), then an exclusive prefix sum of
 * the sorted densities by recursive levels of 256-row block sums, each level one bounded invocation
 * per block. The original three levels and lowest-to-highest offset addition order are retained
 * within the old 16,777,216-cell range. Larger surfaces add levels until the root fits one block.
 * The sums are float32 in a fixed order, so results are reproducible; the relative rounding error
 * of a volume is about 1e-6 for 1e6 cells. Per-frame: `density` and `parameters`.
 * Compile-time: the cell count and which outputs exist.
 *
 * Reference: tests compare with a binary64 sort and cumulative sum.
 */
export class GPUUtilizationVolumeIsopleths implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of cells. */
  readonly cellCount: number;
  /** Upper bound including the published radix sort and its recursive scans. */
  readonly estimatedCommandNodeCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPUUtilizationVolumeIsoplethsProps>;

  constructor(props: GPUUtilizationVolumeIsoplethsProps) {
    const id = props.id ?? 'utilization-volume-isopleths';
    validateSinglePackedView(props.density, ['float32'], `${id} density`);
    this.cellCount = props.density.length;
    validateCapacity(id, 'cellCount', this.cellCount, 2);
    validateMaximumCount(
      id,
      'cellCount',
      this.cellCount,
      GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS
    );
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH,
      `parameters must contain ${GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH} float32 rows`
    );
    const {output} = props;
    validateSinglePackedView(output.volume, ['float32'], `${id} output.volume`);
    validateViewLength(id, 'output.volume', output.volume, this.cellCount);
    if (output.inside) {
      validateSinglePackedView(output.inside, ['uint32'], `${id} output.inside`);
      validateViewLength(id, 'output.inside', output.inside, this.cellCount);
    }
    if (output.totalMass) {
      validateSinglePackedView(output.totalMass, ['float32'], `${id} output.totalMass`);
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
    validateTrajectoryWritableDispatches(id, [
      {volume: output.volume, inside: output.inside},
      {totalMass: output.totalMass}
    ]);
    this.estimatedCommandNodeCount =
      SORT_COMMAND_NODE_ALLOWANCE +
      getUtilizationPrefixLayout(this.cellCount).lengths.length +
      4 +
      (output.totalMass ? 1 : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
    const {id, props, cellCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.density,
      props.parameters,
      output.volume,
      output.inside,
      output.totalMass
    ]);
    const layout = getUtilizationPrefixLayout(cellCount);
    for (const [name, view] of Object.entries({
      density: props.density,
      parameters: props.parameters,
      'output.volume': output.volume,
      'output.inside': output.inside,
      'output.totalMass': output.totalMass
    })) {
      if (view) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          getViewBindingRange(view).size,
          'reduce the view binding range'
        );
      }
    }
    // Radix histograms have sixteen buckets per 256-row workgroup; recursive scan buffers
    // are no larger than their histogram. Preflight every shape before declaration.
    for (const [name, byteLength] of [
      ['row and sort scratch', cellCount * 4],
      ['radix histograms and scan scratch', 16 * Math.ceil(cellCount / 256) * 4],
      ['upper prefixes', layout.upperLength * 4],
      ...layout.lengths.map(
        (length, level) =>
          [`level ${level} totals`, Math.ceil(length / UTILIZATION_PREFIX_BLOCK_SIZE) * 4] as const
      ),
      ['total', 4]
    ] as const) {
      validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, 'reduce cellCount');
    }
    validateRequirement(
      id,
      graph.device.limits.maxUniformBufferBindingSize >= UTILIZATION_PREFIX_PARAMETER_BYTE_LENGTH,
      `prefix parameters need ${UTILIZATION_PREFIX_PARAMETER_BYTE_LENGTH} uniform bytes`
    );
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', cellCount);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', cellCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', cellCount);
    const sortedCells = createTransientView(graph, `${id}-sorted-cells`, 'uint32', cellCount);
    const sortedValues = createTransientView(graph, `${id}-sorted-values`, 'float32', cellCount);
    const localPrefix = createTransientView(graph, `${id}-local-prefix`, 'float32', cellCount);
    const upperPrefix = createTransientView(
      graph,
      `${id}-upper-prefix`,
      'float32',
      layout.upperLength
    );
    // Keep the read-only scatter input private: callers may pack totalMass beside volume.
    // WebGPU forbids read-only and writable bindings of one physical buffer in a dispatch.
    const total = createTransientView(graph, `${id}-total`, 'float32', 1);
    const prefixNodes: GPUCommandNode<Parameters>[] = [];
    let input = sortedValues;
    for (const [level, length] of layout.lengths.entries()) {
      const prefix =
        level === 0
          ? localPrefix
          : graph.createDataView(upperPrefix.buffer, {
              format: 'float32',
              length,
              byteOffset: layout.upperOffsets[level - 1] * 4
            });
      const totals =
        level === layout.lengths.length - 1
          ? total
          : createTransientView(
              graph,
              `${id}-level-${level}-totals`,
              'float32',
              Math.ceil(length / UTILIZATION_PREFIX_BLOCK_SIZE)
            );
      prefixNodes.push(
        createUtilizationPrefixNode(graph, {
          id: `${id}-level-${level}`,
          input,
          prefix,
          totals
        })
      );
      input = totals;
    }
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
      ...prefixNodes,
      createUtilizationOffsetNode(graph, {
        id: `${id}-offsets`,
        localPrefix,
        upperPrefix,
        upperOffsets: layout.upperOffsets
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
  // A cell without mass is the last any isopleth reaches: volume 1, so 'volume <= level' leaves it out.
  volume[volumeOffset + cell] = select(1.0, min((preceding + value) / mass, 1.0), hasMass && value > 0.0);
${
  output.inside
    ? `  inside[insideOffset + cell] = select(0u, 1u, hasMass && value > 0.0 && preceding / mass < parameters[parametersOffset]);`
    : '  _ = parameters[parametersOffset];'
}`
      }),
      ...(output.totalMass
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-publish-total`,
              operation: OPERATION,
              variant: 'publish-total',
              invocationCount: 1,
              bindings: [
                {name: 'total', view: total, type: 'f32', access: 'read'},
                {name: 'totalMass', view: output.totalMass, type: 'f32', access: 'read_write'}
              ],
              body: 'totalMass[totalMassOffset] = total[totalOffset];'
            })
          ]
        : [])
    ];
  }
}
