// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from './gpu-command-graph';
import {getBoundedWorkgroupDispatchLayout, getBoundedDispatchOverrides} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBindingList
} from './gpu-kernel-node';
import {GPUHashIndexQuery, type GPUHashIndexView} from './gpu-hash-index';
import {GPUScan} from './gpu-scan';
import {
  createTransientView,
  createTransientVectorView,
  doGraphDataViewsOverlap,
  getViewElementOffset,
  validatePackedUint32View
} from './graph-data-view-utils';
import {alignGraphVectorViews, getGraphVectorData} from './graph-vector-view-utils';

const HASH_JOIN_WORKGROUP_SIZE = 256;
const MAXIMUM_UINT32 = 0xffffffff;

type DispatchLayout = {x: number; y: number; z: number};

/** Properties for a stable, capacity-bounded sparse inner join. */
export type GPUHashJoinProps = {
  id?: string;
  /** Previously built right-side key-to-row index. */
  index: GPUHashIndexView;
  /** Packed left-side keys in source order. */
  keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Optional left-side row IDs aligned with `keys`. */
  leftRows?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** First generated left row ID. Mutually exclusive with `leftRows`. */
  firstLeftRow?: number;
  /** Stable compacted left row IDs. Its length defines output capacity. */
  outputLeftRows: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Matched right row IDs aligned with `outputLeftRows`. */
  outputRightRows: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Required match count before capacity truncation. */
  count: GraphDataView<'uint32'>;
  /** Nonzero when `count` exceeds output capacity. */
  overflow: GraphDataView<'uint32'>;
  /** Four-row hash-query statistics block. */
  statistics: GraphDataView<'uint32'>;
  /** Optional source-aligned match mask. */
  found?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Optional source-aligned probe counts. */
  probes?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Defaults to the index probe bound. */
  maxProbeCount?: number;
};

/** CPU-visible storage and bounded-work facts for {@link GPUHashJoin}. */
export type GPUHashJoinStats = {
  inputLength: number;
  outputCapacity: number;
  maxProbeCount: number;
  outputByteLength: number;
};

/**
 * Resolves sparse right rows and stably compacts matching row pairs.
 *
 * `count` reports required capacity even when output buffers truncate publication. The workflow
 * adds lookup, scan, and bounded pair-scatter passes without submission or readback.
 */
export class GPUHashJoin {
  readonly id: string;
  readonly index: GPUHashIndexView;
  readonly keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly leftRows?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly firstLeftRow: number;
  readonly outputLeftRows: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly outputRightRows: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly count: GraphDataView<'uint32'>;
  readonly overflow: GraphDataView<'uint32'>;
  readonly statistics: GraphDataView<'uint32'>;
  readonly found?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly probes?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly maxProbeCount: number;
  readonly stats: GPUHashJoinStats;

  constructor(props: GPUHashJoinProps) {
    this.id = props.id ?? 'gpu-hash-join';
    this.index = props.index;
    this.keys = props.keys;
    this.leftRows = props.leftRows;
    this.firstLeftRow = props.firstLeftRow ?? 0;
    this.outputLeftRows = props.outputLeftRows;
    this.outputRightRows = props.outputRightRows;
    this.count = props.count;
    this.overflow = props.overflow;
    this.statistics = props.statistics;
    this.found = props.found;
    this.probes = props.probes;
    this.maxProbeCount = props.maxProbeCount ?? this.index.maxProbeCount;

    for (const [view, name] of [
      [this.index.tableKeys, 'index.tableKeys'],
      [this.index.tableValues, 'index.tableValues'],
      ...(this.index.statistics ? ([[this.index.statistics, 'index.statistics']] as const) : []),
      [this.keys, 'keys'],
      ...(this.leftRows ? ([[this.leftRows, 'leftRows']] as const) : []),
      [this.outputLeftRows, 'outputLeftRows'],
      [this.outputRightRows, 'outputRightRows'],
      [this.count, 'count'],
      [this.overflow, 'overflow'],
      [this.statistics, 'statistics'],
      ...(this.found ? ([[this.found, 'found']] as const) : []),
      ...(this.probes ? ([[this.probes, 'probes']] as const) : [])
    ] as const) {
      for (const chunk of getGraphVectorData(view)) {
        validatePackedUint32View(chunk, `${this.id} ${name}`);
      }
    }
    if (this.leftRows && this.leftRows.length !== this.keys.length) {
      throw new Error(`${this.id} leftRows length must match keys`);
    }
    if (this.leftRows && props.firstLeftRow !== undefined) {
      throw new Error(`${this.id} leftRows and firstLeftRow are mutually exclusive`);
    }
    if (this.outputLeftRows.length !== this.outputRightRows.length) {
      throw new Error(`${this.id} output capacities must match`);
    }
    if (this.outputLeftRows.length > MAXIMUM_UINT32) {
      throw new Error(`${this.id} output capacity must fit in uint32`);
    }
    if (
      this.index.tableKeys.length !== this.index.tableValues.length ||
      !Number.isInteger(Math.log2(this.index.tableKeys.length))
    ) {
      throw new Error(`${this.id} index must have matching positive power-of-two capacities`);
    }
    if (this.index.statistics && this.index.statistics.length < 6) {
      throw new Error(`${this.id} index statistics must contain six uint32 rows`);
    }
    if (
      !Number.isSafeInteger(this.maxProbeCount) ||
      this.maxProbeCount < 1 ||
      this.maxProbeCount > this.index.tableKeys.length
    ) {
      throw new Error(`${this.id} maxProbeCount must be an integer from one through capacity`);
    }
    if (this.keys.length * this.maxProbeCount > MAXIMUM_UINT32) {
      throw new Error(`${this.id} aggregate probe count must fit in uint32 statistics`);
    }
    if (this.count.length < 1 || this.overflow.length < 1) {
      throw new Error(`${this.id} count and overflow must contain one uint32 row`);
    }
    if (this.statistics.length < 4) {
      throw new Error(`${this.id} statistics must contain four uint32 rows`);
    }
    if (this.found && this.found.length !== this.keys.length) {
      throw new Error(`${this.id} found length must match keys`);
    }
    if (this.probes && this.probes.length !== this.keys.length) {
      throw new Error(`${this.id} probes length must match keys`);
    }
    if (
      !Number.isSafeInteger(this.firstLeftRow) ||
      this.firstLeftRow < 0 ||
      this.firstLeftRow > MAXIMUM_UINT32 ||
      (this.keys.length > 0 && this.firstLeftRow + this.keys.length - 1 > MAXIMUM_UINT32)
    ) {
      throw new Error(`${this.id} generated left rows must fit in uint32`);
    }

    const inputs = [
      this.index.tableKeys,
      this.index.tableValues,
      ...(this.index.statistics ? [this.index.statistics] : []),
      this.keys,
      ...(this.leftRows ? [this.leftRows] : [])
    ];
    const outputs = [
      this.outputLeftRows,
      this.outputRightRows,
      this.count,
      this.overflow,
      this.statistics,
      ...(this.found ? [this.found] : []),
      ...(this.probes ? [this.probes] : [])
    ];
    validateDisjointViews(this.id, inputs, outputs);

    this.stats = Object.freeze({
      inputLength: this.keys.length,
      outputCapacity: this.outputLeftRows.length,
      maxProbeCount: this.maxProbeCount,
      outputByteLength:
        this.outputLeftRows.length * 8 +
        (2 + 4 + (this.found ? this.keys.length : 0) + (this.probes ? this.keys.length : 0)) * 4
    });
  }

  /** Adds lookup, exclusive scan, and stable bounded pair publication to a graph. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    const views = [
      this.index.tableKeys,
      this.index.tableValues,
      ...(this.index.statistics ? [this.index.statistics] : []),
      this.keys,
      ...(this.leftRows ? [this.leftRows] : []),
      this.outputLeftRows,
      this.outputRightRows,
      this.count,
      this.overflow,
      this.statistics,
      ...(this.found ? [this.found] : []),
      ...(this.probes ? [this.probes] : [])
    ];
    if (views.flatMap(getGraphVectorData).some(view => view.buffer.graph !== graph)) {
      throw new Error(`${this.id} views must belong to the target graph`);
    }

    const createScratch = (id: string) =>
      this.keys instanceof GraphVectorView
        ? createTransientVectorView(graph, id, this.keys)
        : createTransientView(graph, id, 'uint32', this.keys.length);
    const matchedRightRows = createScratch(`${this.id}-matched-right-rows`);
    const found = this.found ?? createScratch(`${this.id}-found`);
    const probes = this.probes ?? createScratch(`${this.id}-probes`);
    nodes.push(
      ...new GPUHashIndexQuery({
        id: `${this.id}-lookup`,
        index: this.index,
        keys: this.keys,
        values: matchedRightRows,
        found,
        probes,
        statistics: this.statistics,
        maxProbeCount: this.maxProbeCount
      }).getCommandNodes(graph)
    );

    if (this.keys.length === 0) {
      nodes.push(...addEmptyJoinPass(graph, this));
    } else {
      const offsets = createScratch(`${this.id}-offsets`);
      nodes.push(
        ...new GPUScan({id: `${this.id}-scan`, input: found, output: offsets}).getCommandNodes(
          graph
        )
      );
      // Global scan offsets route stable pairs into independently partitioned destinations.
      const sources = alignGraphVectorViews(graph, [
        matchedRightRows,
        found,
        offsets,
        this.leftRows ?? matchedRightRows
      ]);
      const destinations = alignGraphVectorViews(graph, [
        this.outputLeftRows,
        this.outputRightRows
      ]);
      let outputStart = 0;
      for (const [destinationIndex, [outputLeftRows, outputRightRows]] of destinations.entries()) {
        let inputStart = 0;
        for (const [sourceIndex, [matchedRows, flags, positions, leftRows]] of sources.entries()) {
          nodes.push(
            ...addJoinScatterPass(
              graph,
              {
                id: `${this.id}-source-${sourceIndex}-destination-${destinationIndex}`,
                firstLeftRow: this.firstLeftRow + inputStart,
                leftRows: this.leftRows ? leftRows : undefined,
                outputLeftRows,
                outputRightRows,
                outputStart
              },
              matchedRows,
              flags,
              positions
            )
          );
          inputStart += matchedRows.length;
        }
        outputStart += outputLeftRows.length;
      }
      // Publish the global count once, even when output capacity is zero.
      const [, lastFound, lastOffsets] = sources[sources.length - 1];
      nodes.push(...addJoinCountOnlyPass(graph, this, lastFound, lastOffsets));
    }
    if (this.index.statistics) {
      nodes.push(...addSourceOverflowPass(graph, this.id, this.index.statistics, this.overflow));
    }

    return nodes;
  }
}

function addEmptyJoinPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  join: GPUHashJoin
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const bindings: GPUKernelBindingList = [
    {name: 'count', view: join.count, access: 'write'},
    {name: 'overflow', view: join.overflow, access: 'write'}
  ];
  const constants = {
    COUNT_OFFSET: getViewElementOffset(join.count),
    OVERFLOW_OFFSET: getViewElementOffset(join.overflow)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() {
  count[COUNT_OFFSET] = 0u;
  overflow[OVERFLOW_OFFSET] = 0u;
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${join.id}-empty`,
      source,
      bindings,
      constants,
      dispatchSize: {x: 1, y: 1, z: 1},
      record: {moduleId: 'GPUHashJoin/empty', invocationCount: 1, workgroupSize: 1}
    })
  );

  return nodes;
}

function addJoinScatterPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  join: {
    id: string;
    firstLeftRow: number;
    leftRows?: GraphDataView<'uint32'>;
    outputLeftRows: GraphDataView<'uint32'>;
    outputRightRows: GraphDataView<'uint32'>;
    outputStart: number;
  },
  matchedRightRows: GraphDataView<'uint32'>,
  found: GraphDataView<'uint32'>,
  offsets: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const layout = getDispatchLayout(
    matchedRightRows.length,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const leftRowExpression = join.leftRows
    ? 'leftRows[LEFT_ROWS_OFFSET + inputIndex]'
    : 'FIRST_LEFT_ROW + inputIndex';
  const bindings: GPUKernelBindingList = [
    {name: 'matchedRightRows', view: matchedRightRows, access: 'read'},
    {name: 'found', view: found, access: 'read'},
    {name: 'offsets', view: offsets, access: 'read'},
    join.leftRows && {name: 'leftRows', view: join.leftRows, access: 'read'},
    {name: 'outputLeftRows', view: join.outputLeftRows, access: 'write'},
    {name: 'outputRightRows', view: join.outputRightRows, access: 'write'}
  ];
  const constants: Record<string, number> = {
    ELEMENT_COUNT: matchedRightRows.length,
    OUTPUT_START: join.outputStart,
    OUTPUT_END: join.outputStart + join.outputLeftRows.length,
    MATCHED_ROWS_OFFSET: getViewElementOffset(matchedRightRows),
    FOUND_OFFSET: getViewElementOffset(found),
    OFFSETS_OFFSET: getViewElementOffset(offsets),
    OUTPUT_LEFT_OFFSET: getViewElementOffset(join.outputLeftRows),
    OUTPUT_RIGHT_OFFSET: getViewElementOffset(join.outputRightRows),
    ...(join.leftRows
      ? {LEFT_ROWS_OFFSET: getViewElementOffset(join.leftRows)}
      : {FIRST_LEFT_ROW: join.firstLeftRow}),
    ...getBoundedDispatchOverrides(layout)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

@compute @workgroup_size(${HASH_JOIN_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let inputIndex = workgroupIndex * ${HASH_JOIN_WORKGROUP_SIZE}u + localIndex;
  if (inputIndex >= ELEMENT_COUNT) { return; }
  let accepted = min(found[FOUND_OFFSET + inputIndex], 1u);
  let outputIndex = offsets[OFFSETS_OFFSET + inputIndex];
  if (accepted != 0u && outputIndex >= OUTPUT_START && outputIndex < OUTPUT_END) {
    outputLeftRows[OUTPUT_LEFT_OFFSET + outputIndex - OUTPUT_START] = ${leftRowExpression};
    outputRightRows[OUTPUT_RIGHT_OFFSET + outputIndex - OUTPUT_START] =
      matchedRightRows[MATCHED_ROWS_OFFSET + inputIndex];
  }
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${join.id}-scatter`,
      source,
      bindings,
      constants,
      dispatchSize: layout,
      record: {moduleId: 'GPUHashJoin/scatter', invocationCount: matchedRightRows.length}
    })
  );

  return nodes;
}

function addJoinCountOnlyPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  join: GPUHashJoin,
  found: GraphDataView<'uint32'>,
  offsets: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const bindings: GPUKernelBindingList = [
    {name: 'found', view: found, access: 'read'},
    {name: 'offsets', view: offsets, access: 'read'},
    {name: 'count', view: join.count, access: 'write'},
    {name: 'overflow', view: join.overflow, access: 'write'}
  ];
  const constants = {
    LAST_INDEX: found.length - 1,
    OUTPUT_CAPACITY: join.outputLeftRows.length,
    FOUND_OFFSET: getViewElementOffset(found),
    OFFSETS_OFFSET: getViewElementOffset(offsets),
    COUNT_OFFSET: getViewElementOffset(join.count),
    OVERFLOW_OFFSET: getViewElementOffset(join.overflow)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

@compute @workgroup_size(1) fn main() {
  let requiredCount = offsets[OFFSETS_OFFSET + LAST_INDEX] +
    min(found[FOUND_OFFSET + LAST_INDEX], 1u);
  count[COUNT_OFFSET] = requiredCount;
  overflow[OVERFLOW_OFFSET] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY);
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${join.id}-count-only`,
      source,
      bindings,
      constants,
      dispatchSize: {x: 1, y: 1, z: 1},
      record: {moduleId: 'GPUHashJoin/count-only', invocationCount: 1, workgroupSize: 1}
    })
  );

  return nodes;
}

function addSourceOverflowPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  indexStatistics: GraphDataView<'uint32'>,
  overflow: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const bindings: GPUKernelBindingList = [
    {name: 'indexStatistics', view: indexStatistics, access: 'read'},
    {name: 'overflow', view: overflow, access: 'read_write'}
  ];
  const constants = {
    INDEX_STATISTICS_OFFSET: getViewElementOffset(indexStatistics),
    OVERFLOW_OFFSET: getViewElementOffset(overflow)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(1) fn main() {
  if (indexStatistics[INDEX_STATISTICS_OFFSET + 2u] != 0u) {
    overflow[OVERFLOW_OFFSET] = 1u;
  }
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${id}-source-overflow`,
      source,
      bindings,
      constants,
      dispatchSize: {x: 1, y: 1, z: 1},
      record: {moduleId: 'GPUHashJoin/source-overflow', invocationCount: 1, workgroupSize: 1}
    })
  );

  return nodes;
}

function validateDisjointViews(
  id: string,
  inputVectors: readonly (GraphDataView<'uint32'> | GraphVectorView<'uint32'>)[],
  outputVectors: readonly (GraphDataView<'uint32'> | GraphVectorView<'uint32'>)[]
): void {
  const inputs = inputVectors.flatMap(getGraphVectorData);
  const outputs = outputVectors.flatMap(getGraphVectorData);
  for (const input of inputs) {
    for (const output of outputs) {
      if (doGraphDataViewsOverlap(input, output)) {
        throw new Error(`${id} input and output views must not overlap`);
      }
    }
  }
  for (let first = 0; first < outputs.length; first++) {
    for (let second = first + 1; second < outputs.length; second++) {
      if (doGraphDataViewsOverlap(outputs[first], outputs[second])) {
        throw new Error(`${id} output views must not overlap`);
      }
    }
  }
}

function getDispatchLayout(elementCount: number, maximumDimension: number): DispatchLayout {
  return getBoundedWorkgroupDispatchLayout(
    'GPUHashJoin',
    Math.ceil(elementCount / HASH_JOIN_WORKGROUP_SIZE),
    maximumDimension
  );
}

function addKernelPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    source: string;
    bindings: GPUKernelBindingList;
    constants: Readonly<Record<string, number>>;
    dispatchSize: DispatchLayout;
    record: {moduleId: string; invocationCount: number; workgroupSize?: number};
  }
): readonly GPUCommandNode<Parameters>[] {
  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUHashJoin',
      moduleId: props.record.moduleId,
      source: props.source,
      constants: props.constants,
      bindings: props.bindings,
      workgroupSize: props.record.workgroupSize ?? HASH_JOIN_WORKGROUP_SIZE,
      dispatchLayout: props.dispatchSize,
      invocationCount: props.record.invocationCount,
      workloadScope: 'never'
    })
  ];
}
