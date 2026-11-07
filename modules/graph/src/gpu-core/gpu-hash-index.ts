// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {HASH_UINT32_WGSL} from '../utils/hash-uint32';
import {type GPUCommandNode} from './gpu-command-node';
import {GPUCommandGraph, type GraphVectorView, type GraphDataView} from './gpu-command-graph';
import {getBoundedWorkgroupDispatchLayout, getBoundedDispatchOverrides} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBindingList
} from './gpu-kernel-node';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewElementOffset,
  validatePackedUint32View
} from './graph-data-view-utils';
import {alignGraphVectorViews, getGraphVectorData} from './graph-vector-view-utils';

const HASH_INDEX_WORKGROUP_SIZE = 256;
const HASH_INDEX_COMPARE_EXCHANGE_RETRY_COUNT = 4;
const MAXIMUM_UINT32 = 0xffffffff;

/** Empty table-key marker. Input rows containing this key are counted as invalid. */
export const GPU_HASH_INDEX_EMPTY_KEY = MAXIMUM_UINT32;
/** `[unique, duplicates, overflow, invalid, total probes, maximum probes]`. */
export const GPU_HASH_INDEX_STATISTICS_LENGTH = 6;
/** `[found, missing, total probes, maximum probes]`. */
export const GPU_HASH_QUERY_STATISTICS_LENGTH = 4;

type DispatchLayout = {x: number; y: number; z: number};

/** Storage contract consumed by {@link GPUHashIndexQuery}. */
export type GPUHashIndexView = {
  tableKeys: GraphDataView<'uint32'>;
  tableValues: GraphDataView<'uint32'>;
  /** Optional six-row build statistics used by consumers that propagate source overflow. */
  statistics?: GraphDataView<'uint32'>;
  maxProbeCount: number;
};

/** One preserved packed source chunk contributed to a shared hash-index rebuild. @internal */
export type GPUHashIndexBuildBatch = {
  keys: GraphDataView<'uint32'>;
  values?: GraphDataView<'uint32'>;
  validity?: GraphDataView<'uint32'>;
  firstValue: number;
};

type GPUHashIndexBuildTarget = GPUHashIndexView & {
  id: string;
  statistics: GraphDataView<'uint32'>;
};

type GPUHashIndexBuildPass = GPUHashIndexBuildBatch & {
  id: string;
  firstSourceRow: number;
};

/** Properties for one fixed-capacity hash-index rebuild from a logical key sequence. */
export type GPUHashIndexProps = {
  id?: string;
  /** Packed keys. `0xffffffff` is reserved and ignored. */
  keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Optional packed values aligned with keys. Generated row IDs are used by default. */
  values?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** First generated row ID. Mutually exclusive with `values`. */
  firstValue?: number;
  /** Caller-owned power-of-two key table. Its length defines capacity. */
  tableKeys: GraphDataView<'uint32'>;
  /** Caller-owned values aligned with `tableKeys`. */
  tableValues: GraphDataView<'uint32'>;
  /** Caller-owned six-row build-statistics block. */
  statistics: GraphDataView<'uint32'>;
  /** Maximum slots examined per input. Defaults to table capacity. */
  maxProbeCount?: number;
};

/** CPU-visible allocation and bounded-work facts. */
export type GPUHashIndexStats = {
  capacity: number;
  maxProbeCount: number;
  tableByteLength: number;
  statisticsByteLength: number;
  outputByteLength: number;
};

/**
 * Rebuilds a packed `uint32` key/value hash index with bounded linear probing.
 *
 * Duplicate keys deterministically retain the value from the lowest source row. A full or
 * probe-limited table reports overflow; which distinct keys survive overflow is unspecified.
 */
export class GPUHashIndex implements GPUHashIndexView {
  readonly id: string;
  readonly keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly values?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly firstValue: number;
  readonly tableKeys: GraphDataView<'uint32'>;
  readonly tableValues: GraphDataView<'uint32'>;
  readonly statistics: GraphDataView<'uint32'>;
  readonly maxProbeCount: number;
  readonly stats: GPUHashIndexStats;
  readonly updatePolicy = 'rebuild' as const;

  constructor(props: GPUHashIndexProps) {
    this.id = props.id ?? 'gpu-hash-index';
    this.keys = props.keys;
    this.values = props.values;
    this.firstValue = props.firstValue ?? 0;
    this.tableKeys = props.tableKeys;
    this.tableValues = props.tableValues;
    this.statistics = props.statistics;
    this.maxProbeCount = props.maxProbeCount ?? this.tableKeys.length;

    for (const [view, name] of [
      [this.keys, 'keys'],
      ...(this.values ? ([[this.values, 'values']] as const) : []),
      [this.tableKeys, 'tableKeys'],
      [this.tableValues, 'tableValues'],
      [this.statistics, 'statistics']
    ] as const) {
      for (const chunk of getGraphVectorData(view)) {
        validatePackedUint32View(chunk, `${this.id} ${name}`);
      }
    }
    if (!isPowerOfTwo(this.tableKeys.length)) {
      throw new Error(`${this.id} table capacity must be a positive power of two`);
    }
    if (this.tableKeys.length > MAXIMUM_UINT32) {
      throw new Error(`${this.id} table capacity must fit in uint32`);
    }
    if (this.tableValues.length !== this.tableKeys.length) {
      throw new Error(`${this.id} table key and value capacities must match`);
    }
    if (this.statistics.length < GPU_HASH_INDEX_STATISTICS_LENGTH) {
      throw new Error(`${this.id} statistics must contain six uint32 rows`);
    }
    if (this.values && this.values.length !== this.keys.length) {
      throw new Error(`${this.id} keys and values lengths must match`);
    }
    if (this.values && props.firstValue !== undefined) {
      throw new Error(`${this.id} values and firstValue are mutually exclusive`);
    }
    if (
      !Number.isSafeInteger(this.firstValue) ||
      this.firstValue < 0 ||
      this.firstValue > MAXIMUM_UINT32 ||
      (this.keys.length > 0 && this.firstValue + this.keys.length - 1 > MAXIMUM_UINT32)
    ) {
      throw new Error(`${this.id} generated values must fit in uint32`);
    }
    validateProbeCount(this.id, this.maxProbeCount, this.tableKeys.length, this.keys.length);
    validateDisjointViews(
      this.id,
      [this.keys, ...(this.values ? [this.values] : [])],
      [this.tableKeys, this.tableValues, this.statistics]
    );

    this.stats = Object.freeze({
      capacity: this.tableKeys.length,
      maxProbeCount: this.maxProbeCount,
      tableByteLength: this.tableKeys.length * 8,
      statisticsByteLength: GPU_HASH_INDEX_STATISTICS_LENGTH * 4,
      outputByteLength: this.tableKeys.length * 8 + GPU_HASH_INDEX_STATISTICS_LENGTH * 4
    });
  }

  /** Adds initialization, insertion, and deterministic value finalization to a command graph. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    const views = [
      this.keys,
      ...(this.values ? [this.values] : []),
      this.tableKeys,
      this.tableValues,
      this.statistics
    ];
    if (views.flatMap(getGraphVectorData).some(view => view.buffer.graph !== graph)) {
      throw new Error(`${this.id} views must belong to the target graph`);
    }

    // Borrow intersections so explicit values need not share key chunk boundaries.
    const spans = this.values
      ? alignGraphVectorViews(graph, [this.keys, this.values])
      : alignGraphVectorViews(graph, [this.keys]);
    let firstValue = this.firstValue;
    const batches: GPUHashIndexBuildBatch[] = [];
    for (const [keys, values] of spans) {
      batches.push({keys, values, firstValue});
      firstValue += keys.length;
    }
    nodes.push(...getGPUHashIndexBuildBatchesCommandNodes(graph, this, batches));

    return nodes;
  }
}

/** Rebuilds one shared index from ordered source chunks without concatenating their buffers. @internal */
export function getGPUHashIndexBuildBatchesCommandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  index: GPUHashIndexBuildTarget,
  batches: readonly GPUHashIndexBuildBatch[]
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sourceRows = createTransientView(
    graph,
    `${index.id}-source-rows`,
    'uint32',
    index.tableKeys.length
  );
  nodes.push(...addBuildInitializePass(graph, index, sourceRows));

  let firstSourceRow = 0;
  for (const [batchIndex, batch] of batches.entries()) {
    if (batch.keys.length > 0) {
      const pass: GPUHashIndexBuildPass = {
        ...batch,
        id: `${index.id}-batch-${batchIndex}`,
        firstSourceRow
      };
      nodes.push(...addBuildPass(graph, index, sourceRows, pass));
      nodes.push(...addBuildFinalizePass(graph, index, sourceRows, pass));
    }
    firstSourceRow += batch.keys.length;
  }

  return nodes;
}

/** Properties for bounded lookup across independently partitioned query columns. */
export type GPUHashIndexQueryProps = {
  id?: string;
  index: GPUHashIndexView;
  keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned values aligned with query keys. Missing values become `0xffffffff`. */
  values: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned nonzero/zero lookup results aligned with query keys. */
  found: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned number of examined slots per query. */
  probes: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned four-row query-statistics block. */
  statistics: GraphDataView<'uint32'>;
  /** Defaults to the index build probe bound. */
  maxProbeCount?: number;
};

/** Performs bounded hash-index lookups across one logical query sequence. */
export class GPUHashIndexQuery {
  readonly id: string;
  readonly index: GPUHashIndexView;
  readonly keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly values: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly found: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly probes: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly statistics: GraphDataView<'uint32'>;
  readonly maxProbeCount: number;

  constructor(props: GPUHashIndexQueryProps) {
    this.id = props.id ?? 'gpu-hash-index-query';
    this.index = props.index;
    this.keys = props.keys;
    this.values = props.values;
    this.found = props.found;
    this.probes = props.probes;
    this.statistics = props.statistics;
    this.maxProbeCount = props.maxProbeCount ?? this.index.maxProbeCount;

    for (const [view, name] of [
      [this.index.tableKeys, 'index.tableKeys'],
      [this.index.tableValues, 'index.tableValues'],
      [this.keys, 'keys'],
      [this.values, 'values'],
      [this.found, 'found'],
      [this.probes, 'probes'],
      [this.statistics, 'statistics']
    ] as const) {
      for (const chunk of getGraphVectorData(view)) {
        validatePackedUint32View(chunk, `${this.id} ${name}`);
      }
    }
    if (this.index.tableKeys.length !== this.index.tableValues.length) {
      throw new Error(`${this.id} index table key and value capacities must match`);
    }
    if (!isPowerOfTwo(this.index.tableKeys.length)) {
      throw new Error(`${this.id} index capacity must be a positive power of two`);
    }
    if (
      this.values.length !== this.keys.length ||
      this.found.length !== this.keys.length ||
      this.probes.length !== this.keys.length
    ) {
      throw new Error(`${this.id} query outputs must match key length`);
    }
    if (this.statistics.length < GPU_HASH_QUERY_STATISTICS_LENGTH) {
      throw new Error(`${this.id} statistics must contain four uint32 rows`);
    }
    validateProbeCount(this.id, this.maxProbeCount, this.index.tableKeys.length, this.keys.length);
    validateDisjointViews(
      this.id,
      [this.index.tableKeys, this.index.tableValues, this.keys],
      [this.values, this.found, this.probes, this.statistics]
    );
  }

  /** Adds statistics initialization and bounded lookup spans without submission or readback. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    const views = [
      this.index.tableKeys,
      this.index.tableValues,
      this.keys,
      this.values,
      this.found,
      this.probes,
      this.statistics
    ];
    if (views.flatMap(getGraphVectorData).some(view => view.buffer.graph !== graph)) {
      throw new Error(`${this.id} views must belong to the target graph`);
    }
    nodes.push(...addQueryInitializePass(graph, this));
    const spans = alignGraphVectorViews(graph, [this.keys, this.values, this.found, this.probes]);
    for (const [spanIndex, [keys, values, found, probes]] of spans.entries()) {
      nodes.push(
        ...addQueryPass(graph, {
          ...this,
          id: `${this.id}-span-${spanIndex}`,
          keys,
          values,
          found,
          probes
        })
      );
    }

    return nodes;
  }
}

function addBuildInitializePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  index: GPUHashIndexBuildTarget,
  sourceRows: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const layout = getDispatchLayout(
    Math.max(index.tableKeys.length, GPU_HASH_INDEX_STATISTICS_LENGTH),
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const bindings: GPUKernelBindingList = [
    {name: 'tableKeys', view: index.tableKeys, access: 'write'},
    {name: 'tableValues', view: index.tableValues, access: 'write'},
    {name: 'sourceRows', view: sourceRows, access: 'write'},
    {name: 'statistics', view: index.statistics, access: 'write'}
  ];
  const constants = {
    CAPACITY: index.tableKeys.length,
    TABLE_KEYS_OFFSET: getViewElementOffset(index.tableKeys),
    TABLE_VALUES_OFFSET: getViewElementOffset(index.tableValues),
    SOURCE_ROWS_OFFSET: getViewElementOffset(sourceRows),
    STATISTICS_OFFSET: getViewElementOffset(index.statistics),
    ...getBoundedDispatchOverrides(layout)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(${HASH_INDEX_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let elementIndex = workgroupIndex * ${HASH_INDEX_WORKGROUP_SIZE}u + localIndex;
  if (elementIndex < CAPACITY) {
    tableKeys[TABLE_KEYS_OFFSET + elementIndex] = ${GPU_HASH_INDEX_EMPTY_KEY}u;
    tableValues[TABLE_VALUES_OFFSET + elementIndex] = ${GPU_HASH_INDEX_EMPTY_KEY}u;
    sourceRows[SOURCE_ROWS_OFFSET + elementIndex] = ${GPU_HASH_INDEX_EMPTY_KEY}u;
  }
  if (elementIndex < ${GPU_HASH_INDEX_STATISTICS_LENGTH}u) {
    statistics[STATISTICS_OFFSET + elementIndex] = 0u;
  }
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${index.id}-initialize`,
      source,
      bindings,
      constants,
      dispatchSize: layout,
      record: {moduleId: 'GPUHashIndex/initialize', invocationCount: index.tableKeys.length}
    })
  );

  return nodes;
}

function addBuildPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  index: GPUHashIndexBuildTarget,
  sourceRows: GraphDataView<'uint32'>,
  batch: GPUHashIndexBuildPass
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const layout = getDispatchLayout(
    batch.keys.length,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const bindings: GPUKernelBindingList = [
    {name: 'inputKeys', view: batch.keys, access: 'read'},
    batch.validity && {name: 'inputValidity', view: batch.validity, access: 'read'},
    {name: 'tableKeys', view: index.tableKeys, access: 'read_write', atomic: true},
    {name: 'sourceRows', view: sourceRows, access: 'read_write', atomic: true},
    {name: 'statistics', view: index.statistics, access: 'read_write', atomic: true}
  ];
  // Row counts, capacity, probe limit, and view offsets are pipeline constants.
  const constants: Record<string, number> = {
    ELEMENT_COUNT: batch.keys.length,
    CAPACITY_MASK: index.tableKeys.length - 1,
    MAX_PROBES: index.maxProbeCount,
    KEYS_OFFSET: getViewElementOffset(batch.keys),
    TABLE_KEYS_OFFSET: getViewElementOffset(index.tableKeys),
    SOURCE_ROWS_OFFSET: getViewElementOffset(sourceRows),
    STATISTICS_OFFSET: getViewElementOffset(index.statistics),
    FIRST_SOURCE_ROW: batch.firstSourceRow,
    ...(batch.validity && {VALIDITY_OFFSET: getViewElementOffset(batch.validity)}),
    ...getBoundedDispatchOverrides(layout)
  };
  const rejectInvalid = batch.validity
    ? 'if (inputValidity[VALIDITY_OFFSET + inputIndex] == 0u) { return; }'
    : '';
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

${HASH_UINT32_WGSL}

@compute @workgroup_size(${HASH_INDEX_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let inputIndex = workgroupIndex * ${HASH_INDEX_WORKGROUP_SIZE}u + localIndex;
  if (inputIndex >= ELEMENT_COUNT) { return; }
  ${rejectInvalid}
  let key = inputKeys[KEYS_OFFSET + inputIndex];
  if (key == ${GPU_HASH_INDEX_EMPTY_KEY}u) {
    atomicAdd(&statistics[STATISTICS_OFFSET + 3u], 1u);
    return;
  }
  let start = hashUint32(key) & CAPACITY_MASK;
  var probes = 0u;
  var inserted = false;
  var duplicate = false;
  for (var probe = 0u; probe < MAX_PROBES; probe++) {
    probes = probe + 1u;
    let slot = (start + probe) & CAPACITY_MASK;
    var result = atomicCompareExchangeWeak(
      &tableKeys[TABLE_KEYS_OFFSET + slot], ${GPU_HASH_INDEX_EMPTY_KEY}u, key
    );
    // Bound spurious weak-CAS retries without probing onward and risking a duplicate slot.
    var exchangeAttempt = 1u;
    while (
      !result.exchanged &&
      result.old_value == ${GPU_HASH_INDEX_EMPTY_KEY}u &&
      exchangeAttempt < ${HASH_INDEX_COMPARE_EXCHANGE_RETRY_COUNT}u
    ) {
      result = atomicCompareExchangeWeak(
        &tableKeys[TABLE_KEYS_OFFSET + slot], ${GPU_HASH_INDEX_EMPTY_KEY}u, key
      );
      exchangeAttempt++;
    }
    if (!result.exchanged && result.old_value == ${GPU_HASH_INDEX_EMPTY_KEY}u) {
      break;
    }
    if (result.exchanged || result.old_value == key) {
      atomicMin(&sourceRows[SOURCE_ROWS_OFFSET + slot], FIRST_SOURCE_ROW + inputIndex);
      inserted = result.exchanged;
      duplicate = !result.exchanged;
      break;
    }
  }
  atomicAdd(&statistics[STATISTICS_OFFSET + 4u], probes);
  atomicMax(&statistics[STATISTICS_OFFSET + 5u], probes);
  if (inserted) {
    atomicAdd(&statistics[STATISTICS_OFFSET], 1u);
  } else if (duplicate) {
    atomicAdd(&statistics[STATISTICS_OFFSET + 1u], 1u);
  } else {
    atomicAdd(&statistics[STATISTICS_OFFSET + 2u], 1u);
  }
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${batch.id}-build`,
      source,
      bindings,
      constants,
      dispatchSize: layout,
      record: {moduleId: 'GPUHashIndex/build', invocationCount: batch.keys.length}
    })
  );

  return nodes;
}

function addBuildFinalizePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  index: GPUHashIndexBuildTarget,
  sourceRows: GraphDataView<'uint32'>,
  batch: GPUHashIndexBuildPass
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const layout = getDispatchLayout(
    index.tableKeys.length,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const hasExplicitValues = Boolean(batch.values && batch.keys.length > 0);
  const valueExpression = hasExplicitValues
    ? 'inputValues[VALUES_OFFSET + localSourceRow]'
    : 'FIRST_VALUE + localSourceRow';
  const bindings: GPUKernelBindingList = [
    {name: 'tableKeys', view: index.tableKeys, access: 'read'},
    {name: 'sourceRows', view: sourceRows, access: 'read'},
    {name: 'tableValues', view: index.tableValues, access: 'write'},
    hasExplicitValues && {name: 'inputValues', view: batch.values!, access: 'read'}
  ];
  const constants: Record<string, number> = {
    CAPACITY: index.tableKeys.length,
    FIRST_SOURCE_ROW: batch.firstSourceRow,
    SOURCE_ROW_COUNT: batch.keys.length,
    TABLE_KEYS_OFFSET: getViewElementOffset(index.tableKeys),
    TABLE_VALUES_OFFSET: getViewElementOffset(index.tableValues),
    SOURCE_ROWS_OFFSET: getViewElementOffset(sourceRows),
    ...(hasExplicitValues
      ? {VALUES_OFFSET: getViewElementOffset(batch.values!)}
      : {FIRST_VALUE: batch.firstValue}),
    ...getBoundedDispatchOverrides(layout)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(${HASH_INDEX_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let slot = workgroupIndex * ${HASH_INDEX_WORKGROUP_SIZE}u + localIndex;
  if (slot >= CAPACITY || tableKeys[TABLE_KEYS_OFFSET + slot] == ${GPU_HASH_INDEX_EMPTY_KEY}u) { return; }
  let sourceRow = sourceRows[SOURCE_ROWS_OFFSET + slot];
  if (sourceRow < FIRST_SOURCE_ROW || sourceRow - FIRST_SOURCE_ROW >= SOURCE_ROW_COUNT) { return; }
  let localSourceRow = sourceRow - FIRST_SOURCE_ROW;
  tableValues[TABLE_VALUES_OFFSET + slot] = ${valueExpression};
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${batch.id}-finalize`,
      source,
      bindings,
      constants,
      dispatchSize: layout,
      record: {moduleId: 'GPUHashIndex/finalize', invocationCount: index.tableKeys.length}
    })
  );

  return nodes;
}

function addQueryInitializePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUHashIndexQuery
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const bindings: GPUKernelBindingList = [
    {name: 'statistics', view: query.statistics, access: 'write'}
  ];
  const constants = {STATISTICS_OFFSET: getViewElementOffset(query.statistics)};
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(${HASH_INDEX_WORKGROUP_SIZE}) fn main(
  @builtin(global_invocation_id) globalId: vec3u
) {
  if (globalId.x < ${GPU_HASH_QUERY_STATISTICS_LENGTH}u) {
    statistics[STATISTICS_OFFSET + globalId.x] = 0u;
  }
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${query.id}-initialize`,
      source,
      bindings,
      constants,
      dispatchSize: {x: 1, y: 1, z: 1},
      record: {moduleId: 'GPUHashIndex/query-initialize', invocationCount: 1}
    })
  );

  return nodes;
}

function addQueryPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: Pick<GPUHashIndexQuery, 'id' | 'index' | 'statistics' | 'maxProbeCount'> & {
    keys: GraphDataView<'uint32'>;
    values: GraphDataView<'uint32'>;
    found: GraphDataView<'uint32'>;
    probes: GraphDataView<'uint32'>;
  }
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const layout = getDispatchLayout(
    query.keys.length,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const bindings: GPUKernelBindingList = [
    {name: 'queryKeys', view: query.keys, access: 'read'},
    {name: 'tableKeys', view: query.index.tableKeys, access: 'read'},
    {name: 'tableValues', view: query.index.tableValues, access: 'read'},
    {name: 'outputValues', view: query.values, access: 'write'},
    {name: 'outputFound', view: query.found, access: 'write'},
    {name: 'outputProbes', view: query.probes, access: 'write'},
    {name: 'statistics', view: query.statistics, access: 'read_write', atomic: true}
  ];
  const constants = {
    ELEMENT_COUNT: query.keys.length,
    CAPACITY_MASK: query.index.tableKeys.length - 1,
    MAX_PROBES: query.maxProbeCount,
    KEYS_OFFSET: getViewElementOffset(query.keys),
    TABLE_KEYS_OFFSET: getViewElementOffset(query.index.tableKeys),
    TABLE_VALUES_OFFSET: getViewElementOffset(query.index.tableValues),
    VALUES_OFFSET: getViewElementOffset(query.values),
    FOUND_OFFSET: getViewElementOffset(query.found),
    PROBES_OFFSET: getViewElementOffset(query.probes),
    STATISTICS_OFFSET: getViewElementOffset(query.statistics),
    ...getBoundedDispatchOverrides(layout)
  };
  const source = /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

${HASH_UINT32_WGSL}

@compute @workgroup_size(${HASH_INDEX_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let queryIndex = workgroupIndex * ${HASH_INDEX_WORKGROUP_SIZE}u + localIndex;
  if (queryIndex >= ELEMENT_COUNT) { return; }
  let key = queryKeys[KEYS_OFFSET + queryIndex];
  var value = ${GPU_HASH_INDEX_EMPTY_KEY}u;
  var found = false;
  var probes = 0u;
  if (key != ${GPU_HASH_INDEX_EMPTY_KEY}u) {
    let start = hashUint32(key) & CAPACITY_MASK;
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
      probes = probe + 1u;
      let slot = (start + probe) & CAPACITY_MASK;
      let storedKey = tableKeys[TABLE_KEYS_OFFSET + slot];
      if (storedKey == key) {
        value = tableValues[TABLE_VALUES_OFFSET + slot];
        found = true;
        break;
      }
      if (storedKey == ${GPU_HASH_INDEX_EMPTY_KEY}u) { break; }
    }
  }
  outputValues[VALUES_OFFSET + queryIndex] = value;
  outputFound[FOUND_OFFSET + queryIndex] = select(0u, 1u, found);
  outputProbes[PROBES_OFFSET + queryIndex] = probes;
  atomicAdd(&statistics[STATISTICS_OFFSET + select(1u, 0u, found)], 1u);
  atomicAdd(&statistics[STATISTICS_OFFSET + 2u], probes);
  atomicMax(&statistics[STATISTICS_OFFSET + 3u], probes);
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${query.id}-lookup`,
      source,
      bindings,
      constants,
      dispatchSize: layout,
      record: {moduleId: 'GPUHashIndex/lookup', invocationCount: query.keys.length}
    })
  );

  return nodes;
}

function validateProbeCount(id: string, probes: number, capacity: number, rows: number): void {
  if (!Number.isSafeInteger(probes) || probes < 1 || probes > capacity) {
    throw new Error(`${id} maxProbeCount must be an integer from one through capacity`);
  }
  if (rows * probes > MAXIMUM_UINT32) {
    throw new Error(`${id} aggregate probe count must fit in uint32 statistics`);
  }
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

function isPowerOfTwo(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0 && Number.isInteger(Math.log2(value));
}

function getDispatchLayout(elementCount: number, maximumDimension: number): DispatchLayout {
  return getBoundedWorkgroupDispatchLayout(
    'GPUHashIndex',
    Math.ceil(elementCount / HASH_INDEX_WORKGROUP_SIZE),
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
    /** Kernel facts recorded on recording graphs. */
    record?: {moduleId: string; invocationCount: number; workgroupSize?: number};
  }
): readonly GPUCommandNode<Parameters>[] {
  return [
    createGPUKernelNode(graph, {
      id: props.id,
      operation: 'GPUHashIndex',
      moduleId: props.record?.moduleId ?? `GPUHashIndex/${props.id}`,
      source: props.source,
      constants: props.constants,
      bindings: props.bindings,
      workgroupSize: props.record?.workgroupSize ?? HASH_INDEX_WORKGROUP_SIZE,
      dispatchLayout: props.dispatchSize,
      invocationCount: props.record?.invocationCount,
      workloadScope: 'never'
    })
  ];
}
