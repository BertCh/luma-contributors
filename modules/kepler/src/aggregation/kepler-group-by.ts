// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {addKeplerLinearKernel as addKernel} from '../analysis/analysis-scaling';
import {
  createTransientView,
  GraphVectorView,
  GPUGroupAggregation,
  GPUGather,
  GPUSort,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {zipKeplerVectorChunks, getKeplerVectorChunks} from './kepler-vector-chunks';
import type {KeplerDatasetReference} from '../recipes/analysis-recipe';

/** Per-graph limits. Larger group-bys run as several passes (see kepler-group-by-passes.ts). */
export const MAXIMUM_KEPLER_GROUP_BY_SOURCE_ROWS = 67_108_864;
export const MAXIMUM_KEPLER_GROUP_BY_GROUPS = 1_048_576;
export const MAXIMUM_KEPLER_GROUP_BY_MEASURES = 16;

export type KeplerGroupByOperation = 'count' | 'sum' | 'mean' | 'min' | 'max';
export type KeplerGroupBySettings = Readonly<{
  keyColumn: string;
  aggregations: readonly Readonly<{
    column: string;
    operation: KeplerGroupByOperation;
    name: string;
  }>[];
  population: 'all';
  numericalPolicy: 'float32';
}>;

type ScalarColumn<Format extends 'uint32' | 'float32'> =
  | GraphDataView<Format>
  | GraphVectorView<Format>;
export type KeplerGroupByOutputs = {
  /** Dictionary code; dictionaryCount is the single null-key group. Read only the groupCount prefix. */
  keys: GraphDataView<'uint32'>;
  keyValidity: GraphDataView<'uint32'>;
  /** First original source row per group. Empty tail rows contain 0xffffffff. */
  firstSourceRows: GraphDataView<'uint32'>;
  measures: Readonly<
    Record<
      string,
      {
        values: GraphDataView<'uint32'> | GraphDataView<'float32'>;
        validity: GraphDataView<'uint32'>;
      }
    >
  >;
  /** [groupCount, invalidKeyCount, nonfiniteInputCount, nonfiniteOutputCount]. Last three must be zero. */
  status: GraphDataView<'uint32'>;
};
export type KeplerGroupByProps = {
  id?: string;
  source: KeplerDatasetReference;
  settings: KeplerGroupBySettings;
  /** Declared original fields, including nonnumeric fields eligible for native row-count aggregation. */
  sourceFields: readonly string[];
  /** Prepared, globally reconciled string-dictionary codes, preserving original source chunks. */
  key: ScalarColumn<'uint32'>;
  keyValidity?: ScalarColumn<'uint32'>;
  dictionaryCount: number;
  /**
   * Restricts this pass to dictionary slots [start, start + capacity), where slot dictionaryCount
   * is the null group. Output keys are then window-local (add start for the dictionary slot).
   * Rows outside the window are skipped without diagnostics. Defaults to every slot.
   */
  keyWindow?: Readonly<{start: number; capacity: number}>;
  /** Only non-count measure fields are accessed. Nulls do not trigger nonfinite diagnostics. */
  columns: Readonly<
    Record<string, {values: ScalarColumn<'float32'>; validity?: ScalarColumn<'uint32'>}>
  >;
  /** Dedicated caller-owned buffers, each in separate storage, with dictionaryCount+1 rows. */
  outputs: KeplerGroupByOutputs;
};

export function parseKeplerGroupBySettings(
  value: unknown,
  maximumMeasures = MAXIMUM_KEPLER_GROUP_BY_MEASURES
): KeplerGroupBySettings {
  const settings = readObject(value);
  checkKeys(settings, ['keyColumn', 'aggregations', 'population', 'numericalPolicy']);
  if (settings['population'] !== 'all' || settings['numericalPolicy'] !== 'float32')
    throw new Error('Group By requires all source rows and explicit Float32 numerical policy');
  const keyColumn = readName(settings['keyColumn']);
  const rawAggregations = settings['aggregations'];
  if (!Array.isArray(rawAggregations) || rawAggregations.length > maximumMeasures)
    // Measure count exceeds the per-pass limit; the caller must batch measures into passes.
    throw new Error('Group By supports zero to sixteen measures per pass');
  const names = new Set([keyColumn]);
  const aggregations = rawAggregations.map(value => {
    const aggregation = readObject(value);
    checkKeys(aggregation, ['column', 'operation', 'name']);
    const column = readName(aggregation['column']);
    const name = readName(aggregation['name']);
    if (names.has(name)) throw new Error('Group By output names must be unique');
    names.add(name);
    const operation = aggregation['operation'];
    if (
      operation !== 'count' &&
      operation !== 'sum' &&
      operation !== 'mean' &&
      operation !== 'min' &&
      operation !== 'max'
    )
      throw new Error('Unsupported Group By measure');
    return Object.freeze({column, operation, name});
  });
  return Object.freeze({
    keyColumn,
    aggregations: Object.freeze(aggregations),
    population: 'all',
    numericalPolicy: 'float32'
  });
}

/**
 * Pre-admit outputs and graph construction before allocation. chunkCount is the TOTAL number of
 * input chunks across key, key validity, and all distinct numeric columns/validity. After authoring,
 * compile and compare compiled.stats.physicalTransientResourceBytes with the reservation; this estimate
 * does not replace the compiled statistics or include borrowed sources, CPU metadata, pipeline/driver storage or result readback.
 */
export function getKeplerGroupByMemoryEstimate(props: {
  rowCount: number;
  chunkCount: number;
  dictionaryCount: number;
  measureCount: number;
}) {
  const {rowCount, chunkCount, dictionaryCount, measureCount} = props;
  validateInteger(rowCount, 0, MAXIMUM_KEPLER_GROUP_BY_SOURCE_ROWS);
  validateInteger(chunkCount, 0, Number.MAX_SAFE_INTEGER);
  validateInteger(dictionaryCount, 0, MAXIMUM_KEPLER_GROUP_BY_GROUPS - 1);
  validateInteger(measureCount, 0, MAXIMUM_KEPLER_GROUP_BY_MEASURES);
  if (rowCount > 0 && chunkCount === 0) throw new Error('Group By source chunks are required');
  const capacity = dictionaryCount + 1;
  // Logical output bytes; the host also accounts for any allocator-specific buffer padding.
  const outputByteLength = capacity * (12 + measureCount * 8) + 16;
  const scratchByteLength =
    rowCount * (4 + measureCount * 4) +
    capacity * (64 + measureCount * 16) +
    chunkCount * (512 + measureCount * 256) +
    65536;
  const totalByteLength = outputByteLength + scratchByteLength;
  if (!Number.isSafeInteger(totalByteLength))
    throw new KeplerAllocationError('memory-budget', {
      message: 'Group By memory estimate exceeds capacity'
    });
  return Object.freeze({capacity, outputByteLength, scratchByteLength, totalByteLength});
}

/**
 * Native all-source Group By over one prepared string dictionary. Null keys form one real group;
 * groups follow first source occurrence, not dictionary order. Count counts EVERY row regardless
 * of the named measure's nullness/type. Numeric measures ignore source nulls, but any non-null
 * NaN/Infinity rejects publication through status. Nonfinite aggregated results also reject it.
 * Sum/mean use the existing Float32 atomic reductions; no fixed-point substitute or Float64 parity
 * is implied. All-null numeric groups produce NaN plus validity zero; count remains valid.
 *
 * Inputs are borrowed and their chunks are preserved. Outputs are a new group-row table with a
 * valid prefix status[0], not source-row fields. This function never submits, reads back, publishes,
 * allocates parameter buffers or destroys caller outputs. Encode a generation once, validate its
 * status after completion/currentness checks, then publish it immutably through the dataset registry.
 */
export function addKeplerGroupBy<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerGroupByProps
) {
  const settings = parseKeplerGroupBySettings(props.settings);
  const id = props.id ?? 'kepler-group-by';
  const rowCount = props.key.length;
  const keyWindow = props.keyWindow ?? {start: 0, capacity: props.dictionaryCount + 1};
  // The window must lie inside the dictionary slots plus the null slot.
  if (
    !Number.isSafeInteger(keyWindow.start) ||
    keyWindow.start < 0 ||
    keyWindow.capacity < 1 ||
    keyWindow.start + keyWindow.capacity > props.dictionaryCount + 1
  )
    throw new Error('Group By key window is outside the dictionary');
  const sourceFields = new Set(props.sourceFields);
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    !sourceFields.has(settings.keyColumn) ||
    settings.aggregations.some(aggregation => !sourceFields.has(aggregation.column))
  )
    throw new Error('Group By requires a current source and declared fields');
  const numericFields = [
    ...new Set(
      settings.aggregations
        .filter(aggregation => aggregation.operation !== 'count')
        .map(aggregation => aggregation.column)
    )
  ];
  const inputViews: GraphDataView[] = [];
  const validateInput = (
    input: ScalarColumn<'uint32'> | ScalarColumn<'float32'>,
    format: 'uint32' | 'float32'
  ) => {
    if (input.length !== rowCount || input.format !== format)
      throw new Error('Group By source columns must align');
    for (const view of getKeplerVectorChunks(input)) {
      validatePackedView(view, [format], id);
      if (view.buffer.graph !== graph)
        throw new Error('Group By views must belong to the current graph');
      inputViews.push(view);
    }
  };
  validateInput(props.key, 'uint32');
  if (props.keyValidity) validateInput(props.keyValidity, 'uint32');
  for (const field of numericFields) {
    const column = props.columns[field];
    if (!column) throw new Error('Group By numerical measure requires a Float32 source column');
    validateInput(column.values, 'float32');
    if (column.validity) validateInput(column.validity, 'uint32');
  }
  const memoryEstimate = getKeplerGroupByMemoryEstimate({
    rowCount,
    chunkCount: inputViews.length,
    dictionaryCount: keyWindow.capacity - 1,
    measureCount: settings.aggregations.length
  });
  const {capacity} = memoryEstimate;
  const windowStart = keyWindow.start;
  const windowEnd = keyWindow.start + keyWindow.capacity;
  const outputs = props.outputs;
  if (Object.keys(outputs.measures).length !== settings.aggregations.length)
    throw new Error('Group By output measures must match settings');
  const outputSpecifications: readonly [GraphDataView, 'uint32' | 'float32', number][] = [
    [outputs.keys, 'uint32', capacity],
    [outputs.keyValidity, 'uint32', capacity],
    [outputs.firstSourceRows, 'uint32', capacity],
    [outputs.status, 'uint32', 4],
    ...settings.aggregations.flatMap(aggregation => {
      const output = outputs.measures[aggregation.name];
      if (!output) throw new Error('Missing Group By measure output');
      return [
        [output.values, aggregation.operation === 'count' ? 'uint32' : 'float32', capacity],
        [output.validity, 'uint32', capacity]
      ] as [GraphDataView, 'uint32' | 'float32', number][];
    })
  ];
  const buffers = new Set(inputViews.map(view => view.buffer));
  for (const [view, format, length] of outputSpecifications) {
    validatePackedView(view, [format], id);
    if (view.length !== length || view.buffer.graph !== graph || buffers.has(view.buffer))
      throw new Error('Group By outputs require separate matching current-graph buffers');
    buffers.add(view.buffer);
  }
  const normalizedKeys = createUint32Scratch(graph, `${id}-normalized-keys`, props.key);
  const firstRows = createTransientView(graph, `${id}-first-rows`, 'uint32', capacity);
  const denseKeys = createTransientView(graph, `${id}-dense-keys`, 'uint32', capacity);
  const diagnostics = createTransientView(graph, `${id}-diagnostics`, 'uint32', 4);
  addKernel(
    graph,
    `${id}-initialize`,
    capacity,
    [
      {name: 'firstRows', view: firstRows, write: true},
      {name: 'denseKeys', view: denseKeys, write: true},
      {name: 'diagnostics', view: diagnostics, write: true}
    ],
    `firstRows[firstRowsOffset + index] = 0xffffffffu;
denseKeys[denseKeysOffset + index] = index;
if (index == 0u) { diagnostics[diagnosticsOffset] = 0u; diagnostics[diagnosticsOffset + 1u] = 0u; diagnostics[diagnosticsOffset + 2u] = 0u; diagnostics[diagnosticsOffset + 3u] = 0u; }`
  );
  let sourceOffset = 0;
  const keySpans = zipKeplerVectorChunks([
    props.key,
    props.keyValidity ?? props.key,
    normalizedKeys
  ]);
  for (const [spanIndex, [key, validity, destination]] of keySpans.entries()) {
    const offset = sourceOffset;
    sourceOffset += key.length;
    addKernel(
      graph,
      `${id}-prepare-keys-${spanIndex}`,
      key.length,
      [
        {name: 'key', view: key},
        ...(props.keyValidity ? [{name: 'validity', view: validity}] : []),
        {name: 'destination', view: destination, write: true},
        {name: 'firstRows', view: firstRows, atomic: true},
        {name: 'diagnostics', view: diagnostics, atomic: true}
      ],
      `let sourceKey = key[keyOffset + index];
let valid = ${props.keyValidity ? 'validity[validityOffset + index] != 0u' : 'true'};
let slot = select(${props.dictionaryCount}u, sourceKey, valid);
let admitted = !valid || sourceKey < ${props.dictionaryCount}u;
let inWindow = admitted && slot >= ${windowStart}u && slot < ${windowEnd}u;
let group = select(0u, slot - ${windowStart}u, inWindow);
destination[destinationOffset + index] = select(0xffffffffu, group, inWindow);
if (inWindow) { atomicMin(&firstRows[firstRowsOffset + group], ${offset}u + index); }
else if (!admitted) { atomicAdd(&diagnostics[diagnosticsOffset], 1u); }`
    );
  }
  // Sorting the first-row sentinel to the tail also compacts occupied groups without a second copy.
  graph.add(
    new GPUSort({
      id: `${id}-group-order`,
      keys: firstRows,
      values: denseKeys,
      outputKeys: outputs.firstSourceRows,
      outputValues: outputs.keys,
      algorithm: 'bitonic'
    })
  );
  addKernel(
    graph,
    `${id}-key-validity`,
    capacity,
    [
      {name: 'keys', view: outputs.keys},
      {name: 'firstRows', view: outputs.firstSourceRows},
      {name: 'validity', view: outputs.keyValidity, write: true},
      {name: 'diagnostics', view: diagnostics, atomic: true}
    ],
    // Slot 3 counts occupied groups in parallel so the status kernel needs no serial loop.
    `let occupied = firstRows[firstRowsOffset + index] != 0xffffffffu;
validity[validityOffset + index] = select(0u, 1u, occupied && keys[keysOffset + index] + ${windowStart}u < ${props.dictionaryCount}u);
if (occupied) { atomicAdd(&diagnostics[diagnosticsOffset + 3u], 1u); }`
  );
  const groupCounts = createTransientView(graph, `${id}-group-counts`, 'uint32', capacity);
  graph.add(
    new GPUGroupAggregation({id: `${id}-count`, keys: normalizedKeys, output: groupCounts})
  );
  const numericStates = new Map<
    string,
    {mask: GraphVectorView<'uint32'>; counts: GraphDataView<'uint32'>}
  >();
  for (const [fieldIndex, field] of numericFields.entries()) {
    const column = props.columns[field];
    const mask = createUint32Scratch(graph, `${id}-numeric-mask-${fieldIndex}`, column.values);
    const counts = createTransientView(
      graph,
      `${id}-numeric-counts-${fieldIndex}`,
      'uint32',
      capacity
    );
    const spans = zipKeplerVectorChunks([column.values, column.validity ?? normalizedKeys, mask]);
    for (const [spanIndex, [values, validity, destination]] of spans.entries()) {
      addKernel(
        graph,
        `${id}-numeric-validity-${fieldIndex}-${spanIndex}`,
        values.length,
        [
          {name: 'values', view: values},
          ...(column.validity ? [{name: 'validity', view: validity}] : []),
          {name: 'destination', view: destination, write: true},
          {name: 'diagnostics', view: diagnostics, atomic: true}
        ],
        `let included = ${column.validity ? 'validity[validityOffset + index] != 0u' : 'true'};
let finite = isFiniteValue(values[valuesOffset + index]);
destination[destinationOffset + index] = select(0u, 1u, included && finite);
if (included && !finite) { atomicAdd(&diagnostics[diagnosticsOffset + 1u], 1u); }`
      );
    }
    graph.add(
      new GPUGroupAggregation({
        id: `${id}-numeric-count-${fieldIndex}`,
        keys: normalizedKeys,
        mask,
        output: counts
      })
    );
    numericStates.set(field, {mask, counts});
  }
  for (const [measureIndex, aggregation] of settings.aggregations.entries()) {
    const output = outputs.measures[aggregation.name];
    if (aggregation.operation === 'count') {
      if (output.values.format !== 'uint32') throw new Error('Count output must be uint32');
      graph.add(
        new GPUGather({
          id: `${id}-count-gather-${measureIndex}`,
          source: groupCounts,
          indices: outputs.keys,
          output: output.values
        })
      );
      addKernel(
        graph,
        `${id}-count-validity-${measureIndex}`,
        capacity,
        [
          {name: 'firstRows', view: outputs.firstSourceRows},
          {name: 'validity', view: output.validity, write: true}
        ],
        'validity[validityOffset + index] = select(0u, 1u, firstRows[firstRowsOffset + index] != 0xffffffffu);'
      );
      continue;
    }
    if (output.values.format !== 'float32') throw new Error('Numerical output must be Float32');
    const state = numericStates.get(aggregation.column)!;
    const denseValues = createTransientView(
      graph,
      `${id}-measure-${measureIndex}`,
      'float32',
      capacity
    );
    graph.add(
      new GPUGroupAggregation({
        id: `${id}-aggregate-${measureIndex}`,
        keys: normalizedKeys,
        values: props.columns[aggregation.column].values,
        mask: state.mask,
        operation: aggregation.operation,
        output: denseValues
      })
    );
    graph.add(
      new GPUGather({
        id: `${id}-measure-gather-${measureIndex}`,
        source: denseValues,
        indices: outputs.keys,
        output: output.values
      })
    );
    addKernel(
      graph,
      `${id}-measure-validity-${measureIndex}`,
      capacity,
      [
        {name: 'keys', view: outputs.keys},
        {name: 'firstRows', view: outputs.firstSourceRows},
        {name: 'counts', view: state.counts},
        {name: 'values', view: output.values, write: true},
        {name: 'validity', view: output.validity, write: true},
        {name: 'diagnostics', view: diagnostics, atomic: true}
      ],
      `let included = firstRows[firstRowsOffset + index] != 0xffffffffu && counts[countsOffset + keys[keysOffset + index]] > 0u;
let value = values[valuesOffset + index];
let finite = isFiniteValue(value);
if (included && !finite) { atomicAdd(&diagnostics[diagnosticsOffset + 2u], 1u); }
let valid = included && finite;
validity[validityOffset + index] = select(0u, 1u, valid);
values[valuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), value, valid);`
    );
  }
  addKernel(
    graph,
    `${id}-status`,
    1,
    [
      {name: 'diagnostics', view: diagnostics, atomic: true},
      {name: 'status', view: outputs.status, write: true}
    ],
    `status[statusOffset] = atomicLoad(&diagnostics[diagnosticsOffset + 3u]);
status[statusOffset + 1u] = atomicLoad(&diagnostics[diagnosticsOffset]);
status[statusOffset + 2u] = atomicLoad(&diagnostics[diagnosticsOffset + 1u]);
status[statusOffset + 3u] = atomicLoad(&diagnostics[diagnosticsOffset + 2u]);`
  );
  return {
    source: Object.freeze({...props.source}),
    settings,
    capacity,
    outputs,
    memoryEstimate,
    rowIdentity: 'group-key' as const,
    parameterByteLength: 0,
    destroyParameters() {}
  };
}

function createUint32Scratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  template: ScalarColumn<'uint32'> | ScalarColumn<'float32'>
): GraphVectorView<'uint32'> {
  const data = getKeplerVectorChunks(template).map((chunk, index) =>
    createTransientView(graph, `${id}-${index}`, 'uint32', chunk.length)
  );
  return new GraphVectorView({
    id,
    name: id,
    format: 'uint32',
    length: template.length,
    valueLength: template.length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  });
}
function readObject(value: unknown): Record<string, unknown> {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    Object.getPrototypeOf(value) !== Object.prototype
  )
    throw new Error('Expected a plain Group By settings object');
  return value as Record<string, unknown>;
}
function checkKeys(value: Record<string, unknown>, keys: readonly string[]): void {
  if (Object.keys(value).some(key => !keys.includes(key)))
    throw new Error('Unknown Group By setting');
}
function readName(value: unknown): string {
  if (typeof value !== 'string' || !value.length || value.length > 1024)
    throw new Error('Invalid Group By field');
  return value;
}
function validateInteger(value: number, minimum: number, maximum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum)
    throw new KeplerAllocationError('capacity', {message: 'Group By exceeds capacity'});
}
