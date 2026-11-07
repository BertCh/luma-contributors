// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  GPUDataFrame,
  column,
  literal,
  type CompiledGPUDataFrameGlobalSort,
  type GPUDataFrameQueryParameters
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation,
  KeplerSessionExplorer
} from '@luma-contributors/kepler';
import type * as arrow from 'apache-arrow';
import type {OriginalSortDirection} from './original-table-sort';
import {chooseTableSortRoute, type TableOperationRoute} from './resident-table-population-routes';
import {
  classifyTableSortType,
  estimateWideSortBytes,
  fillWideSortKeys,
  makeDictionaryRankTable,
  type WideSortKind
} from './resident-table-sort-keys';

import {clamp} from '@math.gl/core';

type KeyColumns = {value: 'uint32'; selected: 'uint32'};
type PassColumns = {value: 'uint32'};

type SortProgram = {
  key: string;
  revision: number;
  /** Buffer holding the final sorted source row IDs (first `count` entries are meaningful). */
  rows: Buffer;
  rowsOffset: number;
  encode(encoder: ReturnType<KeplerComputeSession['device']['createCommandEncoder']>): void;
  destroy(): void;
  reservation: KeplerMemoryReservation;
};

export type ResidentTableGPUSortRequest = {
  column: string;
  direction: OriginalSortDirection;
  population: 'all' | 'selected';
  offset: number;
  limit: number;
};

/**
 * Exact GPU sorting for columns beyond Int32/Uint32/Float32: dictionary strings through a rank
 * table (one stable pass over remapped codes) and 64-bit/Float64 columns through two stable passes
 * over order-preserving hi/lo words. Only the requested page of sorted row IDs returns to the CPU.
 * Wide words come from the retained Arrow column (GPU-only resident datasets do not retain them).
 */
export class ResidentTableGPUSorter {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private program?: SortProgram;
  private requestRevision = 0;
  private tail: Promise<void> = Promise.resolve();
  private destroyed = false;
  private destruction?: Promise<void>;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      /** Admission ceiling for one sort program's keys, permutations and compiled passes. */
      memoryBudgetBytes: number;
    }
  ) {
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.explorer.dataset.id,
      {allowIncomplete: props.explorer.allowIncompletePopulation}
    );
  }

  /** Route for sorting one source column of a population of this many rows. */
  planSort(columnName: string, populationRows: number): TableOperationRoute {
    const value = this.lease.dataset.value;
    const field = value.source.schema.fields.find(candidate => candidate.name === columnName);
    if (!field) return {route: 'refused', reason: 'Unknown column.'};
    const orderKind = classifyTableSortType(field.type);
    const route = chooseTableSortRoute({
      orderKind,
      hasPreparedGPUColumn: Boolean(
        value.columns[columnName] && value.prepared.dictionaries[columnName]
      ),
      populationRows,
      hasRetainedWideData: !value.resident
    });
    // Only the new families are served here; scalar Float32/Int32/Uint32 sorts stay in the inspector.
    return orderKind === 'float32' || orderKind === 'int32' || orderKind === 'uint32'
      ? {route: 'refused', reason: 'Sorted by the prepared scalar permutation sort.'}
      : route;
  }

  /** One page of source row IDs in exact order, or null when superseded. Serialized. */
  readSortedRows(
    request: ResidentTableGPUSortRequest
  ): Promise<{rows: Uint32Array; totalCount: number} | null> {
    const requestRevision = ++this.requestRevision;
    const pending = this.tail.then(() => this.readCurrent(request, requestRevision));
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.requestRevision++;
    this.destruction = (async () => {
      await this.tail.catch(() => undefined);
      await this.releaseProgram();
      await this.lease.release();
    })();
    return this.destruction;
  }

  private isCurrent(requestRevision: number): boolean {
    return (
      !this.destroyed &&
      requestRevision === this.requestRevision &&
      this.lease.isValid() &&
      this.props.explorer.isCurrent &&
      !this.props.session.deviceLost
    );
  }

  private async readCurrent(
    request: ResidentTableGPUSortRequest,
    requestRevision: number
  ): Promise<{rows: Uint32Array; totalCount: number} | null> {
    if (!this.isCurrent(requestRevision)) return null;
    const selection = await this.props.explorer.readSelectedCount();
    if (!selection || !this.isCurrent(requestRevision)) return null;
    const rowCount = this.lease.dataset.value.source.numRows;
    const totalCount = request.population === 'selected' ? selection.selectedCount : rowCount;
    const length = clamp(totalCount - request.offset, 0, request.limit);
    if (length === 0) return {rows: new Uint32Array(0), totalCount};
    const program = await this.getProgram(request);
    if (!this.isCurrent(requestRevision)) return null;
    if (program.revision !== selection.revision) {
      const encoder = this.props.session.device.createCommandEncoder();
      program.encode(encoder);
      this.props.session.device.submit(encoder.finish());
      program.revision = selection.revision;
    }
    const bytes = await program.rows.readAsync(program.rowsOffset + request.offset * 4, length * 4);
    if (!this.isCurrent(requestRevision)) return null;
    return {
      rows: new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + length * 4)),
      totalCount
    };
  }

  private async releaseProgram(): Promise<void> {
    const program = this.program;
    if (!program) return;
    this.program = undefined;
    const fence = this.props.session.device.createFence();
    try {
      await Promise.race([fence.signaled, this.props.session.device.lost]);
    } finally {
      fence.destroy();
      program.destroy();
      program.reservation.release();
    }
  }

  private async getProgram(request: ResidentTableGPUSortRequest): Promise<SortProgram> {
    const key = `${request.column}:${request.direction}:${request.population}`;
    if (this.program?.key === key) return this.program;
    await this.releaseProgram();
    const value = this.lease.dataset.value;
    const rowCount = value.source.numRows;
    const field = value.source.schema.fields.find(candidate => candidate.name === request.column)!;
    const orderKind = classifyTableSortType(field.type);
    const passes = orderKind === 'dictionary-string' ? 1 : 2;
    const estimatedBytes = estimateWideSortBytes(rowCount, passes);
    if (estimatedBytes > this.props.memoryBudgetBytes)
      throw new Error('GPU table sort exceeds its working memory budget'); // sorting this many rows needs more GPU scratch memory
    const reservation = this.props.session.registry.reserveMemory(estimatedBytes, 'gpu', 'scratch');
    const owned: {destroy(): void}[] = [];
    try {
      const program =
        orderKind === 'dictionary-string'
          ? this.makeDictionaryProgram(key, request, reservation, owned)
          : this.makeWideProgram(key, request, orderKind as WideSortKind, reservation, owned);
      this.program = program;
      return program;
    } catch (error) {
      for (const resource of owned.reverse()) resource.destroy();
      reservation.release();
      throw error;
    }
  }

  private makeFlatVector(
    name: string,
    length: number,
    owned: {destroy(): void}[],
    data?: Uint32Array
  ): {vector: GPUVector<'uint32'>; buffer: Buffer} {
    const buffer = this.props.session.device.createBuffer({
      id: name,
      byteLength: Math.max(4, length * 4),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
      ...(data ? {data} : {})
    });
    const vector = new GPUVector({
      name,
      type: 'data',
      format: 'uint32',
      data: [new GPUData({format: 'uint32', length, ownsBuffer: true, buffer})],
      ownsData: true
    });
    owned.push(vector);
    return {vector, buffer};
  }

  private makeQuery<Columns extends PassColumns | KeyColumns>(
    id: string,
    vectors: Record<string, GPUVector<'uint32'>>,
    validity: GPUVector<'uint32'> | undefined,
    request: ResidentTableGPUSortRequest,
    selectedFilter: boolean,
    owned: {destroy(): void}[]
  ): CompiledGPUDataFrameGlobalSort<Columns> {
    const table = new GPUTable<Columns>({
      vectors: Object.fromEntries(
        Object.entries(vectors).map(([name, vector]) => [
          name,
          new GPUVector({
            name,
            type: 'data',
            format: 'uint32',
            data: vector.data,
            ownsData: false
          })
        ])
      ) as never
    });
    for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
      const valueField = schema.fields.find(candidate => candidate.name === 'value');
      if (valueField) valueField.nullable = Boolean(validity);
    }
    const frame = new GPUDataFrame<Columns>({
      table,
      ...(validity ? {validity: {value: validity}} : {})
    } as never);
    try {
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(this.props.session.device, {
        id: `kepler-table-${id}`
      });
      const filtered = frame.filter(
        selectedFilter ? column('selected' as never).equal(literal(1)) : literal(true)
      );
      const query = (
        filtered as never as {
          sortByGlobal(name: string, options: object): {compile(graph: unknown): never};
        }
      )
        .sortByGlobal('value', {direction: request.direction, nulls: 'last', nans: 'last'})
        .compile(graph) as CompiledGPUDataFrameGlobalSort<Columns>;
      owned.push(query);
      return query;
    } finally {
      frame.destroy();
    }
  }

  /** Copies each source chunk into the flat vector at its global row offset. */
  private encodeFlatCopy(
    encoder: SortProgram extends {encode(encoder: infer Encoder): void} ? Encoder : never,
    source: GPUVector,
    target: Buffer
  ): void {
    let rowStart = 0;
    for (const chunk of source.data) {
      if (chunk.length)
        encoder.copyBufferToBuffer({
          sourceBuffer: getBuffer(chunk),
          sourceOffset: chunk.byteOffset,
          destinationBuffer: target,
          destinationOffset: rowStart * 4,
          size: chunk.length * 4
        });
      rowStart += chunk.length;
    }
  }

  private makeDictionaryProgram(
    key: string,
    request: ResidentTableGPUSortRequest,
    reservation: KeplerMemoryReservation,
    owned: {destroy(): void}[]
  ): SortProgram {
    const value = this.lease.dataset.value;
    const rowCount = value.source.numRows;
    const codes = value.columns[request.column] as GPUVector<'uint32'>;
    const dictionary = value.prepared.dictionaries[request.column]!.values;
    const validity = value.analytics.validity[request.column];
    const keys = this.makeFlatVector('table-sort-rank-keys', rowCount, owned);
    const selected = this.makeFlatVector('table-sort-selected', rowCount, owned);
    const flatValidity = validity
      ? this.makeFlatVector('table-sort-validity', rowCount, owned)
      : undefined;
    const rankTable = this.makeFlatVector(
      'table-sort-rank-table',
      dictionary.length,
      owned,
      makeDictionaryRankTable(dictionary)
    );
    const remap = makeRemapGraph(
      this.props.session.device,
      codes,
      rankTable.vector,
      keys.vector,
      dictionary.length
    );
    owned.push(remap);
    const query = this.makeQuery<KeyColumns>(
      'dictionary-sort',
      {value: keys.vector, selected: selected.vector},
      flatValidity?.vector,
      request,
      request.population === 'selected',
      owned
    );
    reservation.resize(
      reservation.byteLength + query.compilationStats.physicalTransientResourceBytes
    );
    const indices = query.globalRowIndices.data[0];
    return {
      key,
      revision: -1,
      rows: getBuffer(indices),
      rowsOffset: indices.byteOffset,
      reservation,
      encode: encoder => {
        this.encodeFlatCopy(encoder, this.props.explorer.selectionMask, selected.buffer);
        if (validity && flatValidity) this.encodeFlatCopy(encoder, validity, flatValidity.buffer);
        remap.encode(encoder, {parameters: undefined});
        query.encode(encoder);
      },
      destroy: () => {
        for (const resource of owned.reverse()) resource.destroy();
      }
    };
  }

  private makeWideProgram(
    key: string,
    request: ResidentTableGPUSortRequest,
    kind: WideSortKind,
    reservation: KeplerMemoryReservation,
    owned: {destroy(): void}[]
  ): SortProgram {
    const value = this.lease.dataset.value;
    const rowCount = value.source.numRows;
    const words = readWideWords(value.source.getChild(request.column)!, kind, rowCount);
    const lo = this.makeFlatVector('table-sort-lo', rowCount, owned, words.lo);
    const hi = this.makeFlatVector('table-sort-hi', rowCount, owned, words.hi);
    const valid = this.makeFlatVector('table-sort-valid', rowCount, owned, words.valid);
    const selected = this.makeFlatVector('table-sort-selected', rowCount, owned);
    const hiPermuted = this.makeFlatVector('table-sort-hi-permuted', rowCount, owned);
    const validPermuted = this.makeFlatVector('table-sort-valid-permuted', rowCount, owned);
    const final = this.makeFlatVector('table-sort-final', rowCount, owned);
    // Pass 1 orders every selected row by the low word; stable pass 2 orders by the high word.
    const first = this.makeQuery<KeyColumns>(
      'wide-sort-lo',
      {value: lo.vector, selected: selected.vector},
      undefined,
      request,
      request.population === 'selected',
      owned
    );
    const second = this.makeQuery<PassColumns>(
      'wide-sort-hi',
      {value: hiPermuted.vector},
      validPermuted.vector,
      request,
      false,
      owned
    );
    const gather = makeWideGatherGraph(this.props.session.device, {
      permutation: first.globalRowIndices,
      count: first.globalSelectedCount,
      hi: hi.vector,
      valid: valid.vector,
      hiPermuted: hiPermuted.vector,
      validPermuted: validPermuted.vector,
      rowCount
    });
    owned.push(gather);
    const resolve = makeWideResolveGraph(this.props.session.device, {
      permutation: first.globalRowIndices,
      secondPermutation: second.globalRowIndices,
      output: final.vector,
      rowCount
    });
    owned.push(resolve);
    reservation.resize(
      reservation.byteLength +
        first.compilationStats.physicalTransientResourceBytes +
        second.compilationStats.physicalTransientResourceBytes
    );
    return {
      key,
      revision: -1,
      rows: final.buffer,
      rowsOffset: 0,
      reservation,
      encode: encoder => {
        this.encodeFlatCopy(encoder, this.props.explorer.selectionMask, selected.buffer);
        first.encode(encoder);
        gather.encode(encoder, {parameters: undefined});
        second.encode(encoder);
        resolve.encode(encoder, {parameters: undefined});
      },
      destroy: () => {
        for (const resource of owned.reverse()) resource.destroy();
      }
    };
  }
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

/** Retained Arrow words to order-preserving lo/hi keys (CPU transform, GPU sort). */
export function readWideWords(
  vector: arrow.Vector,
  kind: WideSortKind,
  rowCount: number
): {lo: Uint32Array; hi: Uint32Array; valid: Uint32Array} {
  const lo = new Uint32Array(rowCount);
  const hi = new Uint32Array(rowCount);
  const valid = new Uint32Array(rowCount);
  let outputOffset = 0;
  for (const data of vector.data) {
    const values = data.values as ArrayBufferView;
    const bitmap = data.nullBitmap;
    fillWideSortKeys({
      kind,
      words: new Uint32Array(values.buffer, values.byteOffset, values.byteLength / 4),
      firstRow: data.offset,
      count: data.length,
      isPresent:
        data.nullCount && bitmap?.length
          ? row => ((bitmap[(data.offset + row) >> 3] >> ((data.offset + row) & 7)) & 1) === 1
          : undefined,
      lo,
      hi,
      valid,
      outputOffset
    });
    outputOffset += data.length;
  }
  return {lo, hi, valid};
}

function makeComputeGraph(
  device: ConstructorParameters<typeof GPUCommandGraph>[0],
  id: string,
  rowCount: number,
  inputs: readonly {name: string; vector: GPUVector}[],
  outputs: readonly {name: string; vector: GPUVector}[],
  body: string
): CompiledGPUCommandGraph {
  const graph = new GPUCommandGraph(device, {id});
  const imported = [...inputs, ...outputs].map(({name, vector}) => ({
    name,
    output: outputs.some(candidate => candidate.name === name),
    data: graph.importGPUVector(name, vector).data[0]
  }));
  const dispatch = getBoundedDispatchLayout(
    id,
    rowCount,
    256,
    device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id,
    resources: imported.map(({data, output}) => ({
      buffer: data,
      usage: output ? ('storage-write' as const) : ('storage-read' as const)
    })),
    compile: ({device: compileDevice}) => {
      const kernel = new Kernel(compileDevice, {
        source: `
${imported.map(({name, output}, location) => `@group(0) @binding(${location}) var<storage,${output ? 'read_write' : 'read'}> ${name}: array<u32>;`).join('\n')}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${rowCount}u) {return;}
${imported.map(({name, data}) => `  let ${name}Base = ${data.byteOffset / 4}u;`).join('\n')}
${body}
}`,
        shaderLayout: {
          bindings: imported.map(({name}, location) => ({
            name,
            location,
            group: 0,
            type: 'storage' as const
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer: resolve}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              imported.map(({name, data}) => [name, resolve(data.buffer)])
            ),
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
  return graph.compile();
}

/** keys[i] = rank[codes[i]] for each source chunk, chunk-preserving input and a flat output. */
function makeRemapGraph(
  device: ConstructorParameters<typeof GPUCommandGraph>[0],
  codes: GPUVector<'uint32'>,
  ranks: GPUVector<'uint32'>,
  keys: GPUVector<'uint32'>,
  dictionarySize: number
): CompiledGPUCommandGraph {
  const graph = new GPUCommandGraph(device, {id: 'table-sort-rank-remap'});
  const rankData = graph.importGPUVector('ranks', ranks).data[0];
  const keyData = graph.importGPUVector('keys', keys).data[0];
  const chunks = graph.importGPUVector('codes', codes);
  let rowStart = 0;
  for (const [chunkIndex, chunk] of chunks.data.entries()) {
    const start = rowStart;
    rowStart += chunk.length;
    if (!chunk.length) continue;
    const dispatch = getBoundedDispatchLayout(
      'TableSortRankRemap',
      chunk.length,
      256,
      device.limits.maxComputeWorkgroupsPerDimension
    );
    graph.addComputePass({
      id: `rank-remap-${chunkIndex}`,
      resources: [
        {buffer: chunk, usage: 'storage-read'},
        {buffer: rankData, usage: 'storage-read'},
        {buffer: keyData, usage: 'storage-write'}
      ],
      compile: ({device: compileDevice}) => {
        const kernel = new Kernel(compileDevice, {
          source: `
@group(0) @binding(0) var<storage,read> codes: array<u32>;
@group(0) @binding(1) var<storage,read> ranks: array<u32>;
@group(0) @binding(2) var<storage,read_write> keys: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${chunk.length}u) {return;}
  let code = codes[${chunk.byteOffset / 4}u + index];
  keys[${keyData.byteOffset / 4 + start}u + index] = select(0xffffffffu, ranks[${rankData.byteOffset / 4}u + code], code < ${dictionarySize}u);
}`,
          shaderLayout: {
            bindings: ['codes', 'ranks', 'keys'].map((name, location) => ({
              name,
              location,
              group: 0,
              type: 'storage' as const
            }))
          }
        });
        return {
          encode: ({computePass, getBuffer: resolve}) =>
            kernel.dispatch(computePass, {
              bindings: {
                codes: resolve(chunk.buffer),
                ranks: resolve(rankData.buffer),
                keys: resolve(keyData.buffer)
              },
              ...dispatch
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
  }
  return graph.compile();
}

/** Applies the pass-1 permutation to the high word and validity; rows past the pass-1 count are invalid. */
function makeWideGatherGraph(
  device: ConstructorParameters<typeof GPUCommandGraph>[0],
  props: {
    permutation: GPUVector<'uint32'>;
    count: GPUVector<'uint32'>;
    hi: GPUVector<'uint32'>;
    valid: GPUVector<'uint32'>;
    hiPermuted: GPUVector<'uint32'>;
    validPermuted: GPUVector<'uint32'>;
    rowCount: number;
  }
): CompiledGPUCommandGraph {
  return makeComputeGraph(
    device,
    'table-sort-wide-gather',
    props.rowCount,
    [
      {name: 'permutation', vector: props.permutation},
      {name: 'count', vector: props.count},
      {name: 'hi', vector: props.hi},
      {name: 'valid', vector: props.valid}
    ],
    [
      {name: 'hiPermuted', vector: props.hiPermuted},
      {name: 'validPermuted', vector: props.validPermuted}
    ],
    `  let source = permutation[permutationBase + index];
  let present = index < count[countBase] && source < ${props.rowCount}u;
  let safe = select(0u, source, present);
  hiPermuted[hiPermutedBase + index] = select(0u, hi[hiBase + safe], present);
  validPermuted[validPermutedBase + index] = select(0u, valid[validBase + safe], present);`
  );
}

/** final[k] = pass1[pass2[k]]: the exact source row at sorted position k. */
function makeWideResolveGraph(
  device: ConstructorParameters<typeof GPUCommandGraph>[0],
  props: {
    permutation: GPUVector<'uint32'>;
    secondPermutation: GPUVector<'uint32'>;
    output: GPUVector<'uint32'>;
    rowCount: number;
  }
): CompiledGPUCommandGraph {
  return makeComputeGraph(
    device,
    'table-sort-wide-resolve',
    props.rowCount,
    [
      {name: 'permutation', vector: props.permutation},
      {name: 'secondPermutation', vector: props.secondPermutation}
    ],
    [{name: 'output', vector: props.output}],
    `  let position = secondPermutation[secondPermutationBase + index];
  let present = position < ${props.rowCount}u;
  output[outputBase + index] = select(0xffffffffu, permutation[permutationBase + select(0u, position, present)], present);`
  );
}
