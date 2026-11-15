// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MeteredCommandGraph} from '../session/metered-command-graph';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUGather,
  GPUHistogram,
  GPUMask,
  GPUReadbackRing,
  type GPUReadbackTicket,
  GPUReduction,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  GPUCrossfilter,
  type GPUCrossfilterDimension,
  type GPUCrossfilterView
} from '@luma.gl/graph/gpu-crossfilter';
import {
  GPUDataFrame,
  and,
  or,
  column,
  parameter,
  literal,
  type CompiledGPUDataFrameQuery,
  type GPUDataFrameQueryParameters,
  type GPUExpression
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';

/** Crossfilter dimension that keeps dead rows out of every linked view. */
const LIVE_DIMENSION_ID = 'kepler-live-mask';
/** Inclusive uint32 range selecting every nonzero mask flag. */
const SELECTED_MASK_RANGE: [number, number] = [1, 0xffffffff];

export type GPUResidentScalarFormat = 'float32' | 'uint32' | 'sint32';
export type GPUResidentColumns = Readonly<Record<string, GPUVector<GPUResidentScalarFormat>>>;
export type GPUResidentDimension =
  | {id: string; kind: 'range'; column: string; exclusiveMaximum?: boolean}
  | {id: string; kind: 'bounds'; x: string; y: string}
  | {id: string; kind: 'category'; column: string; capacity: number}
  | {
      id: string;
      kind: 'external';
      mask: GPUVector<'uint32'>;
      /** Optional borrowed time/spatial contributor, encoded before mask composition. */
      encode?: (commandEncoder: CommandEncoder) => void;
    };

export type GPUResidentHistogram = {
  id: string;
  column: string;
  dimension?: string;
  includeOwnSelection?: boolean;
  domain: readonly [number, number];
  binCount: number;
};

export type GPUResidentDomain = {
  id: string;
  column: string;
  /** Undefined uses every display filter; [] uses only liveness; names choose domain filters. */
  dimensions?: readonly string[];
};

export type GPUResidentExplorerProps = {
  device: Device;
  columns: GPUResidentColumns;
  /** Source batch lengths, required without scalar columns; never creates a user-visible field. */
  sourceChunkLengths?: readonly number[];
  validity?: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  dimensions: readonly GPUResidentDimension[];
  histograms?: readonly GPUResidentHistogram[];
  domains?: readonly GPUResidentDomain[];
  liveMask?: GPUVector<'uint32'>;
  datasetRevision?: number;
  /** Maximum rows returned by an explicit inspection request. Defaults to 256. */
  maximumPageSize?: number;
  /** Concurrent summary/inspection requests; additional requests return null. Defaults to 2. */
  maximumConcurrentReads?: number;
  /** Maximum payload of each explicit read request. Defaults to 1 MiB. */
  maximumReadbackBytes?: number;
};

export type GPUResidentSummary = {
  revision: number;
  datasetRevision: number;
  selectedCount: number;
  histograms: Record<string, Uint32Array>;
  domains: Record<string, readonly [number, number] | null>;
};

export type GPUResidentPage = {
  revision: number;
  datasetRevision: number;
  sourceRowIndices: Uint32Array;
  columns: Record<string, Float32Array | Int32Array | Uint32Array>;
  validity: Record<string, Uint32Array>;
};

type PredicateState = {
  query: CompiledGPUDataFrameQuery<Record<string, GPUResidentScalarFormat>>;
  parameters: Record<string, number | boolean | null>;
  /** Physical transient scratch of the compiled predicate graph. */
  scratchByteLength: number;
};

/**
 * Application-owned linked exploration over borrowed GPU columns. Control changes only update
 * small parameter buffers. Source masks, compaction and histograms remain GPU resident.
 *
 * The caller submits encode() before requesting summaries/pages and keeps sources alive until
 * destroy(). Explicit inspection submits its own bounded gather; normal frame encoding never submits.
 */
export class GPUResidentExplorer {
  readonly selectionMask: GPUVector<'uint32'>;
  readonly datasetRevision: number;
  readonly rowCount: number;
  readonly maximumPageSize: number;
  readonly maximumConcurrentReads: number;
  readonly maximumReadbackBytes: number;
  private readonly device: Device;
  private readonly columns: GPUResidentColumns;
  private readonly validity: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  private readonly dimensions: readonly GPUResidentDimension[];
  private readonly predicates = new Map<string, PredicateState>();
  private readonly domainPredicates = new Map<string, PredicateState>();
  private readonly vectors: GPUVector[] = [];
  private readonly domainFilters: GPUCrossfilter[] = [];
  private readonly hasLiveMask: boolean;
  private readonly domainBuffers = new Map<
    string,
    {buffer: Buffer; format: GPUResidentScalarFormat}
  >();
  private readonly histogramBuffers = new Map<string, Buffer>();
  private readonly selectedRows: GPUVector<'uint32'>;
  private readonly selectedCount: Buffer;
  private readonly filter: GPUCrossfilter;
  private readonly compiled: CompiledGPUCommandGraph;
  private revision = 0;
  private encodedRevision = -1;
  private destroyed = false;
  /** Admitted reads still using explorer resources; bounded by maximumConcurrentReads. */
  private activeReads = 0;
  /** Single-slot rings grow lazily up to maximumConcurrentReads as reads need staging. */
  private readonly readbackRings: GPUReadbackRing[] = [];
  private transferredBytes = 0;
  private resourcesReleased = false;
  private releaseWaiters: (() => void)[] = [];

  constructor(props: GPUResidentExplorerProps) {
    validateExplorerProps(props);
    this.device = props.device;
    this.columns = Object.fromEntries(
      Object.entries(props.columns).map(([name, vector]) => [
        name,
        new GPUVector({
          name,
          type: 'data',
          format: getScalarFormat(vector),
          data: vector.data,
          ownsData: false
        })
      ])
    );
    this.validity = {...props.validity};
    this.hasLiveMask = Boolean(props.liveMask);
    this.dimensions = props.dimensions.map(dimension => ({...dimension}));
    this.datasetRevision = props.datasetRevision ?? 1;
    this.maximumPageSize = props.maximumPageSize ?? 256;
    this.maximumConcurrentReads = props.maximumConcurrentReads ?? 2;
    this.maximumReadbackBytes = props.maximumReadbackBytes ?? 1024 * 1024;
    const firstColumn = Object.values(this.columns)[0];
    const sourceChunkLengths =
      props.sourceChunkLengths ?? firstColumn.data.map(data => data.length);
    this.rowCount = sourceChunkLengths.reduce((total, length) => total + length, 0);
    const graph = new GPUCommandGraph(props.device, {id: 'kepler-resident-explorer'});
    let frame: GPUDataFrame<Record<string, GPUResidentScalarFormat>> | undefined;
    let filter: GPUCrossfilter | undefined;
    try {
      // Crossfilter needs a typed topology even for an unfiltered, geometry-only source. This
      // zero-initialized carrier is owned/accounted here, never included in columns or exports.
      let neutralSource = firstColumn;
      const getNeutralSource = () =>
        (neutralSource ??= this.allocateVector('source-row-topology', sourceChunkLengths));
      this.selectionMask = this.allocateVector('selection', sourceChunkLengths);
      this.selectedRows = this.allocateVector('selected-rows', sourceChunkLengths);
      const countVector = this.allocateVector('selected-count', [1]);
      this.selectedCount = getBuffer(countVector.data[0]);
      if (firstColumn) {
        const table = new GPUTable<Record<string, GPUResidentScalarFormat>>({
          vectors: this.columns
        });
        for (const field of table.schema.fields)
          field.nullable = Boolean(this.validity[field.name]);
        for (const batch of table.batches) {
          for (const field of batch.schema.fields)
            field.nullable = Boolean(this.validity[field.name]);
        }
        frame = new GPUDataFrame({table, validity: this.validity});
      }
      const dimensions: GPUCrossfilterDimension[] = this.dimensions.map(dimension => {
        if (dimension.kind === 'external') {
          // Published crossfilter has no external-mask kind: a caller-written 0/1 mask is a
          // range dimension over that mask, active for any nonzero flag and bypassed when cleared.
          return {
            id: dimension.id,
            kind: 'range',
            input: graph.importGPUVector(dimension.id, dimension.mask)
          };
        }
        const predicate = createPredicate(dimension);
        const predicateGraph = new MeteredCommandGraph<GPUDataFrameQueryParameters>(props.device);
        const query = frame!.filter(predicate).compile(predicateGraph);
        this.predicates.set(dimension.id, {
          query,
          parameters: {},
          scratchByteLength: predicateGraph.transientByteLength
        });
        return {
          id: dimension.id,
          kind: 'range',
          input: graph.importGPUVector(dimension.id, query.selectionMask)
        };
      });
      // Liveness is a permanently active dimension, so it intersects every linked view.
      const liveDimension: GPUCrossfilterDimension | undefined = props.liveMask
        ? {
            id: LIVE_DIMENSION_ID,
            kind: 'range',
            input: graph.importGPUVector('live', props.liveMask)
          }
        : undefined;
      // A neutral dimension keeps an unfiltered dataset valid without synthesizing CPU rows.
      if (!dimensions.length) {
        dimensions.push({
          id: 'unfiltered',
          kind: 'range',
          input: graph.importGPUVector('unfiltered', getNeutralSource())
        });
      }
      const views: GPUCrossfilterView[] = [
        {
          id: 'visible',
          kind: 'visibility',
          output: graph.importGPUVector('visible', this.selectedRows),
          count: graph.importGPUData('count', countVector.data[0])
        }
      ];
      for (const histogram of props.histograms ?? []) {
        views.push({
          id: histogram.id,
          kind: 'mask',
          dimension: histogram.dimension,
          includeOwnSelection: histogram.includeOwnSelection ?? false,
          output: graph.importGPUVector(
            `${histogram.id}-mask`,
            this.allocateVector(`${histogram.id}-mask`, sourceChunkLengths)
          )
        });
      }
      filter = new GPUCrossfilter(graph, {
        dimensions: liveDimension ? [...dimensions, liveDimension] : dimensions,
        views,
        outputMask: graph.importGPUVector('selection', this.selectionMask)
      });
      if (liveDimension) filter.setRange(LIVE_DIMENSION_ID, SELECTED_MASK_RANGE);
      this.filter = filter;
      filter.addToGraph(graph);
      for (const histogram of props.histograms ?? []) {
        let mask = filter.getViewMask(histogram.id);
        const validity = this.validity[histogram.column];
        if (validity && mask) {
          const validMask = graph.importGPUVector(
            `${histogram.id}-valid`,
            this.allocateVector(`${histogram.id}-valid`, sourceChunkLengths)
          );
          graph.add(
            new GPUMask({
              id: `${histogram.id}-valid`,
              inputs: [mask, graph.importGPUVector(`${histogram.id}-validity`, validity)],
              output: validMask
            })
          );
          mask = validMask;
        }
        const output = this.allocateVector(`${histogram.id}-bins`, [histogram.binCount]);
        this.histogramBuffers.set(histogram.id, getBuffer(output.data[0]));
        graph.add(
          new GPUHistogram({
            id: `${histogram.id}-bins`,
            input: graph.importGPUVector(`${histogram.id}-input`, this.columns[histogram.column]),
            mask,
            domain: histogram.domain,
            output: graph.importGPUData(`${histogram.id}-output`, output.data[0])
          })
        );
      }
      const scopes = new Map<string, GPUCrossfilter>();
      for (const domain of props.domains ?? []) {
        let mask = filter.mask;
        if (domain.dimensions) {
          const key = JSON.stringify([...domain.dimensions].sort());
          let scopedFilter = scopes.get(key);
          if (!scopedFilter) {
            const selectedDimensions = dimensions.filter(dimension =>
              domain.dimensions!.includes(dimension.id)
            );
            scopedFilter = new GPUCrossfilter(graph, {
              id: `domain-scope-${scopes.size}`,
              dimensions:
                selectedDimensions.length || props.liveMask
                  ? [
                      ...selectedDimensions,
                      ...(props.liveMask
                        ? [
                            {
                              id: LIVE_DIMENSION_ID,
                              kind: 'range' as const,
                              input: graph.importGPUVector(
                                `domain-live-${scopes.size}`,
                                props.liveMask
                              )
                            }
                          ]
                        : [])
                    ]
                  : [
                      {
                        id: 'unfiltered',
                        kind: 'range' as const,
                        input: graph.importGPUVector(
                          `domain-source-${scopes.size}`,
                          getNeutralSource()
                        )
                      }
                    ],
              outputMask: graph.importGPUVector(
                `domain-mask-${scopes.size}`,
                this.allocateVector(`domain-mask-${scopes.size}`, sourceChunkLengths)
              )
            });
            if (props.liveMask) scopedFilter.setRange(LIVE_DIMENSION_ID, SELECTED_MASK_RANGE);
            this.domainFilters.push(scopedFilter);
            scopes.set(key, scopedFilter);
            scopedFilter.addToGraph(graph);
          }
          mask = scopedFilter.mask;
        }
        const format = getScalarFormat(this.columns[domain.column]);
        let validity = this.validity[domain.column];
        if (format === 'float32') {
          let finite = this.domainPredicates.get(domain.column);
          if (!finite) {
            const finitePredicate = and(
              column(domain.column).greaterThanOrEqual(literal(-3.4028234663852886e38)),
              column(domain.column).lessThanOrEqual(literal(3.4028234663852886e38))
            );
            const finiteGraph = new MeteredCommandGraph<GPUDataFrameQueryParameters>(props.device);
            finite = {
              query: frame!.filter(finitePredicate).compile(finiteGraph),
              parameters: {},
              scratchByteLength: finiteGraph.transientByteLength
            };
            this.domainPredicates.set(domain.column, finite);
          }
          validity = finite.query.selectionMask;
        }
        if (validity) {
          const validMask = graph.importGPUVector(
            `${domain.id}-domain-valid`,
            this.allocateVector(`${domain.id}-domain-valid`, sourceChunkLengths)
          );
          graph.add(
            new GPUMask({
              id: `${domain.id}-domain-valid`,
              inputs: [mask, graph.importGPUVector(`${domain.id}-domain-validity`, validity)],
              output: validMask
            })
          );
          mask = validMask;
        }
        const output = this.allocateVector(`${domain.id}-extent`, [3]);
        const buffer = getBuffer(output.data[0]);
        this.domainBuffers.set(domain.id, {buffer, format});
        graph.add(
          new GPUReduction({
            id: `${domain.id}-finite-count`,
            operation: 'sum',
            input: mask,
            output: graph.importGPUData(
              `${domain.id}-finite-count-output`,
              new GPUData({buffer, format: 'uint32', byteOffset: 8, length: 1, ownsBuffer: false})
            )
          })
        );
        graph.add(
          new GPUReduction({
            id: `${domain.id}-extent`,
            operation: 'extent',
            input: graph.importGPUVector(`${domain.id}-domain-input`, this.columns[domain.column]),
            mask,
            output: graph.importGPUData(
              `${domain.id}-extent-output`,
              new GPUData({buffer, format, length: 2, ownsBuffer: false})
            )
          })
        );
      }
      this.compiled = graph.compile();
    } catch (error) {
      filter?.destroy();
      for (const scopedFilter of this.domainFilters) scopedFilter.destroy();
      for (const predicate of [...this.predicates.values(), ...this.domainPredicates.values()])
        predicate.query.destroy();
      for (const vector of this.vectors) vector.destroy();
      throw error;
    } finally {
      frame?.destroy();
    }
  }

  get currentRevision(): number {
    return this.revision;
  }

  setRange(id: string, minimum: number, maximum: number): void {
    const dimension = this.requireDimension(id, 'range');
    if (dimension.kind !== 'range') return;
    validateColumnRange(this.columns[dimension.column], minimum, maximum);
    this.setPredicateParameters(id, {minimum, maximum});
  }

  setBounds(id: string, bounds: readonly [number, number, number, number]): void {
    const dimension = this.requireDimension(id, 'bounds');
    if (dimension.kind !== 'bounds') return;
    validateColumnRange(this.columns[dimension.x], bounds[0], bounds[2]);
    validateColumnRange(this.columns[dimension.y], bounds[1], bounds[3]);
    this.setPredicateParameters(id, {
      minimumX: bounds[0],
      minimumY: bounds[1],
      maximumX: bounds[2],
      maximumY: bounds[3]
    });
  }

  setCategories(id: string, categories: readonly number[]): void {
    const dimension = this.requireDimension(id, 'category');
    if (dimension.kind !== 'category') return;
    if (
      categories.length > dimension.capacity ||
      categories.some(value => !Number.isInteger(value) || value < 0 || value > 0xffffffff)
    )
      throw new Error('Category selection exceeds its unsigned-key contract');
    const parameters: Record<string, number | boolean> = {};
    for (let index = 0; index < dimension.capacity; index++) {
      parameters[`enabled${index}`] = index < categories.length;
      parameters[`category${index}`] = categories[index] ?? 0;
    }
    this.setPredicateParameters(id, parameters);
  }

  /** Explicit empty intersection, distinct from clearing/bypassing a dimension. */
  setEmpty(id: string): void {
    const dimension = this.requireDimension(id);
    if (dimension.kind === 'external')
      throw new Error('External empty selections must be written by their owner');
    this.setPredicateParameters(id, {enabled: false});
  }

  /** Any nonzero external or live mask flag selects the row. */
  setExternalActive(id: string, active: boolean): void {
    this.requireDimension(id, 'external');
    this.setDimensionActive(this.filter, id, active);
    this.updateDomainFilters(id, active);
    this.invalidate();
  }

  clear(id: string): void {
    this.requireDimension(id);
    this.filter.clear(id);
    this.updateDomainFilters(id, false);
    this.invalidate();
  }

  clearAll(): void {
    this.assertActive();
    this.filter.clearAll();
    for (const scopedFilter of this.domainFilters) scopedFilter.clearAll();
    // clearAll also bypasses the liveness dimension, which must stay active.
    if (this.hasLiveMask) {
      this.filter.setRange(LIVE_DIMENSION_ID, SELECTED_MASK_RANGE);
      for (const scopedFilter of this.domainFilters)
        scopedFilter.setRange(LIVE_DIMENSION_ID, SELECTED_MASK_RANGE);
    }
    this.invalidate();
  }

  /** Call after rewriting any borrowed source or external mask. */
  invalidate(): void {
    this.assertActive();
    this.revision++;
  }

  /** Encode before Deck draws; the caller retains submission ownership. */
  encode(commandEncoder: CommandEncoder): number {
    this.assertActive();
    try {
      for (const dimension of this.dimensions) {
        if (dimension.kind === 'external') dimension.encode?.(commandEncoder);
      }
      for (const predicate of [...this.predicates.values(), ...this.domainPredicates.values()])
        predicate.query.encode(commandEncoder, predicate.parameters);
      this.compiled.encode(commandEncoder, {parameters: undefined});
      this.encodedRevision = this.revision;
      return this.revision;
    } catch (error) {
      this.encodedRevision = -1;
      throw error;
    }
  }

  /** Call if a recorded frame is abandoned or submission fails. */
  discardEncoding(): void {
    this.assertActive();
    this.encodedRevision = -1;
  }

  /** Read only the four-byte selected-row count for bounded export preflight. */
  async readSelectedCount(): Promise<Pick<
    GPUResidentSummary,
    'revision' | 'datasetRevision' | 'selectedCount'
  > | null> {
    const revision = this.beginRead();
    if (revision === null) return null;
    try {
      this.transferredBytes += 4;
      const selectedCount = await this.readSelectedCountValue();
      if (!this.isCurrent(revision)) return null;
      return {
        revision,
        datasetRevision: this.datasetRevision,
        selectedCount
      };
    } finally {
      this.endRead();
    }
  }

  /** Configured row limit for one inspection or export page. */
  get maximumInspectionPageSize(): number {
    return this.maximumPageSize;
  }

  /** Configured byte limit for one inspection or export page, including its count read. */
  get maximumInspectionReadbackBytes(): number {
    return this.maximumReadbackBytes;
  }

  async readSummary(): Promise<GPUResidentSummary | null> {
    const revision = this.beginRead();
    if (revision === null) return null;
    try {
      const buffers = [
        this.selectedCount,
        ...this.histogramBuffers.values(),
        ...[...this.domainBuffers.values()].map(value => value.buffer)
      ];
      this.transferredBytes += buffers.reduce((total, buffer) => total + buffer.byteLength, 0);
      const results = await this.readThroughRing(
        buffers.map(buffer => ({buffer, byteLength: buffer.byteLength}))
      );
      if (!this.isCurrent(revision)) return null;
      return {
        revision,
        datasetRevision: this.datasetRevision,
        selectedCount: new Uint32Array(results[0].buffer, results[0].byteOffset, 1)[0],
        domains: Object.fromEntries(
          [...this.domainBuffers].map(([id, output], index) => {
            const values = decodeColumn(
              output.format,
              results[1 + this.histogramBuffers.size + index]
            );
            const bytes = results[1 + this.histogramBuffers.size + index];
            const count = new Uint32Array(bytes.buffer, bytes.byteOffset + 8, 1)[0];
            return [id, count ? [values[0], values[1]] : null];
          })
        ),
        histograms: Object.fromEntries(
          [...this.histogramBuffers.keys()].map((id, index) => [
            id,
            new Uint32Array(
              results[index + 1].buffer,
              results[index + 1].byteOffset,
              results[index + 1].byteLength / 4
            )
          ])
        )
      };
    } finally {
      this.endRead();
    }
  }

  /** Read a bounded page in stable selected source order. No full selection array is downloaded. */
  async readPage(
    offset: number,
    limit: number,
    columns: readonly string[] = Object.keys(this.columns)
  ): Promise<GPUResidentPage | null> {
    this.assertActive();
    if (
      !Number.isSafeInteger(offset) ||
      offset < 0 ||
      !Number.isInteger(limit) ||
      limit < 0 ||
      limit > this.maximumPageSize
    )
      throw new Error('Invalid bounded page request');
    if (new Set(columns).size !== columns.length || columns.some(name => !this.columns[name]))
      throw new Error('Unknown or duplicate inspection column');
    this.validateInspectionBudget(limit, columns);
    const inspectionColumns = [...columns];
    const revision = this.beginRead();
    if (revision === null) return null;
    try {
      this.transferredBytes += 4;
      const count = await this.readSelectedCountValue();
      if (!this.isCurrent(revision)) return null;
      const length = Math.min(limit, Math.max(0, count - offset));
      return await this.gatherPage(offset, length, inspectionColumns, revision);
    } finally {
      this.endRead();
    }
  }

  /** Read specific source rows for picking; source order is independent of current selection. */
  async readRows(
    sourceRows: readonly number[],
    columns: readonly string[] = Object.keys(this.columns)
  ): Promise<GPUResidentPage | null> {
    this.assertActive();
    if (
      sourceRows.length > this.maximumPageSize ||
      sourceRows.some(row => !Number.isInteger(row) || row < 0 || row >= this.rowCount)
    )
      throw new Error('Invalid bounded source-row request');
    if (new Set(columns).size !== columns.length || columns.some(name => !this.columns[name]))
      throw new Error('Unknown or duplicate inspection column');
    this.validateInspectionBudget(sourceRows.length, columns);
    const rows = [...sourceRows];
    const inspectionColumns = [...columns];
    const revision = this.beginRead();
    if (revision === null) return null;
    try {
      return await this.gatherPage(0, rows.length, inspectionColumns, revision, rows);
    } finally {
      this.endRead();
    }
  }

  destroy(): void {
    this.destroyed = true;
    if (!this.activeReads) this.releaseResources();
  }

  /** Resolves when pending inspections finish and all owned resources have been released. */
  async destroyAsync(): Promise<void> {
    this.destroy();
    if (this.resourcesReleased) return;
    await new Promise<void>(resolve => this.releaseWaiters.push(resolve));
  }

  /** Exact retained output allocation; excludes compiled graph scratch and borrowed sources. */
  get ownedOutputByteLength(): number {
    return this.vectors.reduce(
      (total, vector) =>
        total + vector.data.reduce((sum, data) => sum + getBuffer(data).byteLength, 0),
      0
    );
  }

  /** Explicit CPU payload read by this runtime, excluding caller readbacks. */
  get readbackByteLength(): number {
    return this.transferredBytes;
  }

  /**
   * Accounted resident outputs and compiled graph transient resources. Excludes borrowed source
   * storage, driver/pipeline/private executable allocations, and command-encoder-owned ordered
   * control uploads; reserve submission/driver headroom separately.
   */
  get accountedResidentByteLength(): number {
    const buffers = new Set<Buffer>();
    let transientBytes = this.compiled.stats.physicalTransientResourceBytes;
    for (const predicate of [...this.predicates.values(), ...this.domainPredicates.values()]) {
      for (const vector of [
        predicate.query.selectionMask,
        predicate.query.rowIndices,
        predicate.query.selectedCounts
      ]) {
        for (const data of vector.data) buffers.add(getBuffer(data));
      }
      transientBytes += predicate.scratchByteLength;
    }
    // Neutral range selections own one 20-byte parameter buffer each.
    const neutralBuffers =
      (this.dimensions.length ? 0 : 20) +
      this.domainFilters.reduce(
        (total, filter) => total + (filter.dimensions[0]?.id === 'unfiltered' ? 20 : 0),
        0
      );
    return (
      this.ownedOutputByteLength +
      transientBytes +
      neutralBuffers +
      [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0)
    );
  }

  /** GPU gather payload plus readback staging payload reserved across allowed inspections. */
  get inspectionReservationByteLength(): number {
    return this.maximumConcurrentReads * this.maximumReadbackBytes * 2;
  }

  private validateInspectionBudget(rows: number, names: readonly string[]): void {
    const valuesPerRow = 1 + names.length + names.filter(name => this.validity[name]).length;
    if (rows * valuesPerRow * 4 + 4 > this.maximumReadbackBytes)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Inspection exceeds readback byte budget'
      });
  }

  private async gatherPage(
    offset: number,
    length: number,
    names: readonly string[],
    revision: number,
    sourceRows?: readonly number[]
  ): Promise<GPUResidentPage | null> {
    if (!length)
      return {
        revision,
        datasetRevision: this.datasetRevision,
        sourceRowIndices: new Uint32Array(),
        columns: Object.fromEntries(
          names.map(name => [
            name,
            decodeColumn(getScalarFormat(this.columns[name]), new Uint8Array())
          ])
        ),
        validity: {}
      };
    const buffers: Buffer[] = [];
    const graph = new GPUCommandGraph(this.device, {id: 'kepler-inspection'});
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const indices = this.device.createBuffer({
        id: 'inspection-indices',
        byteLength: length * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(indices);
      const indicesView = graph.importGPUData(
        'indices',
        new GPUData({buffer: indices, format: 'uint32', length, ownsBuffer: false})
      );
      const outputs: {
        name: string;
        validity: boolean;
        format: GPUResidentScalarFormat;
        buffer: Buffer;
      }[] = [];
      for (const name of names) {
        for (const [isValidity, source] of [
          [false, this.columns[name]],
          [true, this.validity[name]]
        ] as const) {
          if (!source) continue;
          const output = this.device.createBuffer({
            id: `inspection-${name}-${isValidity}`,
            byteLength: length * 4,
            usage: Buffer.STORAGE | Buffer.COPY_SRC
          });
          buffers.push(output);
          outputs.push({
            name,
            validity: isValidity,
            format: getScalarFormat(source),
            buffer: output
          });
          graph.add(
            new GPUGather({
              id: `gather-${name}-${isValidity}`,
              source: graph.importGPUVector(`source-${name}-${isValidity}`, source),
              indices: indicesView,
              output: graph.importGPUData(
                `output-${name}-${isValidity}`,
                new GPUData({buffer: output, format: source.format, length, ownsBuffer: false})
              )
            })
          );
        }
      }
      const gather = await graph.compileAsync();
      compiled = gather;
      this.transferredBytes += buffers.length * length * 4;
      // Index copy, gather and staging copies share one encoder and one submission.
      const bytes = await this.readThroughRing(
        buffers.map(buffer => ({buffer, byteLength: length * 4})),
        encoder => {
          if (sourceRows) indices.write(new Uint32Array(sourceRows));
          else copyVectorRange(encoder, this.selectedRows, offset, length, indices);
          gather.encode(encoder, {parameters: undefined});
        }
      );
      if (!this.isCurrent(revision)) return null;
      const page: GPUResidentPage = {
        revision,
        datasetRevision: this.datasetRevision,
        sourceRowIndices: new Uint32Array(bytes[0].buffer, bytes[0].byteOffset, length),
        columns: Object.create(null),
        validity: Object.create(null)
      };
      for (const [index, output] of outputs.entries()) {
        if (output.validity)
          page.validity[output.name] = new Uint32Array(
            bytes[index + 1].buffer,
            bytes[index + 1].byteOffset,
            length
          );
        else page.columns[output.name] = decodeColumn(output.format, bytes[index + 1]);
      }
      return page;
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }

  private async readSelectedCountValue(): Promise<number> {
    const [bytes] = await this.readThroughRing([{buffer: this.selectedCount, byteLength: 4}]);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
  }

  /**
   * Records `recordWork` and then copies each source range, back to back, into one staging slot
   * on a single encoder; submits once and maps once. Returns one independent copy per source.
   * Admission (beginRead) caps reads at the slot count. Each read holds at most one slot at a
   * time and remains admitted until its mapping, including error cleanup, has settled.
   */
  private async readThroughRing(
    sources: readonly {buffer: Buffer; byteLength: number}[],
    recordWork?: (commandEncoder: CommandEncoder) => void
  ): Promise<Uint8Array[]> {
    const ticket = await this.acquireReadbackTicket();
    let state: 'reserved' | 'encoded' | 'submitted' = 'reserved';
    try {
      const commandEncoder = this.device.createCommandEncoder();
      recordWork?.(commandEncoder);
      let byteLength = 0;
      for (const source of sources) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source.buffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: byteLength,
          size: source.byteLength
        });
        byteLength += source.byteLength;
      }
      ticket.markEncoded({byteOffset: 0, byteLength});
      state = 'encoded';
      this.device.submit(commandEncoder.finish());
      state = 'submitted';
      const bytes = await ticket.read();
      let byteOffset = 0;
      return sources.map(source => {
        const part = bytes.slice(byteOffset, byteOffset + source.byteLength);
        byteOffset += source.byteLength;
        return part;
      });
    } finally {
      // The ring's rule: an unused slot is cancelled; an encoded one is read before it is reused.
      if (state === 'reserved') ticket.cancel();
      else if (state === 'encoded') await ticket.read().catch(() => undefined);
    }
  }

  private acquireReadbackTicket(): GPUReadbackTicket {
    for (const ring of this.readbackRings) {
      const ticket = ring.tryAcquire();
      if (ticket) return ticket;
    }
    // Admission limits concurrent tickets; an unavailable full pool cannot serve this read.
    if (this.readbackRings.length >= this.maximumConcurrentReads)
      throw new KeplerAllocationError('device-out-of-memory', {
        message: 'Readback staging slots are unavailable'
      });
    let ring: GPUReadbackRing;
    try {
      // A one-slot constructor cannot leave earlier allocations behind if allocation fails.
      ring = new GPUReadbackRing(this.device, {
        id: `kepler-explorer-readback-${this.readbackRings.length}`,
        byteLength: this.getReadbackSlotByteLength(),
        slotCount: 1
      });
    } catch (cause) {
      throw new KeplerAllocationError('device-out-of-memory', {cause});
    }
    this.readbackRings.push(ring);
    return ring.tryAcquire()!;
  }

  /**
   * Largest payload one admitted read copies: a summary, or a full page of row IDs, every column
   * and every validity mask. Both are validated against maximumReadbackBytes, so a slot never
   * exceeds the readback reservation.
   */
  private getReadbackSlotByteLength(): number {
    const summaryByteLength = [
      this.selectedCount,
      ...this.histogramBuffers.values(),
      ...[...this.domainBuffers.values()].map(output => output.buffer)
    ].reduce((total, buffer) => total + buffer.byteLength, 0);
    const names = Object.keys(this.columns);
    const pageByteLength =
      this.maximumPageSize *
      (1 + names.length + names.filter(name => this.validity[name]).length) *
      4;
    const budgetByteLength = Math.floor(this.maximumReadbackBytes / 4) * 4;
    return Math.max(4, Math.min(budgetByteLength, Math.max(summaryByteLength, pageByteLength)));
  }

  private allocateVector(name: string, lengths: readonly number[]): GPUVector<'uint32'> {
    const data: GPUData<'uint32'>[] = [];
    try {
      for (const length of lengths)
        data.push(
          new GPUData({
            buffer: this.device.createBuffer({
              id: name,
              byteLength: Math.max(4, length * 4),
              usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
            }),
            format: 'uint32',
            length,
            ownsBuffer: true
          })
        );
      const vector = new GPUVector({name, type: 'data', format: 'uint32', data, ownsData: true});
      this.vectors.push(vector);
      return vector;
    } catch (error) {
      for (const chunk of data) chunk.destroy();
      throw error;
    }
  }

  private setPredicateParameters(id: string, parameters: Record<string, number | boolean>): void {
    this.predicates.get(id)!.parameters = parameters;
    this.setDimensionActive(this.filter, id, true);
    this.updateDomainFilters(id, true);
    this.invalidate();
  }
  private updateDomainFilters(id: string, active: boolean): void {
    for (const scopedFilter of this.domainFilters) {
      if (scopedFilter.dimensions.some(dimension => dimension.id === id))
        this.setDimensionActive(scopedFilter, id, active);
    }
  }

  /** Every dimension is a mask range: active selects nonzero flags, inactive bypasses the mask. */
  private setDimensionActive(filter: GPUCrossfilter, id: string, active: boolean): void {
    if (active) filter.setRange(id, SELECTED_MASK_RANGE);
    else filter.clear(id);
  }

  private requireDimension(id: string, kind?: GPUResidentDimension['kind']): GPUResidentDimension {
    this.assertActive();
    const dimension = this.dimensions.find(candidate => candidate.id === id);
    if (!dimension || (kind && dimension.kind !== kind))
      throw new Error('Unknown or incompatible exploration dimension');
    return dimension;
  }
  private assertActive(): void {
    if (this.destroyed) throw new Error('Explorer is destroyed');
  }
  private isCurrent(revision: number): boolean {
    return !this.destroyed && this.revision === revision && this.encodedRevision === revision;
  }
  private beginRead(): number | null {
    this.assertActive();
    if (this.encodedRevision !== this.revision || this.activeReads >= this.maximumConcurrentReads)
      return null;
    this.activeReads++;
    return this.revision;
  }
  private endRead(): void {
    this.activeReads--;
    if (this.destroyed && !this.activeReads) this.releaseResources();
  }
  private releaseResources(): void {
    if (this.resourcesReleased) return;
    this.resourcesReleased = true;
    for (const ring of this.readbackRings) ring.destroy();
    this.compiled.destroy();
    this.filter.destroy();
    for (const scopedFilter of this.domainFilters) scopedFilter.destroy();
    for (const predicate of [...this.predicates.values(), ...this.domainPredicates.values()])
      predicate.query.destroy();
    for (const vector of this.vectors) vector.destroy();
    for (const resolve of this.releaseWaiters.splice(0)) resolve();
  }
}

function createPredicate(
  dimension: Exclude<GPUResidentDimension, {kind: 'external'}>
): GPUExpression<boolean> {
  let predicate: GPUExpression<boolean>;
  if (dimension.kind === 'range') {
    predicate = and(
      column(dimension.column).greaterThanOrEqual(parameter('minimum', 0)),
      dimension.exclusiveMaximum
        ? column(dimension.column).lessThan(parameter('maximum', 0))
        : column(dimension.column).lessThanOrEqual(parameter('maximum', 0))
    );
  } else if (dimension.kind === 'bounds') {
    predicate = and(
      and(
        column(dimension.x).greaterThanOrEqual(parameter('minimumX', 0)),
        column(dimension.x).lessThanOrEqual(parameter('maximumX', 0))
      ),
      and(
        column(dimension.y).greaterThanOrEqual(parameter('minimumY', 0)),
        column(dimension.y).lessThanOrEqual(parameter('maximumY', 0))
      )
    );
  } else {
    predicate = combineCategories(
      Array.from({length: dimension.capacity}, (_, index) =>
        and(
          parameter(`enabled${index}`, false),
          column(dimension.column).equal(parameter(`category${index}`, 0))
        )
      )
    );
  }
  return and(parameter('enabled', true), predicate);
}

function combineCategories(expressions: GPUExpression<boolean>[]): GPUExpression<boolean> {
  if (expressions.length === 1) return expressions[0];
  const middle = Math.floor(expressions.length / 2);
  return or(
    combineCategories(expressions.slice(0, middle)),
    combineCategories(expressions.slice(middle))
  );
}

function validateExplorerProps(props: GPUResidentExplorerProps): void {
  if (!Number.isSafeInteger(props.datasetRevision ?? 1) || (props.datasetRevision ?? 1) < 1)
    throw new Error('Dataset revision must be positive');
  const maximumReads = props.maximumConcurrentReads ?? 2;
  const maximumBytes = props.maximumReadbackBytes ?? 1024 * 1024;
  if (
    !Number.isInteger(maximumReads) ||
    maximumReads < 1 ||
    maximumReads > 16 ||
    !Number.isSafeInteger(maximumBytes) ||
    maximumBytes < 4
  )
    throw new Error('Invalid bounded readback configuration');
  if (
    4 +
      (props.histograms ?? []).reduce((total, histogram) => total + histogram.binCount * 4, 0) +
      (props.domains?.length ?? 0) * 12 >
    maximumBytes
  )
    throw new KeplerAllocationError('memory-budget', {
      message: 'Summaries exceed readback byte budget'
    });
  const columns = Object.values(props.columns);
  if (!columns.length && !props.sourceChunkLengths)
    throw new Error('Explorer requires source chunk lengths when no scalar columns are selected');
  if (props.device.type !== 'webgpu') throw new Error('Explorer requires WebGPU');
  // The published crossfilter composes every selection with one GPUMask pass that binds each
  // selection mask plus the output as storage buffers, and GPUMask does not split by device limit.
  const composedSelectionCount = Math.max(1, props.dimensions.length + (props.liveMask ? 1 : 0));
  const storageBufferLimit = props.device.limits.maxStorageBuffersPerShaderStage;
  if (composedSelectionCount + 1 > storageBufferLimit)
    throw new Error(
      `Explorer dimensions: ${composedSelectionCount} selection masks need ${composedSelectionCount + 1} storage buffers per shader stage, but the device limit maxStorageBuffersPerShaderStage is ${storageBufferLimit}; request a higher limit when creating the device or use fewer dimensions`
    );
  const lengths = props.sourceChunkLengths ?? columns[0].data.map(data => data.length);
  if (lengths.some(length => !Number.isSafeInteger(length) || length < 0))
    throw new Error('Source chunk lengths must be non-negative safe integers');
  if (
    columns.some(
      vector =>
        !['float32', 'uint32', 'sint32'].includes(vector.format ?? '') ||
        vector.data.length !== lengths.length ||
        vector.data.some(
          (data, index) => data.length !== lengths[index] || getBuffer(data).device !== props.device
        )
    )
  )
    throw new Error('Explorer columns require aligned scalar chunks on one device');
  if (lengths.reduce((total, length) => total + length, 0) > 0xffffffff)
    throw new Error('Source row identities exceed uint32');
  const dimensionIds = new Set(props.dimensions.map(dimension => dimension.id));
  if (dimensionIds.size !== props.dimensions.length || dimensionIds.has('unfiltered'))
    throw new Error('Duplicate or reserved dimension identifier');
  for (const dimension of props.dimensions) {
    const names =
      dimension.kind === 'external'
        ? []
        : dimension.kind === 'bounds'
          ? [dimension.x, dimension.y]
          : [dimension.column];
    if (names.some(name => !props.columns[name])) throw new Error('Unknown dimension column');
    if (
      dimension.kind === 'category' &&
      (props.columns[dimension.column].format !== 'uint32' ||
        !Number.isInteger(dimension.capacity) ||
        dimension.capacity < 1 ||
        dimension.capacity > 64)
    )
      throw new Error('Categories require uint32 keys and capacity from 1 to 64');
  }
  const histograms = props.histograms ?? [];
  if (
    new Set(histograms.map(histogram => histogram.id)).size !== histograms.length ||
    histograms.some(
      histogram =>
        histogram.id === 'visible' ||
        !props.columns[histogram.column] ||
        (histogram.dimension && !dimensionIds.has(histogram.dimension)) ||
        !Number.isInteger(histogram.binCount) ||
        histogram.binCount < 1 ||
        histogram.binCount > 65536
    )
  )
    throw new Error('Invalid histogram configuration');
  const domains = props.domains ?? [];
  if (
    new Set(domains.map(domain => domain.id)).size !== domains.length ||
    domains.some(
      domain =>
        !props.columns[domain.column] ||
        domain.dimensions?.some(id => !dimensionIds.has(id)) ||
        (domain.dimensions && new Set(domain.dimensions).size !== domain.dimensions.length)
    )
  )
    throw new Error('Invalid domain configuration');
  for (const histogram of histograms) validateRange(histogram.domain[0], histogram.domain[1]);
  if (
    !Number.isInteger(props.maximumPageSize ?? 256) ||
    (props.maximumPageSize ?? 256) < 1 ||
    (props.maximumPageSize ?? 256) > 4096
  )
    throw new Error('Page size must be between 1 and 4096');
}

function validateRange(minimum: number, maximum: number): void {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum)
    throw new Error('Invalid finite selection range');
}
function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof DynamicBuffer ? data.buffer.buffer : data.buffer;
}
function decodeColumn(
  format: GPUResidentScalarFormat,
  bytes: Uint8Array
): Float32Array | Int32Array | Uint32Array {
  const ArrayType =
    format === 'float32' ? Float32Array : format === 'sint32' ? Int32Array : Uint32Array;
  return new ArrayType(bytes.slice().buffer);
}
function copyVectorRange(
  encoder: CommandEncoder,
  vector: GPUVector<'uint32'>,
  offset: number,
  length: number,
  destination: Buffer
): void {
  let sourceOffset = 0;
  let destinationOffset = 0;
  for (const data of vector.data) {
    const start = Math.max(offset, sourceOffset);
    const end = Math.min(offset + length, sourceOffset + data.length);
    if (end > start) {
      encoder.copyBufferToBuffer({
        sourceBuffer: getBuffer(data),
        sourceOffset: data.byteOffset + (start - sourceOffset) * 4,
        destinationBuffer: destination,
        destinationOffset,
        size: (end - start) * 4
      });
      destinationOffset += (end - start) * 4;
    }
    sourceOffset += data.length;
  }
}

function getScalarFormat(vector: GPUVector<GPUResidentScalarFormat>): GPUResidentScalarFormat {
  const format = vector.format;
  if (format !== 'float32' && format !== 'uint32' && format !== 'sint32')
    throw new Error('Unsupported scalar format');
  return format;
}

function validateColumnRange(
  vector: GPUVector<GPUResidentScalarFormat>,
  minimum: number,
  maximum: number
): void {
  validateRange(minimum, maximum);
  const format = getScalarFormat(vector);
  if (format === 'float32') {
    if (!Number.isFinite(Math.fround(minimum)) || !Number.isFinite(Math.fround(maximum)))
      throw new Error('Range exceeds Float32 capacity');
  } else {
    const lower = format === 'uint32' ? 0 : -0x80000000;
    const upper = format === 'uint32' ? 0xffffffff : 0x7fffffff;
    if (
      !Number.isInteger(minimum) ||
      !Number.isInteger(maximum) ||
      minimum < lower ||
      maximum > upper
    )
      throw new Error('Range must fit its integer column');
  }
}
