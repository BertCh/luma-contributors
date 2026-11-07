// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {MeteredCommandGraph} from '../session/metered-command-graph';
import {
  GPUDataFrame,
  and,
  or,
  not,
  column,
  literal,
  parameter,
  type GPUExpression,
  type GPUDataFrameQueryParameters,
  type CompiledGPUDataFrameQuery
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import {
  KeplerAllocationError,
  type KeplerDatasetRegistry,
  type KeplerMemoryReservation
} from '../runtime/dataset-registry';
import {
  GPUKeplerTimeSelection,
  convertKeplerTimeInstant,
  type GPUKeplerTimeSelectionProps,
  type GPUKeplerTimeWindow
} from './gpu-time-selection';
import {normalizeKeplerScalarRange} from './scalar-range';

export type GPUKeplerScalarFilter =
  | {filterId: string; kind: 'range'; column: string; exclusiveMaximum?: boolean}
  | {filterId: string; kind: 'category'; column: string; capacity: number}
  | (Omit<GPUKeplerTimeSelectionProps, 'device' | 'id'> & {
      filterId: string;
      kind: 'time';
      /** Trusted native calendar-chart click only. Ordinary time sliders keep null rejection. */
      chartNullAsEpoch?: boolean;
    });

export type GPUKeplerScalarSelectionProps = {
  device: Device;
  columns?: Readonly<Record<string, GPUVector<'float32' | 'uint32' | 'sint32'>>>;
  validity?: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
  filters: readonly GPUKeplerScalarFilter[];
  datasetRevision?: number;
  /** Optional application admission. Supplying a registry requires datasetId and a budget. */
  registry?: KeplerDatasetRegistry;
  datasetId?: string;
  workingMemoryBudgetBytes?: number;
};

/** JSON-safe control snapshots; exact time ticks use decimal strings in their source unit. */
export type GPUKeplerScalarControl = {
  filterId: string;
  enabled: boolean;
  empty: boolean;
} & (
  | {kind: 'range'; minimum: number; maximum: number}
  | {kind: 'category'; categories: readonly number[]}
  | {kind: 'time'; window: {start: string; end: string; unit: GPUKeplerTimeWindow['unit']}}
);

export type GPUKeplerScalarRevision = Readonly<{datasetRevision: number; revision: number}>;

type ScalarFormat = 'float32' | 'uint32' | 'sint32';
type PredicateState = {
  filter: GPUKeplerScalarFilter;
  source: GPUVector<ScalarFormat>;
  query: CompiledGPUDataFrameQuery<Record<string, ScalarFormat>>;
  /** Physical transient scratch of the compiled query graph. */
  scratchByteLength: number;
  mask: GPUVector<'uint32'>;
  time?: GPUKeplerTimeSelection;
  control: GPUKeplerScalarControl;
  parameters: Record<string, number | boolean | null>;
  revision: number;
  /** Every source vector this predicate reads, with the chunk topology it was built against. */
  topology: SourceTopology[];
};
type SourceTopology = {
  vector: GPUVector;
  chunks: readonly GPUData[];
  lengths: number[];
  buffers: Buffer[];
};

/** Columns and validity a single added filter reads; must align with the existing sources. */
export type GPUKeplerScalarFilterSources = {
  columns?: Readonly<Record<string, GPUVector<'float32' | 'uint32' | 'sint32'>>>;
  validity?: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
};

/**
 * One stable, chunk-preserving predicate mask per native Kepler filter ID.
 *
 * Clearing a predicate includes null rows, letting consumers compose any subset of filters without
 * recompilation. Active range/category predicates retain SQL null rejection. All controls and masks
 * are resident: this adapter performs no readback and never traverses source rows on the CPU.
 * Encode before GPUKeplerSelectionScopes; invalidate that consumer after any control/source update.
 * addFilter/removeFilter change a single filter slot in place (identical to a full rebuild of the
 * final filter set); replacing source chunks still requires a new adapter. The caller retains
 * source leases and supplies a completion fence to close after the final dependent submission.
 */
export class GPUKeplerScalarSelection {
  readonly rowCount: number;
  readonly datasetRevision: number;
  private readonly device: Device;
  private readonly states = new Map<string, PredicateState>();
  private readonly chunkLengths: readonly number[];
  private readonly retiringStates = new Set<PredicateState>();
  private readonly destroyedStates = new WeakSet<PredicateState>();
  private readonly registry?: KeplerDatasetRegistry;
  private readonly datasetId?: string;
  private readonly deviceEpoch?: number;
  private readonly reservation?: KeplerMemoryReservation;
  private workingMemoryBudgetBytes?: number;
  private revision = 0;
  private encodedRevision = -1;
  private closing = false;
  private released = false;
  private completion?: Promise<void>;

  constructor(props: GPUKeplerScalarSelectionProps) {
    const sources = validateProps(props);
    this.device = props.device;
    this.rowCount = sources[0].length;
    this.registry = props.registry;
    this.datasetId = props.datasetId;
    this.datasetRevision =
      props.datasetRevision ??
      (props.datasetId ? props.registry?.getDataset(props.datasetId)?.revision : undefined) ??
      1;
    this.deviceEpoch = props.registry?.deviceEpoch;
    this.workingMemoryBudgetBytes = props.workingMemoryBudgetBytes;
    this.chunkLengths = sources[0].data.map(chunk => chunk.length);
    // Reserve before constructing query outputs, time buffers or graph scratch.
    this.reservation = props.registry?.reserveMemory(
      props.workingMemoryBudgetBytes!,
      'gpu',
      'scratch'
    );
    try {
      for (const filter of props.filters) {
        const state = this.createPredicate(props, filter);
        this.states.set(filter.filterId, state);
        this.assertBudget();
      }
    } catch (error) {
      this.releaseResources();
      throw error;
    }
  }

  /** Stable per-filter masks; a fresh frozen snapshot, so re-read after addFilter/removeFilter. */
  get predicateMasks(): Readonly<Record<string, GPUVector<'uint32'>>> {
    const masks: Record<string, GPUVector<'uint32'>> = Object.create(null);
    for (const [filterId, state] of this.states) masks[filterId] = state.mask;
    return Object.freeze(masks);
  }

  get filterIds(): readonly string[] {
    return [...this.states.keys()];
  }

  hasFilter(filterId: string): boolean {
    return this.states.has(filterId);
  }

  get currentRevision(): number {
    return this.revision;
  }

  getMask(filterId: string): GPUVector<'uint32'> {
    return this.getState(filterId).mask;
  }

  getFilterRevision(filterId: string): number {
    return this.getState(filterId).revision;
  }

  acquireRevision(): GPUKeplerScalarRevision {
    this.assertActive();
    return {datasetRevision: this.datasetRevision, revision: this.revision};
  }

  /** Only publish a consumer result if its acquired revision is still encoded and current. */
  isCurrent(snapshot: GPUKeplerScalarRevision): boolean {
    return (
      !this.closing &&
      this.isDatasetCurrent() &&
      snapshot.datasetRevision === this.datasetRevision &&
      snapshot.revision === this.revision &&
      this.encodedRevision === this.revision
    );
  }

  setRange(filterId: string, minimum: number, maximum: number): number {
    const state = this.getState(filterId, 'range');
    return this.applyControl(state, {
      filterId,
      kind: 'range',
      enabled: true,
      empty: false,
      minimum,
      maximum
    });
  }

  setCategories(filterId: string, categories: readonly number[]): number {
    const state = this.getState(filterId, 'category');
    return this.applyControl(state, {
      filterId,
      kind: 'category',
      enabled: true,
      empty: false,
      categories
    });
  }

  setTimeWindow(filterId: string, window: GPUKeplerTimeWindow): number {
    const state = this.getState(filterId, 'time');
    const filter = state.filter;
    if (filter.kind !== 'time') throw new Error('Expected time filter');
    return this.applyControl(state, {
      filterId,
      kind: 'time',
      enabled: true,
      empty: false,
      window: {
        start: convertKeplerTimeInstant(window.start, window.unit, filter.unit).toString(),
        end: convertKeplerTimeInstant(window.end, window.unit, filter.unit).toString(),
        unit: filter.unit
      }
    });
  }

  setEmpty(filterId: string): number {
    const state = this.getState(filterId);
    return this.applyControl(state, {...state.control, enabled: true, empty: true});
  }

  clear(filterId: string): number {
    const state = this.getState(filterId);
    return this.applyControl(state, {...state.control, enabled: false, empty: false});
  }

  clearAll(): number {
    this.assertActive();
    for (const state of this.states.values()) this.clear(state.filter.filterId);
    return this.revision;
  }

  getControls(): readonly GPUKeplerScalarControl[] {
    this.assertActive();
    return [...this.states.values()].map(state => cloneControl(state.control));
  }

  /** Validates a complete control snapshot before applying any changes. Topology stays fixed. */
  setControls(controls: readonly GPUKeplerScalarControl[]): number {
    this.assertActive();
    if (
      controls.length !== this.states.size ||
      new Set(controls.map(control => control.filterId)).size !== this.states.size
    )
      throw new Error('Scalar control snapshot must contain every filter exactly once');
    return this.updateControls(controls);
  }

  /** Validates then applies controls for the listed existing filters only (others keep theirs). */
  updateControls(controls: readonly GPUKeplerScalarControl[]): number {
    this.assertActive();
    const updates = controls.map(control => {
      const state = this.getState(control.filterId, control.kind);
      return {state, control: validateControl(state, control)};
    });
    for (const {state, control} of updates) this.applyControl(state, control);
    return this.revision;
  }

  /**
   * Adds one filter slot without touching the others. The new filter starts cleared, exactly as in
   * the constructor; its sources must match the existing chunk topology. Invalidates encodings.
   * Pass workingMemoryBudgetBytes to grow the reservation before anything is allocated.
   */
  addFilter(
    filter: GPUKeplerScalarFilter,
    sources: GPUKeplerScalarFilterSources = {},
    options: {workingMemoryBudgetBytes?: number} = {}
  ): GPUVector<'uint32'> {
    this.assertActive();
    if (this.states.has(filter.filterId)) throw new Error('Duplicate or empty scalar filter ID');
    const props: GPUKeplerScalarSelectionProps = {
      device: this.device,
      columns: sources.columns,
      validity: sources.validity,
      filters: [filter]
    };
    const vectors = validateProps(props);
    for (const vector of vectors)
      if (
        vector.length !== this.rowCount ||
        vector.data.length !== this.chunkLengths.length ||
        vector.data.some((chunk, index) => chunk.length !== this.chunkLengths[index])
      )
        throw new Error('Scalar source chunks must align on one device');
    const previousBudget = this.workingMemoryBudgetBytes;
    if (options.workingMemoryBudgetBytes !== undefined)
      this.growBudget(options.workingMemoryBudgetBytes);
    try {
      const state = this.createPredicate(props, filter);
      this.states.set(filter.filterId, state);
      try {
        this.assertBudget();
      } catch (error) {
        this.states.delete(filter.filterId);
        this.destroyState(state);
        throw error;
      }
      this.revision++;
      this.encodedRevision = -1;
      return state.mask;
    } catch (error) {
      if (previousBudget !== undefined && this.workingMemoryBudgetBytes !== previousBudget)
        this.shrinkBudget(previousBudget);
      throw error;
    }
  }

  /**
   * Removes one filter slot. Its mask and query are destroyed immediately, or after completion
   * settles when dependent GPU work may still read them. Other filters are untouched.
   */
  removeFilter(filterId: string, completion?: PromiseLike<unknown>): number {
    const state = this.getState(filterId);
    this.states.delete(filterId);
    this.revision++;
    this.encodedRevision = -1;
    if (completion) {
      this.retiringStates.add(state);
      void Promise.race([completion, this.device.lost])
        .catch(() => undefined)
        .finally(() => {
          this.retiringStates.delete(state);
          this.destroyState(state);
        });
    } else this.destroyState(state);
    return this.revision;
  }

  /** Grows the reservation (admission first) or shrinks it down to what is still accounted. */
  setWorkingMemoryBudget(byteLength: number): void {
    this.assertActive();
    if (this.workingMemoryBudgetBytes === undefined) return;
    if (byteLength > this.workingMemoryBudgetBytes) this.growBudget(byteLength);
    else if (byteLength >= this.accountedResidentByteLength) this.shrinkBudget(byteLength);
  }

  private growBudget(byteLength: number): void {
    if (this.workingMemoryBudgetBytes === undefined || byteLength <= this.workingMemoryBudgetBytes)
      return;
    this.reservation?.resize(byteLength);
    this.workingMemoryBudgetBytes = byteLength;
  }

  private shrinkBudget(byteLength: number): void {
    if (this.workingMemoryBudgetBytes === undefined) return;
    this.reservation?.resize(byteLength);
    this.workingMemoryBudgetBytes = byteLength;
  }

  private destroyState(state: PredicateState): void {
    if (this.destroyedStates.has(state)) return;
    this.destroyedStates.add(state);
    state.query.destroy();
    state.mask.destroy();
    state.time?.destroy();
  }

  /** Call after rewriting borrowed source values or validity without changing chunk topology. */
  invalidate(filterId?: string): number {
    this.assertActive();
    const states = filterId === undefined ? [...this.states.values()] : [this.getState(filterId)];
    for (const state of states) {
      state.revision++;
      state.time?.invalidate();
    }
    return ++this.revision;
  }

  encode(commandEncoder: CommandEncoder): number {
    this.assertActive();
    if (commandEncoder.device !== this.device) throw new Error('Scalar encoder device mismatch');
    for (const state of this.states.values())
      for (const {vector, chunks, lengths, buffers} of state.topology) {
        if (
          vector.length !== this.rowCount ||
          vector.data.length !== chunks.length ||
          chunks.some(
            (chunk, index) =>
              vector.data[index] !== chunk ||
              chunk.length !== lengths[index] ||
              getBuffer(chunk) !== buffers[index]
          )
        )
          throw new Error('Scalar source topology changed; recreate the adapter');
      }
    this.encodedRevision = -1;
    for (const state of this.states.values()) {
      state.time?.encode(commandEncoder);
      state.query.encode(commandEncoder, state.parameters);
      // Query masks are storage outputs. The public copies additionally support layer attributes.
      for (const [index, source] of state.query.selectionMask.data.entries()) {
        if (!source.length) continue;
        const destination = state.mask.data[index];
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getBuffer(source),
          sourceOffset: source.byteOffset,
          destinationBuffer: getBuffer(destination),
          destinationOffset: destination.byteOffset,
          size: source.length * 4
        });
      }
    }
    this.encodedRevision = this.revision;
    return this.encodedRevision;
  }

  discardEncoding(): void {
    this.encodedRevision = -1;
  }

  /**
   * Owned masks/query outputs/time controls/scratch. Source, driver/pipeline memory and transient
   * control-upload staging owned by the caller's command encoders are excluded.
   */
  get accountedResidentByteLength(): number {
    if (this.released) return 0;
    let byteLength = 0;
    for (const state of this.states.values()) {
      const buffers = new Set<Buffer>();
      for (const vector of [
        state.mask,
        state.query.selectionMask,
        state.query.rowIndices,
        state.query.selectedCounts
      ])
        for (const chunk of vector.data) buffers.add(getBuffer(chunk));
      byteLength +=
        [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0) +
        state.scratchByteLength +
        (state.time?.accountedResidentByteLength ?? 0);
    }
    return byteLength;
  }

  /** No inspection API: consumers explicitly budget their own bounded readbacks. */
  get readbackByteLength(): number {
    return 0;
  }

  /** Stops updates immediately; outputs and reservation survive until dependent GPU work settles. */
  close(completion: PromiseLike<unknown>): Promise<void> {
    if (!this.completion) {
      this.closing = true;
      this.encodedRevision = -1;
      this.completion = Promise.race([completion, this.device.lost])
        .finally(() => this.releaseResources())
        .then(() => undefined);
    }
    return this.completion;
  }

  private createPredicate(
    props: GPUKeplerScalarSelectionProps,
    filter: GPUKeplerScalarFilter
  ): PredicateState {
    let time: GPUKeplerTimeSelection | undefined;
    let query: PredicateState['query'] | undefined;
    let scratchByteLength = 0;
    let frame: GPUDataFrame<Record<string, ScalarFormat>> | undefined;
    let mask: GPUVector<'uint32'> | undefined;
    try {
      if (filter.kind === 'time') {
        time = new GPUKeplerTimeSelection({
          ...filter,
          device: this.device,
          id: `scalar-${filter.filterId}-time`
        });
        time.prepare();
      }
      const source = filter.kind === 'time' ? time!.mask : props.columns![filter.column];
      const validity = filter.kind === 'time' ? undefined : props.validity?.[filter.column];
      const table = new GPUTable<Record<string, ScalarFormat>>({
        vectors: {
          value: new GPUVector({
            name: 'value',
            type: 'data',
            format: source.format!,
            data: source.data,
            ownsData: false
          }),
          ...(filter.kind === 'time' && filter.chartNullAsEpoch && filter.validity
            ? {
                sourceValidity: new GPUVector({
                  name: 'sourceValidity',
                  type: 'data',
                  format: 'uint32',
                  data: filter.validity.data,
                  ownsData: false
                })
              }
            : {})
        }
      });
      for (const field of table.schema.fields) field.nullable = Boolean(validity);
      for (const batch of table.batches)
        for (const field of batch.schema.fields) field.nullable = Boolean(validity);
      frame = new GPUDataFrame({
        table,
        validity: validity ? {value: validity} : {},
        ownership: 'borrowed'
      });
      const predicate = makePredicate(filter);
      const queryGraph = new MeteredCommandGraph<GPUDataFrameQueryParameters>(this.device);
      query = frame
        .filter(or(not(parameter('active', false)), and(not(parameter('empty', false)), predicate)))
        .compile(queryGraph);
      scratchByteLength = queryGraph.transientByteLength;
      mask = makeMask(
        this.device,
        filter.filterId,
        source.data.map(chunk => chunk.length)
      );
      const control: GPUKeplerScalarControl =
        filter.kind === 'range'
          ? {
              filterId: filter.filterId,
              kind: 'range',
              enabled: false,
              empty: false,
              minimum: 0,
              maximum: 0
            }
          : filter.kind === 'category'
            ? {
                filterId: filter.filterId,
                kind: 'category',
                enabled: false,
                empty: false,
                categories: []
              }
            : {
                filterId: filter.filterId,
                kind: 'time',
                enabled: false,
                empty: false,
                window: {
                  start: convertKeplerTimeInstant(
                    filter.window.start,
                    filter.window.unit,
                    filter.unit
                  ).toString(),
                  end: convertKeplerTimeInstant(
                    filter.window.end,
                    filter.window.unit,
                    filter.unit
                  ).toString(),
                  unit: filter.unit
                }
              };
      const topology = validateProps({...props, filters: [filter]}).map(vector => ({
        vector,
        chunks: [...vector.data],
        lengths: vector.data.map(chunk => chunk.length),
        buffers: vector.data.map(getBuffer)
      }));
      return {
        filter: {...filter},
        source,
        query,
        scratchByteLength,
        mask,
        time,
        control,
        parameters: {},
        revision: 0,
        topology
      };
    } catch (error) {
      mask?.destroy();
      query?.destroy();
      time?.destroy();
      throw error;
    } finally {
      frame?.destroy();
    }
  }

  private applyControl(state: PredicateState, value: GPUKeplerScalarControl): number {
    const control = validateControl(state, value);
    const parameters: PredicateState['parameters'] = {
      active: control.enabled,
      empty: control.empty
    };
    if (control.kind === 'range') {
      const bounds = normalizeKeplerScalarRange(
        state.source.format!,
        control.minimum,
        control.maximum,
        state.filter.kind === 'range' && state.filter.exclusiveMaximum
      );
      parameters['minimum'] = bounds.minimum;
      parameters['maximum'] = bounds.maximum;
      parameters['empty'] = control.empty || bounds.empty;
    } else if (control.kind === 'category' && state.filter.kind === 'category') {
      for (let index = 0; index < state.filter.capacity; index++) {
        parameters[`enabled${index}`] = index < control.categories.length;
        parameters[`category${index}`] = control.categories[index] ?? 0;
      }
    } else if (control.kind === 'time') {
      parameters['includeNullTime'] =
        BigInt(control.window.start) <= 0n && BigInt(control.window.end) >= 0n;
      state.time!.setWindow({
        start: BigInt(control.window.start),
        end: BigInt(control.window.end),
        unit: control.window.unit
      });
    }
    state.control = control;
    state.parameters = parameters;
    state.revision++;
    return ++this.revision;
  }

  private getState(filterId: string, kind?: GPUKeplerScalarFilter['kind']): PredicateState {
    this.assertActive();
    const state = this.states.get(filterId);
    if (!state || (kind && state.filter.kind !== kind))
      throw new Error('Unknown scalar filter or kind');
    return state;
  }

  private assertBudget(): void {
    if (
      this.workingMemoryBudgetBytes !== undefined &&
      this.accountedResidentByteLength > this.workingMemoryBudgetBytes
    )
      throw new KeplerAllocationError('working-memory');
  }

  private isDatasetCurrent(): boolean {
    return (
      !this.registry ||
      (this.registry.deviceEpoch === this.deviceEpoch &&
        this.registry.getDataset(this.datasetId!)?.revision === this.datasetRevision)
    );
  }

  private assertActive(): void {
    if (this.closing || !this.isDatasetCurrent())
      throw new Error('Scalar selection is closed or stale');
  }

  private releaseResources(): void {
    if (this.released) return;
    this.released = true;
    for (const state of [...this.states.values(), ...this.retiringStates]) this.destroyState(state);
    this.reservation?.release();
  }
}

function makePredicate(filter: GPUKeplerScalarFilter): GPUExpression<boolean> {
  if (filter.kind === 'time') {
    const selected = column('value').notEqual(literal(0));
    return filter.chartNullAsEpoch && filter.validity
      ? or(
          selected,
          and(parameter('includeNullTime', false), column('sourceValidity').equal(literal(0)))
        )
      : selected;
  }
  if (filter.kind === 'range')
    return and(
      column('value').greaterThanOrEqual(parameter('minimum', 0)),
      column('value').lessThanOrEqual(parameter('maximum', 0))
    );
  return combineCategories(
    Array.from({length: filter.capacity}, (_, index) =>
      and(
        parameter(`enabled${index}`, false),
        column('value').equal(parameter(`category${index}`, 0))
      )
    )
  );
}

function combineCategories(predicates: GPUExpression<boolean>[]): GPUExpression<boolean> {
  if (predicates.length === 1) return predicates[0];
  const middle = Math.floor(predicates.length / 2);
  return or(
    combineCategories(predicates.slice(0, middle)),
    combineCategories(predicates.slice(middle))
  );
}

function cloneControl(control: GPUKeplerScalarControl): GPUKeplerScalarControl {
  return control.kind === 'category'
    ? {...control, categories: [...control.categories]}
    : control.kind === 'time'
      ? {...control, window: {...control.window}}
      : {...control};
}

function validateControl(
  state: PredicateState,
  control: GPUKeplerScalarControl
): GPUKeplerScalarControl {
  if (
    control.filterId !== state.filter.filterId ||
    control.kind !== state.filter.kind ||
    typeof control.enabled !== 'boolean' ||
    typeof control.empty !== 'boolean'
  )
    throw new Error('Scalar control does not match its filter');
  if (control.kind === 'range') {
    normalizeKeplerScalarRange(
      state.source.format!,
      control.minimum,
      control.maximum,
      state.filter.kind === 'range' && state.filter.exclusiveMaximum
    );
  } else if (control.kind === 'category' && state.filter.kind === 'category') {
    if (control.categories.length > state.filter.capacity)
      throw new KeplerAllocationError('capacity', {message: 'Category selection exceeds capacity'});
    for (const value of control.categories) validateScalar(value, 'uint32');
  } else if (control.kind === 'time' && state.filter.kind === 'time') {
    if (!/^-?\d+$/.test(control.window.start) || !/^-?\d+$/.test(control.window.end))
      throw new Error('Time control requires decimal integer ticks');
    const start = convertKeplerTimeInstant(
      BigInt(control.window.start),
      control.window.unit,
      state.filter.unit
    );
    const end = convertKeplerTimeInstant(
      BigInt(control.window.end),
      control.window.unit,
      state.filter.unit
    );
    if (start > end) throw new Error('Reversed time range; use setEmpty for an empty intersection');
    return {
      ...control,
      window: {start: start.toString(), end: end.toString(), unit: state.filter.unit}
    };
  }
  return cloneControl(control);
}

function validateScalar(value: number, format: ScalarFormat): void {
  if (
    !Number.isFinite(value) ||
    (format === 'float32'
      ? !Number.isFinite(Math.fround(value))
      : !Number.isInteger(value) ||
        value < (format === 'uint32' ? 0 : -0x80000000) ||
        value > (format === 'uint32' ? 0xffffffff : 0x7fffffff))
  )
    throw new Error('Invalid scalar control value');
}

function validateProps(props: GPUKeplerScalarSelectionProps): GPUVector[] {
  if (props.device.type !== 'webgpu') throw new Error('Scalar selection requires WebGPU');
  if (
    props.datasetRevision !== undefined &&
    (!Number.isSafeInteger(props.datasetRevision) || props.datasetRevision < 1)
  )
    throw new Error('Dataset revision must be positive');
  if (
    props.workingMemoryBudgetBytes !== undefined &&
    (!Number.isSafeInteger(props.workingMemoryBudgetBytes) || props.workingMemoryBudgetBytes < 0)
  )
    throw new Error('Invalid scalar working memory budget');
  if (
    props.registry &&
    (!props.datasetId ||
      props.workingMemoryBudgetBytes === undefined ||
      !props.registry.getDataset(props.datasetId))
  )
    throw new Error('Registry admission requires an existing dataset and working memory budget');
  if (
    props.registry &&
    props.datasetRevision !== undefined &&
    props.registry.getDataset(props.datasetId!)?.revision !== props.datasetRevision
  )
    throw new Error('Scalar dataset revision is stale');
  const sources: GPUVector[] = Object.values(props.columns ?? {});
  for (const vector of sources)
    if (!['float32', 'uint32', 'sint32'].includes(vector.format ?? ''))
      throw new Error('Scalar columns require portable numeric formats');
  for (const [name, validity] of Object.entries(props.validity ?? {})) {
    if (!validity) continue;
    if (!props.columns?.[name] || validity.format !== 'uint32')
      throw new Error('Invalid scalar validity column');
    sources.push(validity);
  }
  const identifiers = new Set<string>();
  for (const filter of props.filters) {
    if (!filter.filterId || identifiers.has(filter.filterId))
      throw new Error('Duplicate or empty scalar filter ID');
    identifiers.add(filter.filterId);
    if (filter.kind === 'time') {
      if (filter.chartNullAsEpoch && filter.endTimestamps)
        throw new Error('Chart null-as-epoch selection requires timestamp instants');
      sources.push(
        filter.timestamps,
        ...[filter.endTimestamps, filter.validity, filter.endValidity].filter(
          (vector): vector is GPUVector<'uint32x2'> | GPUVector<'uint32'> => Boolean(vector)
        )
      );
    } else {
      const source = props.columns?.[filter.column];
      if (!source) throw new Error('Scalar filter column is missing');
      if (
        filter.kind === 'category' &&
        (source.format !== 'uint32' ||
          !Number.isInteger(filter.capacity) ||
          filter.capacity < 1 ||
          filter.capacity > 64)
      )
        throw new Error('Category filters require uint32 codes and capacity between 1 and 64');
    }
  }
  if (!sources.length || sources[0].length > 0xffffffff)
    throw new Error('Scalar selection requires a source with uint32 row identities');
  const lengths = sources[0].data.map(chunk => chunk.length);
  for (const source of sources) {
    if (
      source.length !== sources[0].length ||
      source.data.length !== lengths.length ||
      source.data.some(
        (chunk, index) =>
          chunk.length !== lengths[index] || getBuffer(chunk).device !== props.device
      )
    )
      throw new Error('Scalar source chunks must align on one device');
  }
  return [...new Set(sources)];
}

function makeMask(
  device: Device,
  filterId: string,
  lengths: readonly number[]
): GPUVector<'uint32'> {
  const chunks: GPUData<'uint32'>[] = [];
  try {
    for (const [index, length] of lengths.entries())
      chunks.push(
        new GPUData({
          buffer: device.createBuffer({
            id: `scalar-${filterId}-mask-${index}`,
            byteLength: Math.max(4, length * 4),
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format: 'uint32',
          length,
          ownsBuffer: true
        })
      );
    return new GPUVector({
      type: 'data',
      name: `scalar-${filterId}-mask`,
      format: 'uint32',
      data: chunks,
      ownsData: true
    });
  } catch (error) {
    for (const chunk of chunks) chunk.destroy();
    throw error;
  }
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof DynamicBuffer ? data.buffer.buffer : data.buffer;
}
