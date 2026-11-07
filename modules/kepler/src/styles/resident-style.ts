// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {GPUData, GPUVector, type GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  createTransientView,
  type CompiledGPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUColorScale,
  GPUColumnQuantiles,
  GPUScalarScale,
  getGPUColorScaleParameterValues,
  getGPUColumnQuantilesParameterValues,
  getGPUScalarScaleParameterValues,
  prepareGPUScalarScaleControls
} from '@luma-contributors/graph/gpu-dataframe';
import {readGPUColumnStatistics} from '../exploration/gpu-column-statistics';
import type {KeplerComputeSession, KeplerArrowDataset} from '../session/compute-session';
import type {KeplerDatasetLease, KeplerMemoryReservation} from '../runtime/dataset-registry';
import {
  getKeplerResidentStyleKey,
  getKeplerResidentStyleTopologyKey,
  validateKeplerResidentStylePlan,
  type KeplerResidentStylePlan
} from './style-plan';

export type KeplerResidentStyleProps = {
  session: KeplerComputeSession;
  plan: KeplerResidentStylePlan;
  /** Borrowed mask is only the domain population; source nulls independently control output validity. */
  selectionMask?: GPUVector<'uint32'>;
  /**
   * Borrowed Float32 widening of a prepared Int32/Uint32 numeric column. It must preserve the source
   * chunk topology; the host owns, accounts for and destroys it after this style retires.
   */
  sourceOverride?: GPUVector<'float32'>;
  /** Admission reservation includes outputs, explicit global-domain workspace and bounded legend staging. */
  workingMemoryBudgetBytes: number;
  /**
   * Reserve only what the source needs (2 MiB plus 128 bytes per row), capped by
   * `workingMemoryBudgetBytes`, so many small styled datasets do not exhaust the memory budget.
   */
  scaleWorkingMemoryToRows?: boolean;
  maximumDomainCount?: number;
  maximumPaletteCount?: number;
  maximumLegendBytes?: number;
  maximumConcurrentLegendReads?: number;
};

export type KeplerResidentStyleLegend = {
  styleId: string;
  revision: number;
  source: {datasetId: string; revision: number};
  selectionRevision?: number;
  scale: KeplerResidentStylePlan['scale'];
  /** Null denotes an empty GPU-derived finite population; fixed domains retain their declared stops. */
  domain: readonly number[] | null;
  /** Mixed signs and zero upper endpoints are retained for diagnostics, never silently filtered. */
  domainStatus: 'valid' | 'empty' | 'invalid-log-domain';
  finiteCount: number | null;
  /**
   * Rows of the whole source with a null or NaN value (not colored). Null when it cannot be derived
   * on the GPU (non-Float32 source or more than 2^24 rows).
   */
  missingCount?: number | null;
  palette?: readonly number[];
  range?: readonly [number, number];
  labels?: readonly string[];
  codes?: readonly number[];
  numericalPolicy: 'float32';
};

export type KeplerResidentStyleOutputs = {
  colors?: GPUVector<'unorm8x4'>;
  values?: GPUVector<'float32'>;
  validity?: GPUVector<'uint32'>;
};
export type KeplerResidentStyleOutputLease = KeplerResidentStyleOutputs & {
  isCurrent(): boolean;
  release(completion?: PromiseLike<unknown>): Promise<void>;
};

/**
 * One field/channel, one source lease, stable source-chunk outputs. Native translation supplies
 * semantic plans; this runtime never guesses dictionary order or scans source rows on the CPU.
 * Derived quantiles explicitly pack only a temporary GPU workspace and use one global distribution.
 * Output mapping remains per chunk. Plan/source/selection revisions guard bounded legends.
 */
export class KeplerResidentStyle implements KeplerResidentStyleOutputs {
  readonly colors?: GPUVector<'unorm8x4'>;
  readonly values?: GPUVector<'float32'>;
  readonly validity?: GPUVector<'uint32'>;
  readonly maximumDomainCount: number;
  readonly maximumPaletteCount: number;
  private readonly session: KeplerComputeSession;
  private readonly device: Device;
  private readonly sourceLease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly source: GPUVector<'float32'> | GPUVector<'uint32'>;
  private readonly sourceValidity?: GPUVector<'uint32'>;
  private readonly selectionMask?: GPUVector<'uint32'>;
  private readonly sourceOverride?: GPUVector<'float32'>;
  private readonly sourceChunks: readonly GPUData[];
  private readonly ownedBuffers: Buffer[] = [];
  private readonly reservation: KeplerMemoryReservation;
  private readonly domainBuffer: Buffer;
  private readonly paletteBuffer: Buffer;
  private readonly rangeBuffer: Buffer;
  private readonly styleParameters: Buffer;
  private readonly quantileParameters: Buffer;
  private readonly finiteCountBuffer: Buffer;
  private readonly maximumLegendBytes: number;
  private readonly maximumConcurrentLegendReads: number;
  private readonly outputLeases = new Set<object>();
  private readonly outputWaiters: (() => void)[] = [];
  private readonly pendingReads = new Set<Promise<unknown>>();
  private domainGraph?: CompiledGPUCommandGraph;
  private styleGraph?: CompiledGPUCommandGraph;
  private plan: KeplerResidentStylePlan;
  private revision = 1;
  private encodedRevision = -1;
  private domainDirty = true;
  private destroyed = false;
  private destruction?: Promise<void>;
  private transferredBytes = 0;
  private domainEncodings = 0;
  private styleEncodings = 0;
  private uploadedControlBytes = 0;
  private controlUploads: {buffer: Buffer; data: Float32Array | Uint32Array}[] = [];

  constructor(props: KeplerResidentStyleProps) {
    validateKeplerResidentStylePlan(props.plan);
    this.session = props.session;
    this.device = props.session.device;
    this.plan = clonePlan(props.plan);
    this.selectionMask = props.selectionMask;
    this.sourceOverride = props.sourceOverride;
    this.maximumDomainCount =
      props.maximumDomainCount ??
      (props.plan.kind === 'scalar' && props.plan.scale === 'point'
        ? 256
        : props.plan.kind === 'scalar'
          ? 2
          : 64);
    this.maximumPaletteCount = props.maximumPaletteCount ?? 256;
    this.maximumLegendBytes = props.maximumLegendBytes ?? 8192;
    this.maximumConcurrentLegendReads = props.maximumConcurrentLegendReads ?? 1;
    if (
      !Number.isInteger(this.maximumDomainCount) ||
      this.maximumDomainCount < 2 ||
      this.maximumDomainCount > 256 ||
      !Number.isInteger(this.maximumPaletteCount) ||
      this.maximumPaletteCount < 1 ||
      this.maximumPaletteCount > 65536 ||
      !Number.isSafeInteger(this.maximumLegendBytes) ||
      this.maximumLegendBytes < 4 ||
      !Number.isInteger(this.maximumConcurrentLegendReads) ||
      this.maximumConcurrentLegendReads < 1 ||
      this.maximumConcurrentLegendReads > 8
    )
      throw new Error('Invalid resident style capacities');
    this.validateControls(this.plan);
    this.sourceLease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.plan.source.datasetId
    );
    let reservation: KeplerMemoryReservation | undefined;
    try {
      if (this.sourceLease.dataset.revision !== props.plan.source.revision)
        throw new Error('Resident style source revision is stale');
      const original = this.sourceLease.dataset.value.columns[props.plan.column];
      const source = props.sourceOverride ?? original;
      if (
        props.sourceOverride &&
        (original?.length !== source.length ||
          original.data.length !== source.data.length ||
          original.data.some((chunk, index) => chunk.length !== source.data[index].length))
      )
        throw new Error('Style source overrides must preserve source chunk topology');
      const categorical =
        (props.plan.kind === 'color' && props.plan.scale === 'ordinal') ||
        (props.plan.kind === 'scalar' && props.plan.scale === 'point');
      if (
        !source ||
        (source.format !== 'float32' && !(categorical && source.format === 'uint32')) ||
        (props.plan.kind === 'color' &&
          props.plan.scale === 'ordinal' &&
          source.format !== 'uint32')
      )
        throw new Error(
          'Style source requires prepared Float32 numeric values or Uint32 category codes'
        );
      // Format narrowing stays at the validated adapter boundary.
      this.source = source as GPUVector<'float32'> | GPUVector<'uint32'>;
      this.sourceChunks = [...source.data];
      this.sourceValidity = this.sourceLease.dataset.value.analytics.validity[props.plan.column];
      this.validateMask(this.sourceValidity);
      if (props.plan.domain.kind === 'selection') {
        if (!props.selectionMask) throw new Error('Selected style domains require a resident mask');
        this.validateMask(props.selectionMask);
      }
      if (props.plan.domain.kind !== 'fixed' && source.length > 2 ** 24)
        throw new Error('Global style quantiles exceed exact Float32 rank capacity');
      const reservedByteLength = props.scaleWorkingMemoryToRows
        ? Math.min(props.workingMemoryBudgetBytes, 2 * 1024 * 1024 + source.length * 128)
        : props.workingMemoryBudgetBytes;
      reservation = props.session.registry.reserveMemory(reservedByteLength, 'gpu', 'scratch');
      this.reservation = reservation;
      this.domainBuffer = this.allocate(
        'domain',
        new Float32Array(this.maximumDomainCount).fill(NaN)
      );
      this.paletteBuffer = this.allocate('palette', new Uint32Array(this.maximumPaletteCount));
      this.rangeBuffer = this.allocate('range', new Float32Array(2));
      this.styleParameters = this.allocate('style-parameters', new Float32Array(9));
      this.quantileParameters = this.allocate(
        'quantile-parameters',
        new Float32Array(4 + this.quantileCapacity)
      );
      this.finiteCountBuffer = this.allocate('finite-count', new Uint32Array(1));
      if (props.plan.kind === 'color') this.colors = this.allocateOutput('colors', 'unorm8x4');
      else {
        this.values = this.allocateOutput('values', 'float32');
        this.validity = this.allocateOutput('validity', 'uint32');
      }
      this.writeControls(this.plan);
      if (source.length) {
        if (props.plan.domain.kind !== 'fixed') this.domainGraph = this.createDomainGraph();
        this.styleGraph = this.createStyleGraph();
      }
      if (
        this.accountedResidentByteLength +
          this.maximumConcurrentLegendReads * this.maximumLegendBytes >
        reservedByteLength
      )
        throw new KeplerAllocationError('memory-budget', {
          message: 'Resident style allocations exceed its memory reservation'
        });
      void this.device.lost
        .then(() => {
          this.outputLeases.clear();
          this.resolveOutputWaiters();
          return this.destroy(Promise.resolve());
        })
        .catch(() => {});
    } catch (error) {
      this.domainGraph?.destroy();
      this.styleGraph?.destroy();
      for (const buffer of this.ownedBuffers) buffer.destroy();
      reservation?.release();
      void this.sourceLease.release();
      throw error;
    }
  }

  get currentRevision(): number {
    return this.revision;
  }
  get currentPlan(): KeplerResidentStylePlan {
    return clonePlan(this.plan);
  }
  get isCurrent(): boolean {
    return (
      !this.destroyed &&
      this.sourceLease.isValid() &&
      this.session.registry.getDataset(this.plan.source.datasetId) === this.sourceLease.dataset
    );
  }
  get readbackByteLength(): number {
    return this.transferredBytes;
  }
  get controlSnapshotByteLength(): number {
    return this.controlUploads.reduce((sum, upload) => sum + upload.data.byteLength, 0);
  }
  get uploadedControlByteLength(): number {
    return this.uploadedControlBytes;
  }
  get executionCounts(): Readonly<{domain: number; style: number}> {
    return {domain: this.domainEncodings, style: this.styleEncodings};
  }
  /** Excludes borrowed inputs, encoder-owned control staging and driver/private executable allocations. */
  get accountedResidentByteLength(): number {
    return (
      this.ownedBuffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) +
      (this.domainGraph?.stats.physicalTransientResourceBytes ?? 0) +
      (this.styleGraph?.stats.physicalTransientResourceBytes ?? 0)
    );
  }
  canUpdate(
    plan: KeplerResidentStylePlan,
    selectionMask?: GPUVector<'uint32'>,
    sourceOverride?: GPUVector<'float32'>
  ): boolean {
    return (
      this.isCurrent &&
      sourceOverride === this.sourceOverride &&
      getKeplerResidentStyleTopologyKey(plan) === getKeplerResidentStyleTopologyKey(this.plan) &&
      (plan.domain.kind !== 'selection' || selectionMask === this.selectionMask)
    );
  }

  /** Same-topology changes update only small controls; rejected plans leave existing output usable. */
  update(plan: KeplerResidentStylePlan): number {
    this.assertCurrent();
    validateKeplerResidentStylePlan(plan);
    if (getKeplerResidentStyleTopologyKey(plan) !== getKeplerResidentStyleTopologyKey(this.plan))
      throw new Error('Style topology changed; replace through the fenced cache');
    this.validateControls(plan);
    if (getKeplerResidentStyleKey(plan) === getKeplerResidentStyleKey(this.plan))
      return this.revision;
    const next = clonePlan(plan);
    const domainChanged = domainKey(next) !== domainKey(this.plan);
    this.writeControls(next);
    this.plan = next;
    this.domainDirty ||= domainChanged;
    return ++this.revision;
  }

  /** Caller encodes selection producers first and owns submission before any legend read. */
  encode(commandEncoder: CommandEncoder): number {
    this.assertCurrent();
    if (commandEncoder.device !== this.device)
      throw new Error('Style encoder belongs to another device');
    if (
      this.source.data.length !== this.sourceChunks.length ||
      this.sourceChunks.some((chunk, index) => chunk !== this.source.data[index])
    )
      throw new Error('Style source chunks changed; rebuild the style');
    if (this.encodedRevision === this.revision) return this.revision;
    try {
      for (const upload of this.controlUploads)
        this.device.writeBufferViaCommandEncoder(commandEncoder, upload.buffer, upload.data);
      this.uploadedControlBytes += this.controlSnapshotByteLength;
      if (this.domainDirty && this.domainGraph) {
        this.domainGraph.encode(commandEncoder, {parameters: undefined});
        this.domainEncodings++;
      }
      this.styleGraph?.encode(commandEncoder, {parameters: undefined});
      this.styleEncodings++;
      this.domainDirty = false;
      this.encodedRevision = this.revision;
      return this.revision;
    } catch (error) {
      this.discardEncoding();
      throw error;
    }
  }
  discardEncoding(): void {
    this.encodedRevision = -1;
    this.domainDirty = true;
  }

  /** Bounded control-domain readback only; no source rows or per-row style buffers are downloaded. */
  readLegend(): Promise<KeplerResidentStyleLegend | null> {
    if (
      !this.isCurrent ||
      this.encodedRevision !== this.revision ||
      this.pendingReads.size >= this.maximumConcurrentLegendReads
    )
      return Promise.resolve(null);
    const revision = this.revision;
    const plan = clonePlan(this.plan);
    const count = activeDomainCount(plan);
    const pending = (async () => {
      try {
        this.transferredBytes += count * 4 + (plan.domain.kind === 'fixed' ? 0 : 4);
        const results = await Promise.allSettled([
          count ? this.domainBuffer.readAsync(0, count * 4) : Promise.resolve(new Uint8Array()),
          ...(plan.domain.kind === 'fixed' ? [] : [this.finiteCountBuffer.readAsync(0, 4)])
        ]);
        if (!this.isCurrent || revision !== this.revision || this.encodedRevision !== revision)
          return null;
        const failure = results.find(result => result.status === 'rejected');
        if (failure?.status === 'rejected') throw failure.reason;
        const bytes = results.map(result =>
          result.status === 'fulfilled' ? result.value : new Uint8Array()
        );
        const finiteCount =
          plan.domain.kind === 'fixed'
            ? null
            : new Uint32Array(bytes[1].buffer, bytes[1].byteOffset, 1)[0];
        const domain =
          finiteCount === 0
            ? null
            : Array.from(new Float32Array(bytes[0].buffer, bytes[0].byteOffset, count)).slice(
                plan.kind === 'color' && plan.scale === 'threshold' ? 1 : 0,
                plan.kind === 'color' && plan.scale === 'threshold' ? count - 1 : count
              );
        const missingCount = await this.readMissingCount(plan, finiteCount, revision);
        if (!this.isCurrent || revision !== this.revision || this.encodedRevision !== revision)
          return null;
        const domainStatus: KeplerResidentStyleLegend['domainStatus'] =
          domain === null
            ? 'empty'
            : plan.scale === 'log' &&
                !(
                  domain.every(value => Number.isFinite(value) && value > 0) ||
                  domain.every(value => Number.isFinite(value) && value < 0)
                )
              ? 'invalid-log-domain'
              : 'valid';
        return {
          styleId: plan.id,
          revision,
          source: {...plan.source},
          scale: plan.scale,
          ...(plan.domain.kind === 'selection'
            ? {selectionRevision: plan.domain.revision}
            : plan.domain.kind === 'fixed' && plan.domain.selectionRevision !== undefined
              ? {selectionRevision: plan.domain.selectionRevision}
              : {}),
          domain,
          domainStatus,
          finiteCount,
          missingCount,
          numericalPolicy: 'float32' as const,
          ...(plan.kind === 'color'
            ? {palette: [...plan.palette]}
            : {range: [...plan.range] as [number, number]}),
          ...(plan.legend
            ? {
                labels: [...plan.legend.labels],
                ...(plan.legend.codes ? {codes: [...plan.legend.codes]} : {})
              }
            : {})
        };
      } catch (error) {
        if (!this.isCurrent || revision !== this.revision) return null;
        throw error;
      }
    })();
    this.pendingReads.add(pending);
    void pending.finally(() => this.pendingReads.delete(pending)).catch(() => {});
    return pending;
  }

  /**
   * GPU-derived count of null or NaN rows over the whole source. Global-domain styles already
   * reduced the finite population; fixed and selection domains run one fused column profile.
   */
  private async readMissingCount(
    plan: KeplerResidentStylePlan,
    finiteCount: number | null,
    revision: number
  ): Promise<number | null> {
    const rowCount = this.source.length;
    if (plan.domain.kind === 'source' && finiteCount !== null)
      return Math.max(0, rowCount - finiteCount);
    if (this.source.format !== 'float32' || rowCount < 1) return rowCount < 1 ? 0 : null;
    try {
      const statistics = await readGPUColumnStatistics({
        device: this.device,
        values: this.source as GPUVector<'float32'>,
        validity: this.sourceValidity,
        populationRows: rowCount,
        histogramBinCount: 1,
        isCurrent: () => this.isCurrent && revision === this.revision
      });
      return statistics ? statistics.nullCount : null;
    } catch {
      // Over 2^24 rows or a refused profile: the legend states the count is unavailable.
      return null;
    }
  }

  acquireOutputs(): KeplerResidentStyleOutputLease {
    this.assertCurrent();
    const token = {};
    this.outputLeases.add(token);
    let completion: Promise<void> | undefined;
    return {
      colors: this.colors,
      values: this.values,
      validity: this.validity,
      isCurrent: () => !completion && this.isCurrent && this.outputLeases.has(token),
      release: (fence?: PromiseLike<unknown>) => {
        completion ??= Promise.race([Promise.resolve(fence), this.device.lost])
          .finally(() => {
            this.outputLeases.delete(token);
            this.resolveOutputWaiters();
          })
          .then(() => undefined);
        return completion;
      }
    };
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.destruction = Promise.race([completion, this.device.lost])
      .finally(async () => {
        await Promise.allSettled([...this.pendingReads]);
        if (this.outputLeases.size)
          await new Promise<void>(resolve => this.outputWaiters.push(resolve));
        this.domainGraph?.destroy();
        this.styleGraph?.destroy();
        for (const buffer of this.ownedBuffers) buffer.destroy();
        this.colors?.destroy();
        this.values?.destroy();
        this.validity?.destroy();
        await this.sourceLease.release();
        this.reservation.release();
      })
      .then(() => undefined);
    return this.destruction;
  }

  private get quantileCapacity(): number {
    return Math.min(this.maximumDomainCount, 64);
  }
  private validateControls(plan: KeplerResidentStylePlan): void {
    const count = activeDomainCount(plan);
    if (count > this.maximumDomainCount || count * 4 + 4 > this.maximumLegendBytes)
      throw new KeplerAllocationError('capacity', {
        message: 'Style domain exceeds its bounded capacity'
      });
    if (plan.kind === 'color' && plan.palette.length > this.maximumPaletteCount)
      throw new KeplerAllocationError('capacity', {message: 'Style palette exceeds its capacity'});
    if (
      plan.kind === 'color' &&
      plan.scale === 'quantile' &&
      plan.domain.kind !== 'fixed' &&
      count > this.quantileCapacity
    )
      throw new Error('GPU quantiles support at most 63 color classes');
    if (plan.legend && plan.legend.labels.length > this.maximumPaletteCount)
      throw new KeplerAllocationError('capacity', {message: 'Style legend exceeds its capacity'});
    if (plan.kind === 'scalar' && plan.domain.kind === 'fixed')
      prepareGPUScalarScaleControls({
        scale: plan.scale,
        domain: plan.domain.values,
        range: plan.range,
        nullValue: plan.nullValue,
        invert: plan.invert
      });
  }
  private writeControls(plan: KeplerResidentStylePlan): void {
    const uploads: {buffer: Buffer; data: Float32Array | Uint32Array}[] = [];
    if (plan.domain.kind === 'fixed') {
      const values =
        plan.kind === 'color' && plan.scale === 'threshold'
          ? [plan.domain.values[0] ?? 0, ...plan.domain.values, plan.domain.values.at(-1) ?? 0]
          : plan.domain.values;
      if (values.length) uploads.push({buffer: this.domainBuffer, data: new Float32Array(values)});
    }
    if (plan.kind === 'color') {
      uploads.push({buffer: this.paletteBuffer, data: new Uint32Array(plan.palette)});
      uploads.push({
        buffer: this.styleParameters,
        data: getGPUColorScaleParameterValues({
          scale:
            plan.scale === 'equal-interval'
              ? 'quantize'
              : plan.scale === 'threshold'
                ? 'quantile'
                : plan.scale,
          domainCount: activeDomainCount(plan),
          paletteCount: plan.palette.length,
          interpolation: plan.interpolation ?? 'linear',
          noDataColor: plan.noDataColor ?? 0,
          clamp: plan.outsideDomain === 'clamp',
          extrapolate: plan.outsideDomain === 'extrapolate',
          logMode: 'domain-sign'
        })
      });
    } else {
      uploads.push({buffer: this.rangeBuffer, data: new Float32Array(plan.range)});
      uploads.push({
        buffer: this.styleParameters,
        data: getGPUScalarScaleParameterValues({
          scale: plan.scale,
          domainCount: activeDomainCount(plan),
          nullValue: plan.nullValue,
          invert: plan.invert
        })
      });
    }
    if (plan.domain.kind !== 'fixed') {
      const count = activeDomainCount(plan);
      uploads.push({
        buffer: this.quantileParameters,
        data: getGPUColumnQuantilesParameterValues({
          quantiles: Array.from({length: count}, (_, index) => index / (count - 1)),
          quantileCount: this.quantileCapacity,
          interpolation: 'linear'
        })
      });
    }
    this.controlUploads = uploads;
  }
  private createDomainGraph(): CompiledGPUCommandGraph {
    const graph = new GPUCommandGraph(this.device, {id: `${this.plan.id}-domain`});
    const values = graph.importGPUVector('source', this.source);
    const packedValues = packVector(graph, 'global-values', values);
    const masks = [
      this.sourceValidity,
      this.plan.domain.kind === 'selection' ? this.selectionMask : undefined
    ].flatMap((mask, index) => (mask ? [graph.importGPUVector(`mask-${index}`, mask)] : []));
    let packedMask: GraphDataView<'uint32'> | undefined;
    if (masks.length) {
      packedMask = createTransientView(graph, 'global-mask', 'uint32', this.source.length);
      graph.add(new GPUMask({id: 'valid-population', inputs: masks, output: packedMask}));
    }
    // Numeric validation above guarantees Float32; Uint32 categorical modes always use fixed domains.
    graph.add(
      new GPUColumnQuantiles({
        id: 'global-style-quantiles',
        values: packedValues as GraphDataView<'float32'>,
        mask: packedMask,
        parameters: importBuffer(
          graph,
          'quantile-parameters',
          this.quantileParameters,
          'float32',
          4 + this.quantileCapacity
        ),
        quantileCount: this.quantileCapacity,
        output: {
          quantiles: importBuffer(
            graph,
            'domain',
            this.domainBuffer,
            'float32',
            this.maximumDomainCount
          ),
          validCount: importBuffer(graph, 'finite-count', this.finiteCountBuffer, 'uint32', 1)
        }
      })
    );
    return graph.compile();
  }
  private createStyleGraph(): CompiledGPUCommandGraph {
    const graph = new GPUCommandGraph(this.device, {id: `${this.plan.id}-mapping`});
    const values = graph.importGPUVector('source', this.source);
    const validity = this.sourceValidity
      ? graph.importGPUVector('validity', this.sourceValidity)
      : undefined;
    const domain = importBuffer(
      graph,
      'domain',
      this.domainBuffer,
      'float32',
      this.maximumDomainCount
    );
    const parameters = importBuffer(graph, 'parameters', this.styleParameters, 'float32', 9);
    if (this.plan.domain.kind !== 'fixed') {
      const finiteCount = importBuffer(graph, 'finite-count', this.finiteCountBuffer, 'uint32', 1);
      // Native getLogDomain uses the full extent, replacing only a zero minimum with 1e-5.
      // Negative and zero source rows stay in the population. Unsupported mixed-sign domains
      // disable mapping; the bounded legend exposes their endpoints and invalid-log-domain status.
      // Every encoded revision first restores its host parameter snapshot.
      graph.addComputePass({
        id: 'guard-empty-domain',
        resources: [
          {buffer: finiteCount, usage: 'storage-read'},
          {buffer: parameters, usage: 'storage-read-write'},
          {buffer: domain, usage: 'storage-read-write'}
        ],
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id: 'guard-empty-style-domain',
            source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> finiteCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> parameters: array<f32>;
@group(0) @binding(2) var<storage, read_write> domain: array<f32>;
@compute @workgroup_size(1) fn main() {
  if (finiteCount[0] == 0u) { parameters[1] = 0.0; return; }
  if (parameters[0] == ${this.plan.kind === 'color' ? '9.0' : '2.0'}) {
    if (domain[0] == 0.0) { domain[0] = 1e-5; }
    let sameSign = (domain[0] > 0.0 && domain[1] > 0.0) || (domain[0] < 0.0 && domain[1] < 0.0);
    if (!sameSign) { parameters[1] = 0.0; }
  }
}`,
            shaderLayout: {
              bindings: [
                {name: 'finiteCount', type: 'storage', group: 0, location: 0},
                {name: 'parameters', type: 'storage', group: 0, location: 1},
                {name: 'domain', type: 'storage', group: 0, location: 2}
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {
                  finiteCount: getBuffer(finiteCount.buffer),
                  parameters: getBuffer(parameters.buffer),
                  domain: getBuffer(domain.buffer)
                },
                x: 1
              }),
            destroy: () => kernel.destroy()
          };
        }
      });
    }
    const palette = importBuffer(
      graph,
      'palette',
      this.paletteBuffer,
      'uint32',
      this.maximumPaletteCount
    );
    const range = importBuffer(graph, 'range', this.rangeBuffer, 'float32', 2);
    for (const [index, input] of values.data.entries()) {
      if (!input.length) continue;
      if (this.colors) {
        graph.add(
          new GPUColorScale({
            id: `color-${index}`,
            values: input,
            mask: validity?.data[index],
            domain,
            palette,
            parameters,
            maximumDomainCount: this.maximumDomainCount,
            maximumPaletteCount: this.maximumPaletteCount,
            output: {
              colors: importOutputChunk(graph, `color-${index}`, this.colors.data[index], 'uint32')
            }
          })
        );
      } else if (this.values && this.validity) {
        graph.add(
          new GPUScalarScale({
            id: `scalar-${index}`,
            values: input,
            validity: validity?.data[index],
            domain,
            range,
            parameters,
            maximumDomainCount: this.maximumDomainCount,
            output: {
              values: graph.importGPUData(`value-${index}`, this.values.data[index]),
              validity: graph.importGPUData(`output-validity-${index}`, this.validity.data[index])
            }
          })
        );
      }
    }
    return graph.compile();
  }
  private allocate(id: string, values: Float32Array | Uint32Array): Buffer {
    const buffer = this.device.createBuffer({
      id: `${this.plan.id}-${id}`,
      data: values,
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.ownedBuffers.push(buffer);
    return buffer;
  }
  private allocateOutput<Format extends 'float32' | 'uint32' | 'unorm8x4'>(
    id: string,
    format: Format
  ): GPUVector<Format> {
    return new GPUVector({
      type: 'data',
      name: `${this.plan.id}-${id}`,
      format,
      ownsData: false,
      data: this.source.data.map((chunk, index) => {
        const buffer = this.device.createBuffer({
          id: `${this.plan.id}-${id}-${index}`,
          byteLength: Math.max(4, chunk.length * 4),
          usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        this.ownedBuffers.push(buffer);
        return new GPUData({
          format,
          length: chunk.length,
          buffer,
          ownsBuffer: false
        });
      })
    });
  }
  private validateMask(mask?: GPUVector<'uint32'>): void {
    if (!mask) return;
    if (
      mask.format !== 'uint32' ||
      mask.length !== this.source.length ||
      mask.data.length !== this.source.data.length ||
      mask.data.some(
        (chunk, index) =>
          chunk.length !== this.source.data[index].length ||
          chunk.byteStride !== 4 ||
          chunk.buffer.device !== this.device
      )
    )
      throw new Error('Style masks must preserve source chunk topology on the same device');
  }
  private assertCurrent(): void {
    if (!this.isCurrent) throw new Error('Resident style source is stale or closed');
  }
  private resolveOutputWaiters(): void {
    if (!this.outputLeases.size) for (const resolve of this.outputWaiters.splice(0)) resolve();
  }
}

function activeDomainCount(plan: KeplerResidentStylePlan): number {
  if (plan.domain.kind === 'fixed')
    return (
      plan.domain.values.length + (plan.kind === 'color' && plan.scale === 'threshold' ? 2 : 0)
    );
  return plan.kind === 'color' && plan.scale === 'quantile' ? plan.palette.length + 1 : 2;
}
function domainKey(plan: KeplerResidentStylePlan): string {
  return JSON.stringify([
    plan.domain,
    plan.scale === 'log',
    plan.kind === 'color' && plan.scale === 'quantile' ? plan.palette.length : 0
  ]);
}
function clonePlan(plan: KeplerResidentStylePlan): KeplerResidentStylePlan {
  const common = {
    ...plan,
    source: {...plan.source},
    domain:
      plan.domain.kind === 'fixed'
        ? {...plan.domain, values: [...plan.domain.values]}
        : {...plan.domain},
    ...(plan.legend
      ? {
          legend: {
            labels: [...plan.legend.labels],
            ...(plan.legend.codes ? {codes: [...plan.legend.codes]} : {})
          }
        }
      : {})
  };
  return plan.kind === 'color'
    ? {
        ...common,
        ...plan,
        source: common.source,
        domain: common.domain,
        legend: common.legend,
        palette: [...plan.palette]
      }
    : {
        ...common,
        ...plan,
        source: common.source,
        domain: common.domain,
        legend: common.legend,
        range: [...plan.range]
      };
}
function importBuffer<Format extends GPUVectorFormat>(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(
    graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
    {format, length}
  );
}
function importOutputChunk<Format extends 'uint32'>(
  graph: GPUCommandGraph,
  id: string,
  chunk: GPUData,
  format: Format
): GraphDataView<Format> {
  return graph.importGPUData(
    id,
    new GPUData({
      buffer: chunk.buffer,
      byteOffset: chunk.byteOffset,
      format,
      length: chunk.length,
      ownsBuffer: false
    })
  );
}
function packVector<Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph,
  id: string,
  vector: GraphVectorView<Format>
): GraphDataView<Format> {
  if (vector.data.length === 1) return vector.data[0];
  const output = createTransientView(
    graph,
    id,
    vector.format,
    vector.length,
    Buffer.STORAGE | Buffer.COPY_DST
  );
  let rowOffset = 0;
  for (const [index, chunk] of vector.data.entries()) {
    if (chunk.length) {
      const target = graph.createDataView(output.buffer, {
        format: vector.format,
        byteOffset: rowOffset * 4,
        length: chunk.length
      });
      graph.addCopyPass({
        id: `${id}-${index}`,
        resources: [
          {buffer: chunk, usage: 'copy-source'},
          {buffer: target, usage: 'copy-destination'}
        ],
        compile: () => ({
          encode: ({commandEncoder, getBuffer}) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: getBuffer(chunk),
              sourceOffset: chunk.byteOffset,
              destinationBuffer: getBuffer(target),
              destinationOffset: target.byteOffset,
              size: chunk.length * 4
            })
        })
      });
    }
    rowOffset += chunk.length;
  }
  return output;
}
