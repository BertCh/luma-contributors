// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CommandEncoder} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUKeplerScalarSelection,
  GPUKeplerSelectionScopes,
  type GPUKeplerBrushTargets,
  type GPUKeplerScalarFilter,
  type GPUKeplerScalarControl,
  type GPUResidentDimension,
  type GPUResidentSummary,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type KeplerSelectionLayer,
  type KeplerSessionExplorer,
  type KeplerPolygonMaskFilter,
  type KeplerConsumerSelectionScope
} from '@luma-contributors/kepler';
import {
  planKeplerComputeFilters,
  usesNativeTopologyPredicate,
  type KeplerComputeFilter,
  type KeplerComputeFilterCommand,
  type KeplerComputeTableBinding,
  type KeplerComputeTableLike,
  type KeplerComputeTableState,
  type KeplerComputeBin
} from './kepler-table-bridge';
import {prepareNativePolygonFilter} from './resident-table-binding';
import {
  listCategoryCoverageWarnings,
  makeFilterRevisionReport,
  categoryCoverageRegistry,
  type FilterRevisionReport
} from './binding3-filter-runtime';
import {describeUnpreparedFilterField} from './binding2-filter-lifecycle';
import {fieldPreparationCoordinator} from './binding3-field-preparation';
import {prepareKeplerTimeRangeMilliseconds} from './time-runtime';
import {ResidentTimeHistogram, makeResidentTimeHistogramPlan} from './resident-time-histogram';
import {ResidentNumericHistogram} from './resident-numeric-histogram';
import type {
  ResidentPlotGroupingProps,
  ResidentPlotGroupingSummary
} from './resident-plot-grouping';
import {
  makeResidentNumericPlotEdges,
  type ResidentNumericPlotEdges
} from './resident-numeric-plot-edges';
import {
  RESIDENT_PLOT_MEASURE_OPERATIONS,
  getResidentPlotMeasureBudgetBytes,
  type ResidentPlotMeasureOperation,
  type ResidentPlotMeasureSummary
} from './resident-plot-measure';

import {clamp} from '@math.gl/core';

/** Explicit scalar mapping, or exact word-time calendar metadata compiled by this provider. */
export type ResidentScopePlot = {
  column: string;
  domain: readonly [number, number];
  binCount: number;
  nativeDomain?: readonly [number, number];
  interval?: string;
  /** Exact native calendar thresholds; last threshold has a native zero-width terminal bin. */
  timeThresholds?: readonly number[];
  /** Native range-plot edges (d3 ticks or numeric interval) classified exactly on the GPU. */
  numericEdges?: ResidentNumericPlotEdges;
  /** Native line-chart y axis, reduced on the GPU over the same bins as the counts. */
  measure?: {column: string; operation: ResidentPlotMeasureOperation};
  /** Native grouped line series, counted and reduced over dense slot-and-bin keys on the GPU. */
  group?: ResidentPlotGroupingProps;
};

export type ResidentScopeBindingProps = {
  session: KeplerComputeSession;
  datasetId: string;
  fields: KeplerComputeTableBinding['fields'];
  /** Explicit native filtering/inspection of an unfinished stream prefix; analysis remains complete-only. */
  allowIncompletePopulation?: boolean;
  /** Enable only when NativeStyleRuntime owns numeric domains, legends and layer diagnostics. */
  residentStyleDomains?: boolean;
  layers?: readonly KeplerSelectionLayer[];
  /** Numeric fields use the pinned native range-plot edges; time fields compile bounded native UTC calendar bins. */
  plots?: Readonly<Record<string, ResidentScopePlot>>;
  maximumFilters?: number;
  scalarWorkingMemoryBudgetBytes?: number;
  scopeWorkingMemoryBudgetBytes?: number;
  inspectorWorkingMemoryBudgetBytes?: number;
  onUpdate(): void;
  onUnsupported(reason: string): void;
  onPlotUnsupported?(filterId: string, reason: string | null): void;
};

type ScalarPlan = {
  filter: KeplerComputeFilter;
  field: string;
  specification: GPUKeplerScalarFilter;
  command: KeplerComputeFilterCommand;
};
type PlotEntry = {
  field: string;
  key: string;
  plot: ResidentScopePlot;
  explorer: Pick<KeplerSessionExplorer, 'encode' | 'close' | 'invalidate'> & {
    readSummary(): Promise<{
      histograms: Record<string, Uint32Array>;
      measure?: ResidentPlotMeasureSummary;
      groups?: ResidentPlotGroupingSummary;
    } | null>;
  };
};
type HistogramSnapshot = {
  revision: number;
  generation: number;
  bins: readonly KeplerComputeBin[];
  interval?: string;
};
type NativeFilterRequest = {
  filters: readonly KeplerComputeFilter[];
  revision: number;
  fields: KeplerComputeTableLike['fields'];
  layers: readonly unknown[];
};

/**
 * Native Kepler filters over per-filter GPU masks and exact per-layer selection scopes.
 *
 * Construct before the display explorer and include externalDimension, then bind that explorer.
 * The external callback encodes predicates, scope composition, dynamic domains and each native
 * filter's histogram population before the primary explorer consumes the native table union.
 * Layer renderers borrow getLayerMask(id), while table/export/analysis use the primary explorer.
 * Read summaries after submission and publishHistograms into the matching native bridge revision.
 */
export class ResidentScopeBinding {
  readonly scopes: GPUKeplerSelectionScopes;
  readonly externalDimension: Extract<GPUResidentDimension, {kind: 'external'}>;
  readonly domainExplorer: KeplerSessionExplorer;
  private readonly props: ResidentScopeBindingProps;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly plots = new Map<string, PlotEntry>();
  private readonly histograms = new Map<string, HistogramSnapshot>();
  private readonly unsupportedPlots = new Map<string, string>();
  private readonly retiredScalars: GPUKeplerScalarSelection[] = [];
  private readonly retiredExplorers: PlotEntry['explorer'][] = [];
  private readonly retirements = new Set<Promise<void>>();
  private scalar?: GPUKeplerScalarSelection;
  private scalarTopology = '';
  /** Per-filter slot signature of the live scalar selection (drives incremental add/remove). */
  private scalarSlots: ReadonlyMap<string, string> = new Map();
  /** Set when an incremental update could not be undone; the next update rebuilds fully. */
  private scalarBroken = false;
  private layers: readonly KeplerSelectionLayer[];
  private primary?: KeplerSessionExplorer;
  private bridgeState?: KeplerComputeTableState;
  private nativeRequest?: NativeFilterRequest;
  private nativeLayerIdentity = '';
  private chartTimeNullEpochFilters: ReadonlyMap<string, string> = new Map();
  private registrationDepth = 0;
  private registrationFailed = false;
  private pendingNativeRequest?: NativeFilterRequest;
  private pendingNativeLayers?: readonly unknown[];
  private registrationPlanError?: string;
  private nativeRevision = 0;
  /** How the latest native filter revision differed from the previous one and what it rebuilt. */
  lastFilterRevision?: FilterRevisionReport;
  private generation = 0;
  private encodedGeneration = -1;
  private domainSummary?: {
    generation: number;
    revision: number;
    domains: GPUResidentSummary['domains'];
  };
  private pendingRead?: Promise<boolean>;
  private destroyed = false;
  private completion?: Promise<void>;

  constructor(props: ResidentScopeBindingProps) {
    this.props = props;
    this.layers = [...(props.layers ?? [])];
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.datasetId, {
      allowIncomplete: props.allowIncompletePopulation
    });
    let scopes: GPUKeplerSelectionScopes | undefined;
    let domainExplorer: KeplerSessionExplorer | undefined;
    try {
      const rows = Object.values(this.lease.dataset.value.columns)[0];
      const sourceChunkLengths = this.lease.dataset.value.prepared.sourceBatches.map(
        batch => batch.sourceRowCount
      );
      scopes = new GPUKeplerSelectionScopes({
        device: props.session.device,
        rows,
        sourceChunkLengths,
        datasetRevision: this.lease.dataset.revision,
        layers: this.layers,
        registry: props.session.registry,
        workingMemoryBudgetBytes:
          props.scopeWorkingMemoryBudgetBytes ??
          this.lease.dataset.rowCount * 256 + 4 * 1024 * 1024,
        encodePredicates: encoder => {
          this.scalar?.encode(encoder);
        }
      });
      this.scopes = scopes;
      const domains = Object.entries(props.fields)
        .filter(
          ([name, field]) =>
            field.kind === 'range' && Boolean(this.lease.dataset.value.columns[name])
        )
        .map(([column]) => ({id: column, column}));
      domainExplorer = props.session.openExplorer(props.datasetId, {
        allowIncompletePopulation: props.allowIncompletePopulation,
        // Each domain owns validity masks, finite-value query outputs and reduction scratch.
        // Plot inspectors have only one column; their fixed per-row allowance is insufficient here.
        workingMemoryBudgetBytes:
          props.inspectorWorkingMemoryBudgetBytes ??
          this.inspectorBudget + this.lease.dataset.rowCount * domains.length * 64,
        maximumPageSize: 1,
        maximumConcurrentReads: 1,
        maximumReadbackBytes: 4096,
        dimensions: [{id: 'native-domain', kind: 'external', mask: scopes.domainMask}],
        domains
      });
      this.domainExplorer = domainExplorer;
      domainExplorer.setExternalActive('native-domain', true);
      this.externalDimension = {
        id: 'kepler-native-selection',
        kind: 'external',
        mask: scopes.nativeFilteredMask,
        encode: encoder => this.encode(encoder)
      };
    } catch (error) {
      void Promise.allSettled([
        scopes?.destroy(Promise.resolve()),
        domainExplorer?.close(Promise.resolve()),
        this.lease.release()
      ]);
      throw error;
    }
  }

  get currentRevision(): number {
    return this.generation;
  }
  get nativeFilteredMask(): GPUVector<'uint32'> {
    return this.scopes.nativeFilteredMask;
  }

  get nativeTopologyMask(): GPUVector<'uint32'> {
    return this.scopes.nativeTopologyMask;
  }
  get registeredLayerIds(): readonly string[] {
    return this.layers.map(layer => layer.id);
  }

  /** The host suspends mask consumers/rendering until the outermost registration commits. */
  get isLayerRegistrationPending(): boolean {
    return this.registrationDepth > 0;
  }

  /**
   * Current source masks have been encoded and registration succeeded. Consumers must encode
   * after this binding on the same command encoder, submit before readback, and compare the
   * captured currentRevision after awaiting results. This does not imply GPU completion.
   */
  get canConsume(): boolean {
    return (
      !this.destroyed &&
      this.isCurrentDataset &&
      !this.isLayerRegistrationPending &&
      !this.registrationFailed &&
      this.encodedGeneration === this.generation
    );
  }

  /**
   * Authoritative active native chart cross-filters for this dataset. Match both persisted ID and
   * time field; never infer chart ownership from an ID prefix. Synchronize after native chart
   * config changes, including when createOrUpdateFilter ran before updateChart during a click.
   */
  setChartTimeNullEpochFilters(filters: readonly {filterId: string; field: string}[]): void {
    this.assertActive();
    const next = new Map<string, string>();
    for (const {filterId, field} of filters) {
      if (
        !filterId ||
        this.props.fields[field]?.kind !== 'time' ||
        !this.lease.dataset.value.timeColumns[field]
      )
        throw new Error('Chart null-as-epoch ownership requires an exact time field and filter ID');
      if (next.has(filterId) && next.get(filterId) !== field)
        throw new Error('Chart time filter ID belongs to conflicting fields');
      next.set(filterId, field);
    }
    if (
      next.size === this.chartTimeNullEpochFilters.size &&
      [...next].every(([id, field]) => this.chartTimeNullEpochFilters.get(id) === field)
    )
      return;
    const previous = this.chartTimeNullEpochFilters;
    this.chartTimeNullEpochFilters = next;
    try {
      if (this.nativeRequest && !this.registrationPlanError)
        this.applyNativeFilters(this.nativeRequest);
      else this.invalidateConsumers();
    } catch (error) {
      this.chartTimeNullEpochFilters = previous;
      this.invalidateConsumers();
      throw error;
    }
    this.props.onUpdate();
  }

  beginLayerRegistration(): void {
    this.assertActive();
    if (this.registrationDepth++ === 0) {
      this.pendingNativeRequest = undefined;
      this.pendingNativeLayers = undefined;
      this.invalidateConsumers();
    }
  }

  /** Commit only after the native reducer and selector registration have both completed. */
  endLayerRegistration(): void {
    if (!this.registrationDepth) throw new Error('No resident layer registration is pending');
    if (--this.registrationDepth) return;
    try {
      this.assertActive();
      if (this.registrationPlanError) throw new Error(this.registrationPlanError);
      const request = this.pendingNativeRequest ?? this.nativeRequest;
      if (request)
        this.nativeRequest = {...request, layers: this.pendingNativeLayers ?? request.layers};
      this.scopes.setLayers(this.layers);
      if (this.nativeRequest) this.applyNativeFilters(this.nativeRequest);
      else this.invalidateConsumers();
      this.registrationFailed = false;
      this.props.onUpdate();
    } catch (error) {
      this.registrationFailed = true;
      this.invalidateConsumers();
      this.props.onUnsupported(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      this.pendingNativeRequest = undefined;
      this.pendingNativeLayers = undefined;
    }
  }

  getLayerMask(layerId: string): GPUVector<'uint32'> {
    return this.scopes.getLayerMask(layerId);
  }

  getLayerTopologyMask(layerId: string): GPUVector<'uint32'> {
    return this.scopes.getLayerTopologyMask(layerId);
  }

  getConsumerMask(scope: KeplerConsumerSelectionScope): GPUVector<'uint32'> {
    const revision = this.scopes.currentRevision;
    const mask = this.scopes.getConsumerMask(scope);
    if (this.scopes.currentRevision !== revision) {
      this.invalidateConsumers();
      this.props.onUpdate();
    }
    return mask;
  }

  removeConsumerMask(id: string): void {
    const revision = this.scopes.currentRevision;
    this.scopes.removeConsumerMask(id);
    if (this.scopes.currentRevision !== revision) {
      this.invalidateConsumers();
      this.props.onUpdate();
    }
  }

  setLayers(layers: readonly KeplerSelectionLayer[], nativeLayers?: readonly unknown[]): void {
    this.assertActive();
    if (this.isLayerRegistrationPending) {
      this.layers = [...layers];
      if (nativeLayers) this.pendingNativeLayers = [...nativeLayers];
      return;
    }
    const previous = this.scopes.currentRevision;
    this.scopes.setLayers(layers);
    this.layers = [...layers];
    if (this.registrationPlanError) {
      // A selector update cannot repair malformed native geometry. Wait for a valid native plan.
      this.invalidateConsumers();
      return;
    }
    const nativeLayersChanged =
      nativeLayers !== undefined &&
      getNativeLayerIdentity(nativeLayers) !== this.nativeLayerIdentity;
    if (this.scopes.currentRevision !== previous || nativeLayersChanged) {
      if (nativeLayers && this.nativeRequest)
        this.applyNativeFilters({...this.nativeRequest, layers: nativeLayers});
      else {
        if (nativeLayers) this.nativeLayerIdentity = getNativeLayerIdentity(nativeLayers);
        this.invalidateConsumers();
      }
      this.props.onUpdate();
    }
  }

  /** Supplied by the host once its NativeStyleRuntime exists. */
  getDerivedJenksDomain?: (field: string) => readonly number[] | undefined;

  bind(primary: KeplerSessionExplorer): KeplerComputeTableBinding {
    this.assertActive();
    if (
      (this.primary && this.primary !== primary) ||
      primary.dataset.id !== this.props.datasetId ||
      primary.dataset.revision !== this.lease.dataset.revision
    )
      throw new Error('Resident scope binding requires one explorer on the same dataset revision');
    if (!this.primary) primary.setExternalActive(this.externalDimension.id, true);
    this.primary = primary;
    return {
      fields: this.props.fields,
      filterScopedHistograms: true,
      groupedLineCharts: true,
      getTimeHistogramInterval: (field, filter) => this.getTimeHistogramInterval(field, filter),
      prepareNumericPlot: (field, filter) => this.prepareNumericPlot(field, filter),
      residentStyleDomains: this.props.residentStyleDomains,
      getDerivedJenksDomain: field => this.getDerivedJenksDomain?.(field),
      // The native callback below owns every filter. Legacy combined-field controls must not run.
      explorer: {clear() {}, setRange() {}, setCategories() {}, setEmpty() {}},
      applyNativeFilters: options => this.applyNativeFilters(options),
      onSelectionChange: (_plan, revision) => {
        this.nativeRevision = revision;
        this.props.onUpdate();
      },
      onUnsupported: reason => {
        if (this.isLayerRegistrationPending) this.registrationPlanError = reason;
        this.props.onUnsupported(reason);
      },
      onPlotRecovery: () => this.props.onUpdate(),
      onPlotUnsupported: (filterId, reason) => this.props.onPlotUnsupported?.(filterId, reason),
      prepareSpatialFilter: (filter, layers) => ({
        layerIds: this.preparePolygon(filter, layers)?.layerIds ?? [],
        apply() {}
      }),
      clearSpatialFilters() {},
      getDynamicLayerDomain: (field, scale, revision) => {
        if (this.destroyed || !this.isCurrentDataset) return null;
        if (!['linear', 'quantize', 'sqrt', 'custom'].includes(scale)) {
          if (this.props.residentStyleDomains) return [];
          this.props.onUnsupported(
            `Dynamic ${scale} domains require a matching GPU classification provider`
          );
          return null;
        }
        if (this.props.fields[field]?.kind === 'time') {
          if (this.props.residentStyleDomains) return [];
          this.props.onUnsupported(
            'Exact timestamp layer domains require a matching GPU time-domain reduction'
          );
          return null;
        }
        return this.domainSummary?.generation === this.generation &&
          this.domainSummary.revision === revision
          ? (this.domainSummary.domains[field] ?? null)
          : null;
      }
    };
  }

  /**
   * Links (or with null unlinks) a GPU brush mask so the table, charts and legend domains follow
   * the brushed rows. Layer masks are untouched. The brush owner keeps the mask alive.
   */
  setBrushMask(
    mask: GPUVector<'uint32'> | null,
    targets: GPUKeplerBrushTargets = {table: true, charts: true, legends: true}
  ): void {
    if (this.destroyed) return;
    this.scopes.setBrushMask(mask, targets);
    this.invalidateConsumers();
    this.props.onUpdate();
  }

  encode(encoder: CommandEncoder): void {
    this.assertActive();
    this.encodedGeneration = -1;
    if (this.isLayerRegistrationPending || this.registrationFailed) return;
    this.scopes.encode(encoder);
    this.domainExplorer.encode(encoder);
    for (const entry of this.plots.values()) entry.explorer.encode(encoder);
    this.encodedGeneration = this.generation;
  }

  /** Bounded extrema, histograms and spatial completeness only; never source rows or mask arrays. */
  readSummary(): Promise<boolean> {
    if (!this.canConsume) return Promise.resolve(false);
    if (this.isSummaryCurrent) return Promise.resolve(true);
    if (!this.pendingRead) {
      this.pendingRead = this.readLatestSummary().finally(() => {
        this.pendingRead = undefined;
        if (!this.destroyed) this.collectRetired();
      });
    }
    return this.pendingRead;
  }

  /** A coherent snapshot includes every current plot and the current domain/spatial diagnostics. */
  get isSummaryCurrent(): boolean {
    return (
      this.canConsume &&
      this.domainSummary?.generation === this.generation &&
      this.domainSummary.revision === this.nativeRevision
    );
  }

  /** True when the domain mask reduces this field (a range field with a resident column). */
  hasDomainField(field: string): boolean {
    return (
      this.props.fields[field]?.kind === 'range' && Boolean(this.lease.dataset.value.columns[field])
    );
  }

  /**
   * Finite [minimum, maximum] of a range field over the domain mask (dynamic scalar filters plus
   * any linked legend brush), as Kepler's filteredIndexForDomain does. Null until the current
   * summary is read, or when no row passes. Call readSummary() to refresh after changes.
   */
  readFilteredDomain(field: string): readonly [number, number] | null {
    return this.isSummaryCurrent ? (this.domainSummary?.domains[field] ?? null) : null;
  }

  /** Existing native plots must never fall back to an inclusive primary histogram while pending. */
  hasPlot(filterId: string): boolean {
    return this.plots.has(filterId);
  }

  publishHistograms(state: KeplerComputeTableState): boolean {
    this.bridgeState = state;
    if (this.destroyed || !this.isCurrentDataset) return false;
    if (
      this.encodedGeneration !== this.generation ||
      this.domainSummary?.generation !== this.generation ||
      this.domainSummary.revision !== state.revision ||
      this.nativeRevision !== state.revision
    )
      return false;
    let published = false;
    for (const [filterId, snapshot] of this.histograms) {
      if (snapshot.generation === this.generation && snapshot.revision === state.revision)
        published =
          state.setFilterHistogram(filterId, snapshot.bins, snapshot.revision, snapshot.interval) ||
          published;
    }
    return state.completeScope(this.nativeRevision) || published;
  }

  getFilterHistogram(filterId: string): readonly KeplerComputeBin[] | undefined {
    const snapshot = this.histograms.get(filterId);
    return this.isSummaryCurrent &&
      snapshot?.generation === this.generation &&
      snapshot.revision === this.nativeRevision
      ? snapshot.bins
      : undefined;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.completion) return this.completion;
    this.destroyed = true;
    categoryCoverageRegistry.clear(this.props.datasetId, this);
    fieldPreparationCoordinator.markSettled(this.props.datasetId);
    for (const filterId of this.unsupportedPlots.keys())
      this.props.onPlotUnsupported?.(filterId, null);
    this.unsupportedPlots.clear();
    const fence = Promise.race([completion, this.props.session.device.lost]);
    this.completion = (async () => {
      const results = await Promise.allSettled([
        fence,
        this.pendingRead,
        ...this.retirements,
        this.scalar?.close(fence),
        ...this.retiredScalars.splice(0).map(scalar => scalar.close(fence)),
        this.domainExplorer.close(fence),
        ...[...this.plots.values()].map(entry => entry.explorer.close(fence)),
        ...this.retiredExplorers.splice(0).map(explorer => explorer.close(fence)),
        this.scopes.destroy(fence)
      ]);
      this.plots.clear();
      this.histograms.clear();
      await this.lease.release();
      const failure = results.find(result => result.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
    })();
    return this.completion;
  }

  private applyNativeFilters(options: NativeFilterRequest): void {
    this.assertActive();
    const plans: ScalarPlan[] = [];
    const polygons: KeplerPolygonMaskFilter[] = [];
    const desiredPlots = new Map<string, {field: string; plot: ResidentScopePlot}>();
    const unsupportedPlots = new Map<string, string>();
    for (const filter of options.filters) {
      if (filter.type === 'polygon') {
        const polygon = this.preparePolygon(filter, options.layers);
        if (polygon) polygons.push(polygon);
        continue;
      }
      const field = getFilterField(filter, this.props.datasetId, options.fields);
      if (!field || !filter.type) continue;
      if (filter.enabled !== false && !this.props.fields[field]) {
        // The filter reads a column the resident copy lacks: re-prepare once, otherwise say why not.
        const outcome = fieldPreparationCoordinator.request(this.props.datasetId, field);
        const detail =
          outcome.status === 'requested' || outcome.status === 'pending'
            ? 'Preparing it now.'
            : outcome.reason;
        throw new Error(
          `${describeUnpreparedFilterField(filter, this.props.datasetId, field)} ${detail}`
        );
      }
      const planned = planKeplerComputeFilters(
        this.props.datasetId,
        options.fields,
        this.props.fields,
        [filter],
        {independentPredicates: true}
      );
      const command = planned.display[0];
      if (command)
        plans.push({
          filter,
          field,
          command,
          specification: this.makeSpecification(filter.id, field)
        });
      if (filter.type === 'range' || filter.type === 'timeRange') {
        try {
          const plot = this.makePlot(field, filter);
          if (plot) desiredPlots.set(filter.id, {field, plot});
          else if (filter.enabled !== false)
            unsupportedPlots.set(filter.id, 'Exact time histogram metadata is unavailable');
        } catch (error) {
          if (filter.enabled !== false)
            unsupportedPlots.set(filter.id, error instanceof Error ? error.message : String(error));
        }
      }
    }
    if (
      plans.length + polygons.length > (this.props.maximumFilters ?? 16) ||
      desiredPlots.size > (this.props.maximumFilters ?? 16)
    )
      throw new Error('Resident native filter capacity exceeded');
    if (this.isLayerRegistrationPending) {
      // Native IDs and drawn geometry are already validated above. Only readiness of their
      // resident geometry selectors is deferred until the host finishes rebuilding native layers.
      this.pendingNativeRequest = {
        ...options,
        filters: [...options.filters],
        layers: [...options.layers]
      };
      this.registrationPlanError = undefined;
      this.nativeRevision = options.revision;
      this.invalidateConsumers();
      return;
    }
    plans.sort((left, right) => left.filter.id.localeCompare(right.filter.id));
    const topology = JSON.stringify(
      plans.map(plan => [
        plan.filter.id,
        plan.field,
        plan.specification.kind,
        plan.specification.kind === 'category' ? plan.specification.capacity : undefined,
        plan.specification.kind === 'time' ? Boolean(plan.specification.chartNullAsEpoch) : false
      ])
    );
    const signatures = new Map(
      plans.map(plan => [
        plan.filter.id,
        JSON.stringify([
          plan.field,
          plan.specification.kind,
          plan.specification.kind === 'category' ? plan.specification.capacity : undefined,
          plan.specification.kind === 'time' ? Boolean(plan.specification.chartNullAsEpoch) : false
        ])
      ])
    );
    const workingMemoryBudgetBytes =
      this.props.scalarWorkingMemoryBudgetBytes ??
      this.lease.dataset.rowCount * Math.max(1, plans.length) * 64 + 1024 * 1024;
    // A full rebuild is only needed to create the first scalar selection, after it was emptied, or
    // after a failed incremental update; otherwise a single filter slot is added or removed.
    const incremental =
      plans.length > 0 &&
      Boolean(this.scalar) &&
      !this.scalarBroken &&
      topology !== this.scalarTopology;
    const replaceScalar =
      !incremental &&
      (topology !== this.scalarTopology ||
        (plans.length > 0 && (!this.scalar || this.scalarBroken)));
    const scalar =
      plans.length === 0
        ? undefined
        : replaceScalar
          ? new GPUKeplerScalarSelection({
              device: this.props.session.device,
              columns: this.lease.dataset.value.columns,
              validity: this.lease.dataset.value.analytics.validity,
              filters: plans.map(plan => plan.specification),
              registry: this.props.session.registry,
              datasetId: this.props.datasetId,
              datasetRevision: this.lease.dataset.revision,
              workingMemoryBudgetBytes
            })
          : this.scalar!;
    const slotChanges = incremental
      ? {
          obsolete: [...this.scalarSlots]
            .filter(([filterId, signature]) => signatures.get(filterId) !== signature)
            .map(([filterId]) => filterId),
          added: plans.filter(
            plan => this.scalarSlots.get(plan.filter.id) !== signatures.get(plan.filter.id)
          )
        }
      : undefined;
    const addedFilterIds: string[] = [];
    let removalFence: ReturnType<typeof this.props.session.device.createFence> | undefined;
    const createdPlots = new Map<string, PlotEntry>();
    try {
      const controls = plans.map(plan => this.makeControl(plan));
      for (const [filterId, requested] of desiredPlots) {
        const key = JSON.stringify(requested);
        if (this.plots.get(filterId)?.key === key) continue;
        try {
          const mask = this.scopes.getPlotMask(filterId);
          const explorer = requested.plot.numericEdges
            ? new ResidentNumericHistogram({
                session: this.props.session,
                datasetId: this.props.datasetId,
                field: requested.field,
                plan: requested.plot.numericEdges,
                mask,
                workingMemoryBudgetBytes: this.inspectorBudget,
                allowIncompletePopulation: this.props.allowIncompletePopulation,
                measure: requested.plot.measure,
                grouping: requested.plot.group,
                measureWorkingMemoryBudgetBytes: this.measureBudget
              })
            : requested.plot.timeThresholds
              ? new ResidentTimeHistogram({
                  session: this.props.session,
                  datasetId: this.props.datasetId,
                  field: requested.field,
                  plan: {
                    interval: requested.plot.interval!,
                    thresholds: requested.plot.timeThresholds
                  },
                  mask,
                  workingMemoryBudgetBytes: this.inspectorBudget,
                  allowIncompletePopulation: this.props.allowIncompletePopulation,
                  measure: requested.plot.measure,
                  grouping: requested.plot.group,
                  measureWorkingMemoryBudgetBytes: this.measureBudget
                })
              : this.props.session.openExplorer(this.props.datasetId, {
                  allowIncompletePopulation: this.props.allowIncompletePopulation,
                  workingMemoryBudgetBytes: this.inspectorBudget,
                  maximumPageSize: 1,
                  maximumConcurrentReads: 1,
                  maximumReadbackBytes: 4096,
                  dimensions: [{id: 'plot-population', kind: 'external', mask}],
                  histograms: [
                    {
                      id: 'native-plot',
                      column: requested.plot.column,
                      domain: requested.plot.domain,
                      binCount: requested.plot.binCount,
                      includeOwnSelection: true
                    }
                  ]
                });
          if ('setExternalActive' in explorer) explorer.setExternalActive('plot-population', true);
          createdPlots.set(filterId, {...requested, key, explorer});
        } catch (error) {
          desiredPlots.delete(filterId);
          unsupportedPlots.set(filterId, error instanceof Error ? error.message : String(error));
        }
      }
      if (incremental && scalar && slotChanges) {
        // Same-ID slots whose field/kind changed are removed first, then every new slot is added.
        removalFence = this.props.session.device.createFence();
        const changedIds = slotChanges.obsolete.filter(filterId => signatures.has(filterId));
        for (const filterId of changedIds) {
          scalar.removeFilter(filterId, removalFence.signaled);
          this.scalarBroken = true;
        }
        for (const plan of slotChanges.added) {
          scalar.addFilter(
            plan.specification,
            {
              columns: this.lease.dataset.value.columns,
              validity: this.lease.dataset.value.analytics.validity
            },
            {workingMemoryBudgetBytes}
          );
          addedFilterIds.push(plan.filter.id);
        }
        scalar.updateControls(controls);
      } else scalar?.setControls(controls);
      this.scopes.setFilters(
        plans.map(plan => ({
          filterId: plan.filter.id,
          mask: scalar!.getMask(plan.filter.id),
          dynamicDomain: !plan.filter.fixedDomain,
          nativeTopology: usesNativeTopologyPredicate(plan.filter),
          nativeFieldName: plan.field,
          inverted: plan.filter.inverted
        })),
        polygons
      );
      if (incremental && scalar && slotChanges) {
        removalFence ??= this.props.session.device.createFence();
        // Same-ID replacements were already removed before their new slot was added.
        for (const filterId of slotChanges.obsolete)
          if (!signatures.has(filterId) && scalar.hasFilter(filterId))
            scalar.removeFilter(filterId, removalFence.signaled);
        scalar.setWorkingMemoryBudget(workingMemoryBudgetBytes);
        const fenceToDestroy = removalFence;
        void Promise.allSettled([fenceToDestroy.signaled]).finally(() => fenceToDestroy.destroy());
        removalFence = undefined;
        this.scalarBroken = false;
      }
      if (replaceScalar) {
        if (this.scalar) this.retiredScalars.push(this.scalar);
        this.scalar = scalar;
        this.scalarBroken = false;
      }
      this.scalarTopology = topology;
      this.scalarSlots = signatures;
      for (const [filterId, previous] of this.plots) {
        if (!desiredPlots.has(filterId) || createdPlots.has(filterId)) {
          this.retiredExplorers.push(previous.explorer);
          this.plots.delete(filterId);
          this.histograms.delete(filterId);
          if (!desiredPlots.has(filterId)) this.scopes.removePlot(filterId);
        }
      }
      for (const [filterId, entry] of createdPlots) this.plots.set(filterId, entry);
      for (const filterId of this.unsupportedPlots.keys()) {
        if (!unsupportedPlots.has(filterId)) this.props.onPlotUnsupported?.(filterId, null);
      }
      for (const [filterId, reason] of unsupportedPlots) {
        if (this.unsupportedPlots.get(filterId) !== reason)
          this.props.onPlotUnsupported?.(filterId, reason);
      }
      this.unsupportedPlots.clear();
      for (const [filterId, reason] of unsupportedPlots)
        this.unsupportedPlots.set(filterId, reason);
      this.lastFilterRevision = makeFilterRevisionReport(
        this.nativeRequest?.filters ?? [],
        options.filters,
        this.props.datasetId,
        options.fields,
        replaceScalar
      );
      categoryCoverageRegistry.set(
        this.props.datasetId,
        listCategoryCoverageWarnings(
          options.filters,
          this.props.datasetId,
          options.fields,
          this.props.fields
        ),
        this
      );
      this.nativeRevision = options.revision;
      this.nativeRequest = {...options, filters: [...options.filters], layers: [...options.layers]};
      this.nativeLayerIdentity = getNativeLayerIdentity(options.layers);
      this.registrationFailed = false;
      this.registrationPlanError = undefined;
      this.invalidateConsumers();
    } catch (error) {
      if (incremental && scalar) {
        // Undo this call's added slots; if a same-ID slot was already replaced, force a rebuild.
        for (const filterId of addedFilterIds)
          if (scalar.hasFilter(filterId)) scalar.removeFilter(filterId);
        if (removalFence) {
          const fenceToDestroy = removalFence;
          void Promise.allSettled([fenceToDestroy.signaled]).finally(() =>
            fenceToDestroy.destroy()
          );
        }
      }
      const fence = this.props.session.device.createFence();
      void Promise.allSettled([
        replaceScalar ? scalar?.close(fence.signaled) : undefined,
        ...[...createdPlots.values()].map(entry => entry.explorer.close(fence.signaled))
      ]).finally(() => fence.destroy());
      throw error;
    }
  }

  private makeSpecification(filterId: string, field: string): GPUKeplerScalarFilter {
    const binding = this.props.fields[field];
    if (!binding) throw new Error('Native filter has no resident field binding');
    if (binding.kind === 'time') {
      const source = this.lease.dataset.value.timeColumns[field];
      if (!source) throw new Error('Native time filter requires exact source time words');
      const domain = numericDomain(binding.domain);
      return {
        filterId,
        kind: 'time',
        chartNullAsEpoch: this.chartTimeNullEpochFilters.get(filterId) === field,
        timestamps: source.timestamps,
        validity: source.validity,
        unit: source.metadata.unit,
        window: prepareKeplerTimeRangeMilliseconds(domain[0], domain[1], source.metadata.unit)
      };
    }
    if (binding.kind === 'category')
      return {
        filterId,
        kind: 'category',
        column: field,
        capacity: clamp((binding.categories ?? binding.domain).length, 1, 64)
      };
    return {filterId, kind: 'range', column: field};
  }

  private preparePolygon(
    filter: KeplerComputeFilter,
    layers: readonly unknown[]
  ): KeplerPolygonMaskFilter | undefined {
    return prepareNativePolygonFilter(
      filter,
      this.props.datasetId,
      layers,
      this.isLayerRegistrationPending
        ? getNativeDatasetLayerIds(layers, this.props.datasetId)
        : this.registeredLayerIds
    );
  }

  private makeControl(plan: ScalarPlan): GPUKeplerScalarControl {
    const common = {filterId: plan.filter.id, enabled: true, empty: plan.command.kind === 'empty'};
    if (plan.specification.kind === 'category')
      return {
        ...common,
        kind: 'category',
        categories: plan.command.kind === 'category' ? plan.command.value : []
      };
    const range = plan.command.kind === 'range' ? plan.command.value : ([0, 0] as const);
    if (plan.specification.kind === 'time') {
      const window = prepareKeplerTimeRangeMilliseconds(
        range[0],
        range[1],
        plan.specification.unit
      );
      return {
        ...common,
        kind: 'time',
        window: {start: String(window.start), end: String(window.end), unit: window.unit}
      };
    }
    return {...common, kind: 'range', minimum: range[0], maximum: range[1]};
  }

  private getTimeHistogramInterval(
    field: string,
    filter?: KeplerComputeFilter
  ): string | undefined {
    this.assertActive();
    const plot = this.makePlot(field, filter);
    if (!plot?.interval) return undefined;
    if (filter) this.reconcilePlot(field, filter, plot);
    return plot.interval;
  }

  /**
   * Native numeric-interval controls update plot metadata without necessarily running filterTable.
   * Reconcile the retained request before certifying the plot; an explicit equal-width mapping
   * cannot honor a native interval and says so.
   */
  private prepareNumericPlot(field: string, filter: KeplerComputeFilter): void {
    this.assertActive();
    const plot = this.makePlot(field, filter);
    if (!plot) return;
    if (!plot.numericEdges) {
      if (
        Number.isFinite(filter.plotType?.numericInterval) &&
        filter.plotType!.numericInterval! > 0
      )
        throw new Error(
          'This resident plot has a fixed equal-width mapping; native numeric intervals need native edges'
        );
      return;
    }
    this.reconcilePlot(field, filter, plot);
  }

  private reconcilePlot(field: string, filter: KeplerComputeFilter, plot: ResidentScopePlot): void {
    // Reconcile the retained request before certifying a plot, then await a new submitted frame.
    const current = this.plots.get(filter.id);
    if (
      this.nativeRequest &&
      (current?.field !== field ||
        current.plot.interval !== plot.interval ||
        JSON.stringify(current.plot.timeThresholds) !== JSON.stringify(plot.timeThresholds) ||
        JSON.stringify(current.plot.numericEdges) !== JSON.stringify(plot.numericEdges) ||
        JSON.stringify(current.plot.measure) !== JSON.stringify(plot.measure) ||
        JSON.stringify(current.plot.group) !== JSON.stringify(plot.group))
    ) {
      const previous = this.nativeRequest.filters.find(candidate => candidate.id === filter.id);
      if (
        previous &&
        (!this.unsupportedPlots.has(filter.id) ||
          previous.plotType?.interval !== plot.interval ||
          previous.plotType?.numericInterval !== filter.plotType?.numericInterval ||
          previous.plotType?.type !== filter.plotType?.type ||
          previous.plotType?.aggregation !== filter.plotType?.aggregation ||
          previous.yAxis?.name !== filter.yAxis?.name ||
          JSON.stringify(previous.plotType?.groupBy) !== JSON.stringify(filter.plotType?.groupBy) ||
          getFilterField(previous, this.props.datasetId, this.nativeRequest.fields) !== field ||
          JSON.stringify(previous.domain) !== JSON.stringify(filter.domain))
      ) {
        this.applyNativeFilters({
          ...this.nativeRequest,
          filters: this.nativeRequest.filters.map(candidate =>
            candidate.id === filter.id
              ? {
                  ...filter,
                  domain: filter.domain ?? candidate.domain,
                  plotType: plot.interval
                    ? {...filter.plotType, interval: plot.interval}
                    : filter.plotType
                }
              : candidate
          )
        });
        this.props.onUpdate();
      }
    }
    if (this.unsupportedPlots.has(filter.id)) throw new Error(this.unsupportedPlots.get(filter.id));
  }

  private makePlot(field: string, filter?: KeplerComputeFilter): ResidentScopePlot | undefined {
    const explicit = this.props.plots?.[field];
    if (explicit) {
      if (!this.lease.dataset.value.columns[explicit.column])
        throw new Error('Histogram source column is missing');
      return explicit;
    }
    const binding = this.props.fields[field];
    if (!binding || binding.kind === 'category') return undefined;
    if (binding.kind === 'time') {
      const domain = numericDomain(filter?.domain ?? binding.domain);
      const plan = makeResidentTimeHistogramPlan(
        domain,
        filter?.plotType?.interval ?? binding.histogramInterval
      );
      const measure = this.makeMeasure(filter);
      const group = this.makeGroup(filter);
      return {
        column: field,
        domain,
        binCount: plan.thresholds.length,
        interval: plan.interval,
        timeThresholds: plan.thresholds,
        ...(measure ? {measure} : {}),
        ...(group ? {group} : {})
      };
    }
    // Native plots use the filter domain (the merged domain of every linked dataset) with d3 tick
    // edges, or the configured numeric interval; every dataset therefore shares the same edges.
    const numericEdges = makeResidentNumericPlotEdges(
      numericDomain(filter?.domain ?? binding.domain),
      filter?.plotType?.numericInterval
    );
    const measure = this.makeMeasure(filter);
    const group = this.makeGroup(filter);
    return {
      column: field,
      domain: numericEdges.domain,
      binCount: numericEdges.thresholds.length + 1,
      numericEdges,
      ...(measure ? {measure} : {}),
      ...(group ? {group} : {})
    };
  }

  /**
   * Native draws a line chart for the first dataset of a linked filter only; its y axis is a
   * field of that dataset, aggregated per time or numeric bin. Other resident datasets
   * contribute counts alone.
   */
  private makeMeasure(
    filter?: KeplerComputeFilter
  ): {column: string; operation: ResidentPlotMeasureOperation} | undefined {
    const column = filter?.yAxis?.name;
    if (filter?.plotType?.type !== 'lineChart' || !column) return undefined;
    const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
    if (datasetIds[0] !== this.props.datasetId) return undefined;
    const aggregation = filter.plotType.aggregation ?? 'average';
    const operation =
      RESIDENT_PLOT_MEASURE_OPERATIONS[
        aggregation as keyof typeof RESIDENT_PLOT_MEASURE_OPERATIONS
      ];
    // Other native aggregations (mode, variance, unique count, ...) have no exact grouped GPU reduction.
    if (!operation) throw new Error(`Resident line charts do not support ${aggregation}`);
    return {column, operation};
  }

  /** Native groups the first dataset's line series by `String(value)` of a dictionary field. */
  private makeGroup(filter?: KeplerComputeFilter): ResidentPlotGroupingProps | undefined {
    const groupBy = filter?.plotType?.groupBy;
    if (filter?.plotType?.type !== 'lineChart' || !groupBy?.fieldName) return undefined;
    const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
    if (datasetIds[0] !== this.props.datasetId) return undefined;
    const source = this.lease.dataset.value;
    // Numeric, boolean and unprepared group fields have no exact label table on the GPU.
    if (!source.prepared.dictionaries[groupBy.fieldName])
      throw new Error('Resident grouped line charts need a prepared string dictionary field');
    return {
      column: groupBy.fieldName,
      numGroups: groupBy.numGroups,
      groupOthers: Boolean(groupBy.groupOthers)
    };
  }

  private invalidateConsumers(): void {
    this.generation++;
    this.encodedGeneration = -1;
    this.domainSummary = undefined;
    this.histograms.clear();
    this.bridgeState?.invalidateScope();
    this.domainExplorer.invalidate();
    for (const entry of this.plots.values()) entry.explorer.invalidate();
    this.primary?.invalidate();
  }

  private async readCurrentSummary(): Promise<boolean> {
    const generation = this.generation;
    const revision = this.nativeRevision;
    const entries = [...this.plots];
    const [status, domain, summaries] = await Promise.all([
      this.scopes.readStatus(),
      this.domainExplorer.readSummary(),
      Promise.all(entries.map(([, entry]) => entry.explorer.readSummary()))
    ]);
    if (
      this.destroyed ||
      !this.isCurrentDataset ||
      this.generation !== generation ||
      !domain ||
      summaries.some(summary => !summary) ||
      status.stale
    )
      return false;
    if (!status.complete) {
      this.props.onUnsupported('Resident spatial selection is incomplete');
      return false;
    }
    this.domainSummary = {generation, revision, domains: domain.domains};
    for (const [index, [filterId, entry]] of entries.entries()) {
      const counts = summaries[index]!.histograms['native-plot'];
      const measure = summaries[index]!.measure;
      const groups = summaries[index]!.groups;
      const [minimum, maximum] = entry.plot.nativeDomain ?? entry.plot.domain;
      const width = (maximum - minimum) / counts.length;
      const edges = entry.plot.numericEdges?.edges;
      this.histograms.set(filterId, {
        generation,
        revision,
        interval: entry.plot.interval,
        bins: Array.from(counts, (count, bin) => ({
          x0: edges?.[bin] ?? entry.plot.timeThresholds?.[bin] ?? minimum + bin * width,
          x1:
            edges?.[bin + 1] ??
            entry.plot.timeThresholds?.[Math.min(bin + 1, counts.length - 1)] ??
            minimum + (bin + 1) * width,
          count,
          indexes: [],
          ...(measure
            ? {
                // An empty reduction is a missing value for mean/min/max and zero for a sum.
                measure: measure.valid[bin]
                  ? measure.values[bin]
                  : entry.plot.measure?.operation === 'sum'
                    ? 0
                    : null
              }
            : {}),
          ...(groups ? makeGroupedBin(groups, bin, counts.length, entry.plot.measure) : {})
        })).filter(bin => !entry.plot.timeThresholds || bin.count > 0)
      });
    }
    return true;
  }

  private async readLatestSummary(): Promise<boolean> {
    // One coalesced retry handles a newer submitted frame overtaking an outstanding read. Do not
    // spin while dragging: the host's next submitted frame schedules any further attempt.
    for (let attempt = 0; attempt < 2; attempt++) {
      if (this.destroyed || !this.isCurrentDataset || this.encodedGeneration !== this.generation)
        return false;
      const generation = this.generation;
      const ready = await this.readCurrentSummary();
      if (ready && this.isSummaryCurrent) return true;
      if (this.generation === generation) return false;
    }
    return false;
  }

  private collectRetired(): void {
    const fence = this.props.session.device.createFence();
    const completion = Promise.race([fence.signaled, this.props.session.device.lost]);
    const cleanup = Promise.allSettled([
      this.scopes.collectRetired(completion),
      ...this.retiredScalars.splice(0).map(scalar => scalar.close(completion)),
      ...this.retiredExplorers.splice(0).map(explorer => explorer.close(completion))
    ])
      .then(results => {
        const failure = results.find(result => result.status === 'rejected');
        if (failure?.status === 'rejected') throw failure.reason;
      })
      .catch(error => {
        if (!this.destroyed) this.props.onUnsupported(String(error));
      })
      .finally(() => {
        fence.destroy();
        this.retirements.delete(cleanup);
      });
    this.retirements.add(cleanup);
  }

  /** Y-axis measures reserve their own sorted or grouped scratch, bounded by the session budget. */
  private get measureBudget(): number {
    return getResidentPlotMeasureBudgetBytes(this.lease.dataset.rowCount, 64);
  }

  private get inspectorBudget(): number {
    return (
      this.props.inspectorWorkingMemoryBudgetBytes ?? this.lease.dataset.rowCount * 64 + 1024 * 1024
    );
  }

  private assertActive(): void {
    if (this.destroyed || !this.isCurrentDataset)
      throw new Error('Resident scope binding is closed or stale');
  }

  private get isCurrentDataset(): boolean {
    return (
      this.lease.isValid() &&
      this.props.session.registry.getDataset(this.props.datasetId)?.revision ===
        this.lease.dataset.revision
    );
  }
}

function getFilterField(
  filter: KeplerComputeFilter,
  datasetId: string,
  fields: KeplerComputeTableLike['fields']
): string | undefined {
  const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
  const index = datasetIds.indexOf(datasetId);
  if (index < 0) return undefined;
  return filter.name?.[index] ?? fields[filter.fieldIdx?.[index] ?? -1]?.name;
}

function numericDomain(
  domain: readonly number[] | readonly string[] | readonly boolean[]
): readonly [number, number] {
  const minimum = domain[0];
  const maximum = domain[domain.length - 1];
  if (
    typeof minimum !== 'number' ||
    typeof maximum !== 'number' ||
    !Number.isFinite(minimum) ||
    !Number.isFinite(maximum) ||
    minimum > maximum
  )
    throw new Error('Resident numeric fields require finite ordered source domains');
  return [minimum, maximum];
}

function getNativeLayerIdentity(layers: readonly unknown[]): string {
  return JSON.stringify(
    layers.map(layer => {
      if (!layer || typeof layer !== 'object') return null;
      const record = layer as Record<string, unknown>;
      const config = record['config'];
      return [
        record['id'],
        record['type'],
        config && typeof config === 'object'
          ? (config as Record<string, unknown>)['dataId']
          : undefined
      ];
    })
  );
}

function getNativeDatasetLayerIds(layers: readonly unknown[], datasetId: string): string[] {
  return layers.flatMap(layer => {
    if (!layer || typeof layer !== 'object') return [];
    const record = layer as Record<string, unknown>;
    const config = record['config'];
    return typeof record['id'] === 'string' &&
      config &&
      typeof config === 'object' &&
      (config as Record<string, unknown>)['dataId'] === datasetId
      ? [record['id']]
      : [];
  });
}

/** Series present in one plot bin, from dense `slot * binCount + bin` counts and measures. */
function makeGroupedBin(
  groups: ResidentPlotGroupingSummary,
  bin: number,
  binCount: number,
  measure: ResidentScopePlot['measure']
): Pick<KeplerComputeBin, 'groups' | 'groupLabels' | 'groupOthers'> {
  const slotCount = groups.labels.length + (groups.others ? 1 : 0);
  const present: KeplerComputeBin['groups'] = [];
  for (let slot = 0; slot < slotCount; slot++) {
    const key = slot * binCount + bin;
    const count = groups.counts[key];
    if (!count) continue;
    // Without a y axis a series plots its row count; an empty reduction is a gap, or zero for a sum.
    const value = !measure
      ? count
      : groups.measure?.valid[key]
        ? groups.measure.values[key]
        : measure.operation === 'sum'
          ? 0
          : null;
    (present as {slot: number; count: number; measure: number | null}[]).push({
      slot,
      count,
      measure: value
    });
  }
  return {groups: present, groupLabels: groups.labels, groupOthers: groups.others};
}
