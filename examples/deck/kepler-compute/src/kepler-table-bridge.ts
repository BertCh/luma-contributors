// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describeUnpreparedFilterField} from './binding2-filter-lifecycle';
import {installResidentSynchronousGuards} from './resident-synchronous-guards';

/** The subset of pinned Kepler filter state consumed by resident exploration. */
export type KeplerComputeFilter = {
  id: string;
  dataId: readonly string[] | string;
  name?: readonly string[];
  fieldIdx?: readonly number[];
  type?: string | null;
  value?: unknown;
  domain?: readonly number[] | readonly string[] | readonly boolean[] | null;
  fixedDomain?: boolean;
  enabled?: boolean;
  inverted?: boolean;
  /** Native allocation metadata; resident rendering still consumes its independently encoded mask. */
  gpu?: boolean;
  gpuChannel?: readonly number[];
  gpuEndChannel?: readonly number[];
  animationWindow?: string;
  endName?: readonly (string | null | undefined)[];
  plotType?: {
    type?: string;
    interval?: string;
    numericInterval?: number;
    aggregation?: string;
    groupBy?: {fieldName?: string; numGroups?: number | string; groupOthers?: boolean} | null;
  };
  /** Native line-chart measure; null/absent plots row counts. */
  yAxis?: {name?: string} | null;
  layerId?: readonly string[] | null;
  bins?: unknown;
  timeBins?: unknown;
  lineChart?: unknown;
};

/** Pinned getFilterRecord CPU membership; keep native allocation/history authoritative. */
export function usesNativeTopologyPredicate(filter: KeplerComputeFilter): boolean {
  return !filter.gpu || (filter.type === 'range' && filter.animationWindow === 'interval');
}

export type KeplerComputeFieldBinding = {
  dimension: string;
  kind: 'range' | 'category' | 'time';
  /** Full source domain, prepared once during ingestion, never scanned during filtering. */
  domain: readonly number[] | readonly string[] | readonly boolean[];
  step?: number;
  /** Interval used to compute this field's cached GPU time histogram; never relabel its bins. */
  histogramInterval?: string;
  /** Rows are intervals ending in this column (time-contract declaration); native endName filters match it. */
  intervalEndColumn?: string;
  /** Pure source-specific validation (for example exact Timestamp units), before any GPU mutation. */
  validateRange?(minimum: number, maximum: number): void;
  /** Labels in the exact order of GPU category codes. */
  categories?: readonly (number | string | boolean)[];
  /** Optional prepared domains for a scale; quantile/Jenks cannot use min/max as a substitute. */
  scaleDomains?: Readonly<Record<string, readonly number[] | readonly string[]>>;
};

export type KeplerComputeFilterCommand =
  | {kind: 'range'; dimension: string; value: readonly [number, number]}
  | {kind: 'category'; dimension: string; value: readonly number[]}
  | {kind: 'empty'; dimension: string};

export type KeplerComputeFilterPlan = {
  display: readonly KeplerComputeFilterCommand[];
  domain: readonly KeplerComputeFilterCommand[];
  filterIds: readonly string[];
  /** Polygon masks remain layer-specific; the provider declares their supported target layers. */
  spatial?: readonly {filterId: string; layerIds: readonly string[]}[];
};

export type KeplerComputeBin = {
  x0: number;
  x1: number;
  count: number;
  /** Resident histograms expose counts, not source-row arrays. */
  indexes: readonly never[];
  /** Native line-chart y value for this bin (null: no valid measured row), reduced on the GPU. */
  measure?: number | null;
  /** Grouped line series present in this bin, in series order (`slot` indexes `groupLabels`). */
  groups?: readonly KeplerComputeBinGroup[];
  /** Series labels in native first-seen order; a merged Others series follows when `groupOthers`. */
  groupLabels?: readonly string[];
  groupOthers?: boolean;
};

export type KeplerComputeBinGroup = {
  slot: number;
  /** Rows of this series in the bin. */
  count: number;
  /** Reduced y value (the row count without a y axis); null: no valid measured row. */
  measure: number | null;
};

export type KeplerComputeTableBinding = {
  fields: Readonly<Record<string, KeplerComputeFieldBinding>>;
  /** Exact per-filter scope histograms, including multiple predicates on the same field. */
  filterScopedHistograms?: boolean;
  /** Reduces native grouped line series (`plotType.groupBy`) over the plot bins on the GPU. */
  groupedLineCharts?: boolean;
  /** Explicit exact-word calendar runtime. Returns only a supported, current per-filter interval.
   * Without a filter, returns a bounded default for new native time controls. */
  getTimeHistogramInterval?(field: string, filter?: KeplerComputeFilter): string | undefined;
  /**
   * Certify that a range filter's native plot (d3 tick or numeric-interval edges, y-axis measure)
   * is the plot the GPU computes, reconciling the retained request; throws when unsupported.
   */
  prepareNumericPlot?(field: string, filter: KeplerComputeFilter): void;
  /**
   * Explicit host opt-in when a resident renderer owns style computation, legends and per-layer
   * unsupported-style diagnostics. Unavailable native style domains are empty metadata only;
   * they are never distributions, GPU break arrays or permission to scan source rows.
   */
  residentStyleDomains?: boolean;
  /**
   * Current-revision thresholds already derived by the resident style runtime for a Jenks field.
   * They seed Kepler's native custom-breaks editor; absent while pending, never a placeholder.
   */
  getDerivedJenksDomain?(field: string): readonly number[] | undefined;
  /** Replaces legacy field-combined execution with an independently owned native-filter runtime. */
  applyNativeFilters?(options: {
    filters: readonly KeplerComputeFilter[];
    plan: KeplerComputeFilterPlan;
    revision: number;
    fields: KeplerComputeTableLike['fields'];
    layers: readonly unknown[];
  }): void;
  explorer: {
    clear(dimension: string): void;
    setRange(dimension: string, range: readonly [number, number]): void;
    setCategories(dimension: string, categories: readonly number[]): void;
    setEmpty?(dimension: string): void;
  };
  /** Called after every valid change; schedule encode and bounded summary readback here. */
  onSelectionChange(plan: KeplerComputeFilterPlan, revision: number): void;
  /** Must make unsupported state visible and prevent stale resident output being presented as current. */
  onUnsupported(reason: string): void;
  /** A corrected native plot can resume a still-valid resident selection without re-filtering. */
  onPlotRecovery?(): void;
  /** Plot-only diagnostics must not hide a valid map selection; null clears the filter notice. */
  onPlotUnsupported?(filterId: string, reason: string | null): void;
  /** Optional separate GPU domain executor; never substitute the display mask for this scope. */
  applyDomainSelection?(commands: readonly KeplerComputeFilterCommand[], revision: number): void;
  getDynamicLayerDomain?(
    field: string,
    scale: string,
    revision: number
  ): readonly number[] | readonly string[] | null;
  /** Validate and prepare without changing active GPU state; apply runs only after the whole plan is valid. */
  prepareSpatialFilter?(
    filter: KeplerComputeFilter,
    layers: readonly unknown[]
  ): {
    layerIds: readonly string[];
    apply(): void;
  };
  /** Bypasses prior layer-specific masks when native polygon filters are removed. */
  clearSpatialFilters?(): void;
};

export type KeplerComputeTableLike = {
  id: string;
  fields: {name: string; type?: string; filterProps?: unknown}[];
  allIndexes: number[];
  filteredIndex: number[];
  filteredIndexForDomain: number[];
  filteredIndexByLayer: Record<string, number[]>;
  filteredIdxCPU?: number[];
};

export type KeplerComputeTableState = {
  status: 'ready' | 'pending' | 'unsupported';
  reason: string | null;
  revision: number;
  plan: KeplerComputeFilterPlan;
  /** Invalidate plots after a layer/source-scope change without inventing a native filter revision. */
  invalidateScope(): void;
  /** Accept bounded completion even when the current selection has no histogram filters. */
  completeScope(revision: number): boolean;
  setHistogram(field: string, bins: readonly KeplerComputeBin[], revision: number): boolean;
  setFilterHistogram(
    filterId: string,
    bins: readonly KeplerComputeBin[],
    revision: number,
    interval?: string
  ): boolean;
  /**
   * Single-dataset filters return the complete plot. A filter spanning datasets is composed by the
   * pinned seam in three steps: every resident table writes only its own entry (`prepare`), native
   * code computes the remaining CPU datasets, then every resident table clears its pending
   * placeholder (`finalize`). Native plots are keyed by dataset id and never summed.
   */
  updateFilterPlot<Filter extends KeplerComputeFilter>(
    filter: Filter,
    options?: KeplerComputeFilterPlotOptions
  ): Filter;
};

export type KeplerComputeFilterPlotOptions = {
  stage?: 'prepare' | 'finalize';
  /** Native series builder (`getLineChart`), supplied by the seam because it needs the full Kepler runtime. */
  buildLineChart?: (datasets: Record<string, unknown>, filter: unknown) => unknown;
};

/** Truthy placeholder that stops native code recomputing a pending resident entry from CPU rows. */
const PENDING_PLOT_BINS: readonly KeplerComputeBin[] = Object.freeze([]);
/** Pseudo-row labels and fields of grouped series; native names the merged series itself. */
const GROUP_OTHERS_LABEL = '\u0000luma-others';
const GROUP_COUNT_FIELD = '\u0000luma-count';
const PLOT_STATUS_PRIORITY: Record<string, number> = {
  ready: 0,
  disabled: 1,
  pending: 2,
  unsupported: 3
};

type ResidentPlotStatus = {status: string; reason?: string | null; revision: number};

function mergePlotStatus(
  previous: ResidentPlotStatus | undefined,
  next: ResidentPlotStatus
): ResidentPlotStatus {
  return previous && PLOT_STATUS_PRIORITY[previous.status] > PLOT_STATUS_PRIORITY[next.status]
    ? previous
    : next;
}

type ResidentTable = KeplerComputeTableLike & {lumaCompute: KeplerComputeTableState};
type PlannedFilter = {
  field: string;
  command: KeplerComputeFilterCommand;
  fixedDomain: boolean;
  id: string;
};

/**
 * Pure planning validates the complete request before changing any GPU selection. Independent
 * providers receive each positive predicate separately and apply native inversion after validity;
 * their raw filter descriptors remain authoritative for inversion and native GPU allocation.
 */
export function planKeplerComputeFilters(
  datasetId: string,
  fields: KeplerComputeTableLike['fields'],
  bindings: KeplerComputeTableBinding['fields'],
  filters: readonly KeplerComputeFilter[],
  options: {independentPredicates?: boolean} = {}
): KeplerComputeFilterPlan {
  const planned: PlannedFilter[] = [];
  for (const filter of filters) {
    // Pinned getFilterFunction returns an unconditional match for disabled filters.
    if (filter.enabled === false) continue;
    const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
    const datasetIndex = datasetIds.indexOf(datasetId);
    if (datasetIndex < 0 || !filter.type) continue;
    if (filter.type === 'multiSelect' && Array.isArray(filter.value) && filter.value.length === 0)
      continue;
    const field =
      filter.name?.[datasetIndex] ?? fields[filter.fieldIdx?.[datasetIndex] ?? -1]?.name;
    const binding = field ? bindings[field] : undefined;
    if (!field || !binding)
      throw new Error(describeUnpreparedFilterField(filter, datasetId, field));
    if (
      filter.endName?.[datasetIndex] &&
      binding.intervalEndColumn !== filter.endName[datasetIndex]
    )
      throw new Error('Interval filters require an explicit resident interval binding');
    let command: KeplerComputeFilterCommand;
    if (filter.type === 'range' || filter.type === 'timeRange') {
      if (binding.kind === 'category')
        throw new Error('Category columns require categorical filters');
      if (filter.inverted && !options.independentPredicates)
        throw new Error('Inverted range filters require an explicit external GPU mask');
      const value = filter.value;
      if (
        !Array.isArray(value) ||
        value.length !== 2 ||
        !value.every(Number.isFinite) ||
        value[0] > value[1]
      ) {
        throw new Error('Resident ranges require two ordered finite endpoints');
      }
      binding.validateRange?.(value[0], value[1]);
      command = {kind: 'range', dimension: binding.dimension, value: [value[0], value[1]]};
    } else if (filter.type === 'multiSelect' || filter.type === 'select') {
      if (binding.kind !== 'category')
        throw new Error('Categorical filters require a resident dictionary binding');
      const categories = binding.categories ?? binding.domain;
      const values = filter.type === 'select' ? [filter.value] : filter.value;
      if (!Array.isArray(values))
        throw new Error('Resident categories require an array of dictionary labels');
      const selected = categories.flatMap((label, index) => {
        const included = values.some(value => Object.is(value, label));
        return included !== (!options.independentPredicates && Boolean(filter.inverted))
          ? [index]
          : [];
      });
      command = {kind: 'category', dimension: binding.dimension, value: selected};
    } else {
      throw new Error(`Filter type ${filter.type} requires another execution backend`);
    }
    planned.push({field, command, fixedDomain: Boolean(filter.fixedDomain), id: filter.id});
  }
  return {
    display: options.independentPredicates
      ? planned.map(item => item.command)
      : combineCommands(planned.map(item => item.command)),
    domain: options.independentPredicates
      ? planned.filter(item => !item.fixedDomain).map(item => item.command)
      : combineCommands(planned.filter(item => !item.fixedDomain).map(item => item.command)),
    filterIds: planned.map(item => item.id)
  };
}

function combineCommands(
  commands: readonly KeplerComputeFilterCommand[]
): KeplerComputeFilterCommand[] {
  const combined = new Map<string, KeplerComputeFilterCommand>();
  for (const command of commands) {
    const previous = combined.get(command.dimension);
    if (!previous) {
      combined.set(command.dimension, command);
    } else if (previous.kind === 'range' && command.kind === 'range') {
      const minimum = Math.max(previous.value[0], command.value[0]);
      const maximum = Math.min(previous.value[1], command.value[1]);
      combined.set(
        command.dimension,
        minimum <= maximum
          ? {...command, value: [minimum, maximum]}
          : {kind: 'empty', dimension: command.dimension}
      );
    } else if (previous.kind === 'category' && command.kind === 'category') {
      combined.set(command.dimension, {
        ...command,
        value: previous.value.filter(value => command.value.includes(value))
      });
    } else if (previous.kind === 'empty' || command.kind === 'empty') {
      combined.set(command.dimension, {kind: 'empty', dimension: command.dimension});
    } else {
      throw new Error('One resident dimension cannot mix numeric and categorical filters');
    }
  }
  return [...combined.values()];
}

/**
 * Installs explicit resident operations on a real Kepler table. Kepler's copyTable preserves these
 * enumerable methods and state. Source rows remain available for individual original-value tooltips;
 * this adapter never treats synchronous row access as access to GPU-only computed columns.
 */
export function installKeplerComputeTableBridge<Table extends KeplerComputeTableLike>(
  table: Table,
  binding: KeplerComputeTableBinding
): Table & {lumaCompute: KeplerComputeTableState} {
  if (Boolean(binding.prepareSpatialFilter) !== Boolean(binding.clearSpatialFilters)) {
    throw new Error('Spatial filter preparation and clearing must be provided together');
  }
  const histograms = new Map<
    string,
    {revision: number; bins: readonly KeplerComputeBin[]; interval?: string}
  >();
  const filterHistograms = new Map<
    string,
    {revision: number; bins: readonly KeplerComputeBin[]; interval?: string}
  >();
  const plotFailures = new Map<
    string,
    {reason: string; kind: 'configuration' | 'duplicate-field'}
  >();
  const activeFieldFilters = new Map<string, Set<string>>();
  let selectionStatus: 'ready' | 'pending' | 'unsupported' = 'ready';
  let selectionReason: string | null = null;
  const state: KeplerComputeTableState = {
    status: 'ready',
    reason: null,
    revision: 0,
    plan: {display: [], domain: [], filterIds: []},
    invalidateScope() {
      histograms.clear();
      filterHistograms.clear();
      if (selectionStatus !== 'unsupported') selectionStatus = 'pending';
      refreshStatus();
    },
    completeScope(revision) {
      if (revision !== state.revision || selectionStatus === 'unsupported') return false;
      selectionStatus = 'ready';
      refreshStatus();
      return true;
    },
    setHistogram(field, bins, revision) {
      if (revision !== state.revision || selectionStatus === 'unsupported') return false;
      histograms.set(field, {revision, bins, interval: binding.fields[field]?.histogramInterval});
      selectionStatus = 'ready';
      refreshStatus();
      return true;
    },
    setFilterHistogram(filterId, bins, revision, interval) {
      if (
        !binding.filterScopedHistograms ||
        revision !== state.revision ||
        selectionStatus === 'unsupported'
      )
        return false;
      filterHistograms.set(filterId, {revision, bins, interval});
      selectionStatus = 'ready';
      refreshStatus();
      return true;
    },
    updateFilterPlot<Filter extends KeplerComputeFilter>(
      filter: Filter,
      options?: KeplerComputeFilterPlotOptions
    ): Filter {
      const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
      const datasetIndex = datasetIds.indexOf(table.id);
      if (datasetIndex < 0) return filter;
      const linked = datasetIds.length > 1;
      if (options?.stage === 'finalize')
        return attachLineChart(
          clearPendingLinkedPlot(filter),
          datasetIds[0] === table.id,
          options.buildLineChart
        );
      // Callers outside the pinned seam that omit the stage still receive an honest rejection.
      if (linked && !options?.stage)
        return unsupportedPlot(
          filter,
          'Linked dataset plots are composed by the filter-plot seam, one resident entry at a time'
        );
      if (filter.enabled === false && !binding.filterScopedHistograms) {
        clearPlotFailure(filter.id);
        refreshStatus();
        return {
          ...filter,
          bins: undefined,
          timeBins: undefined,
          lineChart: undefined,
          lumaComputePlot: {status: 'disabled', revision: state.revision}
        };
      }
      const lineChart = filter.plotType?.type === 'lineChart';
      if (filter.plotType?.type && filter.plotType.type !== 'histogram' && !lineChart) {
        return unsupportedPlot(
          filter,
          'Resident filter plots support count histograms and count line charts'
        );
      }
      if (lineChart && !options?.buildLineChart) {
        return unsupportedPlot(
          filter,
          'Resident line charts need the native series builder supplied by the filter-plot seam'
        );
      }
      // Native draws the line chart from the first dataset only. Its y axis is a field of that
      // dataset and aggregates each bin's source rows, so the GPU reduces it over the same bins.
      const isSeriesDataset = datasetIds[0] === table.id;
      if (lineChart && isSeriesDataset) {
        if (filter.plotType?.groupBy && !binding.groupedLineCharts) {
          return unsupportedPlot(filter, 'Resident line charts do not support grouped series');
        }
        if (filter.yAxis?.name && filter.type !== 'timeRange' && !binding.prepareNumericPlot) {
          return unsupportedPlot(
            filter,
            'Resident y-axis line charts need a binding that reduces measures over native numeric bins; numeric bins plot counts'
          );
        }
      }
      const numericInterval = filter.plotType?.numericInterval;
      if (
        filter.type === 'range' &&
        Number.isFinite(numericInterval) &&
        numericInterval! > 0 &&
        !binding.prepareNumericPlot
      ) {
        return unsupportedPlot(
          filter,
          'Resident numeric plots use equal-width GPU bins; a native numeric interval needs explicit GPU edges'
        );
      }
      const field =
        filter.name?.[datasetIndex] ?? table.fields[filter.fieldIdx?.[datasetIndex] ?? -1]?.name;
      if (
        !binding.filterScopedHistograms &&
        (filter.type === 'range' || filter.type === 'timeRange') &&
        field &&
        (activeFieldFilters.get(field)?.size ?? 0) > 1
      ) {
        return unsupportedPlot(
          filter,
          `Multiple active filters on ${field} require per-filter GPU histograms`,
          'duplicate-field'
        );
      }
      if (filter.type === 'range' && field && binding.prepareNumericPlot) {
        try {
          binding.prepareNumericPlot(field, filter);
        } catch (error) {
          return unsupportedPlot(filter, error instanceof Error ? error.message : String(error));
        }
      }
      let histogramInterval = field ? binding.fields[field]?.histogramInterval : undefined;
      if (filter.type === 'timeRange' && field && binding.getTimeHistogramInterval) {
        try {
          histogramInterval = binding.getTimeHistogramInterval(field, filter);
        } catch (error) {
          return unsupportedPlot(filter, error instanceof Error ? error.message : String(error));
        }
      }
      const requestedInterval = filter.plotType?.interval ?? histogramInterval;
      if (
        filter.type === 'timeRange' &&
        (!histogramInterval || requestedInterval !== histogramInterval)
      ) {
        return unsupportedPlot(
          filter,
          histogramInterval
            ? `Resident time histogram supports ${histogramInterval}; ${filter.plotType?.interval ?? 'unspecified'} requires matching GPU bins`
            : 'Resident time histograms require explicit compiled interval metadata'
        );
      }
      const recovered = clearPlotFailure(filter.id);
      refreshStatus();
      if (recovered && state.status !== 'unsupported') binding.onPlotRecovery?.();
      const cached = binding.filterScopedHistograms
        ? filterHistograms.get(filter.id)
        : field
          ? histograms.get(field)
          : undefined;
      const bins =
        selectionStatus !== 'unsupported' &&
        cached?.revision === state.revision &&
        (filter.type !== 'timeRange' || cached.interval === histogramInterval)
          ? cached.bins
          : undefined;
      // Linked plots keep every other dataset's entry; native bins are per dataset, never summed.
      const entry = linked ? (bins ?? PENDING_PLOT_BINS) : bins;
      const status: ResidentPlotStatus = {
        status: selectionStatus === 'unsupported' ? 'unsupported' : bins ? 'ready' : 'pending',
        reason: selectionReason,
        revision: state.revision
      };
      const previous = (filter as {lumaComputePlot?: ResidentPlotStatus}).lumaComputePlot;
      const previousBins = filter.bins as Record<string, unknown> | undefined;
      const previousTimeBins = filter.timeBins as Record<string, unknown> | undefined;
      const next = {
        ...filter,
        ...(filter.type === 'timeRange'
          ? {
              // Linked plots keep the native plot type: the seam already resolved the shared interval.
              plotType: linked
                ? {...filter.plotType, interval: filter.plotType?.interval ?? histogramInterval}
                : {...filter.plotType, interval: histogramInterval}
            }
          : {}),
        bins:
          filter.type === 'range'
            ? {...(linked ? previousBins : {}), [table.id]: entry}
            : filter.bins,
        timeBins:
          filter.type === 'timeRange' && histogramInterval
            ? {
                ...(linked ? previousTimeBins : {}),
                [table.id]: {[histogramInterval]: entry}
              }
            : filter.timeBins,
        lineChart: undefined,
        lumaComputePlot: linked ? mergePlotStatus(previous, status) : status
      };
      // Linked filters attach their chart when every entry has been composed (finalize stage).
      return linked ? next : attachLineChart(next, true, options?.buildLineChart);
    }
  };
  /**
   * Native draws a line chart for the first dataset only, from that dataset's bins. The native
   * series builder runs unchanged: a y axis replaces each bin's count by its GPU-reduced measure
   * (null where no valid value exists, which native draws as a gap), and a stub dataset without
   * the y field makes the builder read that value instead of aggregating source rows.
   */
  function attachLineChart<Filter extends KeplerComputeFilter>(
    filter: Filter,
    isSeriesDataset: boolean,
    buildLineChart: KeplerComputeFilterPlotOptions['buildLineChart']
  ): Filter {
    if (!isSeriesDataset || filter.plotType?.type !== 'lineChart' || !buildLineChart) return filter;
    const status = (filter as {lumaComputePlot?: ResidentPlotStatus}).lumaComputePlot?.status;
    if (status === 'unsupported' || status === 'disabled') return {...filter, lineChart: undefined};
    const interval = filter.plotType.interval;
    const bins =
      filter.type === 'range'
        ? (filter.bins as Record<string, readonly KeplerComputeBin[] | undefined> | undefined)?.[
            table.id
          ]
        : (
            filter.timeBins as
              | Record<string, Record<string, readonly KeplerComputeBin[] | undefined>>
              | undefined
          )?.[table.id]?.[interval ?? ''];
    // While the GPU reduction is pending native still needs a chart object, otherwise its plot
    // container flags the plot as loading and never clears it. Empty series draw nothing.
    const available = bins ?? [];
    const sourceFilter =
      filter.type === 'range'
        ? {...filter, bins: {...(filter.bins as object), [table.id]: available}}
        : {
            ...filter,
            timeBins: {...(filter.timeBins as object), [table.id]: {[interval ?? '']: available}}
          };
    const groupField = filter.plotType.groupBy?.fieldName;
    if (groupField && available.some(bin => bin.groups)) {
      return {
        ...filter,
        lineChart: buildGroupedLineChart(
          filter,
          sourceFilter,
          available,
          groupField,
          buildLineChart
        )
      };
    }
    if (!filter.yAxis?.name) {
      return {
        ...filter,
        lineChart: buildLineChart({[table.id]: table}, sourceFilter)
      };
    }
    const measured = available.map(bin => ({
      ...bin,
      count: bin.measure ?? (undefined as unknown as number)
    }));
    const measuredFilter = withPlotBins(sourceFilter, measured);
    return {
      ...filter,
      lineChart: buildLineChart({[table.id]: {id: table.id, fields: []}}, measuredFilter)
    };
  }
  function withPlotBins<Filter extends KeplerComputeFilter>(
    filter: Filter,
    bins: readonly unknown[]
  ): Filter {
    const interval = filter.plotType?.interval ?? '';
    return filter.type === 'range'
      ? {...filter, bins: {...(filter.bins as object), [table.id]: bins}}
      : {
          ...filter,
          timeBins: {...(filter.timeBins as object), [table.id]: {[interval]: bins}}
        };
  }
  /**
   * Native groups series by `String(value)` of a field, discovering labels while walking the bins
   * and each bin's row array. The native builder runs unchanged over one pseudo row per series and
   * bin, each carrying its GPU-reduced value and read back through averages of a single value. The
   * first bin also lists every series (null-valued) in the GPU's first-seen order, which fixes the
   * discovery order without drawing a point. Title and aggregation come from a reference
   * ungrouped build, because the pseudo rows are always averaged.
   */
  function buildGroupedLineChart<Filter extends KeplerComputeFilter>(
    filter: Filter,
    sourceFilter: Filter,
    bins: readonly KeplerComputeBin[],
    groupField: string,
    buildLineChart: NonNullable<KeplerComputeFilterPlotOptions['buildLineChart']>
  ): unknown {
    const labels = bins[0]?.groupLabels ?? [];
    const slotLabels = bins[0]?.groupOthers ? [...labels, GROUP_OTHERS_LABEL] : [...labels];
    const pseudoLabels: string[] = [];
    const pseudoValues: (number | null)[] = [];
    const yName = filter.yAxis?.name ?? GROUP_COUNT_FIELD;
    const pseudoBins = bins.map((bin, binIndex) => {
      const indexes: number[] = [];
      const present = new Map(bin.groups?.map(group => [group.slot, group]));
      for (const [slot, label] of slotLabels.entries()) {
        const group = present.get(slot);
        if (!group && binIndex > 0) continue;
        indexes.push(pseudoLabels.length);
        pseudoLabels.push(label);
        pseudoValues.push(group ? group.measure : null);
      }
      return {...bin, indexes, count: indexes.length} as unknown as KeplerComputeBin;
    });
    const dataset = {
      id: table.id,
      fields: [
        {
          name: groupField,
          type: 'string',
          valueAccessor: ({index}: {index: number}) => pseudoLabels[index]
        },
        {
          name: yName,
          type: 'real',
          valueAccessor: ({index}: {index: number}) => pseudoValues[index]
        }
      ]
    };
    const grouped = buildLineChart(
      {[table.id]: dataset},
      withPlotBins(
        {
          ...sourceFilter,
          yAxis: filter.yAxis?.name ? filter.yAxis : {name: GROUP_COUNT_FIELD, type: 'real'},
          plotType: {...filter.plotType, aggregation: 'average'}
        } as Filter,
        pseudoBins
      )
    ) as Record<string, unknown>;
    const reference = buildLineChart(
      {[table.id]: {id: table.id, fields: []}},
      withPlotBins({...sourceFilter, plotType: {...filter.plotType, groupBy: undefined}}, [])
    ) as Record<string, unknown>;
    return {
      ...grouped,
      aggregation: reference['aggregation'],
      title: reference['title'],
      fieldType: reference['fieldType'],
      yAxis: reference['yAxis'],
      bins
    };
  }
  /** Replace this table's pending placeholder with the single-dataset pending representation. */
  function clearPendingLinkedPlot<Filter extends KeplerComputeFilter>(filter: Filter): Filter {
    const bins = filter.bins as Record<string, unknown> | undefined;
    const timeBins = filter.timeBins as Record<string, Record<string, unknown>> | undefined;
    const pendingRange = bins?.[table.id] === PENDING_PLOT_BINS;
    const pendingTime = Object.values(timeBins?.[table.id] ?? {}).includes(PENDING_PLOT_BINS);
    if (!pendingRange && !pendingTime) return filter;
    return {
      ...filter,
      ...(pendingRange ? {bins: {...bins, [table.id]: undefined}} : {}),
      ...(pendingTime
        ? {
            timeBins: {
              ...timeBins,
              [table.id]: Object.fromEntries(
                Object.entries(timeBins![table.id]).map(([interval, value]) => [
                  interval,
                  value === PENDING_PLOT_BINS ? undefined : value
                ])
              )
            }
          }
        : {})
    };
  }
  function refreshStatus(): void {
    state.reason = selectionReason ?? plotFailures.values().next().value?.reason ?? null;
    state.status =
      selectionStatus === 'unsupported' || plotFailures.size ? 'unsupported' : selectionStatus;
  }
  function unsupported(reason: string): void {
    selectionStatus = 'unsupported';
    selectionReason = reason;
    refreshStatus();
    binding.onUnsupported(reason);
  }
  function clearPlotFailure(filterId: string): boolean {
    const cleared = plotFailures.delete(filterId);
    if (cleared) binding.onPlotUnsupported?.(filterId, null);
    return cleared;
  }
  function unsupportedPlot<Filter extends KeplerComputeFilter>(
    filter: Filter,
    reason: string,
    kind: 'configuration' | 'duplicate-field' = 'configuration'
  ): Filter {
    if (filter.enabled === false) {
      clearPlotFailure(filter.id);
      refreshStatus();
      return {
        ...filter,
        bins: undefined,
        timeBins: undefined,
        lineChart: undefined,
        lumaComputePlot: {status: 'disabled', revision: state.revision}
      };
    }
    const previous = plotFailures.get(filter.id);
    plotFailures.set(filter.id, {reason, kind});
    refreshStatus();
    if (previous?.reason !== reason || previous.kind !== kind)
      binding.onPlotUnsupported?.(filter.id, reason);
    return {
      ...filter,
      bins: undefined,
      timeBins: undefined,
      lineChart: undefined,
      lumaComputePlot: {status: 'unsupported', reason, revision: state.revision}
    };
  }
  function filterTable(
    this: ResidentTable,
    filters: readonly KeplerComputeFilter[],
    layers: readonly unknown[] = []
  ): ResidentTable {
    let plan: KeplerComputeFilterPlan;
    let spatial: {filter: KeplerComputeFilter; layerIds: readonly string[]; apply(): void}[] = [];
    try {
      const relevant = filters.filter(filter => {
        // Native polygon filtering also bypasses disabled filters before geometry validation.
        if (filter.enabled === false) return false;
        const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
        return datasetIds.includes(this.id);
      });
      spatial = relevant
        .filter(filter => filter.type === 'polygon')
        .map(filter => {
          if (!binding.prepareSpatialFilter)
            throw new Error('Polygon filters require an explicit layer-specific GPU mask binding');
          return {filter, ...binding.prepareSpatialFilter(filter, layers)};
        });
      plan = planKeplerComputeFilters(
        this.id,
        this.fields,
        binding.fields,
        relevant.filter(filter => filter.type !== 'polygon'),
        {independentPredicates: Boolean(binding.applyNativeFilters)}
      );
      if (spatial.length) {
        plan = {
          ...plan,
          filterIds: [...plan.filterIds, ...spatial.map(item => item.filter.id)],
          spatial: spatial.map(item => ({filterId: item.filter.id, layerIds: item.layerIds}))
        };
      }
      if (plan.domain.length && !binding.applyDomainSelection && !binding.applyNativeFilters) {
        throw new Error(
          'Dynamic layer domains require a separate GPU domain selection; use fixed-domain filters'
        );
      }
      if (
        plan.display.some(command => command.kind === 'empty') &&
        !binding.explorer.setEmpty &&
        !binding.applyNativeFilters
      ) {
        throw new Error('Disjoint ranges require an explicit empty resident selection');
      }
    } catch (error) {
      unsupported(error instanceof Error ? error.message : String(error));
      return this;
    }
    state.revision++;
    histograms.clear();
    filterHistograms.clear();
    state.plan = plan;
    selectionStatus = 'pending';
    selectionReason = null;
    activeFieldFilters.clear();
    for (const filter of filters) {
      if (filter.type === 'polygon' || !plan.filterIds.includes(filter.id)) continue;
      const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
      const datasetIndex = datasetIds.indexOf(table.id);
      const field =
        filter.name?.[datasetIndex] ?? table.fields[filter.fieldIdx?.[datasetIndex] ?? -1]?.name;
      if (!field) continue;
      const active = activeFieldFilters.get(field) ?? new Set<string>();
      active.add(filter.id);
      activeFieldFilters.set(field, active);
    }
    for (const [id, failure] of plotFailures) {
      const filter = filters.find(
        candidate =>
          candidate.id === id && candidate.enabled !== false && plan.filterIds.includes(id)
      );
      if (
        !filter ||
        (failure.kind === 'duplicate-field' &&
          ![...activeFieldFilters.values()].some(active => active.has(id) && active.size > 1))
      ) {
        clearPlotFailure(id);
      }
    }
    refreshStatus();
    // CPU index lists intentionally cannot masquerade as the result of a resident selection.
    this.filteredIndex = [];
    this.filteredIndexForDomain = [];
    this.filteredIndexByLayer = {};
    this.filteredIdxCPU = undefined;
    try {
      if (binding.applyNativeFilters) {
        binding.applyNativeFilters({
          filters,
          plan,
          revision: state.revision,
          fields: this.fields,
          layers
        });
      } else {
        for (const dimension of new Set(
          Object.values(binding.fields).map(field => field.dimension)
        )) {
          binding.explorer.clear(dimension);
        }
        for (const command of plan.display) {
          if (command.kind === 'range') binding.explorer.setRange(command.dimension, command.value);
          else if (command.kind === 'category')
            binding.explorer.setCategories(command.dimension, command.value);
          else binding.explorer.setEmpty?.(command.dimension);
        }
        binding.clearSpatialFilters?.();
        for (const prepared of spatial) prepared.apply();
        binding.applyDomainSelection?.(plan.domain, state.revision);
      }
      binding.onSelectionChange(plan, state.revision);
    } catch (error) {
      // Allocation/encoding setup can still fail after pure planning. Never present partial state.
      unsupported(error instanceof Error ? error.message : String(error));
    }
    return this;
  }
  function getColumnFilterDomain(field: {
    name: string;
  }): {domain: readonly unknown[]; step?: number} | null {
    const fieldBinding = binding.fields[field.name];
    return fieldBinding ? {domain: [...fieldBinding.domain], step: fieldBinding.step} : null;
  }
  function getColumnFilterProps(name: string): Record<string, unknown> | null {
    const fieldBinding = binding.fields[name];
    if (!fieldBinding) return null;
    const field = table.fields.find(candidate => candidate.name === name);
    const type =
      fieldBinding.kind === 'category'
        ? 'multiSelect'
        : fieldBinding.kind === 'time'
          ? 'timeRange'
          : 'range';
    const histogramInterval =
      type === 'timeRange'
        ? (binding.getTimeHistogramInterval?.(name) ?? fieldBinding.histogramInterval)
        : undefined;
    return {
      ...getColumnFilterDomain({name}),
      type,
      typeOptions: [type],
      fieldType: field?.type,
      lumaComputeDomain: fieldBinding.kind === 'category' ? 'category' : undefined,
      value: type === 'multiSelect' ? [] : [...fieldBinding.domain],
      fixedDomain: !binding.applyDomainSelection && !binding.applyNativeFilters,
      // Preserve native channel allocation/history for filteredIndex topology semantics. The
      // overridden filterTable never builds native GPU attributes; resident layers use masks.
      gpu: type === 'range' || type === 'timeRange',
      view: 'side',
      plotType: {
        type: 'histogram',
        ...(histogramInterval ? {interval: histogramInterval} : {})
      }
    };
  }
  function getColumnLayerDomain(field: {name: string}, scale: string): readonly unknown[] | null {
    const fieldBinding = binding.fields[field.name];
    if (!fieldBinding) return null;
    if (['ordinal', 'customOrdinal', 'point'].includes(scale)) return [...fieldBinding.domain];
    const extentScale = ['linear', 'quantize', 'sqrt', 'custom'].includes(scale);
    // NativeStyleRuntime owns these computed domains and their legends. [] stays truthy in the
    // native layer updater, so it cannot substitute a misleading default sample distribution.
    // Unsupported mappings are reported by that renderer on the affected layer, not as filters.
    if (scale === 'jenks') {
      const derived = binding.getDerivedJenksDomain?.(field.name);
      if (derived) return [...derived];
    }
    if (binding.residentStyleDomains && !extentScale) return [];
    if (state.plan.domain.length) {
      return binding.getDynamicLayerDomain?.(field.name, scale, state.revision) ?? null;
    }
    if (fieldBinding.scaleDomains?.[scale]) return [...fieldBinding.scaleDomains[scale]];
    if (extentScale && fieldBinding.kind !== 'category') {
      return [...fieldBinding.domain];
    }
    unsupported(`Scale ${scale} requires a prepared GPU domain for ${field.name}`);
    return null;
  }
  Object.assign(table, {
    lumaCompute: state,
    filterTable,
    getColumnFilterDomain,
    getColumnFilterProps,
    getColumnLayerDomain,
    filterTableCPU() {
      throw new Error('Resident table export requires explicit asynchronous materialization');
    }
  });
  // A refused synchronous mutation is recorded for diagnostics but does not poison valid selections.
  const blockedOperations: string[] = [];
  Object.assign(table, {lumaBlockedOperations: blockedOperations});
  installResidentSynchronousGuards(table, report => {
    if (!blockedOperations.includes(report.reason)) blockedOperations.push(report.reason);
  });
  return table as Table & {lumaCompute: KeplerComputeTableState};
}

// Kepler's configurable constructor is untyped upstream; this localized constructor adapter retains
// the supplied class's instance and static API without weakening the GPU execution interfaces.
type TableConstructor = new (...arguments_: any[]) => KeplerComputeTableLike;

export function createKeplerComputeTableClass<Constructor extends TableConstructor>(
  BaseTable: Constructor,
  resolveBinding: (datasetId: string) => KeplerComputeTableBinding | undefined
): Constructor {
  return class KeplerComputeTable extends BaseTable {
    constructor(...arguments_: any[]) {
      super(...arguments_);
      const binding = resolveBinding(this.id);
      if (binding) installKeplerComputeTableBridge(this, binding);
    }
  };
}

/** Called by the pinned Kepler filter-utils hook before any CPU plot indices are calculated. */
export function updateKeplerComputeFilterPlot<Filter extends KeplerComputeFilter>(
  datasets: Record<string, KeplerComputeTableLike & {lumaCompute?: KeplerComputeTableState}>,
  filter: Filter,
  updateLinkedPlot?: (filter: Filter) => Filter,
  options?: KeplerComputeFilterPlotOptions
): Filter | undefined {
  const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
  const resident = datasetIds.map(id => datasets[id]).find(dataset => dataset?.lumaCompute);
  // Linked filters need native code for non-resident datasets, so the caller must supply the
  // seam-patched native update; this function never recomputes them from CPU rows.
  if (datasetIds.length > 1) return resident ? updateLinkedPlot?.(filter) : undefined;
  return resident?.lumaCompute?.updateFilterPlot(filter, options);
}

/** Access the explicit table metadata without leaking renderer casts into the host. */
export function getKeplerComputeTableState(table: unknown): KeplerComputeTableState | undefined {
  if (!table || typeof table !== 'object' || !('lumaCompute' in table)) return undefined;
  const state = table.lumaCompute;
  if (
    !state ||
    typeof state !== 'object' ||
    !('setHistogram' in state) ||
    !('updateFilterPlot' in state)
  )
    return undefined;
  return typeof state.setHistogram === 'function' && typeof state.updateFilterPlot === 'function'
    ? (state as KeplerComputeTableState)
    : undefined;
}
