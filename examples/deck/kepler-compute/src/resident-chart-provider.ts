// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {trackDeferredOperation} from './tracked-operations';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerComputeSession, KeplerSessionExplorer} from '@luma-contributors/kepler';
import {
  getResidentChartPlanKey,
  needsResidentChartMeasureWidening,
  planResidentChart,
  type NativeChartConfiguration,
  type ResidentChartPlan,
  type ResidentChartField
} from './resident-chart-plan';
import {
  widenResidentChartMeasure,
  type ResidentChartWidenedMeasure
} from './resident-chart-measure-widening';
import {
  planResidentWideScalarMeasure,
  ResidentWideScalarReader
} from './resident-chart-wide-scalar';
import {makeResidentGridSyntheticChart} from './resident-chart-grid';
import {ResidentScalarChartRuntime, type ResidentScalarChartResult} from './resident-chart-runtime';
import {
  ResidentCategoryChartRuntime,
  type ResidentCategoryChartResult,
  type ResidentChartSelection
} from './resident-chart-group-runtime';
import {
  ResidentNumericChartRuntime,
  type ResidentNumericChartResult
} from './resident-chart-numeric-runtime';

import {ResidentTopGroupsChartRuntime} from './resident-chart-top-runtime';
import {ResidentNumericSeriesChartRuntime} from './resident-chart-numeric-series';
import {
  ResidentTimeChartRuntime,
  type ResidentTimeChartResult
} from './resident-chart-time-runtime';

type ChartEntry = {
  revision: number;
  key?: string;
  reader?:
    | ResidentScalarChartRuntime
    | ResidentCategoryChartRuntime
    | ResidentTopGroupsChartRuntime
    | ResidentNumericSeriesChartRuntime
    | ResidentNumericChartRuntime
    | ResidentTimeChartRuntime;
  tail: Promise<void>;
};
export type ResidentChartReadResult =
  | {
      status: 'ready';
      /** Stated fallbacks, e.g. a requested group count reduced to the 1,000-group cap. */
      notes?: readonly string[];
      result:
        | ResidentScalarChartResult
        | ResidentCategoryChartResult
        | ResidentNumericChartResult
        | ResidentTimeChartResult;
    }
  | {status: 'unsupported'; reason: string};

/** One provider per explorer. The owner must await destroy before retiring that explorer. */
export class ResidentChartProvider {
  readonly datasetId: string;
  readonly fields: readonly ResidentChartField[];
  private readonly entries = new Map<string, ChartEntry>();
  /** Exact Float32 copies of integer measures, shared by charts and freed with the provider. */
  private readonly widenedMeasures = new Map<string, Promise<ResidentChartWidenedMeasure>>();
  /** Exact Int64 and Uint64 scalar measures, shared by charts over one source revision. */
  private readonly wideScalar = new ResidentWideScalarReader();
  private readonly retirements = new Set<Promise<void>>();
  private readonly chartRetirements = new Map<string, Promise<void>>();
  private destroyed = false;
  private destruction?: Promise<void>;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      subscribe(listener: () => void): () => void;
      workingMemoryBudgetBytes?: number;
      selection?: ResidentChartSelection;
    }
  ) {
    const dataset = props.explorer.dataset;
    this.datasetId = dataset.id;
    this.fields = [
      ...Object.entries(dataset.value.columns).map(([name, vector]) => ({
        name,
        format: vector.format!,
        categorical: Boolean(dataset.value.prepared.dictionaries[name]),
        categoryCount: dataset.value.prepared.dictionaries[name]?.values.length
      })),
      ...Object.entries(dataset.value.timeColumns).map(([name, column]) => ({
        kind: 'time' as const,
        name,
        unit: column.metadata.unit,
        sourceType: column.metadata.sourceType
      }))
    ];
  }

  getRevision(): string {
    const current = !this.destroyed && this.props.explorer.isCurrent;
    return `${this.props.explorer.dataset.revision}:${this.props.explorer.currentRevision}:${this.props.selection?.currentRevision ?? 0}:${current ? 'current' : 'retired'}`;
  }

  subscribe(listener: () => void): () => void {
    return this.props.subscribe(listener);
  }

  read(nativeChart: NativeChartConfiguration): Promise<ResidentChartReadResult | null> {
    if (this.destroyed || !this.props.explorer.isCurrent) return Promise.resolve(null);
    // Heatmap and pivot charts are a category summary with the Y axis as series (see resident-chart-grid).
    const chart = makeResidentGridSyntheticChart(nativeChart) ?? nativeChart;
    const planned = planResidentChart(chart, this.datasetId, this.fields);
    if (!planned.supported) {
      void this.remove(chart.id);
      return Promise.resolve({status: 'unsupported', reason: planned.reason});
    }
    const dataset = this.props.explorer.dataset;
    const wideMeasure = dataset.value.resident
      ? null
      : planResidentWideScalarMeasure(planned.plan, dataset.value.source);
    if (wideMeasure) {
      // Int64 and Uint64 measures are exact on the wide-word kernels; resident Float32 would round.
      void this.remove(chart.id);
      return this.wideScalar
        .read({
          device: this.props.session.device,
          source: dataset.value.source,
          measure: wideMeasure,
          sourceRevision: dataset.revision
        })
        .then(({result, note}) =>
          this.destroyed || !this.props.explorer.isCurrent
            ? null
            : {status: 'ready' as const, result, notes: [note]}
        );
    }
    let entry = this.entries.get(chart.id);
    if (!entry) {
      // A remounted chart must not borrow its previous reader's named selection mask.
      // That reader removes the mask only after its pending read and GPU fence complete.
      entry = {revision: 0, tail: this.chartRetirements.get(chart.id) ?? Promise.resolve()};
      this.entries.set(chart.id, entry);
    }
    const currentEntry = entry;
    const requestRevision = ++entry.revision;
    entry.reader?.cancel();
    const key = getResidentChartPlanKey(planned.plan);
    const current = () =>
      !this.destroyed &&
      this.props.explorer.isCurrent &&
      this.entries.get(chart.id) === currentEntry &&
      currentEntry.revision === requestRevision;
    // Slow reads show in the memory manager; cancelling invalidates this request and stops the reader.
    const tracked = trackDeferredOperation(`Chart ${chart.id}`, () => {
      currentEntry.revision++;
      currentEntry.reader?.cancel();
    });
    const pending = entry.tail.then(async (): Promise<ResidentChartReadResult | null> => {
      if (!current()) return null;
      if (currentEntry.key !== key) {
        await currentEntry.reader?.destroy();
        currentEntry.reader = undefined;
        currentEntry.key = undefined;
        if (!current()) return null;
        const measure = await this.getWidenedMeasure(planned.plan);
        if (!current()) return null;
        const runtimeProps = {
          session: this.props.session,
          explorer: this.props.explorer,
          workingMemoryBudgetBytes: this.props.workingMemoryBudgetBytes ?? 64 * 1024 * 1024,
          ...(measure ? {measure: measure.vector} : {})
        };
        currentEntry.reader =
          planned.plan.kind === 'time'
            ? new ResidentTimeChartRuntime({
                ...runtimeProps,
                plan: planned.plan,
                selection: this.props.selection
              })
            : planned.plan.kind === 'numeric'
              ? new ResidentNumericChartRuntime({
                  ...runtimeProps,
                  plan: planned.plan,
                  selection: this.props.selection
                })
              : planned.plan.kind === 'category' && planned.plan.numericSeries
                ? new ResidentNumericSeriesChartRuntime({
                    ...runtimeProps,
                    plan: planned.plan,
                    selection: this.props.selection
                  })
                : planned.plan.kind === 'category' && planned.plan.topGroups
                  ? new ResidentTopGroupsChartRuntime({
                      ...runtimeProps,
                      plan: planned.plan,
                      selection: this.props.selection
                    })
                  : planned.plan.kind === 'category'
                    ? new ResidentCategoryChartRuntime({
                        ...runtimeProps,
                        plan: planned.plan,
                        selection: this.props.selection
                      })
                    : new ResidentScalarChartRuntime({...runtimeProps, plan: planned.plan});
        currentEntry.key = key;
      }
      const result = await currentEntry.reader!.read();
      return result && current()
        ? {
            status: 'ready',
            result,
            ...(planned.notes?.length ? {notes: planned.notes} : {})
          }
        : null;
    });
    entry.tail = pending.then(
      () => undefined,
      () => undefined
    );
    entry.tail.then(tracked.finish);
    return pending;
  }

  /** Int32 and Uint32 measures run as Float32 only when every value is exactly representable. */
  private async getWidenedMeasure(
    plan: ResidentChartPlan
  ): Promise<ResidentChartWidenedMeasure | undefined> {
    const {column} = plan;
    const vector = column ? this.props.explorer.dataset.value.columns[column] : undefined;
    if (
      !column ||
      !vector ||
      !needsResidentChartMeasureWidening(plan.kind, plan.operation, vector.format)
    )
      return undefined;
    let widened = this.widenedMeasures.get(column);
    if (!widened) {
      widened = widenResidentChartMeasure({
        session: this.props.session,
        vector: vector as GPUVector<'sint32'> | GPUVector<'uint32'>
      });
      this.widenedMeasures.set(column, widened);
      // A rejected widening is not cached, so a corrected source can retry.
      widened.catch(() => {
        if (this.widenedMeasures.get(column) === widened) this.widenedMeasures.delete(column);
      });
    }
    return widened;
  }

  remove(chartId: string): Promise<void> {
    const entry = this.entries.get(chartId);
    if (!entry) return this.chartRetirements.get(chartId) ?? Promise.resolve();
    this.entries.delete(chartId);
    entry.revision++;
    entry.reader?.cancel();
    const completion = entry.tail.then(() => entry.reader?.destroy()).then(() => undefined);
    this.retirements.add(completion);
    this.chartRetirements.set(chartId, completion);
    const finishRetirement = () => {
      this.retirements.delete(completion);
      if (this.chartRetirements.get(chartId) === completion) this.chartRetirements.delete(chartId);
    };
    // Keep rejection observed even if a React cleanup cannot await this promise.
    void completion.then(finishRetirement, finishRetirement);
    return completion;
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    for (const chartId of this.entries.keys()) void this.remove(chartId);
    this.destruction = Promise.allSettled(this.retirements).then(async results => {
      const widened = await Promise.allSettled(this.widenedMeasures.values());
      this.widenedMeasures.clear();
      for (const result of widened) if (result.status === 'fulfilled') result.value.destroy();
      const failures = results.filter(result => result.status === 'rejected');
      if (failures.length)
        throw new AggregateError(
          failures.map(result => result.reason),
          'Chart cleanup failed'
        );
    });
    return this.destruction;
  }
}
