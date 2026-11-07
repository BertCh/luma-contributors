// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUVector} from '@luma.gl/graph/gpu-data';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerSessionExplorer
} from '@luma-contributors/kepler';
import {
  ResidentCategoryChartRuntime,
  type ResidentCategoryChartResult,
  type ResidentChartSelection
} from './resident-chart-group-runtime';
import type {ResidentCategoryChartPlan} from './resident-chart-plan';
import {
  MAXIMUM_RESIDENT_TOP_DICTIONARY,
  ResidentGroupRemapKeys,
  selectResidentTopGroups
} from './resident-chart-top-groups';

/**
 * Category chart over a dictionary above the 1,000-group cap: top groups plus one Other bucket.
 * Pass 1 counts every dictionary entry on the GPU (dense groups, only the per-entry summary
 * returns); the top `limit` entries by the chart measure are kept. Pass 2 remaps dictionary codes
 * on the GPU to those buckets (everything else shares the Other bucket) and reduces the chart
 * measure exactly over at most limit + 1 groups. The ranking is fixed when the chart is prepared
 * (it follows the selection of the first read), later reads re-aggregate the same buckets.
 */
export class ResidentTopGroupsChartRuntime {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private counting?: ResidentCategoryChartRuntime;
  private grouped?: ResidentCategoryChartRuntime;
  private remap?: ResidentGroupRemapKeys;
  private tail: Promise<void> = Promise.resolve();
  private requestRevision = 0;
  private destruction?: Promise<void>;
  private countingBytes = 0;
  private failure?: unknown;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      plan: ResidentCategoryChartPlan;
      selection?: ResidentChartSelection;
      workingMemoryBudgetBytes: number;
      measure?: GPUVector<'float32'>;
    }
  ) {
    const {plan} = props;
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(plan.datasetId);
    try {
      // Order statistics cannot be ranked over a huge dictionary cheaply: rank those by row count.
      const rankByMeasure = plan.operation !== 'median' && plan.operation !== 'standardDeviation';
      this.counting = new ResidentCategoryChartRuntime({
        ...props,
        plan: {
          ...plan,
          chartId: plan.chartId,
          topGroups: undefined,
          series: undefined,
          operation: rankByMeasure ? plan.operation : 'count',
          ...(rankByMeasure ? {} : {column: undefined})
        },
        maximumGroups: MAXIMUM_RESIDENT_TOP_DICTIONARY
      });
    } catch (error) {
      void this.lease.release();
      throw error;
    }
  }

  get readbackByteLength(): number {
    return this.countingBytes + (this.grouped?.readbackByteLength ?? 0);
  }
  cancel(): void {
    this.requestRevision++;
    this.counting?.cancel();
    this.grouped?.cancel();
  }
  read(): Promise<ResidentCategoryChartResult | null> {
    const requestRevision = ++this.requestRevision;
    const pending = this.tail.then(() => this.readCurrent(requestRevision));
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.cancel();
    this.destruction = (async () => {
      await this.tail;
      try {
        await this.counting?.destroy();
        await this.grouped?.destroy();
      } finally {
        this.remap?.destroy();
        await this.lease.release();
      }
    })();
    return this.destruction;
  }

  private async readCurrent(requestRevision: number): Promise<ResidentCategoryChartResult | null> {
    if (this.failure) throw this.failure;
    if (!this.grouped) {
      if (!(await this.buildGroups(requestRevision))) return null;
    }
    return this.requestRevision === requestRevision ? this.grouped!.read() : null;
  }

  /** Counts, ranks, then replaces the counting reader by the bucketed reader; false when superseded. */
  private async buildGroups(requestRevision: number): Promise<boolean> {
    const counting = this.counting!;
    const counted = await counting.read();
    if (!counted || this.requestRevision !== requestRevision) return false;
    this.countingBytes = counting.readbackByteLength;
    try {
      const {plan} = this.props;
      const source = this.lease.dataset.value;
      const dictionary = source.prepared.dictionaries[plan.categoryColumn]!.values;
      const selection = selectResidentTopGroups({
        summaries: counted.summaries,
        labels: dictionary,
        limit: plan.topGroups!.limit
      });
      // The counting reader and the bucketed reader share the chart's selection scope name.
      this.counting = undefined;
      await counting.destroy();
      const keys = new ResidentGroupRemapKeys({
        device: this.props.session.device,
        categories: source.columns[plan.categoryColumn],
        validity: source.analytics.validity[plan.categoryColumn],
        bucketOfCode: selection.bucketOfCode
      });
      this.remap = keys;
      const encoder = this.props.session.device.createCommandEncoder();
      keys.encode(encoder);
      this.props.session.device.submit(encoder.finish());
      this.grouped = new ResidentCategoryChartRuntime({
        ...this.props,
        workingMemoryBudgetBytes: Math.max(
          0,
          this.props.workingMemoryBudgetBytes - keys.byteLength
        ),
        plan: {...plan, topGroups: undefined, categoryCount: selection.labels.length},
        groupingInput: {keys: keys.keys, labels: selection.labels}
      });
      return true;
    } catch (error) {
      this.failure = error;
      throw error;
    }
  }
}
