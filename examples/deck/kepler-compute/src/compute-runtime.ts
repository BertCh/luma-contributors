// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Deck, Effect, EffectContext, PreRenderOptions} from '@deck.gl/core';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector, type GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import {
  KeplerComputeSession,
  type KeplerSessionExplorer,
  type KeplerMemoryReservation
} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import type {TripPreparationOptions} from './trip-source-options';
import {AggregationRuntime} from './aggregation-runtime';
import {TimeRuntime, SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS} from './time-runtime';
import {ResidentScopeBinding} from './resident-scope-binding';
import {DuckDBRuntime} from './duckdb-runtime';
import {AnalysisRuntime} from './analysis-runtime';
import {createResidentTableProvider} from './resident-table-provider';
import {NativeStyleRuntime} from './style-host-runtime';
import {ResidentChartProvider} from './resident-chart-provider';
import type {SampleData} from './sample-data';
import {
  PublishedSourceRuntime,
  preparePublishedH3,
  type CoordinateRoles
} from './source-publication';
import type {KeplerArrowDataset, KeplerDatasetSnapshot} from '@luma-contributors/kepler';
import type {KeplerComputeTableState} from './kepler-table-bridge';
import {HeatmapBindingRuntime} from './heatmap-binding-runtime';
import {GPUTripMarkerLayer} from '@deck.gl-community/gpu-layers';
import {ResidentScenegraphPointLayer} from './scenegraph-resident-layer';
import {renderResidentHeatmapPasses} from './resident-heatmap-layer';
import {PointLabelBindingRuntime} from './point-label-binding-runtime';
import {PointAltitudeBindingRuntime} from './point-altitude-binding-runtime';
import {PointPositionVariantRuntime} from './point-position-variants';
import {ResidentUpdateScheduler} from './resident-update-scheduler';
import type {ResidentGroupByProvenance} from './resident-group-by-plan';
import type {ResidentJoinLineage} from './resident-join-lineage';
import type {ResidentSuitabilityLineage} from './resident-suitability-lineage';

export type ExplorerSummary = {
  revision: number;
  selectedCount: number;
  histograms: Record<string, Uint32Array>;
};

type EncodeTiming = {
  startedAt: number;
  finishedAt: number;
  sourceRevision: number;
  explorerRevision: number;
};
type EncodeTelemetry = {
  generation: number;
  count: number;
  totalMilliseconds: number;
  droppedEvents: number;
  events: (EncodeTiming & {sequence: number})[];
};
type SummaryReadTiming = {
  startedAt: number;
  finishedAt: number;
  sourceRevision: number;
  revision: number;
  stage: 'published-linked-summary' | 'fixture-linked-summary';
};

/** Owns source uploads; Deck owns submission and the explorer borrows these columns. */
const DEFAULT_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024;

/** Accounted dataset budget; `?memoryBudgetMebibytes=N` raises it for million-row acceptance runs. */
function getMemoryBudgetBytes(): number {
  const search = typeof location === 'undefined' ? '' : location.search;
  const mebibytes = Number(new URLSearchParams(search).get('memoryBudgetMebibytes'));
  return Number.isInteger(mebibytes) && mebibytes > 0
    ? mebibytes * 1024 * 1024
    : DEFAULT_MEMORY_BUDGET_BYTES;
}

export class ComputeRuntime {
  readonly explorer: KeplerSessionExplorer;
  readonly analysis: AnalysisRuntime;
  readonly nativeTable: ReturnType<typeof createResidentTableProvider>;
  readonly aggregation: AggregationRuntime;
  readonly duckdb: DuckDBRuntime;
  readonly styles: NativeStyleRuntime;
  readonly heatmaps: HeatmapBindingRuntime;
  readonly labels: PointLabelBindingRuntime;
  readonly pointAltitudes: PointAltitudeBindingRuntime;
  readonly pointPositionVariants: PointPositionVariantRuntime;
  readonly tableBinding: ResidentScopeBinding;
  latestSummary: ExplorerSummary | null = null;
  readonly time: TimeRuntime;
  readonly session: KeplerComputeSession;
  readonly positions: GPUVector<'float32x2'>;
  readonly targetPositions: GPUVector<'float32x2'>;
  readonly colors: GPUVector<'unorm8x4'>;
  readonly uploadBytes: number;
  readonly publications = new Map<string, PublishedSourceRuntime>();
  private chartProvider?: ResidentChartProvider;
  getPublicationBridge?: (datasetId: string) => KeplerComputeTableState | undefined;
  private readonly sourceVectors: GPUVector[] = [];
  private readonly renderReservation: KeplerMemoryReservation;
  private dirty = true;
  private updateRevision = 0;
  private readonly updateScheduler = new ResidentUpdateScheduler(
    () => !this.destroyed && !this.device.isLost && !this.session.deviceLost,
    () => {
      for (const listener of this.invalidationListeners) listener();
      if (!this.destroyed && !this.device.isLost && !this.session.deviceLost)
        this.deck.redraw('Kepler compute controls changed');
    }
  );
  private summaryPending = false;
  private destroyed = false;
  private needsSummary = false;
  private encodedFrames = 0;
  private readonly encodedPerformanceRevisions = import.meta.env.DEV
    ? new Map<string, {sourceRevision: number; explorerRevision: number}>()
    : undefined;
  private performanceDiagnosticsEnabled = false;
  private performanceDiagnosticsGeneration = 0;
  private performanceEncodes?: Map<string, EncodeTiming>;
  private performanceEncodeTelemetry?: Map<string, EncodeTelemetry>;
  private performanceSummaryReads?: Map<string, SummaryReadTiming>;
  private listeners = new Set<(summary: ExplorerSummary) => void>();
  private invalidationListeners = new Set<() => void>();

  constructor(
    readonly device: Device,
    readonly deck: Deck,
    readonly source: SampleData,
    callbacks: {
      onSelectionChange(): void;
      onUnsupported(reason: string): void;
      onPlotUnsupported?(filterId: string, reason: string | null): void;
      onStyleUnsupported?(layerId: string, reason: string | null): void;
    }
  ) {
    this.session = new KeplerComputeSession({device, memoryBudgetBytes: getMemoryBudgetBytes()});
    let reservation: KeplerMemoryReservation | undefined;
    let tableBinding: ResidentScopeBinding | undefined;
    let time: TimeRuntime | undefined;
    let duckdb: DuckDBRuntime | undefined;
    let aggregation: AggregationRuntime | undefined;
    let analysis: AnalysisRuntime | undefined;
    let styles: NativeStyleRuntime | undefined;
    let labels: PointLabelBindingRuntime | undefined;
    let pointAltitudes: PointAltitudeBindingRuntime | undefined;
    let pointPositionVariants: PointPositionVariantRuntime | undefined;
    let nativeTable: ReturnType<typeof createResidentTableProvider> | undefined;
    try {
      const dataset = this.session.importArrowDataset({
        id: 'resident-observations',
        table: new arrow.Table({
          longitude: arrow.makeVector(source.longitude),
          latitude: arrow.makeVector(source.latitude),
          value: arrow.makeVector(source.value),
          category: arrow.makeVector(source.category),
          time: arrow.makeVector(source.time),
          timestamp: source.timestamp.column,
          targetLongitude: arrow.makeVector(source.targetLongitude),
          targetLatitude: arrow.makeVector(source.targetLatitude)
        })
      });
      reservation = this.session.registry.reserveMemory(
        source.positions.byteLength + source.targetPositions.byteLength + source.colors.byteLength,
        'gpu',
        'source'
      );
      this.renderReservation = reservation;
      const upload = <Format extends GPUVectorFormat>(
        name: string,
        format: Format,
        values: Float32Array | Uint32Array | Uint8Array,
        length = source.value.length
      ): GPUVector<Format> =>
        new GPUVector({
          type: 'buffer',
          name,
          format,
          length,
          buffer: device.createBuffer({
            id: `kepler-${name}`,
            data: values,
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          ownsBuffer: true
        });
      this.positions = upload('positions', 'float32x2', source.positions);
      this.sourceVectors.push(this.positions);
      this.targetPositions = upload('target-positions', 'float32x2', source.targetPositions);
      this.sourceVectors.push(this.targetPositions);
      this.colors = upload('colors', 'unorm8x4', source.colors);
      this.sourceVectors.push(this.colors);
      this.uploadBytes =
        source.positions.byteLength +
        source.targetPositions.byteLength +
        source.colors.byteLength +
        dataset.value.uploadedBufferBytes;
      time = new TimeRuntime({session: this.session, datasetId: dataset.id, column: 'timestamp'});
      this.time = time;
      tableBinding = new ResidentScopeBinding({
        residentStyleDomains: true,
        session: this.session,
        datasetId: dataset.id,
        layers: [
          {
            id: 'resident-points',
            geometryRevision: dataset.revision,
            geometry: {kind: 'point', positions: this.positions}
          }
        ],
        plots: {
          timestamp: {
            column: 'time',
            domain: [0, 1440],
            nativeDomain: [
              SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS,
              SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS + 86400000
            ],
            binCount: 24,
            interval: '1-hour'
          }
        },
        fields: {
          longitude: {dimension: 'longitude', kind: 'range', domain: [-74.05, -73.92]},
          latitude: {dimension: 'latitude', kind: 'range', domain: [40.69, 40.82]},
          value: {dimension: 'value', kind: 'range', domain: [0, 100], step: 0.1},
          category: {
            dimension: 'category',
            kind: 'category',
            domain: [0, 1, 2, 3],
            categories: [0, 1, 2, 3]
          },
          time: {dimension: 'time', kind: 'range', domain: [0, 1439], step: 1},
          timestamp: {
            dimension: 'timestamp',
            kind: 'time',
            domain: source.timestamp.domainMilliseconds,
            step: 60000,
            histogramInterval: '1-hour',
            validateRange: time.validateRange
          }
        },
        onUpdate: () => {
          callbacks.onSelectionChange();
          this.requestUpdate();
        },
        onUnsupported: callbacks.onUnsupported,
        onPlotUnsupported: callbacks.onPlotUnsupported
      });
      this.tableBinding = tableBinding;
      this.explorer = this.session.openExplorer(dataset.id, {
        workingMemoryBudgetBytes: source.value.length * 160 + 8 * 1024 * 1024,
        dimensions: [
          {id: 'longitude', kind: 'range', column: 'longitude'},
          {id: 'latitude', kind: 'range', column: 'latitude'},
          {id: 'value', kind: 'range', column: 'value'},
          {id: 'time', kind: 'range', column: 'time'},
          {id: 'category', kind: 'category', column: 'category', capacity: 4},
          {id: 'map', kind: 'bounds', x: 'longitude', y: 'latitude'},
          tableBinding.externalDimension,
          time.externalRange.displayDimension
        ],
        histograms: [
          {id: 'value', dimension: 'value', column: 'value', domain: [0, 100], binCount: 32},
          {id: 'timestamp', dimension: 'timestamp', column: 'time', domain: [0, 1440], binCount: 24}
        ]
      });
      aggregation = new AggregationRuntime({
        device,
        longitude: source.longitude,
        latitude: source.latitude,
        selectionMask: this.explorer.selectionMask,
        memoryRegistry: this.session.registry,
        columns: {
          value: {
            values: dataset.value.columns['value'],
            validity: dataset.value.analytics.validity['value']
          }
        },
        requestUpdate: () => this.requestUpdate()
      });
      this.aggregation = aggregation;
      this.uploadBytes += aggregation.uploadBytes;
      duckdb = new DuckDBRuntime({session: this.session, datasetId: this.explorer.dataset.id});
      this.duckdb = duckdb;
      analysis = new AnalysisRuntime({
        session: this.session,
        explorer: this.explorer,
        requestUpdate: () => this.requestUpdate()
      });
      this.analysis = analysis;
      styles = new NativeStyleRuntime({
        session: this.session,
        requestUpdate: () => this.requestUpdate(),
        onUnsupported: callbacks.onStyleUnsupported
      });
      this.styles = styles;
      tableBinding.getDerivedJenksDomain = field =>
        styles?.getDerivedJenksThresholdsForColumn(this.explorer.dataset.id, field)?.values;
      this.heatmaps = new HeatmapBindingRuntime(this.session, () => this.requestUpdate());
      labels = new PointLabelBindingRuntime(this.session, () => this.requestUpdate());
      this.labels = labels;
      pointAltitudes = new PointAltitudeBindingRuntime(this.session, () => this.requestUpdate());
      this.pointAltitudes = pointAltitudes;
      pointPositionVariants = new PointPositionVariantRuntime(this.session, () =>
        this.requestUpdate()
      );
      this.pointPositionVariants = pointPositionVariants;
      nativeTable = createResidentTableProvider(this);
      this.nativeTable = nativeTable;
    } catch (error) {
      for (const vector of this.sourceVectors) vector.destroy();
      reservation?.release();
      void Promise.allSettled([nativeTable?.destroy()])
        .then(() =>
          Promise.allSettled([
            tableBinding?.destroy(Promise.resolve()),
            time?.destroy(Promise.resolve()),
            duckdb?.destroy(),
            aggregation?.destroy(),
            analysis?.destroy(Promise.resolve()),
            styles?.destroy(Promise.resolve()),
            labels?.destroy(Promise.resolve()),
            pointAltitudes?.destroy(Promise.resolve()),
            pointPositionVariants?.destroy(Promise.resolve())
          ])
        )
        .finally(() => this.session.destroy(Promise.resolve()));
      throw error;
    }
  }

  get frameCount(): number {
    return this.encodedFrames;
  }

  getInterpolationLayers() {
    if (this.destroyed || this.session.deviceLost) return [];
    return [
      this.analysis.getInterpolationLayer(),
      ...[...this.publications.values()].map(published => published.getInterpolationLayer())
    ].filter(layer => layer !== null);
  }

  getAnalysisResultLayers() {
    if (this.destroyed || this.session.deviceLost) return [];
    return [
      ...this.analysis.getResultLayers(),
      ...[...this.publications.values()].flatMap(published => published.getAnalysisResultLayers())
    ];
  }

  getCharts(): ResidentChartProvider | undefined {
    if (this.destroyed || !this.explorer.isCurrent) return undefined;
    if (!this.chartProvider)
      this.chartProvider = new ResidentChartProvider({
        session: this.session,
        explorer: this.explorer,
        selection: this.tableBinding,
        subscribe: listener => {
          const unsubscribeSummary = this.subscribe(listener);
          const unsubscribeInvalidation = this.subscribeInvalidation(listener);
          return () => {
            unsubscribeSummary();
            unsubscribeInvalidation();
          };
        }
      });
    return this.chartProvider;
  }

  subscribe(listener: (summary: ExplorerSummary) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  subscribeInvalidation(listener: () => void): () => void {
    this.invalidationListeners.add(listener);
    return () => this.invalidationListeners.delete(listener);
  }

  requestUpdate(): void {
    if (this.destroyed) return;
    this.dirty = true;
    this.updateRevision++;
    this.updateScheduler.request();
  }

  async attachDataset(
    dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
    label: string,
    coordinates: CoordinateRoles | undefined,
    reportPlot: (filterId: string, reason: string | null) => void,
    changed: () => void,
    geometryColumn?: string,
    tripOptions?: TripPreparationOptions,
    h3Column?: string,
    groupBy?: ResidentGroupByProvenance,
    join?: ResidentJoinLineage,
    suitability?: ResidentSuitabilityLineage
  ): Promise<PublishedSourceRuntime> {
    if (this.destroyed || this.session.deviceLost) throw new Error('Resident workspace is closed');
    const previous = this.publications.get(dataset.id);
    if (h3Column && previous) preparePublishedH3(dataset.value.source, h3Column);
    if (previous?.dataset.revision === dataset.revision) return previous;
    this.publications.delete(dataset.id);
    if (previous) {
      const fence = this.device.createFence();
      try {
        await Promise.all([
          this.styles.retireDataset(dataset.id, fence.signaled),
          this.heatmaps.retireDataset(dataset.id, fence.signaled),
          this.labels.retireDataset(dataset.id, fence.signaled),
          this.pointAltitudes.retireDataset(dataset.id, fence.signaled),
          this.pointPositionVariants.retireDataset(dataset.id, fence.signaled)
        ]);
      } finally {
        fence.destroy();
      }
    }
    await previous?.destroy();
    if (
      this.destroyed ||
      this.session.registry.getDataset(dataset.id)?.revision !== dataset.revision
    )
      throw new Error('Dataset changed before publication');
    const published = new PublishedSourceRuntime(
      dataset,
      label,
      coordinates,
      this.session,
      () => this.requestUpdate(),
      reportPlot,
      changed,
      geometryColumn,
      tripOptions,
      h3Column,
      groupBy,
      join,
      suitability
    );
    try {
      await published.prepareFlow();
    } catch (error) {
      await published.destroy();
      throw error;
    }
    this.publications.set(dataset.id, published);
    published.scopes.getDerivedJenksDomain = field =>
      this.styles.getDerivedJenksThresholdsForColumn(dataset.id, field)?.values;
    this.requestUpdate();
    return published;
  }

  async removePublication(datasetId: string): Promise<void> {
    const published = this.publications.get(datasetId);
    if (!published) return;
    this.publications.delete(datasetId);
    this.requestUpdate();
    const styleFence = this.device.createFence();
    try {
      await Promise.all([
        this.styles.retireDataset(datasetId, styleFence.signaled),
        this.heatmaps.retireDataset(datasetId, styleFence.signaled),
        this.labels.retireDataset(datasetId, styleFence.signaled),
        this.pointAltitudes.retireDataset(datasetId, styleFence.signaled),
        this.pointPositionVariants.retireDataset(datasetId, styleFence.signaled)
      ]);
    } finally {
      styleFence.destroy();
    }
    await published.destroy();
    const fence = this.device.createFence();
    try {
      await this.session.removeDataset(datasetId, fence.signaled);
    } finally {
      fence.destroy();
    }
  }

  get isLayerRegistrationPending(): boolean {
    return (
      this.tableBinding.isLayerRegistrationPending ||
      [...this.publications.values()].some(published => published.scopes.isLayerRegistrationPending)
    );
  }

  get isReadyForRender(): boolean {
    return (
      !this.destroyed &&
      !this.device.isLost &&
      !this.session.deviceLost &&
      this.explorer.isCurrent &&
      !this.isLayerRegistrationPending
    );
  }

  getEncodedSourceRevision(
    datasetId: string
  ): {sourceRevision: number; explorerRevision: number} | undefined {
    return this.encodedPerformanceRevisions?.get(datasetId);
  }

  setPerformanceDiagnosticsEnabled(enabled: boolean): void {
    this.performanceDiagnosticsEnabled = import.meta.env.DEV && enabled;
    this.performanceDiagnosticsGeneration++;
    this.performanceEncodes = this.performanceDiagnosticsEnabled ? new Map() : undefined;
    this.performanceEncodeTelemetry = this.performanceDiagnosticsEnabled ? new Map() : undefined;
    this.performanceSummaryReads = this.performanceDiagnosticsEnabled ? new Map() : undefined;
  }

  getPerformanceDiagnostics(datasetId: string) {
    return {
      encode: this.performanceEncodes?.get(datasetId) ?? null,
      encodeTelemetry: this.performanceEncodeTelemetry?.get(datasetId) ?? null,
      summaryRead: this.performanceSummaryReads?.get(datasetId) ?? null
    };
  }

  encode(): void {
    if (!this.isReadyForRender || !this.dirty) return;
    const updateRevision = this.updateRevision;
    const startedAt = this.performanceDiagnosticsEnabled ? performance.now() : undefined;
    this.explorer.encode(this.device.commandEncoder);
    this.analysis.encode(this.device.commandEncoder);
    this.aggregation.encode(this.device.commandEncoder);
    for (const published of this.publications.values())
      published.encode(this.device.commandEncoder);
    this.pointAltitudes.encode(this.device.commandEncoder);
    this.styles.encode(this.device.commandEncoder);
    this.heatmaps.encode(this.device.commandEncoder);
    if (this.encodedPerformanceRevisions) {
      this.encodedPerformanceRevisions.clear();
      this.encodedPerformanceRevisions.set(this.explorer.dataset.id, {
        sourceRevision: this.explorer.dataset.revision,
        explorerRevision: this.explorer.currentRevision
      });
      for (const published of this.publications.values())
        this.encodedPerformanceRevisions.set(published.dataset.id, {
          sourceRevision: published.dataset.revision,
          explorerRevision: published.consumer.explorer.currentRevision
        });
    }
    if (startedAt !== undefined && this.performanceEncodes && this.encodedPerformanceRevisions) {
      const finishedAt = performance.now();
      for (const [datasetId, revision] of this.encodedPerformanceRevisions) {
        this.performanceEncodes.set(datasetId, {startedAt, finishedAt, ...revision});
        if (this.performanceEncodeTelemetry) {
          let telemetry = this.performanceEncodeTelemetry.get(datasetId);
          if (!telemetry) {
            telemetry = {
              generation: this.performanceDiagnosticsGeneration,
              count: 0,
              totalMilliseconds: 0,
              droppedEvents: 0,
              events: []
            };
            this.performanceEncodeTelemetry.set(datasetId, telemetry);
          }
          telemetry.count++;
          telemetry.totalMilliseconds += finishedAt - startedAt;
          telemetry.events.push({sequence: telemetry.count, startedAt, finishedAt, ...revision});
          if (telemetry.events.length > 128) {
            telemetry.events.shift();
            telemetry.droppedEvents++;
          }
        }
      }
    }
    this.encodedFrames++;
    this.dirty = this.updateRevision !== updateRevision;
    this.needsSummary = true;
  }

  /** Called after Deck has submitted the render frame, never from a compute pass. */
  async readSummary(): Promise<void> {
    if (
      this.destroyed ||
      this.isLayerRegistrationPending ||
      this.summaryPending ||
      !this.needsSummary
    )
      return;
    this.summaryPending = true;
    this.needsSummary = false;
    const startedAt = this.performanceDiagnosticsEnabled ? performance.now() : undefined;
    const timingGeneration = this.performanceDiagnosticsGeneration;
    try {
      await this.tableBinding.readSummary();
      await this.analysis.readSummary();
      await this.styles.readLegends();
      await this.heatmaps.readSummary();
      await Promise.all(
        [...this.publications.values()].map(async published => {
          const publicationStartedAt = this.performanceDiagnosticsEnabled
            ? performance.now()
            : undefined;
          await published.readSummary(this.getPublicationBridge?.(published.dataset.id));
          const summary = published.consumer.summary;
          if (
            publicationStartedAt !== undefined &&
            summary &&
            this.performanceSummaryReads &&
            timingGeneration === this.performanceDiagnosticsGeneration &&
            this.publications.get(published.dataset.id) === published
          )
            this.performanceSummaryReads.set(published.dataset.id, {
              startedAt: publicationStartedAt,
              finishedAt: performance.now(),
              sourceRevision: published.dataset.revision,
              revision: summary.revision,
              stage: 'published-linked-summary'
            });
        })
      );
      const summary = await this.explorer.readSummary();
      if (summary && !this.destroyed) {
        if (!this.tableBinding.isSummaryCurrent) {
          // An encode that landed during this read already queued a fresh read in `finally`.
          // Requesting another encode would make that read stale again and livelock.
          if (!this.needsSummary) this.requestUpdate();
          return;
        }
        for (const [field, filterId] of [
          ['value', 'resident-value'],
          ['timestamp', 'resident-timestamp']
        ]) {
          const bins = this.tableBinding.getFilterHistogram(filterId);
          if (this.tableBinding.hasPlot(filterId) && !bins) {
            if (!this.needsSummary) this.requestUpdate();
            return;
          }
          if (bins) summary.histograms[field] = Uint32Array.from(bins, bin => bin.count);
        }
        this.latestSummary = summary;
        if (
          startedAt !== undefined &&
          this.performanceSummaryReads &&
          timingGeneration === this.performanceDiagnosticsGeneration
        )
          this.performanceSummaryReads.set(this.explorer.dataset.id, {
            startedAt,
            finishedAt: performance.now(),
            sourceRevision: this.explorer.dataset.revision,
            revision: summary.revision,
            stage: 'fixture-linked-summary'
          });
        for (const listener of this.listeners) listener(summary);
      }
    } finally {
      this.summaryPending = false;
      // A newer submitted frame may have arrived while the previous readback was pending.
      if (this.needsSummary && !this.destroyed) await this.readSummary();
    }
  }

  async destroy(afterSubmittedWork?: PromiseLike<unknown>): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    this.updateScheduler.cancel();
    this.listeners.clear();
    this.invalidationListeners.clear();
    // All source buffers are owned here, and remain alive through the last submitted frame.
    const fence =
      afterSubmittedWork || this.session.deviceLost ? undefined : this.device.createFence();
    const completion = Promise.resolve(afterSubmittedWork ?? fence?.signaled);
    try {
      const styleOutcomes = await Promise.allSettled([
        this.styles.destroy(completion),
        this.heatmaps.destroy(completion),
        this.labels.destroy(completion),
        this.pointAltitudes.destroy(completion),
        this.pointPositionVariants.destroy(completion)
      ]);
      const tableOutcomes = await Promise.allSettled([
        this.chartProvider?.destroy(),
        this.nativeTable.destroy(),
        ...[...this.publications.values()].map(published => published.destroy())
      ]);
      this.publications.clear();
      const outcomes = await Promise.allSettled([
        this.analysis.destroy(completion),
        this.tableBinding.destroy(completion),
        this.time.destroy(completion),
        this.aggregation.destroy(completion),
        this.duckdb.destroy()
      ]);
      await this.session.destroy(completion);
      const failure = [...styleOutcomes, ...tableOutcomes, ...outcomes].find(
        outcome => outcome.status === 'rejected'
      );
      if (failure?.status === 'rejected') throw failure.reason;
    } finally {
      fence?.destroy();
      for (const vector of this.sourceVectors) vector.destroy();
      this.renderReservation.release();
    }
  }
}

/** Compute is encoded into Deck's encoder before the layers read its mask. */
export class KeplerComputeEffect implements Effect {
  readonly id = 'kepler-resident-compute';
  readonly props = {};
  readonly useInPicking = false;
  readonly order = 0;
  runtime: ComputeRuntime | null = null;
  constructor(private readonly onError: (error: unknown) => void) {}
  setup(_context: EffectContext): void {}
  preRender(options: PreRenderOptions): void {
    try {
      if (!this.runtime?.isReadyForRender) return;
      this.runtime.encode();
      // Trip markers derive instances from the masks and styles encoded just above.
      for (const layer of options.layers)
        if (
          (layer instanceof GPUTripMarkerLayer || layer instanceof ResidentScenegraphPointLayer) &&
          layer.props.visible !== false
        )
          layer.encodeInstances(this.runtime.device.commandEncoder);
      // Density transforms submit work; their inputs must be encoded before this flush.
      renderResidentHeatmapPasses(options.layers);
    } catch (error) {
      this.onError(error);
    }
  }
  cleanup(_context: EffectContext): void {}
}
