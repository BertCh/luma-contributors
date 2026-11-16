// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getAnalysisRampPalette,
  addAnalysisRampPalette,
  addAnalysisInferenceColorIndices,
  getAnalysisHotspotColors
} from './analysis-result-colors';
import {getKeplerResultToneColor} from '@luma-contributors/kepler';
import {KeplerAllocationError} from '@luma-contributors/kepler';
import {analysisDatasetContext} from './analysis-dataset-context';
import {getAnalysisFormInitialization} from './analysis-form-initialization';
import {trackDeferredOperation} from './tracked-operations';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import type {Layer} from '@deck.gl/core';
import {GPUVector, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {
  createTransientView,
  GPUCommandGraph,
  GPUMask,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUColorScale,
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  getGPUColorScaleParameterValues,
  packGPUColor,
  type GPUVariableStandardizationMethod
} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  addKeplerAnalysisRecipe,
  parseKeplerIDWSettings,
  parseKeplerKrigingSettings,
  parseKeplerDBSCANSettings,
  parseKeplerOLSSettings,
  type KeplerOLSSettings,
  type KeplerDorlingSettings,
  type KeplerPolygonAnalysisOperation,
  type KeplerDBSCANSettings,
  type KeplerIDWSettings,
  type KeplerKrigingSettings,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type KeplerMemoryReservation,
  type KeplerSessionExplorer
} from '@luma-contributors/kepler';

import type {KeplerTableDerivedSnapshot} from '@luma-contributors/kepler';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getAnalysisRecipeSettings,
  isPolygonAnalysisOperation,
  normalizePolygonAnalysisOperation,
  validateAnalysisRecipeCapacity,
  MAXIMUM_EXACT_NATURAL_BREAKS_ROWS
} from './analysis-recipe-settings';
export {MAXIMUM_EXACT_NATURAL_BREAKS_ROWS} from './analysis-recipe-settings';
import {
  type AnalysisPolygonWeightsCache,
  type AnalysisPolygonWeightsLease,
  type PolygonAnalysisSource
} from './analysis-polygon-weights-cache';

import {
  normalizeHotspotSettings,
  getHotspotScratchBytes,
  type HotspotSettings
} from './analysis-hotspot-settings';
import {AnalysisHotspotOutputs, makeAnalysisOutputVector} from './analysis-hotspot-outputs';
export type {HotspotSettings} from './analysis-hotspot-settings';
import {
  normalizePointWeightsSettings,
  makePointWeightsRecipe,
  type PointWeightsSettings
} from './analysis-point-weights-settings';
import {
  AnalysisStreamedPointWeights,
  choosePointWeightsPath,
  getStreamedWeightsSettings,
  type AnalysisPointWeightsPath,
  type AnalysisStreamedWeightsOptions
} from './analysis5-streamed-weights';
import {
  type AnalysisPointWeightsCache,
  packAnalysisVector,
  type AnalysisPointWeightsLease
} from './analysis-point-weights-cache';
import {getAnalysisWeightsStore, type AnalysisWeightsSettings} from './analysis-weights-store';
import {getPointTopologySettings} from './analysis-point-weights-settings';

export type {PointWeightsSettings} from './analysis-point-weights-settings';
import {
  normalizePointHotspotSettings,
  type PointHotspotSettings
} from './analysis-point-hotspot-settings';
export type {PointHotspotSettings} from './analysis-point-hotspot-settings';
import {
  normalizeLocalMoranSettings,
  type LocalMoranSettings
} from './analysis-local-moran-settings';
import {AnalysisLocalMoranOutputs} from './analysis-local-moran-outputs';
export type {LocalMoranSettings} from './analysis-local-moran-settings';
import {
  normalizeQuantileLisaSettings,
  type QuantileLisaSettings
} from './analysis-quantile-lisa-settings';
import {AnalysisQuantileLisaOutputs} from './analysis-quantile-lisa-outputs';
import {getPointCoordinateColumns} from './analysis-point-coordinate-settings';
export type {QuantileLisaSettings} from './analysis-quantile-lisa-settings';
import {
  InterpolationRuntime,
  type InterpolationMetric,
  type KrigingSummary
} from './interpolation-runtime';
import {
  interpolationSurfaceHover,
  registerInterpolationSurfaceProvider
} from './interpolation-surface-hover';
import type {DeclaredProjectedCartogramOptions} from './analysis-cartogram-declared-crs';
import {describeRateSourceExclusions} from './analysis2-narrowing-summary';
import {describeGpuRateCauseCounts, readGpuRateCauseCounts} from './analysis6-rate-cause-gpu';
import {CartogramRuntime, type CartogramSummary} from './cartogram-runtime';
import {normalizeCartogramSettings} from './cartogram-settings';
import {getCartogramMercatorUnavailableReason} from './cartogram-mercator-layer';
import type {CartogramPick} from './cartogram-picking';
import {AnalysisDBSCANOutputs} from './analysis-dbscan-outputs';
import {AnalysisOLSOutputs} from './analysis-ols-outputs';
import {suggestKrigingVariogram} from './kriging-diagnostics';
import {makeOLSAnalysisSummary, type OLSAnalysisSummary} from './analysis-ols-report';

let weightsConsumerSequence = 0;

export type AnalysisMode =
  | 'off'
  | GPUVariableStandardizationMethod
  | 'quantile'
  | 'equal-interval'
  | 'natural-breaks'
  | 'excess-risk'
  | 'hot-spots-lattice'
  | 'spatial-rate-points'
  | 'hot-spots-points'
  | 'local-moran-points'
  | 'spatial-rate-polygons'
  | 'hot-spots-polygons'
  | 'local-moran-polygons'
  | 'quantile-lisa-points'
  | 'quantile-lisa-polygons'
  | 'idw-points'
  | 'kriging-points'
  | 'dorling-cartogram'
  | 'dbscan-points'
  | 'ols';

export function isStandardizationMode(mode: string): mode is GPUVariableStandardizationMethod {
  return ['deviation-from-mean', 'z-score', 'mad', 'range-adjust', 'range-standardize'].includes(
    mode
  );
}

function isQuantileLisaMode(mode: string): boolean {
  return mode === 'quantile-lisa-points' || mode === 'quantile-lisa-polygons';
}

export function isAnalysisMode(value: string): value is AnalysisMode {
  return (
    value === 'off' ||
    value === 'quantile' ||
    value === 'equal-interval' ||
    value === 'natural-breaks' ||
    value === 'excess-risk' ||
    value === 'hot-spots-lattice' ||
    value === 'spatial-rate-points' ||
    value === 'hot-spots-points' ||
    value === 'local-moran-points' ||
    value === 'spatial-rate-polygons' ||
    value === 'hot-spots-polygons' ||
    value === 'local-moran-polygons' ||
    value === 'quantile-lisa-points' ||
    value === 'quantile-lisa-polygons' ||
    value === 'idw-points' ||
    value === 'kriging-points' ||
    value === 'dorling-cartogram' ||
    value === 'dbscan-points' ||
    value === 'ols' ||
    isStandardizationMode(value)
  );
}
export type AnalysisSettings = {
  mode: AnalysisMode;
  population: 'all' | 'selected';
  classCount: number;
  variance: 'sample' | 'population';
  naturalBreaksMode: 'approximate' | 'exact';
  column: string;
  denominatorColumn: string;
  hotspot?: HotspotSettings;
  pointWeights?: PointWeightsSettings;
  pointHotspot?: PointHotspotSettings;
  localMoran?: LocalMoranSettings;
  quantileLisa?: QuantileLisaSettings;
  idw?: KeplerIDWSettings;
  kriging?: KeplerKrigingSettings;
  cartogram?: KeplerDorlingSettings;
  dbscan?: KeplerDBSCANSettings;
  ols?: KeplerOLSSettings;
  polygon?: KeplerPolygonAnalysisOperation;
};
export type AnalysisSummary = {
  cartogram?: CartogramSummary['cartogram'];
  ols?: OLSAnalysisSummary;
  dbscan?: {
    clusterCount: number;
    noiseCount: number;
    validCount: number;
    excludedCount: number;
    complete: boolean;
  };
  kriging?: KrigingSummary;
  interpolation?: {width: number; height: number; validSourceCount: number; cellCount: number};
  datasetId: string;
  column: string;
  denominatorColumn?: string;
  localMoran?: {invalidCount: number};
  quantileLisa?: {
    boundaries: readonly number[];
    validCount: number;
    islandCount: number;
    invalidQuantile: number;
    overflow: number;
  };
  hotspot?: {weightsOverflow: number; weightsRequiredCount: number; invalidCount: number};
  pointWeights?: {weightsOverflow: number; weightsRequiredCount: number; invalidCount: number};
  /** Which storage built the point weights: the single CSR or streamed row chunks. */
  pointWeightsPath?: AnalysisPointWeightsPath;
  polygonWeights?: {
    weightsOverflow: number;
    weightsRequiredCount: number;
    candidateOverflow: number;
    invalidCount: number;
  };
  revision: number;
  sourceRevision: number;
  selectionRevision?: number;
  operation: AnalysisMode;
  numericalPolicy: 'float32';
  count: number;
  nullCount: number;
  minimum: number;
  maximum: number;
  mean: number;
  breaks: readonly number[];
  classCounts: readonly number[];
  classCount?: number;
};
export type AnalysisRow = {
  sourceRowIndex: number;
  sourceRevision: number;
  analysisRevision: number;
  value: number;
  kind:
    | GPUVariableStandardizationMethod
    | 'class'
    | 'excess-risk'
    | 'hot-spots-lattice'
    | 'spatial-rate-points'
    | 'hot-spots-points'
    | 'local-moran-points'
    | 'spatial-rate-polygons'
    | 'hot-spots-polygons'
    | 'local-moran-polygons'
    | 'quantile-lisa-points'
    | 'quantile-lisa-polygons'
    | 'dbscan-points'
    | 'dorling-cartogram'
    | 'ols';
};

type AnalysisProgram = {
  key: string;
  compiled: CompiledGPUCommandGraph;
  pointWeights?: AnalysisPointWeightsLease;
  polygonWeights?: AnalysisPolygonWeightsLease;
  destroy: () => void;
};

/** A concrete Kepler consumer: saved recipes -> resident colors -> bounded diagnostics. */
export class AnalysisRuntime {
  readonly colors: GPUVector<'unorm8x4'>;
  private readonly session: KeplerComputeSession;
  private readonly explorer: KeplerSessionExplorer;
  private readonly requestUpdate: () => void;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly buffers: Buffer[] = [];
  private reservation: KeplerMemoryReservation | undefined;
  private readonly resultValues: Buffer;
  private readonly resultClasses: Buffer;
  private readonly resultColors: Buffer;
  private readonly resultValidity: Buffer;
  private readonly tableValues: GPUVector<'float32'>;
  private readonly tableClasses: GPUVector<'uint32'>;
  private readonly tableValidity: GPUVector<'uint32'>;
  private readonly breaks: Buffer;
  private readonly classCount: Buffer;
  private readonly classCounts: Buffer;
  private readonly statistics: Buffer;
  private readonly counts: Buffer;
  private readonly palette: Buffer;
  private readonly standardDomain: Buffer;
  private program?: AnalysisProgram;
  private hotspotOutputs?: AnalysisHotspotOutputs;
  private localMoranOutputs?: AnalysisLocalMoranOutputs;
  private quantileLisaOutputs?: AnalysisQuantileLisaOutputs;
  private interpolation?: InterpolationRuntime;
  private interpolationEncoder?: CommandEncoder;
  private unregisterSurfaceProvider?: () => void;
  private cartogram?: CartogramRuntime;
  private requestedCartogramDeclaration?: DeclaredProjectedCartogramOptions;
  private cartogramEncoder?: CommandEncoder;
  private currentCartogramInspection: CartogramPick | null = null;
  private currentCartogramInspectionPinned = false;
  private dbscanOutputs?: AnalysisDBSCANOutputs;
  private olsOutputs?: AnalysisOLSOutputs;
  private readonly weightsConsumerId = `analysis-weights-consumer-${++weightsConsumerSequence}`;
  private selectedWeightsId?: string;
  private releaseWeightsSource?: () => void;
  private releaseWeightsListener?: () => void;
  private pointWeightsCache?: AnalysisPointWeightsCache;
  private streamedPointWeights?: AnalysisStreamedPointWeights;
  private streamedWeightsOptions: AnalysisStreamedWeightsOptions = {};
  private streamedWeightsBuilding = false;
  private rateCauseDescription?: {key: string; text: string};
  private rateCausePendingKey?: string;
  private polygonWeightsCache?: AnalysisPolygonWeightsCache;
  private readonly polygonSource?: PolygonAnalysisSource;
  private lastEncodedEncoder?: CommandEncoder;
  private lastSubmittedEncoder?: CommandEncoder;
  private retiredPrograms: AnalysisProgram[] = [];
  private retirementPromises = new Set<Promise<void>>();
  private activeReads = new Set<Promise<unknown>>();
  private listeners = new Set<() => void>();
  /** A fit belongs to the response selected when it was supplied or replayed. */
  private krigingFitResponseColumn?: string;
  private requestedSettings: AnalysisSettings = {
    mode: 'off',
    population: 'all',
    classCount: 5,
    variance: 'sample',
    naturalBreaksMode: 'approximate',
    column: 'value',
    denominatorColumn: 'value'
  };
  private readonly publicationPointColumns?: {longitude: string; latitude: string};
  private releaseDatasetContext?: () => void;
  private valueColumnOverridden = false;
  private denominatorColumnOverridden = false;
  private restoredRecipeRevision = 0;
  private revision = 1;
  private encodedRevision = -1;
  private encodedSelectionRevision = -1;
  private encodedRecipe?: KeplerAnalysisRecipe;
  private summaryRevision = -1;
  private summaryPending = false;
  /** Encoded revision whose summary read the user cancelled from the memory manager. */
  private summaryCancelledRevision = -1;
  private destroyed = false;
  private destruction?: Promise<void>;
  private currentSummary: AnalysisSummary | null = null;
  private currentError = '';
  private readbackBytes = 0;

  constructor(props: {
    session: KeplerComputeSession;
    explorer: KeplerSessionExplorer;
    requestUpdate: () => void;
    column?: string;
    polygonSource?: PolygonAnalysisSource;
    pointColumns?: {longitude: string; latitude: string};
  }) {
    this.session = props.session;
    this.explorer = props.explorer;
    this.requestUpdate = props.requestUpdate;
    this.polygonSource = props.polygonSource;
    this.publicationPointColumns = props.pointColumns;
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.explorer.dataset.id
    );
    try {
      const columns = this.availableColumns;
      const initialization = this.formInitialization;
      const column = props.column ?? initialization.column;
      this.valueColumnOverridden = props.column !== undefined;
      if (!columns.length || (props.column !== undefined && !columns.includes(props.column)))
        throw new Error('Analysis requires a nonempty prepared Float32 column');
      this.requestedSettings.column = column;
      this.requestedSettings.denominatorColumn =
        initialization.valueColumns.find(name => name !== column) ?? column;
      // Account outputs here; individual programs reserve their scratch separately.
      this.reservation = props.session.registry.reserveMemory(
        this.rowCount * 16 + 512,
        'gpu',
        'output'
      );
      const allocate = (
        name: string,
        length: number,
        values?: Float32Array | Uint32Array
      ): Buffer => {
        const buffer = props.session.device.createBuffer({
          id: `kepler-analysis-${name}`,
          byteLength: Math.max(1, length) * 4,
          usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST,
          ...(values ? {data: values} : {})
        });
        this.buffers.push(buffer);
        return buffer;
      };
      this.resultValues = allocate('values', this.rowCount);
      this.resultClasses = allocate('classes', this.rowCount);
      this.resultColors = allocate('colors', this.rowCount);
      this.resultValidity = allocate(
        'validity',
        this.rowCount,
        new Uint32Array(this.rowCount).fill(1)
      );
      const chunkLengths = this.sourceChunkLengths;
      this.tableValues = makeAnalysisOutputVector(
        'analysis-values',
        'float32',
        this.resultValues,
        chunkLengths
      );
      this.tableClasses = makeAnalysisOutputVector(
        'analysis-classes',
        'uint32',
        this.resultClasses,
        chunkLengths
      );
      this.tableValidity = makeAnalysisOutputVector(
        'analysis-validity',
        'uint32',
        this.resultValidity,
        chunkLengths
      );
      this.breaks = allocate('breaks', 9);
      this.classCount = allocate('class-count', 1);
      this.classCounts = allocate('class-counts', 8);
      this.statistics = allocate('statistics', GPU_COLUMN_PROFILE_STATISTIC_COUNT);
      this.counts = allocate('counts', 2);
      this.palette = allocate(
        'palette',
        72,
        new Uint32Array([
          ...getAnalysisRampPalette(),
          ...getAnalysisHotspotColors().map(color => packGPUColor(...color)),
          0
        ])
      );
      this.releaseDatasetContext = analysisDatasetContext.subscribe(() => {
        if (this.destroyed || this.requestedSettings.mode !== 'off') return;
        const initialization = this.formInitialization;
        if (!this.valueColumnOverridden && initialization.column)
          this.requestedSettings.column = initialization.column;
        if (!this.denominatorColumnOverridden)
          this.requestedSettings.denominatorColumn =
            initialization.valueColumns.find(name => name !== this.requestedSettings.column) ??
            this.requestedSettings.column;
        this.notify();
      });
      this.standardDomain = allocate('standard-domain', 2, new Float32Array([-3, 3]));
      this.colors = makeAnalysisOutputVector(
        'analysis-colors',
        'unorm8x4',
        this.resultColors,
        chunkLengths
      );
      this.releaseWeightsSource = this.weightsStore.retainSource(
        this.lease.dataset,
        this.polygonSource
      );
      this.releaseWeightsListener = this.weightsStore.subscribe(() => this.notify());
    } catch (error) {
      this.releaseDatasetContext?.();
      for (const buffer of this.buffers) buffer.destroy();
      this.reservation?.release();
      void this.lease.release();
      throw error;
    }
  }

  /** Shared, revision-scoped metadata; this getter never reads source rows. */
  get formInitialization() {
    const candidate = analysisDatasetContext.get(this.datasetId);
    const entry =
      candidate && candidate.dataset.owner === this.lease.dataset ? candidate : undefined;
    const source = this.lease.dataset.value.source;
    let metadata: unknown = this.lease.dataset.source?.metadata;
    const sample = source.schema.metadata.get('luma:sandbox-sample');
    if (sample) {
      try {
        metadata = JSON.parse(sample);
      } catch {
        /* Unknown metadata leaves topology explicit. */
      }
    }
    return getAnalysisFormInitialization(
      entry,
      this.availableColumns,
      metadata,
      source.numRows,
      this.publicationPointColumns
    );
  }

  get availablePolygonColumns(): readonly string[] {
    return this.polygonSource ? [this.polygonSource.column] : [];
  }
  get weightsStore() {
    return getAnalysisWeightsStore(this.session);
  }
  get weightsReference() {
    return this.weightsStore.getConsumer(this.weightsConsumerId);
  }
  clearNamedWeightsReference(): void {
    this.selectedWeightsId = undefined;
    this.weightsStore.unbind(this.weightsConsumerId);
  }
  restoreNamedWeightsReference(id: string, version: number): void {
    const definition = this.weightsStore.get(id);
    if (
      definition.source.datasetId !== this.datasetId ||
      definition.source.revision !== this.lease.dataset.revision
    )
      throw new Error('Saved weights require this dataset revision');
    this.selectedWeightsId = id;
    this.weightsStore.bind(this.weightsConsumerId, id, version);
    this.notify();
  }
  get weightsStaleReason(): string | undefined {
    return this.weightsReference?.staleReason;
  }
  get namedWeightsSource() {
    return {datasetId: this.datasetId, revision: this.lease.dataset.revision};
  }
  /** Selecting a version explicitly accepts rebuilding every dependent result. */
  selectNamedWeights(id: string): void {
    const definition = this.weightsStore.get(id);
    if (
      definition.source.datasetId !== this.datasetId ||
      definition.source.revision !== this.lease.dataset.revision
    )
      throw new Error('Named weights require this dataset revision');
    const settings = definition.settings;
    const current = this.requestedSettings;
    if (settings.kind === 'contiguity') {
      if (!current.polygon || current.mode !== current.polygon.kind)
        throw new Error('Choose a polygon analysis before selecting polygon weights');
      this.configurePolygon({...current.polygon, weights: settings});
    } else {
      const topology = getPointTopologySettings(settings);
      // Replace the entire coordinate/topology union; retain only the statistic's own settings.
      if (current.mode === 'spatial-rate-points' && current.pointWeights) {
        const {scale, islandPolicy} = current.pointWeights;
        this.configurePointWeights({...topology, scale, islandPolicy});
      } else if (current.mode === 'hot-spots-points' && current.pointHotspot) {
        const {minimumNeighbors, selfWeight, multipleTesting, islands} = current.pointHotspot;
        this.configurePointHotspots({
          ...topology,
          minimumNeighbors,
          selfWeight,
          multipleTesting,
          islands
        });
      } else if (current.mode === 'local-moran-points' && current.localMoran) {
        const {
          minimumNeighbors,
          islands,
          multipleTesting,
          significanceLevel,
          inference,
          permutationCount,
          seed,
          secondColumn,
          variableColumns
        } = current.localMoran;
        this.configureLocalMoran({
          ...topology,
          minimumNeighbors,
          islands,
          multipleTesting,
          significanceLevel,
          inference,
          permutationCount,
          seed,
          secondColumn,
          variableColumns
        });
      } else if (current.mode === 'quantile-lisa-points' && current.quantileLisa) {
        const {quantileCount, selectedQuantile, seed, permutationCount, significanceLevel} =
          current.quantileLisa;
        this.configureQuantileLisa({
          ...topology,
          quantileCount,
          selectedQuantile,
          seed,
          permutationCount,
          significanceLevel
        });
      } else throw new Error('Choose a point analysis before selecting point weights');
    }
    this.selectedWeightsId = id;
    this.weightsStore.bind(this.weightsConsumerId, id);
    this.revision++;
    this.currentSummary = null;
    this.notify();
    this.requestUpdate();
  }
  get polygonWeightsCacheStats() {
    return (
      this.polygonWeightsCache?.stats ?? {
        buildCount: 0,
        encodeCount: 0,
        entryCount: 0,
        byteLength: 0
      }
    );
  }
  /** Debug and test override of the streamed point-weights path; the next encode rebuilds. */
  getStreamedWeightsOptions(): AnalysisStreamedWeightsOptions {
    return {...this.streamedWeightsOptions};
  }
  setStreamedWeightsOptions(options: AnalysisStreamedWeightsOptions): void {
    this.streamedWeightsOptions = {...options};
    this.revision++;
    this.encodedRevision = -1;
    this.currentSummary = null;
    this.notify();
    this.requestUpdate();
  }
  /** True while chunked point weights are being built (the count pass is read back). */
  get isBuildingStreamedWeights(): boolean {
    return this.streamedWeightsBuilding;
  }
  get pointWeightsCacheStats() {
    return (
      this.pointWeightsCache?.stats ?? {buildCount: 0, encodeCount: 0, entryCount: 0, byteLength: 0}
    );
  }
  get rowCount(): number {
    return this.lease.dataset.rowCount;
  }
  private get sourceChunkLengths(): number[] {
    return this.lease.dataset.value.source.batches.map(batch => batch.numRows);
  }
  /** Explicit restoration resets coordinate drafts even when the saved coordinates are unchanged. */
  get coordinateRestorationKey(): string {
    return `${this.datasetId}:${this.restoredRecipeRevision}`;
  }
  get datasetId(): string {
    return this.lease.dataset.id;
  }
  get supportsExactNaturalBreaks(): boolean {
    return this.rowCount <= MAXIMUM_EXACT_NATURAL_BREAKS_ROWS;
  }
  get resultColumn(): string {
    const mode = this.requestedSettings.mode;
    if (mode === 'idw-points' || mode === 'kriging-points') return 'interpolation_grid_value';
    if (mode === 'dorling-cartogram')
      return this.getResultColumnName('analysis_cartogram_radius_meters');
    if (mode === 'ols') return this.getResultColumnName('analysis_ols_residual');
    if (mode === 'dbscan-points') return this.getResultColumnName('analysis_dbscan_cluster');
    if (isQuantileLisaMode(mode))
      return this.getResultColumnName('analysis_quantile_lisa_pseudo_p_value');
    if (mode === 'local-moran-points' || mode === 'local-moran-polygons')
      return this.getResultColumnName('analysis_local_moran_i');
    if (mode === 'spatial-rate-points' || mode === 'spatial-rate-polygons')
      return this.getResultColumnName('analysis_spatial_rate');
    if (
      mode === 'hot-spots-lattice' ||
      mode === 'hot-spots-points' ||
      mode === 'hot-spots-polygons'
    )
      return this.getResultColumnName('analysis_hot_spot_z_score');
    let name =
      isStandardizationMode(mode) || mode === 'excess-risk'
        ? `analysis_${mode.replaceAll('-', '_')}`
        : 'analysis_class';
    return this.getResultColumnName(name);
  }
  /**
   * Cause-specific exclusion counts for the rate event and population columns over every source
   * row. The counts come from one GPU reduction over the resident columns (nine words read back),
   * so GPU-resident results work too; until it finishes, a retained CPU source is scanned instead.
   */
  describeRateExclusions(): string | undefined {
    const dataset = this.lease.dataset.value;
    const eventColumn = this.requestedSettings.column;
    const populationColumn = this.requestedSettings.denominatorColumn;
    // The selected population counts only selected rows, so the cache key carries the selection revision.
    const selected = this.requestedSettings.population === 'selected';
    const key = `${this.lease.dataset.id}:${this.lease.dataset.revision}:${eventColumn}:${populationColumn}:${
      selected ? `selection${this.explorer.currentRevision}` : 'all'
    }`;
    if (this.rateCauseDescription?.key === key) return this.rateCauseDescription.text;
    if (this.rateCausePendingKey !== key && dataset.columns[eventColumn] && populationColumn) {
      this.rateCausePendingKey = key;
      const events = dataset.columns[eventColumn];
      const populations = dataset.columns[populationColumn];
      if (events?.format === 'float32' && populations?.format === 'float32') {
        void this.trackRead(
          readGpuRateCauseCounts(this.session.device, {
            events: borrowFloat32Column(events),
            populations: borrowFloat32Column(populations),
            eventValidity: dataset.analytics.validity[eventColumn],
            populationValidity: dataset.analytics.validity[populationColumn],
            selection: selected ? this.explorer.selectionMask : undefined
          })
            .then(counts => {
              if (this.destroyed || this.rateCausePendingKey !== key) return;
              this.rateCauseDescription = {key, text: describeGpuRateCauseCounts(counts)};
              this.notify();
            })
            .catch(() => {
              // The CPU scan below remains the fallback when the reduction cannot run.
            })
        );
      }
    }
    if (dataset.resident || selected) return undefined;
    return describeRateSourceExclusions(dataset.source, eventColumn, populationColumn);
  }
  get availableColumns(): readonly string[] {
    return Object.entries(this.lease.dataset.value.columns)
      .filter(([, column]) => column.format === 'float32' && column.length > 0)
      .map(([name]) => name);
  }
  private get value(): GPUVector<'float32'> {
    return borrowFloat32Column(this.lease.dataset.value.columns[this.requestedSettings.column]);
  }
  get settings(): Readonly<AnalysisSettings> {
    return {
      ...this.requestedSettings,
      ...(this.requestedSettings.cartogram
        ? {cartogram: normalizeCartogramSettings(this.requestedSettings.cartogram, this.rowCount)}
        : {}),
      ...(this.requestedSettings.polygon
        ? {polygon: normalizePolygonAnalysisOperation(this.requestedSettings.polygon)}
        : {}),
      ...(this.requestedSettings.kriging
        ? {kriging: parseKeplerKrigingSettings(this.requestedSettings.kriging)}
        : {}),
      ...(this.requestedSettings.idw
        ? {idw: parseKeplerIDWSettings(this.requestedSettings.idw)}
        : {}),
      ...(this.requestedSettings.ols
        ? {ols: parseKeplerOLSSettings(this.requestedSettings.ols)}
        : {}),
      ...(this.requestedSettings.dbscan ? {dbscan: {...this.requestedSettings.dbscan}} : {}),
      ...(this.requestedSettings.quantileLisa
        ? {
            quantileLisa: normalizeQuantileLisaSettings(
              this.requestedSettings.quantileLisa,
              this.rowCount
            )
          }
        : {}),
      ...(this.requestedSettings.localMoran
        ? {
            localMoran: normalizeLocalMoranSettings(
              this.requestedSettings.localMoran,
              this.rowCount
            )
          }
        : {}),
      ...(this.requestedSettings.hotspot ? {hotspot: {...this.requestedSettings.hotspot}} : {}),
      ...(this.requestedSettings.pointWeights
        ? {
            pointWeights: normalizePointWeightsSettings(
              this.requestedSettings.pointWeights,
              this.rowCount
            )
          }
        : {}),
      ...(this.requestedSettings.pointHotspot
        ? {
            pointHotspot: normalizePointHotspotSettings(
              this.requestedSettings.pointHotspot,
              this.rowCount
            )
          }
        : {})
    };
  }
  get cartogramSummary(): CartogramSummary | null {
    return this.requestedSettings.mode === 'dorling-cartogram'
      ? (this.cartogram?.summary ?? null)
      : null;
  }
  get cartogramUnavailableReason(): string | undefined {
    const settings = this.requestedSettings.cartogram;
    if (this.cartogram?.getOverlayDecision()) return this.cartogram.unavailableReason;
    return (
      this.cartogramSummary?.cartogram.overlayUnavailableReason ??
      (settings ? getCartogramMercatorUnavailableReason(settings.crs) : undefined)
    );
  }
  /** Declared local-plane georeferencing for non-EPSG:3857 cartogram overlays (display only). */
  get cartogramDeclaration(): DeclaredProjectedCartogramOptions | undefined {
    return this.requestedCartogramDeclaration;
  }
  setCartogramDeclaration(declaration: DeclaredProjectedCartogramOptions | undefined): void {
    this.requestedCartogramDeclaration = declaration;
    this.cartogram?.setDeclaration(declaration);
    this.notify();
  }
  get cartogramInspection(): CartogramPick | null {
    return this.requestedSettings.mode === 'dorling-cartogram' &&
      this.currentCartogramInspection?.isCurrent()
      ? this.currentCartogramInspection
      : null;
  }
  get cartogramInspectionPinned(): boolean {
    return Boolean(this.cartogramInspection && this.currentCartogramInspectionPinned);
  }
  clearCartogramInspection(): void {
    this.cartogram?.clearInspection();
    this.resetCartogramInspection();
    this.notify();
  }
  private resetCartogramInspection(): void {
    this.currentCartogramInspection = null;
    this.currentCartogramInspectionPinned = false;
  }
  get summary(): AnalysisSummary | null {
    if (this.weightsStaleReason) return null;
    if (this.requestedSettings.mode === 'dorling-cartogram') return this.cartogramSummary;
    return this.requestedSettings.mode === 'idw-points' ||
      this.requestedSettings.mode === 'kriging-points'
      ? (this.interpolation?.summary ?? null)
      : this.currentSummary;
  }
  get error(): string {
    if (this.weightsStaleReason) return this.weightsStaleReason;
    if (this.requestedSettings.mode === 'dorling-cartogram')
      return this.cartogram?.error || this.currentError;
    return this.requestedSettings.mode === 'idw-points' ||
      this.requestedSettings.mode === 'kriging-points'
      ? this.interpolation?.error || this.currentError
      : this.currentError;
  }
  get transferredReadbackBytes(): number {
    return this.readbackBytes + (this.cartogram?.transferredReadbackBytes ?? 0);
  }
  get active(): boolean {
    if (this.weightsStaleReason) return false;
    const settings = this.requestedSettings;
    const topology =
      settings.mode === 'spatial-rate-points'
        ? settings.pointWeights
        : settings.mode === 'hot-spots-points'
          ? settings.pointHotspot
          : settings.mode === 'local-moran-points'
            ? settings.localMoran
            : settings.mode === 'quantile-lisa-points'
              ? settings.quantileLisa
              : undefined;
    const requiresSummary =
      settings.mode === 'spatial-rate-polygons' ||
      settings.mode === 'hot-spots-polygons' ||
      settings.mode === 'local-moran-polygons' ||
      settings.mode === 'quantile-lisa-polygons' ||
      settings.mode === 'ols' ||
      topology?.kind === 'distance-band' ||
      (settings.mode === 'local-moran-points' &&
        settings.localMoran?.inference === 'permutation-conditional-two-sided');
    return (
      this.enabled &&
      (!requiresSummary ||
        Boolean(
          this.currentSummary &&
            this.isEncodedCurrent() &&
            this.summaryRevision === this.encodedRevision &&
            this.currentSummary.selectionRevision === this.encodedRecipe?.selectionRevision
        ))
    );
  }
  private get enabled(): boolean {
    return (
      !this.destroyed &&
      this.explorer.isCurrent &&
      this.requestedSettings.mode !== 'off' &&
      this.requestedSettings.mode !== 'idw-points' &&
      this.requestedSettings.mode !== 'kriging-points' &&
      this.requestedSettings.mode !== 'dorling-cartogram' &&
      !this.currentError
    );
  }
  /** Borrowed result columns; table inspectors must finish reads before this runtime is destroyed. */
  getTableColumns(): KeplerTableDerivedSnapshot | undefined {
    if (this.requestedSettings.mode === 'dorling-cartogram')
      return this.cartogram?.getTableColumns();
    if (!this.active) return undefined;
    const recipe = this.encodedRecipe;
    const revision = this.revision;
    const selectionRevision = this.explorer.currentRevision;
    const floatingResult =
      isStandardizationMode(this.requestedSettings.mode) ||
      this.requestedSettings.mode === 'excess-risk' ||
      this.requestedSettings.mode === 'hot-spots-lattice' ||
      this.requestedSettings.mode === 'spatial-rate-points' ||
      this.requestedSettings.mode === 'spatial-rate-polygons' ||
      this.requestedSettings.mode === 'hot-spots-points' ||
      this.requestedSettings.mode === 'hot-spots-polygons' ||
      this.requestedSettings.mode === 'local-moran-points' ||
      this.requestedSettings.mode === 'local-moran-polygons' ||
      isQuantileLisaMode(this.requestedSettings.mode) ||
      this.requestedSettings.mode === 'dbscan-points' ||
      this.requestedSettings.mode === 'ols';
    const name = this.resultColumn;
    const hotspot =
      this.requestedSettings.mode === 'hot-spots-lattice' ||
      this.requestedSettings.mode === 'hot-spots-points' ||
      this.requestedSettings.mode === 'hot-spots-polygons'
        ? this.hotspotOutputs
        : undefined;
    if (
      (this.requestedSettings.mode === 'hot-spots-lattice' ||
        this.requestedSettings.mode === 'hot-spots-points' ||
        this.requestedSettings.mode === 'hot-spots-polygons') &&
      !hotspot
    )
      return undefined;
    const localMoran =
      this.requestedSettings.mode === 'local-moran-points' ||
      this.requestedSettings.mode === 'local-moran-polygons'
        ? this.localMoranOutputs
        : undefined;
    if (
      (this.requestedSettings.mode === 'local-moran-points' ||
        this.requestedSettings.mode === 'local-moran-polygons') &&
      !localMoran
    )
      return undefined;
    const quantileLisa = isQuantileLisaMode(this.requestedSettings.mode)
      ? this.quantileLisaOutputs
      : undefined;
    if (isQuantileLisaMode(this.requestedSettings.mode) && !quantileLisa) return undefined;
    const dbscan = this.requestedSettings.mode === 'dbscan-points' ? this.dbscanOutputs : undefined;
    if (this.requestedSettings.mode === 'dbscan-points' && !dbscan) return undefined;
    const ols = this.requestedSettings.mode === 'ols' ? this.olsOutputs : undefined;
    if (this.requestedSettings.mode === 'ols' && !ols) return undefined;
    const extraColumns = ols
      ? {[this.getResultColumnName('analysis_ols_fitted')]: ols.fitted}
      : dbscan
        ? {
            [this.getResultColumnName('analysis_dbscan_label')]: dbscan.labels,
            [this.getResultColumnName('analysis_dbscan_core_distance_meters')]:
              dbscan.coreDistances,
            [this.getResultColumnName('analysis_dbscan_reachability_meters')]: dbscan.reachability,
            [this.getResultColumnName('analysis_dbscan_predecessor_row')]: dbscan.predecessors
          }
        : quantileLisa
          ? {
              [this.getResultColumnName('analysis_quantile_lisa_join_count')]:
                quantileLisa.observed,
              [this.getResultColumnName('analysis_quantile_lisa_membership')]:
                quantileLisa.membership,
              [this.getResultColumnName('analysis_quantile_lisa_class')]: quantileLisa.classes,
              [this.getResultColumnName('analysis_quantile_lisa_tail_count')]:
                quantileLisa.exceedances,
              [this.getResultColumnName('analysis_quantile_lisa_significant')]:
                quantileLisa.significant
            }
          : localMoran
            ? {
                [this.getResultColumnName('analysis_local_moran_z_score')]: localMoran.zScores,
                [this.getResultColumnName('analysis_local_moran_spatial_lag')]:
                  localMoran.spatialLag,
                [this.getResultColumnName('analysis_local_moran_p_value')]: localMoran.pValues,
                [this.getResultColumnName('analysis_local_moran_adjusted_p_value')]:
                  localMoran.adjustedPValues,
                [this.getResultColumnName('analysis_local_moran_quadrant')]: localMoran.quadrants,
                [this.getResultColumnName('analysis_local_moran_neighbor_count')]:
                  localMoran.neighborCounts
              }
            : hotspot
              ? {
                  [this.getResultColumnName('analysis_hot_spot_p_value')]: hotspot.pValues,
                  [this.getResultColumnName('analysis_hot_spot_bin')]: hotspot.bins,
                  [this.getResultColumnName('analysis_hot_spot_neighbor_count')]:
                    hotspot.neighborCounts
                }
              : {};
    return {
      revision: `${revision}:${selectionRevision}`,
      columns: {[name]: floatingResult ? this.tableValues : this.tableClasses, ...extraColumns},
      validity: Object.fromEntries(
        [name, ...Object.keys(extraColumns)].map(column => [column, this.tableValidity])
      ),
      isCurrent: () =>
        this.isEncodedCurrent() &&
        this.revision === revision &&
        this.encodedRecipe === recipe &&
        this.explorer.currentRevision === selectionRevision
    };
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Retain buffers through an external reader's operation and inspector cleanup. */
  retainTableRead(completion: PromiseLike<unknown>): void {
    if (this.requestedSettings.mode === 'dorling-cartogram') {
      this.cartogram?.retainTableRead(completion);
      return;
    }
    if (this.destroyed) throw new Error('Analysis has been destroyed');
    this.trackRead(Promise.resolve(completion));
  }

  /** Keep borrowed result buffers alive through the reader's complete operation and cleanup. */
  withResultSnapshot<Result>(
    read: (snapshot: KeplerTableDerivedSnapshot) => Promise<Result>
  ): Promise<Result> {
    if (this.requestedSettings.mode === 'dorling-cartogram')
      return (
        this.cartogram?.withResultSnapshot(read) ??
        Promise.reject(new Error('Cartogram results are unavailable'))
      );
    const snapshot = this.getTableColumns();
    if (!snapshot?.isCurrent())
      return Promise.reject(new Error('Analysis results are unavailable or stale'));
    // Register before invoking user code so synchronous teardown cannot outrun this read.
    return this.trackRead(
      Promise.resolve().then(() => {
        if (!snapshot.isCurrent()) throw new Error('Analysis results are unavailable or stale');
        return read(snapshot);
      })
    );
  }

  getResultLayers(): Layer[] {
    if (this.requestedSettings.mode === 'dorling-cartogram') {
      const layer = this.cartogram?.getLayer();
      return layer ? [layer] : [];
    }
    const layer = this.getInterpolationLayer();
    return layer ? [layer] : [];
  }
  /** Hover inspection makes the surface pickable; turn it off to hover points beneath it. */
  setSurfaceInspection(enabled: boolean): void {
    interpolationSurfaceHover.setInspectionEnabled(enabled);
    this.notify();
    this.requestUpdate();
  }
  getInterpolationLayer() {
    return this.requestedSettings.mode === 'idw-points' ||
      this.requestedSettings.mode === 'kriging-points'
      ? (this.interpolation?.getLayer() ?? null)
      : null;
  }
  readInterpolationPage(offset: number, limit = 16) {
    if (
      (this.requestedSettings.mode !== 'idw-points' &&
        this.requestedSettings.mode !== 'kriging-points') ||
      !this.interpolation
    )
      return Promise.reject(new Error('Enable interpolation before reading grid cells'));
    return this.interpolation.readPage(offset, limit);
  }
  /** Whether a current IDW or kriging grid can be read as a derived raster. */
  hasInterpolationGrid(): boolean {
    return (
      (this.requestedSettings.mode === 'idw-points' ||
        this.requestedSettings.mode === 'kriging-points') &&
      Boolean(this.interpolation?.summary)
    );
  }
  readInterpolationGrid() {
    if (!this.hasInterpolationGrid() || !this.interpolation)
      return Promise.reject(new Error('Run IDW or kriging before deriving terrain from its grid'));
    return this.interpolation.readGridValues();
  }
  exportInterpolationGeoTIFF(metric: InterpolationMetric = 'prediction') {
    if (
      (this.requestedSettings.mode !== 'idw-points' &&
        this.requestedSettings.mode !== 'kriging-points') ||
      !this.interpolation
    )
      return Promise.reject(new Error('Enable interpolation before exporting its grid'));
    return this.interpolation.exportGeoTIFF(metric);
  }
  configurePolygon(operation: KeplerPolygonAnalysisOperation): void {
    const polygon = normalizePolygonAnalysisOperation(operation);
    this.update({
      ...this.requestedSettings,
      mode: polygon.kind,
      polygon,
      column: polygon.kind === 'spatial-rate-polygons' ? polygon.events : polygon.column,
      ...(polygon.kind === 'spatial-rate-polygons' ? {denominatorColumn: polygon.populations} : {})
    });
  }
  configureOLS(ols: KeplerOLSSettings): void {
    const settings = parseKeplerOLSSettings(ols);
    this.update({
      ...this.requestedSettings,
      mode: 'ols',
      column: settings.responseColumn,
      ols: settings
    });
  }
  configureDBSCAN(dbscan: KeplerDBSCANSettings): void {
    const settings = parseKeplerDBSCANSettings(dbscan);
    this.update({
      ...this.requestedSettings,
      mode: 'dbscan-points',
      dbscan: settings
    });
  }
  /** Fits kriging variogram parameters to the declared coordinate fields and response field. */
  suggestKrigingVariogram(
    kriging: KeplerKrigingSettings,
    options?: Parameters<typeof suggestKrigingVariogram>[2]
  ) {
    const {columns} = this.lease.dataset.value;
    return suggestKrigingVariogram(
      {
        device: this.session.device,
        longitude: borrowFloat32Column(columns[kriging.longitudeColumn]),
        latitude: borrowFloat32Column(columns[kriging.latitudeColumn]),
        values: this.value
      },
      kriging,
      options
    );
  }
  configureKriging(kriging: KeplerKrigingSettings): void {
    this.update(
      {
        ...this.requestedSettings,
        mode: 'kriging-points',
        kriging: parseKeplerKrigingSettings(kriging)
      },
      true,
      this.requestedSettings.column
    );
  }
  configureInterpolation(idw: KeplerIDWSettings): void {
    this.update({...this.requestedSettings, mode: 'idw-points', idw: parseKeplerIDWSettings(idw)});
  }
  configureCartogram(cartogram: KeplerDorlingSettings): void {
    const settings = normalizeCartogramSettings(cartogram, this.rowCount);
    this.update({
      ...this.requestedSettings,
      mode: 'dorling-cartogram',
      column: settings.valueColumn,
      cartogram: settings
    });
  }

  configureQuantileLisa(quantileLisa: QuantileLisaSettings): void {
    const normalized = normalizeQuantileLisaSettings(quantileLisa, this.rowCount);
    this.update({
      ...this.requestedSettings,
      mode: 'quantile-lisa-points',
      quantileLisa: normalized
    });
  }
  configureLocalMoran(localMoran: LocalMoranSettings): void {
    const normalized = normalizeLocalMoranSettings(localMoran, this.rowCount);
    this.update({
      ...this.requestedSettings,
      mode: 'local-moran-points',
      ...(normalized.variableColumns ? {column: normalized.variableColumns[0]} : {}),
      localMoran: normalized
    });
  }
  configurePointHotspots(pointHotspot: PointHotspotSettings): void {
    const normalized = normalizePointHotspotSettings(pointHotspot, this.rowCount);
    this.update({...this.requestedSettings, mode: 'hot-spots-points', pointHotspot: normalized});
  }
  configurePointWeights(pointWeights: PointWeightsSettings, denominatorColumn?: string): void {
    const normalized = normalizePointWeightsSettings(pointWeights, this.rowCount);
    if (denominatorColumn !== undefined && !this.availableColumns.includes(denominatorColumn))
      throw new Error('Choose a prepared Float32 population column');
    this.update({
      ...this.requestedSettings,
      mode: 'spatial-rate-points',
      pointWeights: normalized,
      denominatorColumn: denominatorColumn ?? this.requestedSettings.denominatorColumn
    });
    if (denominatorColumn !== undefined) this.denominatorColumnOverridden = true;
  }
  configureHotspots(hotspot: HotspotSettings): void {
    const normalized = normalizeHotspotSettings(hotspot, this.rowCount);
    this.update({...this.requestedSettings, mode: 'hot-spots-lattice', hotspot: normalized});
  }
  setColumn(column: string): void {
    if (!this.availableColumns.includes(column))
      throw new Error('Choose a prepared Float32 analysis column');
    this.update({...this.requestedSettings, column});
    this.valueColumnOverridden = true;
  }
  setDenominatorColumn(denominatorColumn: string): void {
    if (!this.availableColumns.includes(denominatorColumn))
      throw new Error('Choose a prepared Float32 population column');
    this.update({...this.requestedSettings, denominatorColumn});
    this.denominatorColumnOverridden = true;
  }
  setMode(mode: AnalysisMode): void {
    if (!isAnalysisMode(mode)) throw new Error('Unsupported example analysis');
    this.update({...this.requestedSettings, mode});
  }
  setPopulation(population: 'all' | 'selected'): void {
    if (population !== 'all' && population !== 'selected')
      throw new Error('Unsupported analysis population');
    this.update({...this.requestedSettings, population});
  }
  setVariance(variance: 'sample' | 'population'): void {
    if (variance !== 'sample' && variance !== 'population')
      throw new Error('Choose sample or population variance');
    this.update({...this.requestedSettings, variance});
  }
  setNaturalBreaksMode(naturalBreaksMode: 'approximate' | 'exact'): void {
    if (naturalBreaksMode !== 'approximate' && naturalBreaksMode !== 'exact')
      throw new Error('Choose approximate or exact natural breaks');
    this.update({...this.requestedSettings, naturalBreaksMode});
  }
  setClassCount(classCount: number): void {
    if (!Number.isInteger(classCount) || classCount < 2 || classCount > 8)
      throw new Error('Choose 2 to 8 classes');
    this.update({...this.requestedSettings, classCount});
  }

  /** Call after explorer.encode on the same encoder. Never submits or reads result rows. */
  encode(commandEncoder: CommandEncoder): void {
    this.assertCurrent();
    if (this.requestedSettings.mode === 'off') return;
    if (this.requestedSettings.mode === 'dorling-cartogram') {
      this.cartogram ??= new CartogramRuntime({
        session: this.session,
        explorer: this.explorer,
        onChange: () => {
          if (this.currentCartogramInspection && !this.currentCartogramInspection.isCurrent())
            this.resetCartogramInspection();
          this.notify();
          this.requestUpdate();
        },
        picking: {
          onHover: pick => {
            if (!this.cartogramInspectionPinned) {
              this.currentCartogramInspectionPinned = false;
              this.currentCartogramInspection = pick;
              this.notify();
            }
          },
          onClick: pick => {
            this.currentCartogramInspection = pick;
            this.currentCartogramInspectionPinned = Boolean(pick);
            this.notify();
          },
          onError: error => {
            this.currentError = error instanceof Error ? error.message : String(error);
            this.notify();
          }
        }
      });
      if (this.requestedCartogramDeclaration)
        this.cartogram.setDeclaration(this.requestedCartogramDeclaration);
      this.cartogramEncoder = commandEncoder;
      this.cartogram.encode(commandEncoder, this.createRecipe());
      return;
    }
    if (
      this.requestedSettings.mode === 'idw-points' ||
      this.requestedSettings.mode === 'kriging-points'
    ) {
      this.interpolation ??= new InterpolationRuntime({
        session: this.session,
        explorer: this.explorer,
        onChange: () => {
          this.notify();
          this.requestUpdate();
        },
        surfaceHover: {
          isEnabled: () => interpolationSurfaceHover.isInspectionEnabled,
          onHover: pick =>
            interpolationSurfaceHover.setPick(pick && {surfaceId: this.datasetId, ...pick})
        }
      });
      this.unregisterSurfaceProvider ??= registerInterpolationSurfaceProvider(this.datasetId, this);
      this.interpolationEncoder = commandEncoder;
      this.interpolation.encode(commandEncoder, this.createRecipe());
      return;
    }
    const selectionRevision = this.explorer.currentRevision + 1;
    if (
      this.encodedRevision === this.revision &&
      (this.lastEncodedEncoder === commandEncoder ||
        this.lastEncodedEncoder === this.lastSubmittedEncoder) &&
      (this.requestedSettings.population === 'all' ||
        selectionRevision === this.encodedSelectionRevision)
    )
      return;
    const recipe = this.createRecipe();
    if (this.weightsStaleReason) {
      this.currentError = this.weightsStaleReason;
      this.notify();
      return;
    }
    const key = JSON.stringify([
      this.requestedSettings,
      this.streamedWeightsOptions,
      this.weightsReference
    ]);
    try {
      if (this.program?.key !== key && this.isAwaitingStreamedWeights(recipe)) return;
      if (this.program?.key !== key) {
        const program = this.createProgram(recipe, key);
        if (this.program) this.retiredPrograms.push(this.program);
        this.program = program;
      }
      this.program.pointWeights?.encode(commandEncoder);
      this.program.polygonWeights?.encode(commandEncoder);
      this.program.compiled.encode(commandEncoder, {parameters: undefined});
      this.program.pointWeights?.markConsumerEncoded(commandEncoder);
      this.program.polygonWeights?.markConsumerEncoded(commandEncoder);
      this.lastEncodedEncoder = commandEncoder;
      this.encodedRevision = this.revision;
      this.encodedSelectionRevision = selectionRevision;
      this.encodedRecipe = recipe;
      this.currentError = '';
    } catch (error) {
      this.encodedRevision = -1;
      this.currentError = error instanceof Error ? error.message : String(error);
      this.notify();
      throw error;
    }
  }

  /** Call after Deck submission, including when disabled, to retire previous program resources. */
  async readSummary(): Promise<void> {
    if (this.cartogramEncoder) this.cartogram?.markSubmitted(this.cartogramEncoder);
    if (this.interpolationEncoder) this.interpolation?.markSubmitted(this.interpolationEncoder);
    await Promise.all([
      this.readSourceSummary(),
      this.interpolation?.readSummary(),
      this.cartogram?.readSummary()
    ]);
  }
  private readSourceSummary(): Promise<void> {
    if (this.lastEncodedEncoder && this.lastEncodedEncoder !== this.lastSubmittedEncoder) {
      this.pointWeightsCache?.markSubmitted(this.lastEncodedEncoder);
      this.polygonWeightsCache?.markSubmitted(this.lastEncodedEncoder);
      this.lastSubmittedEncoder = this.lastEncodedEncoder;
    }
    this.retireSubmittedPrograms();
    if (
      !this.enabled ||
      this.summaryPending ||
      this.summaryCancelledRevision === this.encodedRevision ||
      !this.isEncodedCurrent() ||
      (this.summaryRevision === this.encodedRevision &&
        this.currentSummary?.selectionRevision === this.encodedRecipe?.selectionRevision)
    )
      return Promise.resolve();
    this.summaryPending = true;
    const revision = this.encodedRevision;
    const recipe = this.encodedRecipe!;
    // A slow summary read is a tracked operation; cancelling keeps this revision's summary stale
    // until the analysis settings change.
    const tracked = trackDeferredOperation(`Analysis ${recipe.operation.kind}`, () => {
      this.summaryCancelledRevision = revision;
      this.notify();
    });
    const pointWeights = this.program?.pointWeights;
    const releasePointWeightsRead = pointWeights?.retain();
    const polygonWeights = this.program?.polygonWeights;
    const releasePolygonWeightsRead = polygonWeights?.retain();
    const pending = (async () => {
      try {
        const classified = recipe.operation.kind === 'classify';
        const hotspot =
          recipe.operation.kind === 'hot-spots-lattice' ||
          recipe.operation.kind === 'hot-spots-points' ||
          recipe.operation.kind === 'hot-spots-polygons'
            ? this.hotspotOutputs
            : undefined;
        const localMoran =
          recipe.operation.kind === 'local-moran-points' ||
          recipe.operation.kind === 'multivariate-local-geary-points' ||
          recipe.operation.kind === 'local-moran-polygons'
            ? this.localMoranOutputs
            : undefined;
        const quantileLisa = isQuantileLisaMode(recipe.operation.kind)
          ? this.quantileLisaOutputs
          : undefined;
        const dbscan = recipe.operation.kind === 'dbscan-points' ? this.dbscanOutputs : undefined;
        const ols = recipe.operation.kind === 'ols' ? this.olsOutputs : undefined;
        const coefficientCount =
          recipe.operation.kind === 'ols' ? recipe.operation.predictorColumns.length + 1 : 0;
        const bytes = await Promise.all([
          this.statistics.readAsync(0, GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4),
          this.counts.readAsync(0, 8),
          ...(classified
            ? [
                this.breaks.readAsync(0, (this.requestedSettings.classCount + 1) * 4),
                this.classCounts.readAsync(0, this.requestedSettings.classCount * 4),
                this.classCount.readAsync(0, 4)
              ]
            : []),
          ...(hotspot?.diagnostics.map(buffer => buffer.readAsync(0, 4)) ?? []),
          ...(localMoran
            ? [
                localMoran.invalidCount.readAsync(0, 4),
                localMoran.inferenceOverflow.readAsync(0, 4)
              ]
            : []),
          ...(quantileLisa?.diagnostics.map(buffer => buffer.readAsync()) ?? []),
          ...(dbscan?.diagnostics.map(buffer => buffer.readAsync()) ?? []),
          ...(ols
            ? [
                ols.coefficients.readAsync(0, coefficientCount * 4),
                ols.standardErrors.readAsync(0, coefficientCount * 4),
                ols.tStatistics.readAsync(0, coefficientCount * 4),
                ols.summary.readAsync(),
                ols.status.readAsync()
              ]
            : []),
          ...(pointWeights?.diagnostics.map(buffer => buffer.readAsync(0, 4)) ?? []),
          ...(polygonWeights?.diagnostics.map(buffer => buffer.readAsync(0, 4)) ?? [])
        ]);
        this.readbackBytes += bytes.reduce((sum, values) => sum + values.byteLength, 0);
        if (
          !this.isEncodedCurrent() ||
          revision !== this.encodedRevision ||
          recipe !== this.encodedRecipe
        )
          return;
        const polygonWeightsStatus = polygonWeights
          ? bytes.slice(-4).map(buffer => unsignedValues(buffer)[0])
          : undefined;
        if (polygonWeightsStatus && (polygonWeightsStatus[0] || polygonWeightsStatus[2])) {
          this.currentSummary = null;
          this.currentError = polygonWeightsStatus[2]
            ? `Polygon candidate storage overflowed; at least ${polygonWeightsStatus[1]} distinct neighbor slots were observed. Increase candidate capacity; the required count is only a lower bound.`
            : `Polygon neighbor storage overflowed; ${polygonWeightsStatus[1]} slots are required`;
          this.notify();
          this.requestUpdate();
          return;
        }
        const hotspotStatus = hotspot
          ? bytes.slice(2, 5).map(buffer => unsignedValues(buffer)[0])
          : undefined;
        if (hotspotStatus?.[0]) {
          this.currentSummary = null;
          this.currentError = `Hotspot neighbor storage overflowed; ${hotspotStatus[1]} slots are required`;
          this.notify();
          this.requestUpdate();
          return;
        }
        const pointWeightsStatus = pointWeights
          ? bytes
              .slice(2 + (hotspot ? 3 : 0) + (localMoran ? 2 : 0) + (quantileLisa ? 5 : 0))
              .map(buffer => unsignedValues(buffer)[0])
          : undefined;
        if (pointWeightsStatus?.[0]) {
          this.currentSummary = null;
          this.currentError = `Point neighbor storage overflowed; ${pointWeightsStatus[1]} slots are required`;
          this.notify();
          this.requestUpdate();
          return;
        }
        if (localMoran && unsignedValues(bytes[3])[0] !== 0) {
          this.currentSummary = null;
          this.currentError =
            'Local Moran permutation capacity was exceeded; results are unavailable';
          this.notify();
          this.requestUpdate();
          return;
        }
        const quantileLisaStatus = quantileLisa
          ? {
              boundaries: Array.from(floatValues(bytes[2])),
              validCount: unsignedValues(bytes[3])[0],
              islandCount: unsignedValues(bytes[4])[0],
              invalidQuantile: unsignedValues(bytes[5])[0],
              overflow: unsignedValues(bytes[6])[0]
            }
          : undefined;
        if (
          quantileLisaStatus &&
          (quantileLisaStatus.invalidQuantile || quantileLisaStatus.overflow)
        ) {
          this.currentSummary = null;
          this.currentError = quantileLisaStatus.invalidQuantile
            ? 'Quantile LISA quantile settings are invalid'
            : 'Quantile LISA inference capacity was exceeded; results are unavailable';
          this.notify();
          this.requestUpdate();
          return;
        }
        const dbscanStatus = dbscan
          ? {
              clusterCount: unsignedValues(bytes[3])[0],
              noiseCount: unsignedValues(bytes[3])[1],
              validCount: unsignedValues(bytes[3])[2],
              excludedCount: unsignedValues(bytes[2])[2],
              complete: unsignedValues(bytes[2])[1] === 1
            }
          : undefined;
        if (dbscanStatus && !dbscanStatus.complete) {
          this.currentSummary = null;
          this.currentError = 'DBSCAN traversal did not complete; results are unavailable';
          this.notify();
          return;
        }
        let olsSummary: OLSAnalysisSummary | undefined;
        if (ols && recipe.operation.kind === 'ols') {
          const status = unsignedValues(bytes[6]);
          if (status[0] !== 0) {
            this.currentSummary = null;
            this.currentError =
              status[0] === 2
                ? 'OLS needs more complete observations than coefficients, including its intercept'
                : 'OLS design is singular or cannot be resolved at Float32 precision; remove constant or collinear predictors';
            this.notify();
            this.requestUpdate();
            return;
          }
          const {kind, ...settings} = recipe.operation;
          olsSummary = makeOLSAnalysisSummary(
            {
              coefficients: floatValues(bytes[2]),
              standardErrors: floatValues(bytes[3]),
              tStatistics: floatValues(bytes[4]),
              summary: floatValues(bytes[5]),
              status
            },
            settings
          );
        }
        if (this.summaryCancelledRevision === revision) return;
        const statistics = floatValues(bytes[0]);
        const counts = unsignedValues(bytes[1]);
        this.currentSummary = {
          ...(polygonWeightsStatus
            ? {
                polygonWeights: {
                  weightsOverflow: polygonWeightsStatus[0],
                  weightsRequiredCount: polygonWeightsStatus[1],
                  candidateOverflow: polygonWeightsStatus[2],
                  invalidCount: polygonWeightsStatus[3]
                }
              }
            : {}),
          ...(olsSummary ? {ols: olsSummary} : {}),
          ...(dbscanStatus ? {dbscan: dbscanStatus} : {}),
          ...(quantileLisaStatus ? {quantileLisa: quantileLisaStatus} : {}),
          ...(localMoran ? {localMoran: {invalidCount: unsignedValues(bytes[2])[0]}} : {}),
          ...(pointWeights?.path ? {pointWeightsPath: pointWeights.path} : {}),
          ...(pointWeightsStatus
            ? {
                pointWeights: {
                  weightsOverflow: pointWeightsStatus[0],
                  weightsRequiredCount: pointWeightsStatus[1],
                  invalidCount: pointWeightsStatus[2]
                }
              }
            : {}),
          ...(hotspotStatus
            ? {
                hotspot: {
                  weightsOverflow: hotspotStatus[0],
                  weightsRequiredCount: hotspotStatus[1],
                  invalidCount: hotspotStatus[2]
                }
              }
            : {}),
          datasetId: recipe.source.datasetId,
          column: this.requestedSettings.column,
          ...(recipe.operation.kind === 'excess-risk' ||
          recipe.operation.kind === 'spatial-rate-points' ||
          recipe.operation.kind === 'spatial-rate-polygons'
            ? {denominatorColumn: recipe.operation.populations}
            : {}),
          revision,
          sourceRevision: recipe.source.revision,
          selectionRevision: recipe.selectionRevision,
          operation: this.requestedSettings.mode,
          numericalPolicy: 'float32',
          count: counts[0],
          nullCount: counts[1],
          minimum: statistics[GPU_COLUMN_PROFILE_STATISTIC.minimum],
          maximum: statistics[GPU_COLUMN_PROFILE_STATISTIC.maximum],
          mean: statistics[GPU_COLUMN_PROFILE_STATISTIC.mean],
          ...(classified ? {classCount: unsignedValues(bytes[4])[0]} : {}),
          breaks: classified
            ? Array.from(floatValues(bytes[2])).slice(0, unsignedValues(bytes[4])[0] + 1)
            : [],
          classCounts: classified
            ? Array.from(unsignedValues(bytes[3])).slice(0, unsignedValues(bytes[4])[0])
            : []
        };
        this.summaryRevision = revision;
        this.notify();
        if (olsSummary || localMoran || pointWeights || polygonWeights) this.requestUpdate();
      } finally {
        tracked.finish();
        releasePointWeightsRead?.();
        releasePolygonWeightsRead?.();
        this.summaryPending = false;
        // A newer frame can finish while the previous asynchronous read is pending. Refresh its
        // compact summary even if no further render frame is scheduled.
        if (
          this.isEncodedCurrent() &&
          (revision !== this.encodedRevision || recipe !== this.encodedRecipe)
        ) {
          queueMicrotask(() => {
            void this.readSummary().catch(error => {
              if (this.destroyed) return;
              this.currentError = error instanceof Error ? error.message : String(error);
              this.notify();
            });
          });
        }
      }
    })();
    return this.trackRead(pending);
  }

  /** Explicit four-byte result inspection for the already-picked original source row. */
  readRow(sourceRowIndex: number): Promise<AnalysisRow | null> {
    if (this.requestedSettings.mode === 'dorling-cartogram')
      return (
        this.cartogram
          ?.readRow(sourceRowIndex)
          .then(row =>
            row ? {...row, kind: 'dorling-cartogram' as const, value: row.radius ?? NaN} : null
          ) ?? Promise.resolve(null)
      );
    if (!this.active || !this.isEncodedCurrent() || this.activeReads.size >= 4)
      return Promise.resolve(null);
    if (!Number.isInteger(sourceRowIndex) || sourceRowIndex < 0 || sourceRowIndex >= this.rowCount)
      return Promise.reject(new Error('Invalid analysis source row'));
    const revision = this.encodedRevision;
    const recipe = this.encodedRecipe!;
    const floatingResult = recipe.operation.kind !== 'classify';
    const pending = (async (): Promise<AnalysisRow | null> => {
      const bytes = await (floatingResult ? this.resultValues : this.resultClasses).readAsync(
        sourceRowIndex * 4,
        4
      );
      this.readbackBytes += bytes.byteLength;
      if (
        !this.isEncodedCurrent() ||
        revision !== this.encodedRevision ||
        recipe !== this.encodedRecipe
      )
        return null;
      return {
        sourceRowIndex,
        sourceRevision: recipe.source.revision,
        analysisRevision: revision,
        value: floatingResult ? floatValues(bytes)[0] : unsignedValues(bytes)[0],
        kind: isPolygonAnalysisOperation(recipe.operation)
          ? recipe.operation.kind
          : recipe.operation.kind === 'ols'
            ? 'ols'
            : recipe.operation.kind === 'dbscan-points'
              ? 'dbscan-points'
              : recipe.operation.kind === 'standardize'
                ? recipe.operation.method
                : recipe.operation.kind === 'excess-risk'
                  ? ('excess-risk' as const)
                  : recipe.operation.kind === 'hot-spots-lattice'
                    ? ('hot-spots-lattice' as const)
                    : recipe.operation.kind === 'spatial-rate-points'
                      ? ('spatial-rate-points' as const)
                      : recipe.operation.kind === 'hot-spots-points'
                        ? ('hot-spots-points' as const)
                        : recipe.operation.kind === 'local-moran-points' ||
                            recipe.operation.kind === 'multivariate-local-geary-points'
                          ? ('local-moran-points' as const)
                          : recipe.operation.kind === 'quantile-lisa-points'
                            ? ('quantile-lisa-points' as const)
                            : ('class' as const)
      };
    })();
    return this.trackRead(pending);
  }

  serializeRecipe(): string {
    const recipe = this.createRecipe();
    // The declaration is display-only, so it is saved with the recipe but never enters execution.
    return serializeKeplerAnalysisRecipe(
      this.requestedCartogramDeclaration && recipe.operation.kind === 'dorling-cartogram'
        ? {...recipe, display: {cartogramDeclaration: this.requestedCartogramDeclaration}}
        : recipe
    );
  }
  restoreRecipe(serialized: string, weightsReference?: {weightsId: string; version: number}): void {
    const recipe = restoreKeplerAnalysisRecipe(serialized, {
      resolveDatasetRevision: datasetId => this.session.registry.getDataset(datasetId)?.revision,
      resolveSelectionRevision: datasetId =>
        datasetId === this.lease.dataset.id ? this.explorer.currentRevision + 1 : undefined
    });
    if (recipe.source.datasetId !== this.lease.dataset.id)
      throw new Error('Recipe belongs to another dataset');
    this.update(
      {...this.requestedSettings, ...getAnalysisRecipeSettings(recipe)},
      !weightsReference,
      recipe.operation.kind === 'kriging-points' ? recipe.operation.column : undefined
    );
    if (weightsReference)
      this.restoreNamedWeightsReference(weightsReference.weightsId, weightsReference.version);
    this.valueColumnOverridden = true;
    this.denominatorColumnOverridden = true;
    this.restoredRecipeRevision++;
    this.notify();
    if (recipe.operation.kind === 'dorling-cartogram') {
      // A restored cartogram replaces the declaration, so one without it clears a previous overlay.
      const declaration = recipe.display?.cartogramDeclaration;
      this.setCartogramDeclaration(
        declaration && {
          metersPerUnit: declaration.metersPerUnit,
          planeOrigin: [declaration.planeOrigin[0], declaration.planeOrigin[1]],
          anchor: [declaration.anchor[0], declaration.anchor[1]]
        }
      );
    }
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.releaseDatasetContext?.();
    this.releaseWeightsListener?.();
    this.weightsStore.unbind(this.weightsConsumerId);
    this.resetCartogramInspection();
    this.listeners.clear();
    this.destruction = (async () => {
      try {
        await completion;
      } finally {
        await Promise.allSettled([...this.activeReads, ...this.retirementPromises]);
        this.program?.destroy();
        this.hotspotOutputs?.destroy();
        this.localMoranOutputs?.destroy();
        this.quantileLisaOutputs?.destroy();
        this.dbscanOutputs?.destroy();
        this.olsOutputs?.destroy();
        this.unregisterSurfaceProvider?.();
        interpolationSurfaceHover.clear();
        await this.interpolation?.destroy(completion);
        await this.cartogram?.destroy(completion);
        for (const program of this.retiredPrograms) program.destroy();
        this.retiredPrograms = [];
        this.releaseWeightsSource?.();
        this.streamedPointWeights?.destroy();

        this.colors.destroy();
        this.tableValues.destroy();
        this.tableClasses.destroy();
        this.tableValidity.destroy();
        for (const buffer of this.buffers) buffer.destroy();
        this.reservation?.release();
        await this.lease.release();
      }
    })();
    return this.destruction;
  }

  private createRecipe(settings: AnalysisSettings = this.requestedSettings): KeplerAnalysisRecipe {
    if (settings.mode === 'dorling-cartogram') {
      if (!settings.cartogram) throw new Error('Configure a cartogram before running it');
      const cartogram = normalizeCartogramSettings(
        {...settings.cartogram, valueColumn: settings.column},
        this.rowCount
      );
      return {
        schemaVersion: 1,
        algorithmVersion: 1,
        id: `kepler-analysis-${this.lease.dataset.id}`,
        source: {datasetId: this.lease.dataset.id, revision: this.lease.dataset.revision},
        population: settings.population,
        ...(settings.population === 'selected'
          ? {selectionRevision: this.explorer.currentRevision + 1}
          : {}),
        backendPolicy: 'require-gpu',
        operation: {kind: 'dorling-cartogram', ...cartogram}
      };
    }
    if (
      settings.mode === 'spatial-rate-polygons' ||
      settings.mode === 'hot-spots-polygons' ||
      settings.mode === 'local-moran-polygons' ||
      settings.mode === 'quantile-lisa-polygons'
    ) {
      if (!settings.polygon || settings.polygon.kind !== settings.mode)
        throw new Error('Configure polygon contiguity before polygon analysis');
      const operation = normalizePolygonAnalysisOperation(
        settings.polygon.kind === 'spatial-rate-polygons'
          ? {...settings.polygon, events: settings.column, populations: settings.denominatorColumn}
          : {...settings.polygon, column: settings.column}
      );
      return {
        schemaVersion: 1,
        algorithmVersion: 1,
        id: `kepler-analysis-${this.lease.dataset.id}`,
        source: {datasetId: this.lease.dataset.id, revision: this.lease.dataset.revision},
        population: settings.population,
        ...(settings.population === 'selected'
          ? {selectionRevision: this.explorer.currentRevision + 1}
          : {}),
        backendPolicy: 'require-gpu',
        operation
      };
    }
    this.assertCurrent();
    if (settings.mode === 'off') throw new Error('Enable an analysis before saving its recipe');
    const ols =
      settings.mode === 'ols' && settings.ols ? parseKeplerOLSSettings(settings.ols) : undefined;
    if (settings.mode === 'ols' && !ols)
      throw new Error('Configure response and predictor fields before OLS');
    const dbscan =
      settings.mode === 'dbscan-points' && settings.dbscan
        ? parseKeplerDBSCANSettings(settings.dbscan)
        : undefined;
    if (settings.mode === 'dbscan-points' && !dbscan)
      throw new Error('Configure geographic clustering before DBSCAN');
    const kriging =
      settings.mode === 'kriging-points' && settings.kriging
        ? parseKeplerKrigingSettings(settings.kriging)
        : undefined;
    if (settings.mode === 'kriging-points' && !kriging)
      throw new Error('Configure the interpolation grid and variogram before kriging');
    const idw =
      settings.mode === 'idw-points' && settings.idw
        ? parseKeplerIDWSettings(settings.idw)
        : undefined;
    if (settings.mode === 'idw-points' && !idw)
      throw new Error('Configure the interpolation grid before running IDW');
    const hotspot =
      settings.mode === 'hot-spots-lattice'
        ? settings.hotspot && normalizeHotspotSettings(settings.hotspot, this.rowCount)
        : undefined;
    const quantileLisa =
      settings.mode === 'quantile-lisa-points' && settings.quantileLisa
        ? normalizeQuantileLisaSettings(settings.quantileLisa, this.rowCount)
        : undefined;
    if (settings.mode === 'quantile-lisa-points' && !quantileLisa)
      throw new Error(
        'Configure explicit quantile and geographic point settings before Quantile LISA'
      );
    const localMoran =
      settings.mode === 'local-moran-points' && settings.localMoran
        ? normalizeLocalMoranSettings(settings.localMoran, this.rowCount)
        : undefined;
    if (settings.mode === 'local-moran-points' && !localMoran)
      throw new Error('Configure explicit geographic point weights before Local Moran analysis');
    const pointWeights =
      settings.mode === 'spatial-rate-points'
        ? settings.pointWeights &&
          normalizePointWeightsSettings(settings.pointWeights, this.rowCount)
        : undefined;
    const pointHotspot =
      settings.mode === 'hot-spots-points'
        ? settings.pointHotspot &&
          normalizePointHotspotSettings(settings.pointHotspot, this.rowCount)
        : undefined;
    if (settings.mode === 'hot-spots-points' && !pointHotspot)
      throw new Error('Configure explicit geographic point weights before hotspot analysis');
    if (settings.mode === 'spatial-rate-points' && !pointWeights)
      throw new Error('Configure explicit geographic point weights before spatial-rate analysis');
    if (settings.mode === 'hot-spots-lattice' && !hotspot)
      throw new Error('Explicitly configure the source row-major lattice before hotspot analysis');
    return {
      schemaVersion: 1,
      algorithmVersion: 1,
      id: `kepler-analysis-${this.lease.dataset.id}`,
      source: {datasetId: this.lease.dataset.id, revision: this.lease.dataset.revision},
      population: settings.population,
      ...(settings.population === 'selected'
        ? {selectionRevision: this.explorer.currentRevision + 1}
        : {}),
      backendPolicy: 'require-gpu',
      operation:
        settings.mode === 'ols'
          ? {kind: 'ols', ...ols!}
          : settings.mode === 'dbscan-points'
            ? {kind: 'dbscan-points', ...dbscan!}
            : settings.mode === 'kriging-points'
              ? {kind: 'kriging-points', column: settings.column, ...kriging!}
              : settings.mode === 'idw-points'
                ? {kind: 'idw-points', column: settings.column, ...idw!}
                : settings.mode === 'quantile-lisa-points'
                  ? {
                      kind: 'quantile-lisa-points',
                      column: settings.column,
                      weights: makePointWeightsRecipe(quantileLisa!),
                      quantileCount: quantileLisa!.quantileCount,
                      selectedQuantile: quantileLisa!.selectedQuantile,
                      seed: quantileLisa!.seed,
                      permutationCount: quantileLisa!.permutationCount,
                      significanceLevel: quantileLisa!.significanceLevel,
                      inference: 'legacy-directed-pseudo',
                      multipleTesting: 'none',
                      islands: 'exclude',
                      quantileMethod: 'type-5'
                    }
                  : settings.mode === 'local-moran-points' && localMoran!.variableColumns
                    ? {
                        kind: 'multivariate-local-geary-points',
                        columns: [...localMoran!.variableColumns],
                        weights: makePointWeightsRecipe(localMoran!),
                        significanceLevel: localMoran!.significanceLevel,
                        permutationCount: localMoran!.permutationCount!,
                        seed: localMoran!.seed!,
                        correction:
                          localMoran!.multipleTesting === 'bonferroni' ? 'bonferroni' : 'none',
                        tail: 'equal-tail'
                      }
                    : settings.mode === 'local-moran-points'
                      ? {
                          kind: 'local-moran-points',
                          column: settings.column,
                          ...(localMoran!.secondColumn === undefined
                            ? {}
                            : {secondColumn: localMoran!.secondColumn}),
                          weights: makePointWeightsRecipe(localMoran!),
                          minimumNeighbors: localMoran!.minimumNeighbors,
                          multipleTesting: localMoran!.multipleTesting,
                          islands: localMoran!.islands,
                          significanceLevel: localMoran!.significanceLevel,
                          inference:
                            localMoran!.inference ?? 'analytic-conditional-normal-two-sided',
                          ...(localMoran!.inference === 'permutation-conditional-two-sided'
                            ? {
                                permutationCount: localMoran!.permutationCount!,
                                seed: localMoran!.seed!
                              }
                            : {})
                        }
                      : settings.mode === 'hot-spots-points'
                        ? {
                            kind: 'hot-spots-points',
                            column: settings.column,
                            weights: makePointWeightsRecipe(pointHotspot!),
                            minimumNeighbors: pointHotspot!.minimumNeighbors,
                            selfWeight: pointHotspot!.selfWeight,
                            multipleTesting: pointHotspot!.multipleTesting,
                            islands: pointHotspot!.islands,
                            inference: 'analytic-two-sided'
                          }
                        : settings.mode === 'spatial-rate-points'
                          ? {
                              kind: 'spatial-rate-points',
                              events: settings.column,
                              populations: settings.denominatorColumn,
                              weights: makePointWeightsRecipe(pointWeights!),
                              scale: pointWeights!.scale,
                              islandPolicy: pointWeights!.islandPolicy
                            }
                          : settings.mode === 'hot-spots-lattice'
                            ? {
                                kind: 'hot-spots-lattice',
                                column: settings.column,
                                ...hotspot!,
                                inference: 'analytic-two-sided'
                              }
                            : settings.mode === 'excess-risk'
                              ? {
                                  kind: 'excess-risk',
                                  events: settings.column,
                                  populations: settings.denominatorColumn
                                }
                              : isStandardizationMode(settings.mode)
                                ? {
                                    kind: 'standardize',
                                    column: settings.column,
                                    method: settings.mode,
                                    variance: settings.variance
                                  }
                                : {
                                    kind: 'classify',
                                    column: settings.column,
                                    method: settings.mode,
                                    classCount: settings.classCount,
                                    naturalBreaksMode: settings.naturalBreaksMode
                                  }
    };
  }

  private createProgram(recipe: KeplerAnalysisRecipe, key: string): AnalysisProgram {
    const device = this.session.device;
    const reservation = this.session.registry.reserveMemory(
      this.rowCount * 128 +
        4 * 1024 * 1024 +
        (recipe.operation.kind === 'ols'
          ? this.rowCount * recipe.operation.predictorColumns.length * 8
          : 0) +
        (recipe.operation.kind === 'classify' &&
        recipe.operation.method === 'natural-breaks' &&
        recipe.operation.naturalBreaksMode === 'exact'
          ? this.rowCount * this.rowCount * 4
          : 0) +
        (recipe.operation.kind === 'hot-spots-lattice'
          ? getHotspotScratchBytes(recipe.operation, this.rowCount)
          : 0) +
        (recipe.operation.kind === 'hot-spots-points' ||
        recipe.operation.kind === 'hot-spots-polygons' ||
        recipe.operation.kind === 'local-moran-points' ||
        recipe.operation.kind === 'multivariate-local-geary-points' ||
        recipe.operation.kind === 'local-moran-polygons' ||
        isQuantileLisaMode(recipe.operation.kind)
          ? this.rowCount * 64
          : 0),
      'gpu',
      'scratch'
    );
    const graph = new GPUCommandGraph(device, {id: `kepler-analysis-${this.revision}`});
    let compiled: CompiledGPUCommandGraph | undefined;
    let result: ReturnType<typeof addKeplerAnalysisRecipe> | undefined;
    let colorParameters: GPUParameterBuffer<'float32'> | undefined;
    let pointWeights: AnalysisPointWeightsLease | undefined;
    let polygonWeights: AnalysisPolygonWeightsLease | undefined;
    try {
      // DBSCAN consumes coordinates without selecting one as the analysis value.
      const sourceColumn =
        recipe.operation.kind === 'dbscan-points'
          ? recipe.operation.longitudeColumn
          : this.requestedSettings.column;
      // Global statistics use explicit transient packing; borrowed source chunks remain unchanged.
      const values = packAnalysisVector(
        graph,
        'global-values',
        graph.importGPUVector(
          'source',
          borrowFloat32Column(this.lease.dataset.value.columns[sourceColumn])
        )
      );
      const validity =
        recipe.operation.kind === 'dbscan-points'
          ? undefined
          : this.lease.dataset.value.analytics.validity[this.requestedSettings.column];
      let validityMask = validity
        ? packAnalysisVector(graph, 'global-validity', graph.importGPUVector('validity', validity))
        : undefined;
      const columns: Record<string, GraphDataView<'float32'>> = Object.create(null);
      columns[sourceColumn] = values;
      if (
        (recipe.operation.kind === 'excess-risk' ||
          recipe.operation.kind === 'spatial-rate-points' ||
          recipe.operation.kind === 'spatial-rate-polygons') &&
        recipe.operation.populations !== recipe.operation.events
      ) {
        const populationColumn = recipe.operation.populations;
        columns[populationColumn] = packAnalysisVector(
          graph,
          'global-populations',
          graph.importGPUVector(
            'population-source',
            borrowFloat32Column(this.lease.dataset.value.columns[populationColumn])
          )
        );
        const populationValidity = this.lease.dataset.value.analytics.validity[populationColumn];
        if (populationValidity) {
          const populationMask = packAnalysisVector(
            graph,
            'global-population-validity',
            graph.importGPUVector('population-validity', populationValidity)
          );
          if (validityMask) {
            const combinedValidity = createTransientView(
              graph,
              'rate-validity',
              'uint32',
              this.rowCount
            );
            graph.add(
              new GPUMask({
                id: 'rate-valid-inputs',
                inputs: [validityMask, populationMask],
                output: combinedValidity
              })
            );
            validityMask = combinedValidity;
          } else validityMask = populationMask;
        }
      }
      if (recipe.operation.kind === 'multivariate-local-geary-points') {
        for (const [variableIndex, variableColumn] of recipe.operation.columns.entries()) {
          if (!columns[variableColumn])
            columns[variableColumn] = packAnalysisVector(
              graph,
              `geary-values-${variableIndex}`,
              graph.importGPUVector(
                `geary-source-${variableIndex}`,
                borrowFloat32Column(this.lease.dataset.value.columns[variableColumn])
              )
            );
          const variableValidity = this.lease.dataset.value.analytics.validity[variableColumn];
          if (!variableValidity) continue;
          const variableMask = packAnalysisVector(
            graph,
            `geary-validity-${variableIndex}`,
            graph.importGPUVector(`geary-null-${variableIndex}`, variableValidity)
          );
          if (validityMask) {
            const combinedValidity = createTransientView(
              graph,
              `geary-complete-${variableIndex}`,
              'uint32',
              this.rowCount
            );
            graph.add(
              new GPUMask({
                id: `geary-valid-${variableIndex}`,
                inputs: [validityMask, variableMask],
                output: combinedValidity
              })
            );
            validityMask = combinedValidity;
          } else validityMask = variableMask;
        }
      }
      if (recipe.operation.kind === 'local-moran-points' && recipe.operation.secondColumn) {
        const secondColumn = recipe.operation.secondColumn;
        columns[secondColumn] = packAnalysisVector(
          graph,
          'moran-second-values',
          graph.importGPUVector(
            'moran-second-source',
            borrowFloat32Column(this.lease.dataset.value.columns[secondColumn])
          )
        );
        const secondValidity = this.lease.dataset.value.analytics.validity[secondColumn];
        if (secondValidity) {
          const secondMask = packAnalysisVector(
            graph,
            'moran-second-validity',
            graph.importGPUVector('moran-second-null', secondValidity)
          );
          if (validityMask) {
            const combinedValidity = createTransientView(
              graph,
              'moran-bivariate-validity',
              'uint32',
              this.rowCount
            );
            graph.add(
              new GPUMask({
                id: 'moran-bivariate-valid-inputs',
                inputs: [validityMask, secondMask],
                output: combinedValidity
              })
            );
            validityMask = combinedValidity;
          } else validityMask = secondMask;
        }
      }
      if (recipe.operation.kind === 'ols') {
        for (const [predictorIndex, column] of recipe.operation.predictorColumns.entries()) {
          columns[column] = packAnalysisVector(
            graph,
            `ols-values-${predictorIndex}`,
            graph.importGPUVector(
              `ols-source-${predictorIndex}`,
              borrowFloat32Column(this.lease.dataset.value.columns[column])
            )
          );
          const columnValidity = this.lease.dataset.value.analytics.validity[column];
          if (!columnValidity) continue;
          const columnMask = packAnalysisVector(
            graph,
            `ols-validity-${predictorIndex}`,
            graph.importGPUVector(`ols-null-${predictorIndex}`, columnValidity)
          );
          if (validityMask) {
            const combined = createTransientView(
              graph,
              `ols-complete-${predictorIndex}`,
              'uint32',
              this.rowCount
            );
            graph.add(
              new GPUMask({
                id: `ols-valid-${predictorIndex}`,
                inputs: [validityMask, columnMask],
                output: combined
              })
            );
            validityMask = combined;
          } else validityMask = columnMask;
        }
      }
      if (recipe.operation.kind === 'dbscan-points') {
        const coordinateMasks: GraphDataView<'uint32'>[] = [];
        for (const [coordinateIndex, column] of [
          ...new Set([recipe.operation.longitudeColumn, recipe.operation.latitudeColumn])
        ].entries()) {
          if (!columns[column])
            columns[column] = packAnalysisVector(
              graph,
              `dbscan-values-${coordinateIndex}`,
              graph.importGPUVector(
                `dbscan-source-${coordinateIndex}`,
                borrowFloat32Column(this.lease.dataset.value.columns[column])
              )
            );
          const coordinateValidity = this.lease.dataset.value.analytics.validity[column];
          if (coordinateValidity)
            coordinateMasks.push(
              packAnalysisVector(
                graph,
                `dbscan-validity-${coordinateIndex}`,
                graph.importGPUVector(`dbscan-null-${coordinateIndex}`, coordinateValidity)
              )
            );
        }
        if (coordinateMasks.length) {
          validityMask = createTransientView(
            graph,
            'dbscan-coordinate-validity',
            'uint32',
            this.rowCount
          );
          graph.add(
            new GPUMask({
              id: 'dbscan-valid-coordinates',
              inputs: coordinateMasks,
              output: validityMask
            })
          );
        }
      }
      const selectionMask = packAnalysisVector(
        graph,
        'global-selection',
        graph.importGPUVector('selection', this.explorer.selectionMask)
      );
      let mask = selectionMask;
      if (validityMask && recipe.population === 'selected') {
        mask = createTransientView(graph, 'valid-selected-population', 'uint32', this.rowCount);
        graph.add(
          new GPUMask({
            id: 'analysis-valid-selection',
            inputs: [selectionMask, validityMask],
            output: mask
          })
        );
      }
      const resultValues = importBuffer(
        graph,
        'result-values',
        this.resultValues,
        'float32',
        this.rowCount
      );
      const breaks = importBuffer(
        graph,
        'breaks',
        this.breaks,
        'float32',
        recipe.operation.kind === 'classify' ? recipe.operation.classCount + 1 : 2
      );
      const classCount = importBuffer(graph, 'class-count', this.classCount, 'uint32', 1);
      const classCounts = importBuffer(
        graph,
        'class-counts',
        this.classCounts,
        'uint32',
        recipe.operation.kind === 'classify' ? recipe.operation.classCount : 8
      );
      if (
        recipe.operation.kind === 'hot-spots-lattice' ||
        recipe.operation.kind === 'hot-spots-points' ||
        recipe.operation.kind === 'hot-spots-polygons'
      )
        this.hotspotOutputs ??= new AnalysisHotspotOutputs(this.session, this.sourceChunkLengths);
      if (
        recipe.operation.kind === 'local-moran-points' ||
        recipe.operation.kind === 'multivariate-local-geary-points' ||
        recipe.operation.kind === 'local-moran-polygons'
      )
        this.localMoranOutputs ??= new AnalysisLocalMoranOutputs(
          this.session,
          this.sourceChunkLengths
        );
      if (isQuantileLisaMode(recipe.operation.kind))
        this.quantileLisaOutputs ??= new AnalysisQuantileLisaOutputs(
          this.session,
          this.sourceChunkLengths
        );
      if (recipe.operation.kind === 'ols')
        this.olsOutputs ??= new AnalysisOLSOutputs(this.session, this.sourceChunkLengths);
      const olsViews =
        recipe.operation.kind === 'ols'
          ? this.olsOutputs!.importToGraph(
              graph,
              resultValues,
              recipe.operation.predictorColumns.length
            )
          : undefined;
      if (recipe.operation.kind === 'dbscan-points')
        this.dbscanOutputs ??= new AnalysisDBSCANOutputs(this.session, this.sourceChunkLengths);
      const dbscanViews =
        recipe.operation.kind === 'dbscan-points'
          ? this.dbscanOutputs!.importToGraph(graph, resultValues)
          : undefined;
      const quantileLisaViews = isQuantileLisaMode(recipe.operation.kind)
        ? this.quantileLisaOutputs!.importToGraph(graph, resultValues)
        : undefined;
      const hotspotViews =
        recipe.operation.kind === 'hot-spots-lattice' ||
        recipe.operation.kind === 'hot-spots-points' ||
        recipe.operation.kind === 'hot-spots-polygons'
          ? this.hotspotOutputs!.importToGraph(graph, resultValues)
          : undefined;
      const localMoranViews =
        recipe.operation.kind === 'local-moran-points' ||
        recipe.operation.kind === 'multivariate-local-geary-points' ||
        recipe.operation.kind === 'local-moran-polygons'
          ? this.localMoranOutputs!.importToGraph(graph, resultValues)
          : undefined;
      if (
        recipe.operation.kind === 'spatial-rate-points' ||
        recipe.operation.kind === 'hot-spots-points' ||
        recipe.operation.kind === 'local-moran-points' ||
        recipe.operation.kind === 'multivariate-local-geary-points' ||
        recipe.operation.kind === 'quantile-lisa-points'
      ) {
        const streamedSettings = getStreamedWeightsSettings(recipe.operation);
        const weightsPath = streamedSettings
          ? choosePointWeightsPath(
              this.session.device.limits,
              streamedSettings,
              this.streamedWeightsOptions
            )
          : undefined;
        if (weightsPath?.path === 'streamed') {
          // isAwaitingStreamedWeights() built the chunks before this program was created.
          pointWeights = this.streamedPointWeights!.acquire(graph, recipe.operation.weights);
          this.streamedPointWeights!.retireOthers(recipe.operation.weights);
        } else {
          this.pointWeightsCache ??= this.weightsStore.pointCache(this.lease.dataset);
          pointWeights = this.pointWeightsCache.acquire(graph, recipe.operation.weights);
        }
      }
      if (isPolygonAnalysisOperation(recipe.operation)) {
        if (!this.polygonSource) throw new Error('Polygon source is unavailable');
        this.polygonWeightsCache ??= this.weightsStore.polygonCache(this.lease.dataset);
        polygonWeights = this.polygonWeightsCache.acquire(graph, recipe.operation.weights);
      }
      result = addKeplerAnalysisRecipe(graph, recipe, {
        source: recipe.source,
        columns,
        validityMask,
        pointWeights: pointWeights?.binding,
        polygonWeights: polygonWeights?.binding,
        mask,
        selectionRevision: this.explorer.currentRevision + 1,
        outputs: {
          values: resultValues,
          breaks,
          classCount,
          classCounts,
          ...(olsViews ? {ols: olsViews} : {}),
          ...(dbscanViews ? {dbscan: dbscanViews} : {}),
          ...(quantileLisaViews ? {quantileLisa: quantileLisaViews} : {}),
          ...(localMoranViews ? {localMoran: localMoranViews} : {}),
          ...(hotspotViews ? {hotSpots: hotspotViews} : {})
        }
      });
      const dbscanColorIndices = dbscanViews
        ? this.dbscanOutputs!.addColorIndices(graph, dbscanViews.labels!)
        : undefined;
      const classified = recipe.operation.kind === 'classify';
      const profileValues = classified ? values : resultValues;
      const effectiveMask =
        'effectiveMask' in result
          ? result.effectiveMask
          : recipe.population === 'selected'
            ? mask
            : validityMask;
      // Publish validity beside derived values so excluded source rows cannot expose stale results.
      if (effectiveMask)
        graph.add(
          new GPUMask({
            id: 'analysis-result-validity',
            inputs: [effectiveMask],
            output: importBuffer(
              graph,
              'result-validity',
              this.resultValidity,
              'uint32',
              this.rowCount
            )
          })
        );
      else this.resultValidity.write(new Uint32Array(this.rowCount).fill(1));
      const statistics = importBuffer(
        graph,
        'statistics',
        this.statistics,
        'float32',
        GPU_COLUMN_PROFILE_STATISTIC_COUNT
      );
      graph.add(
        new GPUColumnProfile({
          id: 'analysis-profile',
          columns: [{values: profileValues}],
          mask: effectiveMask,
          histogramBinCount: 1,
          output: {
            statistics,
            counts: importBuffer(graph, 'counts', this.counts, 'uint32', 2)
          }
        })
      );
      let colorDomain = classified
        ? breaks
        : importBuffer(graph, 'standard-domain', this.standardDomain, 'float32', 2);
      if (
        recipe.operation.kind === 'ols' ||
        recipe.operation.kind === 'excess-risk' ||
        recipe.operation.kind === 'spatial-rate-points' ||
        recipe.operation.kind === 'spatial-rate-polygons' ||
        (recipe.operation.kind === 'standardize' && recipe.operation.method !== 'z-score')
      ) {
        // The compact output profile supplies the displayed domain without a CPU readback.
        colorDomain = createTransientView(
          graph,
          'output-domain',
          'float32',
          2,
          Buffer.STORAGE | Buffer.COPY_DST
        );
        const domain = colorDomain;
        graph.addCopyPass({
          id: 'analysis-output-domain',
          resources: [
            {buffer: statistics, usage: 'copy-source'},
            {buffer: domain, usage: 'copy-destination'}
          ],
          compile: () => ({
            encode: ({commandEncoder, getBuffer}) =>
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: getBuffer(statistics),
                sourceOffset: statistics.byteOffset + GPU_COLUMN_PROFILE_STATISTIC.minimum * 4,
                destinationBuffer: getBuffer(domain),
                destinationOffset: domain.byteOffset,
                size: 8
              })
          })
        });
      }
      const inferenceColorIndices = hotspotViews
        ? addAnalysisInferenceColorIndices(graph, hotspotViews.bins!, hotspotViews.pValues!, true)
        : localMoranViews &&
            (recipe.operation.kind === 'local-moran-points' ||
              recipe.operation.kind === 'local-moran-polygons')
          ? addAnalysisInferenceColorIndices(
              graph,
              localMoranViews.quadrants!,
              localMoranViews.pValues!,
              false
            )
          : undefined;
      const rampPalettes = importBuffer(graph, 'palette', this.palette, 'uint32', 72);
      const activePalette = classified
        ? addAnalysisRampPalette(graph, rampPalettes, classCount)
        : graph.createDataView(rampPalettes.buffer, {
            format: 'uint32',
            byteOffset: 7 * 8 * 4,
            length: 8
          });
      const hotspotPalette = hotspotViews
        ? graph.createDataView(rampPalettes.buffer, {
            format: 'uint32',
            byteOffset: 64 * 4,
            length: 7
          })
        : undefined;
      colorParameters = new GPUParameterBuffer(device, {
        id: 'kepler-analysis-color-parameters',
        format: 'float32',
        length: 9,
        values: getGPUColorScaleParameterValues({
          scale:
            hotspotViews || dbscanViews || localMoranViews || quantileLisaViews
              ? 'ordinal'
              : classified
                ? 'threshold'
                : 'linear',
          domainCount: classified ? this.requestedSettings.classCount + 1 : 2,
          paletteCount: hotspotViews
            ? 7
            : dbscanViews
              ? 16
              : quantileLisaViews
                ? 4
                : localMoranViews
                  ? 6
                  : classified
                    ? this.requestedSettings.classCount
                    : 8,
          interpolation: 'linear',
          ...(hotspotViews || localMoranViews
            ? {noDataColor: packGPUColor(...getKeplerResultToneColor('excluded'))}
            : {}),
          clamp: true
        })
      });
      graph.add(
        new GPUColorScale({
          id: 'analysis-colors',
          values:
            inferenceColorIndices ??
            dbscanColorIndices ??
            quantileLisaViews?.classes ??
            localMoranViews?.quadrants ??
            profileValues,
          mask: effectiveMask,
          domain: colorDomain,
          ...(classified ? {domainCount: classCount} : {}),
          palette:
            hotspotPalette ??
            (dbscanViews
              ? importBuffer(graph, 'dbscan-palette', this.dbscanOutputs!.palette, 'uint32', 16)
              : quantileLisaViews
                ? importBuffer(
                    graph,
                    'quantile-lisa-palette',
                    this.quantileLisaOutputs!.palette,
                    'uint32',
                    4
                  )
                : localMoranViews
                  ? importBuffer(
                      graph,
                      'moran-palette',
                      this.localMoranOutputs!.palette,
                      'uint32',
                      6
                    )
                  : activePalette),
          parameters: colorParameters.importToGraph(graph),
          maximumDomainCount: classified ? this.requestedSettings.classCount + 1 : 2,
          maximumPaletteCount: hotspotViews
            ? 7
            : dbscanViews
              ? 16
              : quantileLisaViews
                ? 4
                : localMoranViews
                  ? 6
                  : 8,
          output: {
            colors: importBuffer(
              graph,
              'result-colors',
              this.resultColors,
              'uint32',
              this.rowCount
            ),
            classIndices: importBuffer(
              graph,
              'result-classes',
              this.resultClasses,
              'uint32',
              this.rowCount
            )
          }
        })
      );
      compiled = graph.compile();
      const parameterBytes =
        colorParameters.buffer.byteLength +
        (recipe.operation.kind === 'ols' ? 4 : 0) +
        ((recipe.operation.kind === 'local-moran-points' ||
          recipe.operation.kind === 'local-moran-polygons') &&
        recipe.operation.inference === 'permutation-conditional-two-sided'
          ? 16
          : 0) +
        (isQuantileLisaMode(recipe.operation.kind) ? 24 : 0) +
        (recipe.operation.kind === 'classify' ? (recipe.operation.classCount + 9) * 4 : 0) +
        (recipe.operation.kind === 'hot-spots-lattice' ||
        recipe.operation.kind === 'hot-spots-points' ||
        recipe.operation.kind === 'hot-spots-polygons' ||
        recipe.operation.kind === 'local-moran-points' ||
        recipe.operation.kind === 'multivariate-local-geary-points' ||
        recipe.operation.kind === 'local-moran-polygons'
          ? GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH * 4
          : 0) +
        ('pointWeights' in result ? result.pointWeights.parameterByteLength : 0);
      if (compiled.stats.physicalTransientResourceBytes + parameterBytes > reservation.byteLength) {
        throw new KeplerAllocationError('memory-budget', {
          message: 'Compiled analysis storage exceeds its memory reservation'
        });
      }
      const ownedCompiled = compiled;
      const ownedResult = result;
      const ownedColorParameters = colorParameters;
      const ownedPointWeights = pointWeights;
      const ownedPolygonWeights = polygonWeights;
      return {
        key,
        compiled,
        pointWeights: ownedPointWeights,
        polygonWeights: ownedPolygonWeights,
        destroy() {
          ownedCompiled.destroy();
          ownedResult.destroyParameters();
          ownedColorParameters.destroy();
          ownedPointWeights?.release();
          ownedPolygonWeights?.release();
          reservation.release();
        }
      };
    } catch (error) {
      compiled?.destroy();
      result?.destroyParameters();
      colorParameters?.destroy();
      pointWeights?.release();
      polygonWeights?.release();
      reservation.release();
      throw error;
    }
  }

  /**
   * Decides the point-weights storage of `recipe`. When streamed row chunks are required (the
   * single CSR would exceed the device storage binding limit) and are not built yet, starts the
   * asynchronous build and returns true: the caller skips this encode and the build's completion
   * requests another. A failed build throws its error.
   */
  private isAwaitingStreamedWeights(recipe: KeplerAnalysisRecipe): boolean {
    const settings = getStreamedWeightsSettings(recipe.operation);
    if (!settings) return false;
    const path = choosePointWeightsPath(
      this.session.device.limits,
      settings,
      this.streamedWeightsOptions
    );
    if (path.path !== 'streamed') return false;
    this.streamedPointWeights ??= new AnalysisStreamedPointWeights(
      this.session,
      this.lease.dataset
    );
    const failure = this.streamedPointWeights.getFailure(settings);
    if (failure) throw new Error(failure);
    if (this.streamedPointWeights.isReady(settings)) return false;
    if (!this.streamedPointWeights.isPending(settings)) {
      this.streamedWeightsBuilding = true;
      this.notify();
      void this.streamedPointWeights
        .prepare(settings, {
          maximumSlotsPerChunk: path.maximumSlotsPerChunk,
          reason: path.reason
        })
        .finally(() => {
          this.streamedWeightsBuilding = false;
          if (this.destroyed) return;
          this.notify();
          this.requestUpdate();
        });
    }
    return true;
  }

  private update(
    settings: AnalysisSettings,
    manageWeightsReference = true,
    fittedResponseColumn = this.krigingFitResponseColumn
  ): void {
    this.assertCurrent();
    if (settings.mode === 'ols' && settings.ols)
      settings = {...settings, column: settings.ols.responseColumn};
    if (settings.mode === 'dorling-cartogram') {
      if (!settings.cartogram) throw new Error('Configure a cartogram before running it');
      const cartogram = normalizeCartogramSettings(settings.cartogram, this.rowCount);
      if (
        [cartogram.xColumn, cartogram.yColumn, cartogram.valueColumn].some(
          column => !this.availableColumns.includes(column)
        )
      )
        throw new Error('Cartograms require prepared Float32 coordinate and value fields');
      settings = {...settings, column: cartogram.valueColumn, cartogram};
    }
    if (
      settings.mode === 'spatial-rate-polygons' ||
      settings.mode === 'hot-spots-polygons' ||
      settings.mode === 'local-moran-polygons' ||
      settings.mode === 'quantile-lisa-polygons'
    ) {
      if (!settings.polygon || settings.polygon.kind !== settings.mode)
        throw new Error('Configure polygon contiguity before polygon analysis');
      const polygon = normalizePolygonAnalysisOperation(
        settings.polygon.kind === 'spatial-rate-polygons'
          ? {...settings.polygon, events: settings.column, populations: settings.denominatorColumn}
          : {...settings.polygon, column: settings.column}
      );
      if (!this.availablePolygonColumns.includes(polygon.weights.geometryColumn))
        throw new Error('Polygon analysis requires the retained published geometry field');
      settings = {...settings, polygon};
    }

    if (settings.kriging?.variogramFit && settings.column !== fittedResponseColumn) {
      settings = {
        ...settings,
        kriging: parseKeplerKrigingSettings({...settings.kriging, variogramFit: undefined})
      };
    }

    const requiresValue = settings.mode !== 'off' && settings.mode !== 'dbscan-points';
    if (
      (requiresValue || settings.column !== '') &&
      !this.availableColumns.includes(settings.column)
    )
      throw new Error('Choose a prepared Float32 analysis column');
    const requiresPopulation = [
      'excess-risk',
      'spatial-rate-points',
      'spatial-rate-polygons'
    ].includes(settings.mode);
    if (
      (requiresPopulation || settings.denominatorColumn !== '') &&
      !this.availableColumns.includes(settings.denominatorColumn)
    )
      throw new Error('Choose a prepared Float32 population column');
    if (
      settings.mode === 'spatial-rate-points' &&
      settings.pointWeights &&
      getPointCoordinateColumns(settings.pointWeights).some(
        column => !this.availableColumns.includes(column)
      )
    )
      throw new Error('Point weights require prepared Float32 coordinate columns');
    if (
      settings.mode === 'hot-spots-points' &&
      settings.pointHotspot &&
      getPointCoordinateColumns(settings.pointHotspot).some(
        column => !this.availableColumns.includes(column)
      )
    )
      throw new Error('Point hotspots require prepared Float32 coordinate columns');
    if (
      settings.mode === 'local-moran-points' &&
      settings.localMoran &&
      getPointCoordinateColumns(settings.localMoran).some(
        column => !this.availableColumns.includes(column)
      )
    )
      throw new Error('Local Moran requires prepared Float32 coordinate columns');
    if (
      settings.mode === 'quantile-lisa-points' &&
      settings.quantileLisa &&
      getPointCoordinateColumns(settings.quantileLisa).some(
        column => !this.availableColumns.includes(column)
      )
    )
      throw new Error('Quantile LISA requires prepared Float32 coordinate columns');
    if (
      settings.mode === 'kriging-points' &&
      settings.kriging &&
      (!this.availableColumns.includes(settings.kriging.longitudeColumn) ||
        !this.availableColumns.includes(settings.kriging.latitudeColumn))
    )
      throw new Error('Kriging requires prepared Float32 coordinate fields');
    if (
      settings.mode === 'idw-points' &&
      settings.idw &&
      (!this.availableColumns.includes(settings.idw.longitudeColumn) ||
        !this.availableColumns.includes(settings.idw.latitudeColumn))
    )
      throw new Error('Interpolation requires prepared Float32 coordinate fields');
    if (
      settings.mode === 'dbscan-points' &&
      settings.dbscan &&
      (!this.availableColumns.includes(settings.dbscan.longitudeColumn) ||
        !this.availableColumns.includes(settings.dbscan.latitudeColumn))
    )
      throw new Error('DBSCAN requires prepared Float32 coordinate fields');
    if (
      settings.mode === 'ols' &&
      settings.ols &&
      [settings.ols.responseColumn, ...settings.ols.predictorColumns].some(
        column => !this.availableColumns.includes(column)
      )
    )
      throw new Error('OLS requires prepared Float32 response and predictor fields');
    if (settings.mode !== 'off')
      validateAnalysisRecipeCapacity(this.createRecipe(settings), this.rowCount);
    if (settings.mode !== 'off') {
      this.valueColumnOverridden = true;
      this.denominatorColumnOverridden = true;
    }
    const previousOperation =
      this.requestedSettings.mode === 'off' ? undefined : this.createRecipe().operation;
    const previousWeights =
      previousOperation && 'weights' in previousOperation ? previousOperation.weights : undefined;
    if (JSON.stringify(settings) === JSON.stringify(this.requestedSettings)) {
      // Legacy replay can remove the weights store while leaving this recipe unchanged.
      if (manageWeightsReference) this.synchronizeWeightsReference(previousWeights);
      return;
    }
    this.requestedSettings = {
      ...settings,
      ...(settings.cartogram
        ? {cartogram: normalizeCartogramSettings(settings.cartogram, this.rowCount)}
        : {}),
      ...(settings.polygon ? {polygon: normalizePolygonAnalysisOperation(settings.polygon)} : {}),
      ...(settings.ols ? {ols: parseKeplerOLSSettings(settings.ols)} : {}),
      ...(settings.dbscan ? {dbscan: parseKeplerDBSCANSettings(settings.dbscan)} : {}),
      ...(settings.kriging ? {kriging: parseKeplerKrigingSettings(settings.kriging)} : {}),
      ...(settings.idw ? {idw: parseKeplerIDWSettings(settings.idw)} : {}),
      ...(settings.quantileLisa
        ? {quantileLisa: normalizeQuantileLisaSettings(settings.quantileLisa, this.rowCount)}
        : {}),
      ...(settings.localMoran
        ? {localMoran: normalizeLocalMoranSettings(settings.localMoran, this.rowCount)}
        : {}),
      ...(settings.hotspot ? {hotspot: {...settings.hotspot}} : {}),
      ...(settings.pointWeights
        ? {pointWeights: normalizePointWeightsSettings(settings.pointWeights, this.rowCount)}
        : {}),
      ...(settings.pointHotspot
        ? {pointHotspot: normalizePointHotspotSettings(settings.pointHotspot, this.rowCount)}
        : {})
    };
    this.krigingFitResponseColumn = this.requestedSettings.kriging?.variogramFit
      ? fittedResponseColumn
      : undefined;
    if (manageWeightsReference) this.synchronizeWeightsReference(previousWeights);
    this.interpolation?.invalidate();
    this.resetCartogramInspection();
    this.cartogram?.invalidate();
    this.revision++;
    this.currentSummary = null;
    this.currentError = '';
    this.notify();
    this.requestUpdate();
  }
  private synchronizeWeightsReference(previousWeights?: AnalysisWeightsSettings): void {
    if (this.requestedSettings.mode !== 'off') {
      const operation = this.createRecipe().operation;
      if ('weights' in operation) {
        const selected = this.selectedWeightsId && this.weightsStore.get(this.selectedWeightsId);
        // Value, inference and population edits do not silently accept a newer topology version.
        if (
          !selected ||
          !this.weightsReference ||
          JSON.stringify(previousWeights) !== JSON.stringify(operation.weights)
        ) {
          const definition =
            selected && JSON.stringify(selected.settings) === JSON.stringify(operation.weights)
              ? selected
              : this.weightsStore.findOrCreate(this.namedWeightsSource, operation.weights);
          this.selectedWeightsId = definition.id;
          this.weightsStore.bind(this.weightsConsumerId, definition.id);
        }
      } else this.clearNamedWeightsReference();
    } else this.clearNamedWeightsReference();
  }
  private getResultColumnName(base: string): string {
    let name = base;
    while (Object.hasOwn(this.lease.dataset.value.columns, name)) name += '_';
    return name;
  }
  private isEncodedCurrent(): boolean {
    return (
      this.enabled &&
      !this.weightsStaleReason &&
      this.encodedRevision === this.revision &&
      (this.requestedSettings.population === 'all' ||
        this.encodedSelectionRevision === this.explorer.currentRevision + 1)
    );
  }
  private assertCurrent(): void {
    if (this.destroyed || !this.lease.isValid() || !this.explorer.isCurrent)
      throw new Error('Analysis source is unavailable or stale');
  }
  private notify(): void {
    for (const listener of this.listeners) listener();
  }
  private trackRead<Result>(promise: Promise<Result>): Promise<Result> {
    this.activeReads.add(promise);
    void promise.finally(() => this.activeReads.delete(promise)).catch(() => {});
    return promise;
  }
  private retireSubmittedPrograms(): void {
    if (!this.retiredPrograms.length) return;
    const programs = this.retiredPrograms.splice(0);
    const fence = this.session.device.createFence();
    const completion = fence.signaled.finally(() => {
      for (const program of programs) program.destroy();
      fence.destroy();
    });
    this.retirementPromises.add(completion);
    void completion.finally(() => this.retirementPromises.delete(completion)).catch(() => {});
  }
}

function borrowFloat32Column(column: GPUVector | undefined): GPUVector<'float32'> {
  if (!column || column.format !== 'float32')
    throw new Error('Analysis requires a prepared Float32 column');
  // Generic narrowing is local to this validated adapter boundary; no data conversion occurs.
  return column as GPUVector<'float32'>;
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
function floatValues(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
function unsignedValues(bytes: Uint8Array): Uint32Array {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
