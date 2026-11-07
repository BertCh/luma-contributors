// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture, type CommandEncoder} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import {BitmapLayer} from '@deck.gl/layers';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  createTransientView,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUColorScale,
  GPUColumnProfile,
  GPUParameterBuffer,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '@luma-contributors/graph/gpu-dataframe';
import {
  addKeplerAnalysisRecipe,
  getKrigingConfidenceInterval,
  parseKeplerAnalysisRecipe,
  summarizeKrigingCrossValidation,
  type KeplerAnalysisRecipe,
  type KeplerKrigingCrossValidationSummary,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type KeplerMemoryReservation,
  type KeplerSessionExplorer
} from '@luma-contributors/kepler';
import {packAnalysisVector} from './analysis-point-weights-cache';
import {readFloat32Vector} from './kriging-diagnostics';
import {encodeInterpolationGeoTIFF} from './interpolation-geotiff';
import {getInterpolationCellIndex} from './interpolation-tooltip';

type InterpolationRecipe = KeplerAnalysisRecipe & {
  operation: Extract<KeplerAnalysisRecipe['operation'], {kind: 'idw-points' | 'kriging-points'}>;
};
export type InterpolationGridExport = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly filename: string;
  readonly readbackBytes: number;
  isCurrent(): boolean;
  destroy(): void;
};
export type InterpolationCell = {
  cellIndex: number;
  longitude: number;
  latitude: number;
  value: number | null;
} & (
  | {operation?: 'idw-points'; neighbors: number}
  | {
      operation: 'kriging-points';
      variance: number | null;
      /** Prediction interval from the kriging variance when the recipe declares a level. */
      confidenceLevel?: number;
      lower?: number | null;
      upper?: number | null;
    }
);
export type InterpolationMetric = 'prediction' | 'variance';
export type KrigingSummary = {
  invalidCount: number;
  uncertainCount: number;
  variance: {count: number; nullCount: number; minimum: number; maximum: number; mean: number};
  /** Interval half-widths at the minimum, mean and maximum variance (not a mean half-width). */
  interval?: {
    level: number;
    multiplier: number;
    halfWidthAtMinimumVariance: number;
    halfWidthAtMeanVariance: number;
    halfWidthAtMaximumVariance: number;
  };
  /** Leave-one-out diagnostics over the retained samples. */
  crossValidation?: KeplerKrigingCrossValidationSummary;
};
export type InterpolationSummary = {
  datasetId: string;
  column: string;
  revision: number;
  sourceRevision: number;
  selectionRevision?: number;
  operation: 'idw-points' | 'kriging-points';
  numericalPolicy: 'float32';
  count: number;
  nullCount: number;
  minimum: number;
  maximum: number;
  mean: number;
  breaks: readonly number[];
  classCounts: readonly number[];
  kriging?: KrigingSummary;
  interpolation: {width: number; height: number; validSourceCount: number; cellCount: number};
};
type Program = {
  key: string;
  recipe: InterpolationRecipe;
  compiled: CompiledGPUCommandGraph;
  values: Buffer;
  counts?: Buffer;
  variance?: Buffer;
  varianceStatistics?: Buffer;
  crossValidation?: {prediction: Buffer; variance: Buffer};
  statistics: Buffer;
  diagnostics: Buffer;
  texture: Texture;
  revision: number;
  reads: Set<Promise<unknown>>;
  lastEncoder?: CommandEncoder;
  submitted: boolean;
  destroy(): void;
};

/** A separate cell-indexed result, never attached to source-row table or picking identities. */
export class InterpolationRuntime {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private current?: Program;
  private retired: Program[] = [];
  private retirements = new Set<Promise<void>>();
  private pendingSummary?: Promise<void>;
  private currentSummary: InterpolationSummary | null = null;
  private currentError = '';
  private invalidated = true;
  private destroyed = false;
  private destruction?: Promise<void>;
  private revision = 0;
  private failedKey?: string;
  private errorPending = false;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      onChange(): void;
      /** Surface inspection: when set, the bitmap is pickable and reports the hovered grid cell. */
      surfaceHover?: {
        isEnabled(): boolean;
        onHover(pick: {cellIndex: number; x: number; y: number} | null): boolean;
      };
    }
  ) {
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.explorer.dataset.id
    );
  }
  get summary(): InterpolationSummary | null {
    return this.isCurrent(this.current) ? this.currentSummary : null;
  }
  get error(): string {
    return this.currentError;
  }
  invalidate(): void {
    this.invalidated = true;
    this.currentSummary = null;
    this.currentError = '';
    this.failedKey = undefined;
    this.errorPending = false;
  }
  encode(encoder: CommandEncoder, recipeValue: KeplerAnalysisRecipe): void {
    if (this.destroyed || !this.lease.isValid() || !this.props.explorer.isCurrent)
      throw new Error('Interpolation source is unavailable');
    const recipe = parseKeplerAnalysisRecipe(recipeValue);
    if (recipe.operation.kind !== 'idw-points' && recipe.operation.kind !== 'kriging-points')
      throw new Error('Interpolation requires an IDW or kriging recipe');
    if (
      recipe.source.datasetId !== this.lease.dataset.id ||
      recipe.source.revision !== this.lease.dataset.revision ||
      (recipe.population === 'selected' &&
        recipe.selectionRevision !== this.props.explorer.currentRevision + 1)
    )
      throw new Error('Interpolation recipe source or selection revision is stale');
    const typedRecipe: InterpolationRecipe = {...recipe, operation: recipe.operation};
    const key = JSON.stringify(recipe);
    if (this.failedKey === key) return;
    try {
      if (this.current?.key !== key || this.invalidated) {
        const next = this.createProgram(typedRecipe, key);
        if (this.current) this.retired.push(this.current);
        this.current = next;
      }
      const program = this.current;
      if (!this.invalidated && (program.submitted || program.lastEncoder === encoder)) return;
      program.lastEncoder = encoder;
      program.compiled.encode(encoder, {parameters: undefined});
      program.submitted = false;
      this.invalidated = false;
      this.currentSummary = null;
      this.currentError = '';
    } catch (error) {
      this.invalidated = true;
      this.currentError = error instanceof Error ? error.message : String(error);
      this.failedKey = key;
      this.errorPending = true;
      throw error;
    }
  }
  markSubmitted(encoder: CommandEncoder): void {
    for (const program of [...this.retired, ...(this.current ? [this.current] : [])]) {
      if (program.lastEncoder === encoder) program.submitted = true;
    }
  }
  /** Called only after the shared host submission, also while another analysis is selected. */
  readSummary(): Promise<void> {
    this.retireSubmitted();
    const program = this.current;
    if (this.errorPending) {
      this.errorPending = false;
      return Promise.resolve().then(() => {
        if (!this.destroyed && this.currentError) this.props.onChange();
      });
    }
    if (!this.isCurrent(program) || this.currentSummary || this.pendingSummary)
      return this.pendingSummary ?? Promise.resolve();
    const reservation = this.props.session.registry.reserveMemory(128, 'cpu', 'scratch');
    const pending = this.trackRead(
      program,
      (async () => {
        try {
          const [
            rangeBytes,
            meanBytes,
            diagnosticBytes,
            varianceRangeBytes,
            varianceMeanBytes,
            crossValidationPredictionBytes,
            crossValidationVarianceBytes
          ] = await readSettled([
            () => program.statistics.readAsync(GPU_COLUMN_PROFILE_STATISTIC.minimum * 4, 8),
            () => program.statistics.readAsync(GPU_COLUMN_PROFILE_STATISTIC.mean * 4, 4),
            () => program.diagnostics.readAsync(),
            () =>
              program.varianceStatistics?.readAsync(GPU_COLUMN_PROFILE_STATISTIC.minimum * 4, 8) ??
              Promise.resolve(new Uint8Array()),
            () =>
              program.varianceStatistics?.readAsync(GPU_COLUMN_PROFILE_STATISTIC.mean * 4, 4) ??
              Promise.resolve(new Uint8Array()),
            () =>
              program.crossValidation?.prediction.readAsync() ?? Promise.resolve(new Uint8Array()),
            () => program.crossValidation?.variance.readAsync() ?? Promise.resolve(new Uint8Array())
          ]);
          const observed =
            program.crossValidation && program.recipe.operation.kind === 'kriging-points'
              ? await readFloat32Vector(
                  this.lease.dataset.value.columns[
                    program.recipe.operation.column
                  ] as GPUVector<'float32'>
                )
              : undefined;
          if (!this.isCurrent(program)) return;
          const range = new Float32Array(rangeBytes.buffer, rangeBytes.byteOffset, 2);
          const mean = new Float32Array(meanBytes.buffer, meanBytes.byteOffset, 1)[0];
          const varianceRange = new Float32Array(
            varianceRangeBytes.buffer,
            varianceRangeBytes.byteOffset
          );
          const varianceMean = new Float32Array(
            varianceMeanBytes.buffer,
            varianceMeanBytes.byteOffset
          )[0];
          const diagnostics = new Uint32Array(
            diagnosticBytes.buffer,
            diagnosticBytes.byteOffset,
            diagnosticBytes.byteLength / 4
          );
          const {grid} = program.recipe.operation;
          const level =
            program.recipe.operation.kind === 'kriging-points'
              ? program.recipe.operation.confidenceLevel
              : undefined;
          const multiplier =
            level === undefined
              ? undefined
              : getKrigingConfidenceInterval([0], [0], level).multiplier;
          this.currentSummary = {
            datasetId: program.recipe.source.datasetId,
            column: program.recipe.operation.column,
            revision: program.revision,
            sourceRevision: program.recipe.source.revision,
            selectionRevision: program.recipe.selectionRevision,
            operation: program.recipe.operation.kind,
            numericalPolicy: 'float32',
            count: diagnostics[1],
            nullCount: diagnostics[2],
            minimum: range[0],
            maximum: range[1],
            mean,
            breaks: [],
            classCounts: [],
            ...(program.recipe.operation.kind === 'kriging-points'
              ? {
                  kriging: {
                    invalidCount: diagnostics[3],
                    uncertainCount: diagnostics[4],
                    variance: {
                      count: diagnostics[1],
                      nullCount: diagnostics[2],
                      minimum: varianceRange[0],
                      maximum: varianceRange[1],
                      mean: varianceMean
                    },
                    ...(level !== undefined && multiplier !== undefined
                      ? {
                          interval: {
                            level,
                            multiplier,
                            halfWidthAtMinimumVariance: multiplier * Math.sqrt(varianceRange[0]),
                            halfWidthAtMeanVariance: multiplier * Math.sqrt(varianceMean),
                            halfWidthAtMaximumVariance: multiplier * Math.sqrt(varianceRange[1])
                          }
                        }
                      : {}),
                    ...(observed
                      ? {
                          crossValidation: summarizeKrigingCrossValidation(
                            observed,
                            new Float32Array(
                              crossValidationPredictionBytes.buffer,
                              crossValidationPredictionBytes.byteOffset,
                              observed.length
                            ),
                            new Float32Array(
                              crossValidationVarianceBytes.buffer,
                              crossValidationVarianceBytes.byteOffset,
                              observed.length
                            ),
                            level ?? 0.95
                          )
                        }
                      : {})
                  }
                }
              : {}),
            interpolation: {
              width: grid.width,
              height: grid.height,
              validSourceCount: diagnostics[0],
              cellCount: grid.width * grid.height
            }
          };
          this.props.onChange();
        } catch (error) {
          if (this.isCurrent(program)) {
            this.currentError = error instanceof Error ? error.message : String(error);
            this.invalidated = true;
            this.failedKey = program.key;
            this.props.onChange();
          }
          throw error;
        } finally {
          reservation.release();
        }
      })()
    );
    this.pendingSummary = pending.finally(() => {
      this.pendingSummary = undefined;
      if (this.current !== program && this.isCurrent(this.current)) this.props.onChange();
    });
    return this.pendingSummary;
  }
  getLayer(): BitmapLayer | null {
    const program = this.current;
    if (!this.isCurrent(program) || !this.currentSummary) return null;
    return new BitmapLayer({
      id: `resident-${program.recipe.operation.kind}-${program.recipe.source.datasetId}`,
      image: program.texture,
      bounds: [...program.recipe.operation.grid.bounds],
      _imageCoordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      textureParameters: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      },
      opacity: 0.8,
      pickable: Boolean(this.props.surfaceHover?.isEnabled()),
      onHover: this.props.surfaceHover
        ? information => {
            const coordinate = information.coordinate;
            const cellIndex =
              information.picked && coordinate
                ? getInterpolationCellIndex(
                    program.recipe.operation.grid,
                    coordinate[0],
                    coordinate[1]
                  )
                : null;
            return this.props.surfaceHover!.onHover(
              cellIndex === null ? null : {cellIndex, x: information.x, y: information.y}
            );
          }
        : undefined
    });
  }
  readPage(offset: number, limit = 16): Promise<readonly InterpolationCell[]> {
    const program = this.current;
    if (!this.isCurrent(program))
      return Promise.reject(new Error('Interpolation grid is unavailable or stale'));
    const {grid} = program.recipe.operation;
    const cellCount = grid.width * grid.height;
    if (
      !Number.isSafeInteger(offset) ||
      offset < 0 ||
      offset >= cellCount ||
      !Number.isSafeInteger(limit) ||
      limit < 1 ||
      limit > 128
    )
      return Promise.reject(new Error('Request 1 to 128 cells within the interpolation grid'));
    const count = Math.min(limit, cellCount - offset);
    return this.trackRead(
      program,
      (async () => {
        const secondary = program.variance ?? program.counts;
        if (!secondary) throw new Error('Interpolation cell output is unavailable');
        const [valueBytes, secondaryBytes] = await readSettled([
          () => program.values.readAsync(offset * 4, count * 4),
          () => secondary.readAsync(offset * 4, count * 4)
        ]);
        if (!this.isCurrent(program))
          throw new Error('Interpolation grid changed during inspection');
        const values = new Float32Array(valueBytes.buffer, valueBytes.byteOffset, count);
        const secondaryValues = program.variance
          ? new Float32Array(secondaryBytes.buffer, secondaryBytes.byteOffset, count)
          : new Uint32Array(secondaryBytes.buffer, secondaryBytes.byteOffset, count);
        const [west, south, east, north] = grid.bounds;
        const level =
          program.recipe.operation.kind === 'kriging-points'
            ? program.recipe.operation.confidenceLevel
            : undefined;
        const interval =
          level !== undefined && program.variance
            ? getKrigingConfidenceInterval(values, secondaryValues, level)
            : undefined;
        return Array.from({length: count}, (_, index): InterpolationCell => {
          const cellIndex = offset + index;
          return {
            cellIndex,
            longitude: west + (((cellIndex % grid.width) + 0.5) * (east - west)) / grid.width,
            latitude:
              north - ((Math.floor(cellIndex / grid.width) + 0.5) * (north - south)) / grid.height,
            value: Number.isFinite(values[index]) ? values[index] : null,
            ...(program.recipe.operation.kind === 'kriging-points'
              ? {
                  operation: 'kriging-points' as const,
                  variance: Number.isFinite(secondaryValues[index]) ? secondaryValues[index] : null,
                  ...(interval && level !== undefined
                    ? {
                        confidenceLevel: level,
                        lower: Number.isFinite(interval.lower[index])
                          ? interval.lower[index]
                          : null,
                        upper: Number.isFinite(interval.upper[index]) ? interval.upper[index] : null
                      }
                    : {})
                }
              : {neighbors: secondaryValues[index]})
          };
        });
      })()
    );
  }
  /** Reads the prediction grid (row-major, north-up, NaN where invalid) for terrain derivatives. */
  readGridValues(): Promise<{
    width: number;
    height: number;
    bounds: readonly [number, number, number, number];
    values: Float32Array;
    sourceId: string;
    operation: 'idw-points' | 'kriging-points';
  }> {
    const program = this.current;
    if (!this.isCurrent(program))
      return Promise.reject(new Error('Interpolation grid is unavailable or stale'));
    const {grid} = program.recipe.operation;
    const byteLength = grid.width * grid.height * 4;
    const reservation = this.props.session.registry.reserveMemory(
      byteLength * 2,
      'cpu',
      'readback'
    );
    return this.trackRead(
      program,
      (async () => {
        try {
          const bytes = await program.values.readAsync(0, byteLength);
          if (!this.isCurrent(program)) throw new Error('Interpolation grid changed during read');
          return {
            width: grid.width,
            height: grid.height,
            bounds: grid.bounds,
            values: new Float32Array(bytes.slice().buffer),
            sourceId: program.recipe.source.datasetId,
            operation: program.recipe.operation.kind
          };
        } finally {
          reservation.release();
        }
      })()
    );
  }
  /** Explicit full-grid export. Caller destroys the result after preparing its download. */
  async exportGeoTIFF(
    metric: InterpolationMetric = 'prediction'
  ): Promise<InterpolationGridExport> {
    const program = this.current;
    if (!this.isCurrent(program))
      return Promise.reject(new Error('Interpolation grid is unavailable or stale'));
    if (metric !== 'prediction' && metric !== 'variance')
      throw new Error('Unknown interpolation export metric');
    const output = metric === 'variance' ? program.variance : program.values;
    if (!output) throw new Error('Variance is available only for kriging');
    const grid = program.recipe.operation.grid;
    const readbackBytes = grid.width * grid.height * 4;
    // Cover the bounded readback, file copy and metadata while the caller prepares its download.
    const reservation = this.props.session.registry.reserveMemory(
      readbackBytes * 3 + 64 * 1024,
      'cpu',
      'readback'
    );
    return this.trackRead(
      program,
      (async () => {
        let adopted = false;
        try {
          const bytes = await output.readAsync(0, readbackBytes);
          if (!this.isCurrent(program)) throw new Error('Interpolation grid changed during export');
          const values = new Float32Array(bytes.buffer, bytes.byteOffset, readbackBytes / 4);
          const data = encodeInterpolationGeoTIFF({
            ...grid,
            values,
            recipe: program.recipe,
            metric
          });
          let destroyed = false;
          const result: InterpolationGridExport = {
            get data() {
              if (destroyed) throw new Error('Interpolation export has been released');
              return data;
            },
            filename: `${program.recipe.source.datasetId.replace(/[^a-zA-Z0-9_-]/g, '_')}-${program.recipe.operation.kind === 'idw-points' ? 'idw' : 'kriging'}-${metric}.tif`,
            readbackBytes,
            isCurrent: () => !destroyed && this.isCurrent(program),
            destroy() {
              if (destroyed) return;
              destroyed = true;
              reservation.release();
            }
          };
          adopted = true;
          return result;
        } finally {
          if (!adopted) reservation.release();
        }
      })()
    );
  }
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.currentSummary = null;
    const programs = [...this.retired, ...(this.current ? [this.current] : [])];
    this.retired = [];
    this.current = undefined;
    this.destruction = (async () => {
      await Promise.allSettled([
        Promise.resolve(completion),
        ...this.retirements,
        ...programs.flatMap(program => [...program.reads])
      ]);
      for (const program of programs) program.destroy();
      await this.lease.release();
    })();
    return this.destruction;
  }
  private isCurrent(program: Program | undefined): program is Program {
    return Boolean(
      program &&
        program === this.current &&
        program.submitted &&
        !this.invalidated &&
        !this.destroyed &&
        this.lease.isValid() &&
        this.props.explorer.isCurrent &&
        (program.recipe.population === 'all' ||
          program.recipe.selectionRevision === this.props.explorer.currentRevision + 1)
    );
  }
  private trackRead<Result>(program: Program, promise: Promise<Result>): Promise<Result> {
    program.reads.add(promise);
    void promise.finally(() => program.reads.delete(promise)).catch(() => {});
    return promise;
  }
  private retireSubmitted(): void {
    if (!this.retired.length) return;
    const device = this.props.session.device;
    const fence = device.isLost ? undefined : device.createFence();
    const programs = this.retired.splice(0);
    const completion = fence ? Promise.race([fence.signaled, device.lost]) : Promise.resolve();
    const retirement = (async () => {
      await Promise.allSettled([completion, ...programs.flatMap(program => [...program.reads])]);
      try {
        for (const program of programs) program.destroy();
      } finally {
        fence?.destroy();
      }
    })();
    this.retirements.add(retirement);
    void retirement.finally(() => this.retirements.delete(retirement)).catch(() => {});
  }
  private createProgram(recipe: InterpolationRecipe, key: string): Program {
    const {session, explorer} = this.props;
    const {device} = session;
    const operation = recipe.operation;
    const kriging = operation.kind === 'kriging-points';
    const cellCount = operation.grid.width * operation.grid.height;
    const reservation: KeplerMemoryReservation = session.registry.reserveMemory(
      this.lease.dataset.rowCount * 256 +
        (operation.kind === 'kriging-points' && operation.crossValidation
          ? this.lease.dataset.rowCount * 8
          : 0) +
        cellCount * (kriging ? 80 : 64) +
        4 * 1024 * 1024,
      'gpu',
      'scratch'
    );
    const buffers: Buffer[] = [];
    let compiled: CompiledGPUCommandGraph | undefined;
    let result: ReturnType<typeof addKeplerAnalysisRecipe> | undefined;
    let texture: Texture | undefined;
    let colorParameters: GPUParameterBuffer<'float32'> | undefined;
    try {
      const graph = new GPUCommandGraph(device, {
        id: `idw-${recipe.source.datasetId}-${this.revision + 1}`
      });
      const allocate = (
        name: string,
        length: number,
        data?: Float32Array | Uint32Array
      ): Buffer => {
        const buffer = device.createBuffer({
          id: `idw-output-${name}`,
          byteLength: Math.max(length, 1) * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
          ...(data ? {data} : {})
        });
        buffers.push(buffer);
        return buffer;
      };
      const view = <Format extends 'float32' | 'uint32'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) =>
        graph.createDataView(
          graph.importBuffer(
            {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          ),
          {format, length}
        );
      const columns: Record<string, GraphDataView<'float32'>> = Object.create(null);
      const validityViews: GraphDataView<'uint32'>[] = [];
      for (const name of new Set([
        operation.column,
        operation.longitudeColumn,
        operation.latitudeColumn
      ])) {
        const column = this.lease.dataset.value.columns[name];
        if (!column || column.format !== 'float32')
          throw new Error('Interpolation requires prepared Float32 source fields');
        // Narrow only after checking canonical GPU format; source chunks stay borrowed.
        columns[name] = packAnalysisVector(
          graph,
          `idw-packed-${name}`,
          graph.importGPUVector(`idw-source-${name}`, column as GPUVector<'float32'>)
        );
        const validity = this.lease.dataset.value.analytics.validity[name];
        if (validity)
          validityViews.push(
            packAnalysisVector(
              graph,
              `idw-valid-${name}`,
              graph.importGPUVector(`idw-validity-${name}`, validity)
            )
          );
      }
      if (recipe.population === 'selected')
        validityViews.push(
          packAnalysisVector(
            graph,
            'idw-selected',
            graph.importGPUVector('idw-selection', explorer.selectionMask)
          )
        );
      let mask: GraphDataView<'uint32'> | undefined;
      if (validityViews.length) {
        mask = createTransientView(
          graph,
          'idw-combined-validity',
          'uint32',
          this.lease.dataset.rowCount
        );
        graph.add(new GPUMask({id: 'idw-source-validity', inputs: validityViews, output: mask}));
      }
      const values = allocate('values', cellCount);
      const counts = kriging ? undefined : allocate('counts', cellCount);
      const variance = kriging ? allocate('variance', cellCount) : undefined;
      const varianceStatistics = kriging
        ? allocate('variance-statistics', GPU_COLUMN_PROFILE_STATISTIC_COUNT)
        : undefined;
      const validity = allocate('validity', cellCount);
      const colors = allocate('colors', cellCount);
      const statistics = allocate('statistics', GPU_COLUMN_PROFILE_STATISTIC_COUNT);
      const profileCounts = allocate('profile-counts', 2);
      const crossValidation =
        operation.kind === 'kriging-points' && operation.crossValidation
          ? {
              prediction: allocate('cross-validation-prediction', this.lease.dataset.rowCount),
              variance: allocate('cross-validation-variance', this.lease.dataset.rowCount)
            }
          : undefined;
      const diagnosticCount = kriging ? 5 : 3;
      const diagnostics = allocate('diagnostics', diagnosticCount);
      const valuesView = view('idw-result-values', values, 'float32', cellCount);
      const varianceView = variance
        ? view('kriging-result-variance', variance, 'float32', cellCount)
        : undefined;
      const validityView = view('idw-result-validity', validity, 'uint32', cellCount);
      const statisticView = view(
        'idw-result-statistics',
        statistics,
        'float32',
        GPU_COLUMN_PROFILE_STATISTIC_COUNT
      );
      const diagnosticView = view('idw-result-diagnostics', diagnostics, 'uint32', diagnosticCount);
      const diagnostic = (index: number) =>
        graph.createDataView(diagnosticView.buffer, {
          format: 'uint32',
          byteOffset: index * 4,
          length: 1
        });
      result = addKeplerAnalysisRecipe(graph, recipe, {
        source: recipe.source,
        columns,
        mask,
        validityMask: mask,
        selectionRevision: recipe.selectionRevision,
        outputs: varianceView
          ? {
              kriging: {
                values: valuesView,
                variance: varianceView,
                validity: validityView,
                ...(crossValidation
                  ? {
                      crossValidation: {
                        prediction: view(
                          'kriging-cv-prediction',
                          crossValidation.prediction,
                          'float32',
                          this.lease.dataset.rowCount
                        ),
                        variance: view(
                          'kriging-cv-variance',
                          crossValidation.variance,
                          'float32',
                          this.lease.dataset.rowCount
                        )
                      }
                    }
                  : {}),
                status: {
                  validSourceCount: diagnostic(0),
                  validCellCount: diagnostic(1),
                  noDataCellCount: diagnostic(2),
                  invalidCount: diagnostic(3),
                  uncertainCount: diagnostic(4)
                }
              }
            }
          : counts
            ? {
                idw: {
                  values: valuesView,
                  counts: view('idw-result-counts', counts, 'uint32', cellCount),
                  validity: validityView,
                  status: {
                    validSourceCount: diagnostic(0),
                    validCellCount: diagnostic(1),
                    noDataCellCount: diagnostic(2)
                  }
                }
              }
            : undefined
      });
      graph.add(
        new GPUColumnProfile({
          id: 'idw-profile',
          columns: [{values: valuesView}],
          mask: validityView,
          histogramBinCount: 1,
          output: {
            statistics: statisticView,
            counts: view('idw-profile-counts', profileCounts, 'uint32', 2)
          }
        })
      );
      if (varianceView && varianceStatistics) {
        graph.add(
          new GPUColumnProfile({
            id: 'kriging-variance-profile',
            columns: [{values: varianceView}],
            mask: validityView,
            histogramBinCount: 1,
            output: {
              statistics: view(
                'kriging-variance-statistics',
                varianceStatistics,
                'float32',
                GPU_COLUMN_PROFILE_STATISTIC_COUNT
              ),
              counts: view(
                'kriging-variance-profile-counts',
                allocate('variance-profile-counts', 2),
                'uint32',
                2
              )
            }
          })
        );
      }
      const domain = graph.createDataView(statisticView.buffer, {
        format: 'float32',
        byteOffset: GPU_COLUMN_PROFILE_STATISTIC.minimum * 4,
        length: 2
      });
      const palette = allocate(
        'palette',
        5,
        new Uint32Array([
          packGPUColor(49, 54, 149),
          packGPUColor(116, 173, 209),
          packGPUColor(255, 255, 191),
          packGPUColor(244, 109, 67),
          packGPUColor(165, 0, 38)
        ])
      );
      colorParameters = new GPUParameterBuffer(device, {
        id: 'idw-color-parameters',
        format: 'float32',
        length: 9,
        values: getGPUColorScaleParameterValues({
          scale: 'linear',
          domainCount: 2,
          paletteCount: 5,
          interpolation: 'linear',
          clamp: true
        })
      });
      const colorsView = view('idw-colors', colors, 'uint32', cellCount);
      graph.add(
        new GPUColorScale({
          id: 'idw-color-scale',
          values: valuesView,
          mask: validityView,
          domain,
          palette: view('idw-palette', palette, 'uint32', 5),
          parameters: colorParameters.importToGraph(graph),
          maximumDomainCount: 2,
          maximumPaletteCount: 5,
          output: {colors: colorsView}
        })
      );
      texture = device.createTexture({
        id: 'idw-raster-texture',
        format: 'rgba8unorm',
        width: operation.grid.width,
        height: operation.grid.height,
        usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC,
        sampler: {minFilter: 'nearest', magFilter: 'nearest'}
      });
      const image = graph.createTextureView(
        graph.importTexture(
          {
            id: 'idw-image',
            format: 'rgba8unorm',
            width: operation.grid.width,
            height: operation.grid.height,
            usage: texture.props.usage
          },
          texture
        )
      );
      graph.addComputePass({
        id: 'idw-image-colors',
        resources: [
          {buffer: colorsView, usage: 'storage-read'},
          {texture: image, usage: 'storage-write'}
        ],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            id: 'idw-image-colors',
            source: `@group(0) @binding(0) var<storage, read> colors: array<u32>;
@group(0) @binding(1) var image: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) index: vec3<u32>) {
  if (index.x >= ${operation.grid.width}u || index.y >= ${operation.grid.height}u) { return; }
  textureStore(image, vec2<i32>(index.xy), unpack4x8unorm(colors[index.y * ${operation.grid.width}u + index.x]));
}`,
            shaderLayout: {
              bindings: [
                {name: 'colors', type: 'read-only-storage', group: 0, location: 0},
                {
                  name: 'image',
                  type: 'storage',
                  group: 0,
                  location: 1,
                  access: 'write-only',
                  format: 'rgba8unorm'
                }
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer, getTextureView}) =>
              kernel.dispatch(computePass, {
                bindings: {colors: getBuffer(colorsView), image: getTextureView(image)},
                x: Math.ceil(operation.grid.width / 8),
                y: Math.ceil(operation.grid.height / 8),
                z: 1
              }),
            destroy: () => kernel.destroy()
          };
        }
      });
      if (!('parameterByteLength' in result))
        throw new Error('Interpolation result must declare its parameter allocation');
      const plan = graph.planResources();
      const ownedBytes =
        buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
        cellCount * 4 +
        colorParameters.buffer.byteLength +
        result.parameterByteLength;
      const plannedBytes = plan.stats.physicalTransientResourceBytes + ownedBytes;
      if (plannedBytes > reservation.byteLength)
        throw new Error('Interpolation exceeds its working-memory reservation');
      // Admit graph scratch before compilation allocates it; driver-private shader memory is not measured.
      reservation.resize(plannedBytes);
      compiled = graph.compile();
      const ownedCompiled = compiled;
      const ownedResult = result;
      const ownedTexture = texture;
      const ownedParameters = colorParameters;
      return {
        key,
        recipe,
        compiled,
        values,
        counts,
        variance,
        varianceStatistics,
        crossValidation,
        statistics,
        diagnostics,
        texture,
        revision: ++this.revision,
        reads: new Set(),
        submitted: false,
        destroy() {
          ownedCompiled.destroy();
          ownedResult.destroyParameters();
          ownedParameters.destroy();
          ownedTexture.destroy();
          for (const buffer of buffers) buffer.destroy();
          reservation.release();
        }
      };
    } catch (error) {
      compiled?.destroy();
      result?.destroyParameters();
      colorParameters?.destroy();
      texture?.destroy();
      for (const buffer of buffers) buffer.destroy();
      reservation.release();
      throw error;
    }
  }
}

/** Keep every sibling GPU read alive even when one mapping rejects. */
async function readSettled(reads: (() => Promise<Uint8Array>)[]): Promise<Uint8Array[]> {
  const results = await Promise.allSettled(reads.map(read => Promise.resolve().then(read)));
  const failure = results.find(result => result.status === 'rejected');
  if (failure?.status === 'rejected') throw failure.reason;
  return results.map(result => {
    if (result.status === 'rejected') throw result.reason;
    return result.value;
  });
}
