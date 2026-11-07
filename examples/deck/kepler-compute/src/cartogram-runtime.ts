// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  createTransientView,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '@luma-contributors/graph/gpu-dataframe';
import {
  addKeplerDorlingCartogram,
  estimateKeplerDorlingByteLength,
  parseKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type KeplerDorlingSettings,
  type KeplerMemoryReservation,
  type KeplerSessionExplorer,
  type KeplerTableDerivedSnapshot
} from '@luma-contributors/kepler';
import {packAnalysisVector} from './analysis-point-weights-cache';
import {CartogramOutputs} from './cartogram-outputs';
import {
  getCartogramBoundsUnavailableReason,
  getCartogramBoundsWorkingBytes,
  getCartogramCoordinateResolution,
  normalizeCartogramSettings
} from './cartogram-settings';
import {
  CartogramMercatorLayer,
  getCartogramMercatorUnavailableReason
} from './cartogram-mercator-layer';
import {
  CartogramDeclaredPlaneLayer,
  planCartogramOverlay,
  type CartogramOverlayDecision
} from './analysis2-cartogram-overlay';
import type {DeclaredProjectedCartogramOptions} from './analysis-cartogram-declared-crs';
import {createCartogramPickingHandlers, type CartogramPickingCallbacks} from './cartogram-picking';

type CartogramRecipe = KeplerAnalysisRecipe & {
  operation: {kind: 'dorling-cartogram'} & KeplerDorlingSettings;
};
export type CartogramSummary = {
  datasetId: string;
  column: string;
  revision: number;
  sourceRevision: number;
  selectionRevision?: number;
  operation: 'dorling-cartogram';
  numericalPolicy: 'float32';
  count: number;
  nullCount: number;
  /** Radius statistics, in the declared projected meters. */
  minimum: number;
  maximum: number;
  mean: number;
  breaks: readonly number[];
  classCounts: readonly number[];
  cartogram: {
    iterations: number;
    converged: boolean;
    budgetExhausted: boolean;
    /** Includes unselected rows as well as invalid coordinates or values. */
    excludedCount: number;
    maximumPenetration: number;
    maximumDisplacement: number;
    coordinateResolutionMeters: number;
    precisionWarning?: string;
    crs: string;
    units: 'meters';
    /** Float32 circle extents [minX, minY, maxX, maxY], including radius; absent when empty. */
    resultBounds?: readonly [number, number, number, number];
    overlayUnavailableReason?: string;
    display: 'table-only' | 'web-mercator-overlay' | 'declared-plane-overlay';
  };
};
export type CartogramRow = {
  sourceRowIndex: number;
  sourceRevision: number;
  analysisRevision: number;
  centerX: number | null;
  centerY: number | null;
  radius: number | null;
  sourceId: number;
};
export type CartogramResultSnapshot = {
  readonly positions: GPUVector<'float32x2'>;
  readonly radii: GPUVector<'float32'>;
  readonly sourceIds: GPUVector<'uint32'>;
  readonly validity: GPUVector<'uint32'>;
  readonly settings: KeplerDorlingSettings;
  readonly revision: number;
  readonly sourceRevision: number;
  readonly selectionRevision?: number;
  isCurrent(): boolean;
};
type Program = {
  key: string;
  recipe: CartogramRecipe;
  compiled: CompiledGPUCommandGraph;
  outputs: CartogramOutputs;
  statistics: Buffer;
  revision: number;
  reads: Set<Promise<unknown>>;
  lastEncoder?: CommandEncoder;
  encoded: boolean;
  submitted: boolean;
  summary?: CartogramSummary;
  pendingSummary?: Promise<void>;
  destroy(): void;
};

/** Source-row cartogram results; projected coordinates never become implicit map geometry. */
export class CartogramRuntime {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private current?: Program;
  private retired: Program[] = [];
  private readonly retirements = new Set<Promise<void>>();
  private invalidated = true;
  private destroyed = false;
  private destruction?: Promise<void>;
  private revision = 0;
  private readbackBytes = 0;
  private currentError = '';
  private declaration?: DeclaredProjectedCartogramOptions;
  private failedKey?: string;
  private errorPending = false;
  private picking?: {revision: number; handlers: ReturnType<typeof createCartogramPickingHandlers>};

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      onChange(): void;
      picking?: CartogramPickingCallbacks;
    }
  ) {
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.explorer.dataset.id
    );
  }
  get summary(): CartogramSummary | null {
    return this.isCurrent(this.current) ? (this.current.summary ?? null) : null;
  }
  get error(): string {
    return this.currentError;
  }
  get transferredReadbackBytes(): number {
    return this.readbackBytes;
  }
  get resultColumn(): string {
    return this.columnName('analysis_cartogram_radius_meters');
  }
  /** Declared local-plane georeferencing for a non-EPSG:3857 projected CRS; display only. */
  setDeclaration(declaration: DeclaredProjectedCartogramOptions | undefined): void {
    if (this.declaration === declaration) return;
    this.declaration = declaration;
    this.clearPicking();
    this.props.onChange();
  }
  getOverlayDecision(): CartogramOverlayDecision | undefined {
    const settings = this.current?.recipe.operation;
    const summary = this.summary;
    if (!settings || !summary) return undefined;
    if (getCartogramMercatorUnavailableReason(settings.crs) === undefined)
      return summary.cartogram.overlayUnavailableReason
        ? {kind: 'table-only', reason: summary.cartogram.overlayUnavailableReason}
        : {kind: 'web-mercator'};
    return planCartogramOverlay(settings.crs, summary.cartogram.resultBounds, this.declaration);
  }
  get unavailableReason(): string | undefined {
    const settings = this.current?.recipe.operation;
    const decision = this.getOverlayDecision();
    if (decision) return decision.kind === 'table-only' ? decision.reason : undefined;
    return settings ? getCartogramMercatorUnavailableReason(settings.crs) : undefined;
  }
  invalidate(): void {
    this.invalidated = true;
    this.currentError = '';
    this.failedKey = undefined;
    this.errorPending = false;
    this.clearPicking();
  }
  /** Cancel pending pick publication without releasing the reads or their buffer fences. */
  clearInspection(): void {
    this.clearPicking();
  }
  encode(encoder: CommandEncoder, recipeValue: KeplerAnalysisRecipe): void {
    if (this.destroyed || !this.lease.isValid() || !this.props.explorer.isCurrent)
      throw new Error('Cartogram source is unavailable');
    const recipe = parseKeplerAnalysisRecipe(recipeValue);
    if (recipe.operation.kind !== 'dorling-cartogram')
      throw new Error('Cartogram requires a Dorling recipe');
    const {kind, ...settings} = recipe.operation;
    normalizeCartogramSettings(settings, this.lease.dataset.rowCount);
    if (
      recipe.source.datasetId !== this.lease.dataset.id ||
      recipe.source.revision !== this.lease.dataset.revision ||
      (recipe.population === 'selected' &&
        recipe.selectionRevision !== this.props.explorer.currentRevision + 1)
    )
      throw new Error('Cartogram source or selection revision is stale');
    const typedRecipe: CartogramRecipe = {...recipe, operation: recipe.operation};
    const key = JSON.stringify(recipe);
    if (this.failedKey === key) return;
    if (
      !this.invalidated &&
      this.current?.key === key &&
      this.current.encoded &&
      (this.current.submitted || this.current.lastEncoder === encoder)
    )
      return;
    try {
      if (this.retired.length >= 4)
        throw new Error('Submit pending cartograms before changing settings again');
      // Each new execution gets new outputs so asynchronous readers retain an immutable result.
      const next = this.createProgram(typedRecipe, key);
      this.clearPicking();
      if (this.current) this.retired.push(this.current);
      this.current = next;
      next.lastEncoder = encoder;
      next.compiled.encode(encoder, {parameters: undefined});
      next.encoded = true;
      this.invalidated = false;
      this.currentError = '';
      this.failedKey = undefined;
      this.errorPending = false;
    } catch (error) {
      this.invalidated = true;
      this.currentError = error instanceof Error ? error.message : String(error);
      this.failedKey = key;
      this.errorPending = true;
      // Encoding runs inside Deck preRender; publishing here would synchronously redraw it.
      throw error;
    }
  }
  /** Acknowledge only the encoder that the host actually submitted. */
  markSubmitted(encoder: CommandEncoder): void {
    if (this.destroyed) return;
    for (const program of [...this.retired, ...(this.current ? [this.current] : [])])
      if (program.lastEncoder === encoder) program.submitted = true;
    this.retireSubmitted();
  }
  readSummary(): Promise<void> {
    if (this.errorPending) {
      this.errorPending = false;
      return Promise.resolve().then(() => {
        if (!this.destroyed && this.currentError) this.props.onChange();
      });
    }
    const program = this.current;
    if (!this.isCurrent(program) || program.summary) return Promise.resolve();
    if (program.pendingSummary) return program.pendingSummary;
    const reservation = this.props.session.registry.reserveMemory(128, 'cpu', 'scratch');
    const pending = this.trackRead(
      program,
      (async () => {
        try {
          const [metricBytes, statusBytes, minimumBytes, maximumBytes, meanBytes, boundsBytes] =
            await readBuffers([
              () => program.outputs.metrics.readAsync(0, 8),
              () => program.outputs.status.readAsync(0, 16),
              () => program.statistics.readAsync(GPU_COLUMN_PROFILE_STATISTIC.minimum * 4, 4),
              () => program.statistics.readAsync(GPU_COLUMN_PROFILE_STATISTIC.maximum * 4, 4),
              () => program.statistics.readAsync(GPU_COLUMN_PROFILE_STATISTIC.mean * 4, 4),
              () => program.outputs.bounds.readAsync(0, 16)
            ]);
          this.readbackBytes += 52;
          if (!this.isCurrent(program)) return;
          const metrics = floatValues(metricBytes);
          const status = unsignedValues(statusBytes);
          if (!Number.isFinite(metrics[0]) || !Number.isFinite(metrics[1]))
            throw new Error('Cartogram diagnostics are nonfinite; results are unavailable');
          const {kind, ...settings} = program.recipe.operation;
          const coordinateResolutionMeters = getCartogramCoordinateResolution(settings);
          const bounds = floatValues(boundsBytes);
          const resultBounds: readonly [number, number, number, number] | undefined =
            Number.isFinite(floatValues(minimumBytes)[0])
              ? [bounds[0], bounds[1], bounds[2], bounds[3]]
              : undefined;
          // Other projected CRSs are decided at read time from the declared plane (getOverlayDecision).
          const overlayUnavailableReason =
            getCartogramMercatorUnavailableReason(settings.crs) === undefined
              ? getCartogramBoundsUnavailableReason(resultBounds)
              : undefined;
          program.summary = {
            datasetId: program.recipe.source.datasetId,
            column: settings.valueColumn,
            revision: program.revision,
            sourceRevision: program.recipe.source.revision,
            selectionRevision: program.recipe.selectionRevision,
            operation: 'dorling-cartogram',
            numericalPolicy: 'float32',
            count: this.lease.dataset.rowCount - status[3],
            nullCount: status[3],
            minimum: floatValues(minimumBytes)[0],
            maximum: floatValues(maximumBytes)[0],
            mean: floatValues(meanBytes)[0],
            breaks: [],
            classCounts: [],
            cartogram: {
              iterations: status[0],
              converged: status[1] === 1,
              budgetExhausted: status[2] === 1,
              excludedCount: status[3],
              maximumPenetration: metrics[0],
              maximumDisplacement: metrics[1],
              coordinateResolutionMeters,
              ...(settings.overlapTolerance < coordinateResolutionMeters
                ? {
                    precisionWarning:
                      'Overlap tolerance is smaller than Float32 coordinate spacing at the declared source extent'
                  }
                : {}),
              crs: settings.crs,
              units: 'meters',
              ...(resultBounds ? {resultBounds} : {}),
              ...(overlayUnavailableReason ? {overlayUnavailableReason} : {}),
              display:
                overlayUnavailableReason || getCartogramMercatorUnavailableReason(settings.crs)
                  ? 'table-only'
                  : 'web-mercator-overlay'
            }
          };
          this.props.onChange();
        } catch (error) {
          if (this.isCurrent(program)) {
            this.currentError = error instanceof Error ? error.message : String(error);
            this.props.onChange();
          }
          throw error;
        } finally {
          reservation.release();
        }
      })()
    );
    program.pendingSummary = pending.finally(() => {
      program.pendingSummary = undefined;
    });
    return program.pendingSummary;
  }
  getTableColumns(): KeplerTableDerivedSnapshot | undefined {
    const program = this.current;
    if (!this.isPublished(program)) return undefined;
    const selectionRevision = this.props.explorer.currentRevision;
    const columns = {
      [this.columnName('analysis_cartogram_center_x')]: program.outputs.centerX,
      [this.columnName('analysis_cartogram_center_y')]: program.outputs.centerY,
      [this.resultColumn]: program.outputs.radii,
      [this.columnName('analysis_cartogram_source_id')]: program.outputs.sourceIds
    };
    return {
      revision: `${program.revision}:${selectionRevision}`,
      columns,
      validity: {
        [this.columnName('analysis_cartogram_center_x')]: program.outputs.validity,
        [this.columnName('analysis_cartogram_center_y')]: program.outputs.validity,
        [this.resultColumn]: program.outputs.validity
      },
      isCurrent: () =>
        this.isPublished(program) && this.props.explorer.currentRevision === selectionRevision
    };
  }
  /** Borrowed GPU vectors for an explicitly approved result view; this method never creates a map layer. */
  getOutputs(): CartogramResultSnapshot | undefined {
    const program = this.current;
    if (!this.isPublished(program)) return undefined;
    const {kind, ...settings} = program.recipe.operation;
    return {
      positions: program.outputs.positions,
      radii: program.outputs.radii,
      sourceIds: program.outputs.sourceIds,
      validity: program.outputs.validity,
      settings: normalizeCartogramSettings(settings, this.lease.dataset.rowCount),
      revision: program.revision,
      sourceRevision: program.recipe.source.revision,
      selectionRevision: program.recipe.selectionRevision,
      isCurrent: () => this.isPublished(program)
    };
  }
  getLayer(): CartogramMercatorLayer | CartogramDeclaredPlaneLayer | null {
    const result = this.getOutputs();
    if (!result || this.unavailableReason) {
      this.clearPicking();
      return null;
    }
    if (this.picking?.revision !== result.revision) {
      this.clearPicking();
      this.picking = {
        revision: result.revision,
        handlers: createCartogramPickingHandlers({
          // Disposing an obsolete handler must not publish during encoding or layer creation.
          onHover: pick => {
            if (this.picking?.revision === result.revision) this.props.picking?.onHover?.(pick);
          },
          onClick: pick => {
            if (this.picking?.revision === result.revision) this.props.picking?.onClick?.(pick);
          },
          onError: error => {
            if (this.picking?.revision === result.revision) this.props.picking?.onError?.(error);
          },
          datasetId: this.lease.dataset.id,
          rowCount: this.lease.dataset.rowCount,
          revision: result.revision,
          sourceRevision: result.sourceRevision,
          selectionRevision: result.selectionRevision,
          isCurrent: result.isCurrent,
          ...(this.declaration && this.getOverlayDecision()?.kind === 'declared-plane'
            ? {isViewportSupported: () => true}
            : {}),
          readRow: sourceRowIndex => this.readRow(sourceRowIndex)
        })
      };
    }
    const layerProps = {
      id: `resident-cartogram-${this.lease.dataset.id}`,
      getPosition: result.positions,
      getRadius: result.radii,
      selectionMask: result.validity,
      getFillColor: [49, 130, 189, 180] as [number, number, number, number],
      opacity: 0.8,
      pickable: true,
      onHover: this.picking.handlers.onHover,
      onClick: this.picking.handlers.onClick
    };
    return this.declaration && this.getOverlayDecision()?.kind === 'declared-plane'
      ? new CartogramDeclaredPlaneLayer(this.declaration, layerProps)
      : new CartogramMercatorLayer(layerProps);
  }
  retainTableRead(completion: PromiseLike<unknown>): void {
    const program = this.current;
    if (!this.isPublished(program)) throw new Error('Cartogram results are unavailable or stale');
    this.trackRead(program, Promise.resolve(completion));
  }
  withResultSnapshot<Result>(
    read: (snapshot: KeplerTableDerivedSnapshot) => Promise<Result>
  ): Promise<Result> {
    const program = this.current;
    const snapshot = this.getTableColumns();
    if (!program || !snapshot)
      return Promise.reject(new Error('Cartogram results are unavailable or stale'));
    return this.trackRead(
      program,
      Promise.resolve().then(() => {
        if (!snapshot.isCurrent()) throw new Error('Cartogram results are unavailable or stale');
        return read(snapshot);
      })
    );
  }
  readRow(sourceRowIndex: number): Promise<CartogramRow | null> {
    const program = this.current;
    if (!this.isPublished(program) || program.reads.size >= 4) return Promise.resolve(null);
    if (
      !Number.isSafeInteger(sourceRowIndex) ||
      sourceRowIndex < 0 ||
      sourceRowIndex >= this.lease.dataset.rowCount
    )
      return Promise.reject(new Error('Invalid cartogram source row'));
    const reservation = this.props.session.registry.reserveMemory(64, 'cpu', 'scratch');
    return this.trackRead(
      program,
      (async () => {
        try {
          const outputs = program.outputs;
          const [positionBytes, radiusBytes, identityBytes, validityBytes] = await readBuffers([
            () => outputs.positions.data[0].buffer.readAsync(sourceRowIndex * 8, 8),
            () => outputs.radii.data[0].buffer.readAsync(sourceRowIndex * 4, 4),
            () => outputs.sourceIds.data[0].buffer.readAsync(sourceRowIndex * 4, 4),
            () => outputs.validity.data[0].buffer.readAsync(sourceRowIndex * 4, 4)
          ]);
          this.readbackBytes += 20;
          if (!this.isPublished(program)) return null;
          const position = floatValues(positionBytes);
          const included = unsignedValues(validityBytes)[0] !== 0;
          return {
            sourceRowIndex,
            sourceRevision: program.recipe.source.revision,
            analysisRevision: program.revision,
            centerX: included ? position[0] : null,
            centerY: included ? position[1] : null,
            radius: included ? floatValues(radiusBytes)[0] : null,
            sourceId: unsignedValues(identityBytes)[0]
          };
        } finally {
          reservation.release();
        }
      })()
    );
  }
  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.clearPicking();
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
        program.encoded &&
        program.submitted &&
        !this.invalidated &&
        !this.destroyed &&
        !this.currentError &&
        this.lease.isValid() &&
        this.props.explorer.isCurrent &&
        (program.recipe.population === 'all' ||
          program.recipe.selectionRevision === this.props.explorer.currentRevision + 1)
    );
  }
  private isPublished(program: Program | undefined): program is Program {
    return this.isCurrent(program) && Boolean(program.summary);
  }
  private columnName(base: string): string {
    let name = base;
    while (Object.hasOwn(this.lease.dataset.value.columns, name)) name += '_';
    return name;
  }
  private clearPicking(): void {
    const picking = this.picking;
    this.picking = undefined;
    picking?.handlers.dispose();
  }
  private trackRead<Result>(program: Program, promise: Promise<Result>): Promise<Result> {
    program.reads.add(promise);
    void promise.finally(() => program.reads.delete(promise)).catch(() => {});
    return promise;
  }
  private retireSubmitted(): void {
    const programs = this.retired.filter(program => program.submitted);
    if (!programs.length) return;
    const device = this.props.session.device;
    const fence = device.isLost ? undefined : device.createFence();
    const completion = fence ? Promise.race([fence.signaled, device.lost]) : Promise.resolve();
    this.retired = this.retired.filter(program => !program.submitted);
    const retirement = (async () => {
      try {
        await Promise.allSettled([completion, ...programs.flatMap(program => [...program.reads])]);
        for (const program of programs) program.destroy();
      } finally {
        fence?.destroy();
      }
    })();
    this.retirements.add(retirement);
    void retirement.finally(() => this.retirements.delete(retirement)).catch(() => {});
  }
  private createProgram(recipe: CartogramRecipe, key: string): Program {
    const {session, explorer} = this.props;
    const rowCount = this.lease.dataset.rowCount;
    const {kind, ...settings} = recipe.operation;
    const reservation = session.registry.reserveMemory(
      estimateKeplerDorlingByteLength(rowCount) +
        getCartogramBoundsWorkingBytes(rowCount) +
        rowCount * 256 +
        4 * 1024 * 1024,
      'gpu',
      'scratch'
    );
    const graph = new GPUCommandGraph(session.device, {
      id: `cartogram-${recipe.source.datasetId}-${this.revision + 1}`
    });
    const buffers: Buffer[] = [];
    let outputs: CartogramOutputs | undefined;
    let preparationReservation: KeplerMemoryReservation | undefined;
    let compiled: CompiledGPUCommandGraph | undefined;
    let result: ReturnType<typeof addKeplerDorlingCartogram> | undefined;
    try {
      preparationReservation = session.registry.reserveMemory(
        rowCount * 64 + (settings.maximumIterations + 1) * 32768 + 64 * 1024,
        'cpu',
        'scratch'
      );
      const columns: Record<string, GraphDataView<'float32'>> = Object.create(null);
      const validityViews: GraphDataView<'uint32'>[] = [];
      for (const name of new Set([settings.xColumn, settings.yColumn, settings.valueColumn])) {
        const source = this.lease.dataset.value.columns[name];
        if (!source || source.format !== 'float32' || source.length !== rowCount)
          throw new Error('Cartograms require prepared Float32 coordinate and value fields');
        columns[name] = packAnalysisVector(
          graph,
          `cartogram-packed-${name}`,
          graph.importGPUVector(`cartogram-source-${name}`, borrowFloat32Column(source))
        );
        const validity = this.lease.dataset.value.analytics.validity[name];
        if (validity)
          validityViews.push(
            packAnalysisVector(
              graph,
              `cartogram-valid-${name}`,
              graph.importGPUVector(`cartogram-validity-${name}`, validity)
            )
          );
      }
      let validityMask: GraphDataView<'uint32'> | undefined;
      if (validityViews.length) {
        validityMask = createTransientView(graph, 'cartogram-complete-cases', 'uint32', rowCount);
        graph.add(
          new GPUMask({
            id: 'cartogram-source-validity',
            inputs: validityViews,
            output: validityMask
          })
        );
      }
      const mask =
        recipe.population === 'selected'
          ? packAnalysisVector(
              graph,
              'cartogram-selected',
              graph.importGPUVector('cartogram-selection', explorer.selectionMask)
            )
          : undefined;
      outputs = new CartogramOutputs(
        session,
        this.lease.dataset.value.columns[settings.valueColumn].data.map(chunk => chunk.length)
      );
      const outputViews = outputs.importToGraph(graph);
      result = addKeplerDorlingCartogram(graph, {
        source: recipe.source,
        population: recipe.population,
        selectionRevision: recipe.selectionRevision,
        settings,
        x: columns[settings.xColumn],
        y: columns[settings.yColumn],
        values: columns[settings.valueColumn],
        validityMask,
        mask,
        outputs: outputViews
      });
      const outputValidity = outputs.addScalarOutputs(graph, outputViews);
      const allocate = (name: string, length: number): Buffer => {
        const buffer = session.device.createBuffer({
          id: `cartogram-${name}`,
          byteLength: length * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return buffer;
      };
      const statistics = allocate('radius-statistics', GPU_COLUMN_PROFILE_STATISTIC_COUNT);
      const counts = allocate('radius-counts', 2);
      const importView = <Format extends 'float32' | 'uint32'>(
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
      graph.add(
        new GPUColumnProfile({
          id: 'cartogram-radius-profile',
          columns: [{values: outputViews.radii}],
          mask: outputValidity,
          histogramBinCount: 1,
          output: {
            statistics: importView(
              'cartogram-statistics',
              statistics,
              'float32',
              GPU_COLUMN_PROFILE_STATISTIC_COUNT
            ),
            counts: importView('cartogram-counts', counts, 'uint32', 2)
          }
        })
      );
      compiled = graph.compile();
      if (
        compiled.stats.physicalTransientResourceBytes +
          buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
          result.parameterByteLength >
        reservation.byteLength
      )
        throw new KeplerAllocationError('memory-budget', {
          message: 'Cartogram exceeds its working-memory reservation'
        });
      const ownedCompiled = compiled;
      const ownedOutputs = outputs;
      const ownedResult = result;
      return {
        key,
        recipe,
        compiled,
        outputs,
        statistics,
        revision: ++this.revision,
        reads: new Set(),
        encoded: false,
        submitted: false,
        destroy() {
          ownedCompiled.destroy();
          ownedResult.destroyParameters();
          ownedOutputs.destroy();
          for (const buffer of buffers) buffer.destroy();
          preparationReservation?.release();
          reservation.release();
        }
      };
    } catch (error) {
      compiled?.destroy();
      result?.destroyParameters();
      outputs?.destroy();
      for (const buffer of buffers) buffer.destroy();
      preparationReservation?.release();
      reservation.release();
      throw error;
    }
  }
}

function borrowFloat32Column(column: GPUVector): GPUVector<'float32'> {
  if (column.format !== 'float32') throw new Error('Cartogram requires a prepared Float32 field');
  // TypeScript cannot narrow the generic format after checking the canonical runtime metadata.
  return column as GPUVector<'float32'>;
}
function floatValues(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
function unsignedValues(bytes: Uint8Array): Uint32Array {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** Keep every staging read alive even when one readback rejects early. */
async function readBuffers(reads: readonly (() => Promise<Uint8Array>)[]): Promise<Uint8Array[]> {
  const results = await Promise.allSettled(reads.map(read => Promise.resolve().then(read)));
  return results.map(result => {
    if (result.status === 'rejected') throw result.reason;
    return result.value;
  });
}
