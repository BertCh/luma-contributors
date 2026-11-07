// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  createTransientVectorView,
  type CompiledGPUCommandGraph,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  KeplerAllocationError,
  type KeplerDatasetRegistry,
  type KeplerMemoryReservation
} from '../runtime/dataset-registry';
import {addKeplerLinearKernel} from '../analysis/analysis-scaling';
import {GPUKeplerSpatialSelection} from './gpu-spatial-selection';
import {
  prepareKeplerSpatialSelection,
  type KeplerSelectionCoordinateSpace
} from './prepare-spatial-selection';
import {
  planKeplerSelectionScopes,
  planKeplerConsumerFilterIds,
  type KeplerConsumerSelectionScope,
  type KeplerPolygonMaskFilter,
  type KeplerScalarMaskFilter,
  type KeplerSelectionGeometry,
  type KeplerSelectionLayer,
  type KeplerSelectionScopePlan
} from './selection-scope-plan';

export type GPUKeplerSelectionScopesProps = {
  device: Device;
  /** Borrowed row topology only. Each result preserves these physical source chunks. */
  rows?: GPUVector;
  /** Explicit source topology when there are no portable scalar columns (including empty chunks). */
  sourceChunkLengths?: readonly number[];
  datasetRevision?: number;
  liveMask?: GPUVector<'uint32'>;
  layers?: readonly KeplerSelectionLayer[];
  scalarFilters?: readonly KeplerScalarMaskFilter[];
  polygonFilters?: readonly KeplerPolygonMaskFilter[];
  coordinateSpace?: KeplerSelectionCoordinateSpace;
  maximumSpatialPredicates?: number;
  /** Bounded named consumer populations (Charts); outputs also count against the memory budget. */
  maximumConsumerScopes?: number;
  registry?: KeplerDatasetRegistry;
  /** Includes owned masks, containment scratch, composition scratch and resources awaiting fences. */
  workingMemoryBudgetBytes?: number;
  /** Borrowed scalar owner, encoded before spatial and scope composition on the same encoder. */
  encodePredicates?(encoder: CommandEncoder): void;
};

/** Populations that follow a linked brush mask. */
export type GPUKeplerBrushTargets = {table?: boolean; charts?: boolean; legends?: boolean};

export type GPUKeplerSelectionScopesStatus = {
  revision: number;
  datasetRevision: number;
  stale: boolean;
  complete: boolean;
  spatialPredicateCount: number;
  uncertainCount: number;
  overflow: number;
};

type SpatialPredicate = {
  geometry: KeplerSelectionGeometry;
  geometryRevision: number;
  topologyKey: string;
  boundaryRule: 'open' | 'closed';
  contributors: readonly GPUKeplerSpatialSelection[];
};

/**
 * GPU-native Kepler selection scopes over borrowed source rows and independent scalar predicates.
 *
 * B is the common scalar display mask; each layer receives B AND its polygon predicates. Native
 * table/plot population is B AND the UNION of polygon-targeted layers, or B when none are targeted.
 * Hidden targeted layers participate, while untargeted layers never widen that native union.
 * Domain selection applies only dynamic scalar predicates. A plot excludes exactly one filter id.
 * Native topology selection applies only native CPU filterRecord predicates and the specific
 * layer's polygons, preserving filteredIndexByLayer[id] ?? filteredIndex without CPU row scans.
 *
 * Geometry buffers must contain finite Float32 coordinates. For null positions, ingest a finite
 * placeholder plus an explicit validity mask; validity is applied before per-filter inversion.
 * Sources, scalar masks, and consumers remain caller-owned. Submit encode before reading status;
 * collectRetired/destroy require the caller's GPU completion fence and completed consumer reads.
 */
export class GPUKeplerSelectionScopes {
  readonly displayMask: GPUVector<'uint32'>;
  readonly domainMask: GPUVector<'uint32'>;
  readonly nativeFilteredMask: GPUVector<'uint32'>;
  /** Native filteredIndex population, before any per-layer polygon predicate. */
  readonly nativeTopologyMask: GPUVector<'uint32'>;
  readonly datasetRevision: number;
  private readonly props: GPUKeplerSelectionScopesProps;
  private readonly zeroMask: GPUVector<'uint32'>;
  private readonly layerMasks = new Map<string, GPUVector<'uint32'>>();
  private readonly layerTopologyMasks = new Map<string, GPUVector<'uint32'>>();
  private readonly plotMasks = new Map<string, GPUVector<'uint32'>>();
  private readonly consumerMasks = new Map<
    string,
    {
      mask: GPUVector<'uint32'>;
      scope: KeplerConsumerSelectionScope;
      key: string;
    }
  >();
  private readonly spatial = new Map<string, SpatialPredicate>();
  private readonly retired: {dispose(): void; byteLength: number}[] = [];
  private readonly rowChunks?: readonly GPUData[];
  private readonly rowChunkLengths: readonly number[];
  private readonly reservation?: KeplerMemoryReservation;
  private readonly workingMemoryBudgetBytes: number;
  private constructing = true;
  private maskByteLength = 0;
  private retiredByteLength = 0;
  private layers: readonly KeplerSelectionLayer[] = [];
  private scalarFilters: readonly KeplerScalarMaskFilter[] = [];
  private polygonFilters: readonly KeplerPolygonMaskFilter[] = [];
  private plan: KeplerSelectionScopePlan = {
    scalarFilterIds: [],
    domainFilterIds: [],
    nativeTopologyFilterIds: [],
    layerFilterIds: {},
    targetedLayerIds: []
  };
  private compiled?: CompiledGPUCommandGraph;
  private compositionDirty = true;
  private spatialDirty = true;
  private brushMask?: GPUVector<'uint32'>;
  private brushTargets: GPUKeplerBrushTargets = {};
  private revision = 0;
  private encodedRevision = -1;
  private pendingStatus?: Promise<GPUKeplerSelectionScopesStatus>;
  private readonly pendingRetirements = new Set<Promise<void>>();
  private destroyed = false;
  private completion?: Promise<void>;

  constructor(props: GPUKeplerSelectionScopesProps) {
    this.props = props;
    this.datasetRevision = props.datasetRevision ?? 1;
    if (!props.rows && !props.sourceChunkLengths)
      throw new Error('Selection scopes require source row topology');
    this.rowChunks = props.rows ? [...props.rows.data] : undefined;
    this.rowChunkLengths = props.sourceChunkLengths
      ? [...props.sourceChunkLengths]
      : this.rowChunks!.map(chunk => chunk.length);
    if (
      this.rowChunkLengths.some(length => !Number.isSafeInteger(length) || length < 0) ||
      !Number.isSafeInteger(this.rowChunkLengths.reduce((total, length) => total + length, 0)) ||
      (this.rowChunks &&
        (this.rowChunks.length !== this.rowChunkLengths.length ||
          this.rowChunks.some((chunk, index) => chunk.length !== this.rowChunkLengths[index])))
    )
      throw new Error('Selection source topology must match nonnegative source chunk lengths');
    this.workingMemoryBudgetBytes = props.workingMemoryBudgetBytes ?? 32 * 1024 * 1024;
    if (
      !Number.isSafeInteger(props.maximumConsumerScopes ?? 32) ||
      (props.maximumConsumerScopes ?? 32) < 1
    )
      throw new Error('Consumer scope capacity must be positive');
    if (!Number.isSafeInteger(this.workingMemoryBudgetBytes) || this.workingMemoryBudgetBytes < 1)
      throw new Error('Selection scope memory budget must be positive');
    if (props.liveMask) this.validateRows(props.liveMask);
    this.reservation = props.registry?.reserveMemory(
      this.workingMemoryBudgetBytes,
      'gpu',
      'scratch'
    );
    const allocated: GPUVector<'uint32'>[] = [];
    try {
      this.zeroMask = this.allocateMask('zero');
      allocated.push(this.zeroMask);
      this.displayMask = this.allocateMask('display');
      allocated.push(this.displayMask);
      this.domainMask = this.allocateMask('domain');
      allocated.push(this.domainMask);
      this.nativeFilteredMask = this.allocateMask('native-filtered');
      allocated.push(this.nativeFilteredMask);
      this.nativeTopologyMask = this.allocateMask('native-topology');
      allocated.push(this.nativeTopologyMask);
      this.setLayers(props.layers ?? []);
      this.setFilters(props.scalarFilters ?? [], props.polygonFilters ?? []);
      this.constructing = false;
      this.settleReservation();
    } catch (error) {
      for (const mask of allocated) mask.destroy();
      for (const mask of this.layerMasks.values()) mask.destroy();
      for (const mask of this.layerTopologyMasks.values()) mask.destroy();
      // Predicates created by setLayers/setFilters before the failure also own GPU buffers.
      for (const predicate of this.spatial.values()) destroyPredicate(predicate);
      this.spatial.clear();
      for (const mask of this.plotMasks.values()) mask.destroy();
      for (const {mask} of this.consumerMasks.values()) mask.destroy();
      this.compiled?.destroy();
      this.reservation?.release();
      throw error;
    }
  }

  get workingMemoryByteLength(): number {
    return (
      this.maskByteLength +
      this.retiredByteLength +
      (this.compiled?.stats.physicalTransientResourceBytes ?? 0) +
      [...this.spatial.values()].reduce((bytes, predicate) => bytes + predicateBytes(predicate), 0)
    );
  }

  /** Current registry charge, including retired generations still protected by consumer fences. */
  get reservedMemoryByteLength(): number {
    return this.reservation?.isActive ? this.reservation.byteLength : 0;
  }

  get currentRevision(): number {
    return this.revision;
  }
  get currentPlan(): KeplerSelectionScopePlan {
    return this.plan;
  }

  /** Stable for a layer id until that layer is removed; source ownership never transfers. */
  getLayerMask(layerId: string): GPUVector<'uint32'> {
    this.assertActive();
    const mask = this.layerMasks.get(layerId);
    if (!mask) throw new Error('Selection layer is not registered');
    return mask;
  }

  /** Native filteredIndexByLayer[id] ?? filteredIndex, without ordinary GPU range predicates. */
  getLayerTopologyMask(layerId: string): GPUVector<'uint32'> {
    this.assertActive();
    const mask = this.layerTopologyMasks.get(layerId);
    if (!mask) throw new Error('Selection layer is not registered');
    return mask;
  }

  /** Disabled/missing filter ids exclude nothing; other filters on that field remain active. */
  getPlotMask(filterId: string): GPUVector<'uint32'> {
    this.assertActive();
    let mask = this.plotMasks.get(filterId);
    if (!mask) {
      mask = this.allocateMask(`plot-${filterId}`);
      this.plotMasks.set(filterId, mask);
      this.compositionDirty = true;
      this.invalidate();
    }
    return mask;
  }

  /** Retire an unused plot scope only after dependent inspectors/renderers have been released. */
  removePlot(filterId: string): void {
    this.assertActive();
    const mask = this.plotMasks.get(filterId);
    if (!mask) return;
    this.plotMasks.delete(filterId);
    this.retireMask(mask);
    this.compositionDirty = true;
    this.invalidate();
  }

  /** Stable named Charts population; exclusions follow native CPU-id/GPU-field policy. */
  getConsumerMask(request: KeplerConsumerSelectionScope): GPUVector<'uint32'> {
    this.assertActive();
    const scope = {
      id: request.id,
      excludedCpuFilterIds: [...new Set(request.excludedCpuFilterIds ?? [])].sort(),
      excludedGpuFieldNames: [...new Set(request.excludedGpuFieldNames ?? [])].sort()
    };
    const key = JSON.stringify(scope);
    planKeplerConsumerFilterIds(this.scalarFilters, scope);
    const existing = this.consumerMasks.get(scope.id);
    if (existing?.key === key) return existing.mask;
    if (!existing && this.consumerMasks.size >= (this.props.maximumConsumerScopes ?? 32))
      throw new KeplerAllocationError('capacity', {message: 'Consumer scope capacity exceeded'});
    const mask = existing?.mask ?? this.allocateMask(`consumer-${scope.id}`);
    this.consumerMasks.set(scope.id, {mask, scope, key});
    this.compositionDirty = true;
    this.invalidate();
    return mask;
  }

  /** Release dependent consumers first, then collect retired output storage using their fence. */
  removeConsumerMask(id: string): void {
    this.assertActive();
    const entry = this.consumerMasks.get(id);
    if (!entry) return;
    this.consumerMasks.delete(id);
    this.retireMask(entry.mask);
    this.compositionDirty = true;
    this.invalidate();
  }

  setLayers(layers: readonly KeplerSelectionLayer[]): number {
    this.assertActive();
    const plan = planKeplerSelectionScopes(layers, this.scalarFilters, this.polygonFilters);
    for (const layer of layers) this.validateGeometry(layer.geometry);
    const unchanged =
      layers.length === this.layers.length &&
      layers.every((layer, index) => {
        const previous = this.layers[index];
        return (
          layer.id === previous.id &&
          layer.geometryRevision === previous.geometryRevision &&
          sameGeometry(layer.geometry, previous.geometry)
        );
      });
    if (unchanged) return this.revision;
    const newMasks = new Map<string, GPUVector<'uint32'>>();
    const newTopologyMasks = new Map<string, GPUVector<'uint32'>>();
    try {
      for (const layer of layers) {
        if (!this.layerMasks.has(layer.id)) {
          newMasks.set(layer.id, this.allocateMask(`layer-${layer.id}`));
          newTopologyMasks.set(layer.id, this.allocateMask(`layer-topology-${layer.id}`));
        }
      }
    } catch (error) {
      for (const mask of [...newMasks.values(), ...newTopologyMasks.values()]) {
        mask.destroy();
        this.maskByteLength -= maskBytes(mask);
      }
      this.settleReservation();
      throw error;
    }
    for (const [id, mask] of this.layerMasks) {
      if (!layers.some(layer => layer.id === id)) {
        this.layerMasks.delete(id);
        this.retireMask(mask);
        this.retireMask(this.layerTopologyMasks.get(id)!);
        this.layerTopologyMasks.delete(id);
      }
    }
    for (const [id, mask] of newMasks) this.layerMasks.set(id, mask);
    for (const [id, mask] of newTopologyMasks) this.layerTopologyMasks.set(id, mask);
    this.layers = layers.map(layer => ({...layer, geometry: {...layer.geometry}}));
    this.plan = plan;
    this.compositionDirty = true;
    this.spatialDirty = true;
    return this.invalidate();
  }

  setFilters(
    scalarFilters: readonly KeplerScalarMaskFilter[],
    polygonFilters: readonly KeplerPolygonMaskFilter[]
  ): number {
    this.assertActive();
    const plan = planKeplerSelectionScopes(this.layers, scalarFilters, polygonFilters);
    for (const filter of scalarFilters) this.validateRows(filter.mask);
    for (const layer of this.layers) {
      for (const id of plan.layerFilterIds[layer.id]) {
        const filter = polygonFilters.find(candidate => candidate.id === id)!;
        prepareKeplerSpatialSelection(
          getSelection(layer.geometry, filter),
          this.props.coordinateSpace ?? 'native-linear-degrees'
        );
      }
    }
    const sameMasks =
      scalarFilters.length === this.scalarFilters.length &&
      scalarFilters.every((filter, index) => {
        const previous = this.scalarFilters[index];
        return (
          previous.filterId === filter.filterId &&
          previous.mask === filter.mask &&
          previous.nativeFieldName === filter.nativeFieldName &&
          Boolean(previous.inverted) === Boolean(filter.inverted)
        );
      });
    const sameInversions =
      polygonFilters.length === this.polygonFilters.length &&
      polygonFilters.every((filter, index) => {
        const previous = this.polygonFilters[index];
        return filter.id === previous.id && Boolean(filter.inverted) === Boolean(previous.inverted);
      });
    this.compositionDirty ||=
      !sameMasks || !sameInversions || JSON.stringify(plan) !== JSON.stringify(this.plan);
    this.scalarFilters = scalarFilters.map(filter => ({...filter}));
    this.polygonFilters = polygonFilters.map(filter => ({
      ...filter,
      layerIds: [...filter.layerIds]
    }));
    this.plan = plan;
    this.spatialDirty = true;
    return this.invalidate();
  }

  /**
   * Links a borrowed source-aligned brush mask into the table/chart population (nativeFilteredMask,
   * plot and consumer masks) and/or the legend domain mask. Layer and topology masks never change,
   * so rendering keeps every filtered row. Pass null to unlink. Changing which mask or targets
   * apply recompiles the composition; a mask whose contents changed only needs this call again.
   */
  setBrushMask(mask: GPUVector<'uint32'> | null, targets: GPUKeplerBrushTargets = {}): number {
    this.assertActive();
    if (mask) this.validateRows(mask);
    const nextTargets = mask ? {...targets} : {};
    if (
      mask !== this.brushMask ||
      Boolean(nextTargets.table) !== Boolean(this.brushTargets.table) ||
      Boolean(nextTargets.charts) !== Boolean(this.brushTargets.charts) ||
      Boolean(nextTargets.legends) !== Boolean(this.brushTargets.legends)
    )
      this.compositionDirty = true;
    this.brushMask = mask ?? undefined;
    this.brushTargets = nextTargets;
    return this.invalidate();
  }

  /** Call after changing any borrowed predicate/source values without changing topology. */
  invalidate(): number {
    this.assertActive();
    return ++this.revision;
  }

  encode(encoder: CommandEncoder): number {
    this.assertActive();
    try {
      if (
        this.props.rows &&
        (this.props.rows.data.length !== this.rowChunks!.length ||
          this.props.rows.data.some(
            (chunk, index) =>
              chunk !== this.rowChunks![index] || chunk.length !== this.rowChunkLengths[index]
          ))
      )
        throw new Error('Selection source topology changed; create a new dataset-revision scope');
      this.props.encodePredicates?.(encoder);
      if (this.spatialDirty) this.reconcileSpatial();
      for (const predicate of this.spatial.values())
        for (const contributor of predicate.contributors) contributor.encode(encoder);
      if (this.compositionDirty || !this.compiled) this.compileComposition();
      this.compiled!.encode(encoder, {parameters: undefined});
      this.encodedRevision = this.revision;
      return this.revision;
    } catch (error) {
      this.encodedRevision = -1;
      throw error;
    }
  }

  /** Bounded diagnostics only: three words per spatial contributor, never source rows or masks. */
  readStatus(): Promise<GPUKeplerSelectionScopesStatus> {
    this.assertActive();
    if (this.pendingStatus) return this.pendingStatus;
    const revision = this.encodedRevision;
    const contributors = [...this.spatial.values()].flatMap(predicate => predicate.contributors);
    this.pendingStatus = Promise.all(contributors.map(contributor => contributor.readStatus()))
      .then(statuses => {
        const stale =
          this.destroyed ||
          revision < 0 ||
          revision !== this.revision ||
          statuses.some(status => status.stale);
        return {
          revision,
          datasetRevision: this.datasetRevision,
          stale,
          complete: !stale && statuses.every(status => status.complete),
          spatialPredicateCount: statuses.length,
          uncertainCount: statuses.reduce((count, status) => count + status.uncertainCount, 0),
          overflow: statuses.reduce((count, status) => count + status.overflow, 0)
        };
      })
      .finally(() => {
        this.pendingStatus = undefined;
      });
    return this.pendingStatus;
  }

  collectRetired(completion: PromiseLike<unknown>): Promise<void> {
    const retired = this.retired.splice(0);
    const cleanup = (async () => {
      const outcomes = await Promise.allSettled([
        Promise.race([Promise.resolve(completion), this.props.device.lost]),
        this.pendingStatus
      ]);
      for (const entry of retired) {
        entry.dispose();
        this.retiredByteLength -= entry.byteLength;
      }
      this.settleReservation();
      const failure = outcomes.find(outcome => outcome.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
    })();
    const tracked = cleanup.finally(() => this.pendingRetirements.delete(tracked));
    this.pendingRetirements.add(tracked);
    return tracked;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.completion) return this.completion;
    this.destroyed = true;
    const compiled = this.compiled;
    if (compiled)
      this.retire(() => compiled.destroy(), compiled.stats.physicalTransientResourceBytes);
    this.compiled = undefined;
    for (const predicate of this.spatial.values())
      this.retire(() => destroyPredicate(predicate), predicateBytes(predicate));
    this.spatial.clear();
    for (const mask of [
      this.zeroMask,
      this.displayMask,
      this.domainMask,
      this.nativeFilteredMask,
      this.nativeTopologyMask,
      ...this.layerMasks.values(),
      ...this.layerTopologyMasks.values(),
      ...this.plotMasks.values(),
      ...[...this.consumerMasks.values()].map(entry => entry.mask)
    ])
      this.retireMask(mask);
    this.layerMasks.clear();
    this.layerTopologyMasks.clear();
    this.plotMasks.clear();
    this.consumerMasks.clear();
    const previousRetirements = [...this.pendingRetirements];
    this.completion = Promise.allSettled([...previousRetirements, this.collectRetired(completion)])
      .then(outcomes => {
        const failure = outcomes.find(outcome => outcome.status === 'rejected');
        if (failure?.status === 'rejected') throw failure.reason;
      })
      .finally(() => this.reservation?.release());
    return this.completion;
  }

  private reconcileSpatial(): void {
    const required = new Set<string>();
    let contributorCount = 0;
    for (const layer of this.layers)
      contributorCount +=
        this.plan.layerFilterIds[layer.id].length *
        (layer.geometry.kind === 'endpoint-pair' ? 2 : 1);
    if (contributorCount > (this.props.maximumSpatialPredicates ?? 128))
      throw new KeplerAllocationError('capacity', {message: 'Spatial predicate capacity exceeded'});
    for (const layer of this.layers) {
      for (const id of this.plan.layerFilterIds[layer.id]) {
        const key = JSON.stringify([layer.id, id]);
        required.add(key);
        if (layer.geometry.kind === 'no-representative') continue;
        const filter = this.polygonFilters.find(candidate => candidate.id === id)!;
        const selection = getSelection(layer.geometry, filter);
        const boundaryRule = usesRectangleBounds(layer.geometry, filter) ? 'closed' : 'open';
        const prepared = prepareKeplerSpatialSelection(
          selection,
          this.props.coordinateSpace ?? 'native-linear-degrees'
        );
        const previous = this.spatial.get(key);
        if (
          previous &&
          previous.geometryRevision === layer.geometryRevision &&
          sameGeometry(previous.geometry, layer.geometry) &&
          previous.topologyKey === prepared.topologyKey &&
          previous.boundaryRule === boundaryRule
        ) {
          for (const contributor of previous.contributors) contributor.updateSelection(selection);
        } else {
          const contributors: GPUKeplerSpatialSelection[] = [];
          const currentByteLength = this.workingMemoryByteLength;
          let addedByteLength = 0;
          try {
            for (const positions of getPositions(layer.geometry)) {
              const contributor = new GPUKeplerSpatialSelection({
                device: this.props.device,
                positions,
                selection,
                coordinateSpace: this.props.coordinateSpace ?? 'native-linear-degrees',
                boundaryRule,
                admitMemory: byteLength =>
                  this.admitMemory(currentByteLength + addedByteLength + byteLength)
              });
              contributors.push(contributor);
              addedByteLength += contributor.workingMemoryByteLength;
            }
          } catch (error) {
            for (const contributor of contributors) contributor.destroy();
            this.settleReservation();
            throw error;
          }
          if (previous) this.retire(() => destroyPredicate(previous), predicateBytes(previous));
          this.spatial.set(key, {
            geometry: {...layer.geometry},
            geometryRevision: layer.geometryRevision,
            topologyKey: prepared.topologyKey,
            boundaryRule,
            contributors
          });
          this.compositionDirty = true;
          this.settleReservation();
        }
      }
    }
    for (const [key, predicate] of this.spatial)
      if (!required.has(key)) {
        this.spatial.delete(key);
        this.retire(() => destroyPredicate(predicate), predicateBytes(predicate));
        this.compositionDirty = true;
      }
    this.spatialDirty = false;
  }

  private compileComposition(): void {
    const graph = new GPUCommandGraph(this.props.device, {id: 'kepler-selection-scopes'});
    let sequence = 0;
    const importMask = (mask: GPUVector<'uint32'>) =>
      graph.importGPUVector(`mask-${sequence++}`, mask);
    const display = importMask(this.displayMask);
    const nativeTopology = importMask(this.nativeTopologyMask);
    const transient = (template: GraphVectorView<'uint32'> = display) =>
      createTransientVectorView(graph, `scope-${sequence++}`, template);
    const compose = (
      inputs: readonly GraphVectorView<'uint32'>[],
      output: GraphVectorView<'uint32'>,
      operation: 'and' | 'or' | 'not' = 'and'
    ): void => {
      if (operation === 'not') {
        graph.add(new GPUMask({id: `not-${sequence++}`, inputs, output, operation}));
        return;
      }
      let previous = inputs[0];
      if (inputs.length === 1) {
        graph.add(new GPUMask({id: `copy-${sequence++}`, inputs, output}));
        return;
      }
      for (let index = 1; index < inputs.length; index++) {
        const target = index === inputs.length - 1 ? output : transient(output);
        graph.add(
          new GPUMask({
            id: `compose-${sequence++}`,
            inputs: [previous, inputs[index]],
            output: target,
            operation
          })
        );
        previous = target;
      }
    };
    const allRows = transient();
    compose([importMask(this.zeroMask)], allRows, 'not');
    const live = this.props.liveMask ? importMask(this.props.liveMask) : allRows;
    const scalarMasks = new Map<string, GraphVectorView<'uint32'>>();
    for (const filter of this.scalarFilters) {
      const mask = importMask(filter.mask);
      const predicate = filter.inverted ? transient() : mask;
      if (filter.inverted) compose([mask], predicate, 'not');
      scalarMasks.set(filter.filterId, predicate);
    }
    compose([live, ...this.plan.scalarFilterIds.map(id => scalarMasks.get(id)!)], display);
    compose(
      [live, ...this.plan.nativeTopologyFilterIds.map(id => scalarMasks.get(id)!)],
      nativeTopology
    );
    const brush = this.brushMask ? importMask(this.brushMask) : undefined;
    const brushFor = (target: keyof GPUKeplerBrushTargets) =>
      brush && this.brushTargets[target] ? [brush] : [];
    compose(
      [live, ...this.plan.domainFilterIds.map(id => scalarMasks.get(id)!), ...brushFor('legends')],
      importMask(this.domainMask)
    );
    const polygonMasks = new Map<string, GraphVectorView<'uint32'>>();
    for (const layer of this.layers) {
      const predicates: GraphVectorView<'uint32'>[] = [];
      for (const id of this.plan.layerFilterIds[layer.id]) {
        const filter = this.polygonFilters.find(candidate => candidate.id === id)!;
        const geometry = layer.geometry;
        const predicate = this.spatial.get(JSON.stringify([layer.id, id]));
        const masks = (predicate?.contributors ?? []).map(contributor =>
          importMask(contributor.mask)
        );
        let rowMask: GraphVectorView<'uint32'>;
        if (geometry.kind === 'no-representative') {
          // Native has no centroid for these rows: never inside; inversion below selects them.
          rowMask = importMask(this.zeroMask);
        } else if (geometry.kind === 'multipoint') {
          let pointMask = masks[0];
          if (geometry.pointValidity) {
            const valid = transient(pointMask);
            compose([pointMask, importMask(geometry.pointValidity)], valid);
            pointMask = valid;
          }
          rowMask = transient();
          const offsets = graph.importGPUVector(`offsets-${sequence++}`, geometry.rowOffsets);
          for (const [index, output] of rowMask.data.entries()) {
            if (output.length === 0) continue;
            // Row accepts when any of its vertices is accepted (segmented maximum of 0/1 flags).
            addKeplerLinearKernel(
              graph,
              `multipoint-any-${sequence++}`,
              output.length,
              [
                {name: 'pointMask', view: pointMask.data[index]},
                {name: 'segmentOffsets', view: offsets.data[index]},
                {name: 'output', view: output, write: true}
              ],
              `let segmentStart = segmentOffsets[segmentOffsetsOffset + index];
let segmentEnd = segmentOffsets[segmentOffsetsOffset + index + 1u];
var accepted = 0u;
for (var position = segmentStart; position < segmentEnd; position++) {
  if (pointMask[pointMaskOffset + position] != 0u) { accepted = 1u; break; }
}
output[outputOffset + index] = accepted;`
            );
          }
          if (geometry.validity) {
            const valid = transient();
            compose([rowMask, importMask(geometry.validity)], valid);
            rowMask = valid;
          }
        } else {
          const validity =
            geometry.kind === 'endpoint-pair'
              ? [geometry.sourceValidity, geometry.targetValidity]
              : [geometry.validity];
          rowMask = transient();
          compose(
            [...masks, ...validity.flatMap(mask => (mask ? [importMask(mask)] : []))],
            rowMask
          );
        }
        if (filter.inverted) {
          const inverted = transient();
          compose([rowMask], inverted, 'not');
          rowMask = inverted;
        }
        predicates.push(rowMask);
      }
      const polygonMask = predicates.length ? transient() : allRows;
      if (predicates.length) compose(predicates, polygonMask);
      polygonMasks.set(layer.id, polygonMask);
      compose([display, polygonMask], importMask(this.layerMasks.get(layer.id)!));
      compose([nativeTopology, polygonMask], importMask(this.layerTopologyMasks.get(layer.id)!));
    }
    const nativeSpatial = this.plan.targetedLayerIds.length ? transient() : allRows;
    if (this.plan.targetedLayerIds.length)
      compose(
        this.plan.targetedLayerIds.map(id => polygonMasks.get(id)!),
        nativeSpatial,
        'or'
      );
    compose([display, nativeSpatial, ...brushFor('table')], importMask(this.nativeFilteredMask));
    for (const [filterId, mask] of this.plotMasks) {
      compose(
        [
          live,
          ...this.plan.scalarFilterIds
            .filter(id => id !== filterId)
            .map(id => scalarMasks.get(id)!),
          nativeSpatial,
          ...brushFor('charts')
        ],
        importMask(mask)
      );
    }
    for (const {mask, scope} of this.consumerMasks.values()) {
      compose(
        [
          live,
          ...planKeplerConsumerFilterIds(this.scalarFilters, scope).map(id => scalarMasks.get(id)!),
          nativeSpatial,
          ...brushFor('charts')
        ],
        importMask(mask)
      );
    }
    let compiled: CompiledGPUCommandGraph;
    try {
      compiled = graph.compile();
    } catch (error) {
      this.settleReservation();
      throw error;
    }
    try {
      // Published graphs report transient bytes only after compilation, so admit and release on failure.
      this.checkAdditionalMemory(compiled.stats.physicalTransientResourceBytes);
    } catch (error) {
      compiled.destroy();
      this.settleReservation();
      throw error;
    }
    const previous = this.compiled;
    if (previous)
      this.retire(() => previous.destroy(), previous.stats.physicalTransientResourceBytes);
    this.compiled = compiled;
    this.compositionDirty = false;
    this.settleReservation();
  }

  private validateRows(vector: GPUVector): void {
    if (
      vector.length !== this.rowChunkLengths.reduce((count, length) => count + length, 0) ||
      vector.data.length !== this.rowChunkLengths.length ||
      vector.data.some((chunk, index) => chunk.length !== this.rowChunkLengths[index])
    )
      throw new Error('Selection mask/geometry must preserve source row chunks');
  }

  private validateGeometry(geometry: KeplerSelectionGeometry): void {
    if (geometry.kind === 'multipoint') {
      if (
        geometry.positions.data.length !== this.rowChunkLengths.length ||
        geometry.rowOffsets.data.length !== this.rowChunkLengths.length ||
        geometry.rowOffsets.data.some(
          (chunk, index) => chunk.length !== this.rowChunkLengths[index] + 1
        )
      )
        throw new Error(
          'MultiPoint requires per-source-chunk flattened vertices and local row offsets'
        );
      if (
        geometry.pointValidity &&
        (geometry.pointValidity.length !== geometry.positions.length ||
          geometry.pointValidity.data.some(
            (chunk, index) => chunk.length !== geometry.positions.data[index]?.length
          ))
      )
        throw new Error('MultiPoint validity must match flattened point chunks');
      if (geometry.validity) this.validateRows(geometry.validity);
    } else if (geometry.kind !== 'no-representative') {
      for (const positions of getPositions(geometry)) this.validateRows(positions);
      const validity =
        geometry.kind === 'endpoint-pair'
          ? [geometry.sourceValidity, geometry.targetValidity]
          : [geometry.validity];
      for (const mask of validity) if (mask) this.validateRows(mask);
    }
  }

  private allocateMask(name: string): GPUVector<'uint32'> {
    const byteLength = this.rowChunkLengths.reduce(
      (bytes, length) => bytes + Math.max(4, length * 4),
      0
    );
    this.checkAdditionalMemory(byteLength);
    const chunks: GPUData<'uint32'>[] = [];
    try {
      for (const [index, length] of this.rowChunkLengths.entries())
        chunks.push(
          new GPUData({
            buffer: this.props.device.createBuffer({
              id: `kepler-scope-${name}-${index}`,
              byteLength: Math.max(4, length * 4),
              usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
            }),
            format: 'uint32',
            length,
            ownsBuffer: true
          })
        );
      const mask = new GPUVector({
        name,
        type: 'data',
        format: 'uint32',
        data: chunks,
        ownsData: true
      });
      this.maskByteLength += byteLength;
      return mask;
    } catch (error) {
      for (const chunk of chunks) chunk.destroy();
      this.settleReservation();
      throw error;
    }
  }

  private checkAdditionalMemory(bytes: number): void {
    this.admitMemory(this.workingMemoryByteLength + bytes);
  }

  private admitMemory(byteLength: number): void {
    if (this.reservation && !this.reservation.isActive)
      throw new Error('Selection scope memory reservation is inactive');
    if (byteLength > this.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('working-memory', {
        message:
          'Selection scopes exceed working memory reservation; collect retired generations or increase the budget'
      });
    if (this.reservation && byteLength > this.reservation.byteLength)
      this.reservation.resize(byteLength);
  }

  private settleReservation(): void {
    if (!this.constructing && this.reservation?.isActive)
      this.reservation.resize(this.workingMemoryByteLength);
  }

  private retire(dispose: () => void, byteLength: number): void {
    this.retired.push({dispose, byteLength});
    this.retiredByteLength += byteLength;
  }

  private retireMask(mask: GPUVector<'uint32'>): void {
    const bytes = maskBytes(mask);
    this.maskByteLength -= bytes;
    this.retire(() => mask.destroy(), bytes);
  }

  private assertActive(): void {
    if (this.destroyed) throw new Error('Selection scopes are destroyed');
  }
}

function maskBytes(mask: GPUVector<'uint32'>): number {
  return mask.data.reduce((bytes, chunk) => bytes + Math.max(4, chunk.length * 4), 0);
}

function predicateBytes(predicate: SpatialPredicate): number {
  return predicate.contributors.reduce(
    (bytes, contributor) => bytes + contributor.workingMemoryByteLength,
    0
  );
}

function destroyPredicate(predicate: SpatialPredicate): void {
  for (const contributor of predicate.contributors) contributor.destroy();
}

function getPositions(geometry: KeplerSelectionGeometry): readonly GPUVector<'float32x2'>[] {
  if (geometry.kind === 'no-representative') return [];
  return geometry.kind === 'endpoint-pair'
    ? [geometry.sourcePositions, geometry.targetPositions]
    : [geometry.positions];
}

function usesRectangleBounds(
  geometry: KeplerSelectionGeometry,
  filter: KeplerPolygonMaskFilter
): boolean {
  return (
    geometry.kind === 'representative-point' &&
    Boolean(geometry.nativeRectangleBounds && filter.rectangleBounds)
  );
}

function getSelection(geometry: KeplerSelectionGeometry, filter: KeplerPolygonMaskFilter) {
  return usesRectangleBounds(geometry, filter)
    ? {kind: 'rectangle' as const, bounds: filter.rectangleBounds!}
    : filter.selection;
}

function sameGeometry(left: KeplerSelectionGeometry, right: KeplerSelectionGeometry): boolean {
  if (left.kind !== right.kind) return false;
  return (
    Object.keys(left).every(key => Reflect.get(left, key) === Reflect.get(right, key)) &&
    Object.keys(left).length === Object.keys(right).length
  );
}
