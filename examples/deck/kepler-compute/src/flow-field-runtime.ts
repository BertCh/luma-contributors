// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import type {KeplerMemoryReservation} from '@luma-contributors/kepler';
import {
  FlowFieldViewportTracker,
  getFlowFieldViewportBounds,
  type FlowFieldBounds
} from '@luma-contributors/analysis/raster-analysis';
import {FlowFieldViewportTexture} from './flow-field-viewport-texture';
import {
  FlowParticleSimulation,
  isFlowParticleSimulationSupported,
  type FlowParticleStepResult
} from '@luma.gl/graph/experimental/simulation';
import {FlowParticleLayer, type FlowParticleLayerProps} from '../gpu-layers/index';
import type {FlowFieldGrid} from './flow-field-grid';
import type {FlowFieldGpuGrid} from './flow-field-gpu';
import type {FlowFieldSelectionContext} from './flow-field-filter';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  getFlowFieldVelocityNumerator,
  isFlowFieldVisible,
  normalizeFlowFieldColor,
  type FlowFieldPlan
} from './flow-field-plan';

/** Deterministic simulation seed from a layer id (0..16777215). */
export function getFlowFieldSeed(layerId: string): number {
  let hash = 2166136261;
  for (let index = 0; index < layerId.length; index++)
    hash = Math.imul(hash ^ layerId.charCodeAt(index), 16777619);
  return (hash >>> 0) & 0xffffff;
}

/** Deck calls getNeedsRedraw before each frame, outside any render pass: the simulation steps there. */
class AnimatedFlowParticleLayer extends FlowParticleLayer {
  static override layerName = 'AnimatedFlowParticleLayer';
  /** Skips drawing in viewports that do not intersect the field (split maps, panned away). */
  override draw(options: Parameters<FlowParticleLayer['draw']>[0]): void {
    const viewport = this.context.viewport as {getBounds?: () => number[]} | undefined;
    try {
      const bounds = viewport?.getBounds?.();
      if (
        bounds &&
        bounds.length === 4 &&
        bounds.every(Number.isFinite) &&
        !isFlowFieldVisible(bounds as [number, number, number, number], this.props.bounds)
      )
        return;
    } catch {
      // Non-geospatial viewports have no bounds: draw.
    }
    super.draw(options);
  }
  override getNeedsRedraw(options?: {clearRedrawFlags: boolean}): string | false {
    const props = this.props as FlowParticleLayerProps & {
      advance?: (viewportBounds?: [number, number, number, number]) => void;
    };
    let viewportBounds: [number, number, number, number] | undefined;
    try {
      // One simulation feeds every view. A crop for the active viewport can exclude another
      // split camera, so retain the full field whenever more than one viewport is active.
      if ((this.context.deck?.getViewports().length ?? 1) <= 1)
        viewportBounds = (this.context.viewport as {getBounds?: () => number[]})?.getBounds?.() as
          | [number, number, number, number]
          | undefined;
    } catch {
      // Non-geospatial viewports keep the full field.
    }
    props.advance?.(viewportBounds);
    super.getNeedsRedraw(options);
    return 'flow-field-animation';
  }
}

export type FlowFieldRuntimeStats = {
  stepCount: number;
  particleCount: number;
  byteLength: number;
  /** Null while the GPU resolves the visual-policy scale from the reduced maximum speed. */
  velocityScale: number | null;
  /** `gpu`: binned and smoothed on the GPU, no grid readback. `cpu`: built on the CPU and uploaded. */
  gridMode: 'gpu' | 'cpu';
  /** Compute rebuilds of a GPU grid (filter revisions included). */
  gridBuildCount: number;
  /** A live grid awaits the current selection reduction/build; the preceding field stays usable. */
  gridPending: boolean;
  /** Physical field has finished building for the current selection mask and revision. */
  gridReady: boolean;
  /** Readbacks of grid, mask or scalars performed by rendering (always 0 for the GPU grid). */
  gridReadbackCount: number;
  /** Selection revision the grid was last built under, or null when unfiltered. */
  selectionRevision: number | null;
  /** Bounds currently mapping the grid (they follow the filtered rows on the GPU path). */
  bounds: [number, number, number, number];
  /** 20-byte filtered-bounds reductions read back (one per filter revision; GPU path only). */
  extentReadbackCount: number;
  /** Failure of the last filtered-bounds reduction, if any (the full extent is used then). */
  extentFailure?: string;
  /** Public simulation domain, cropped to the settled padded viewport. */
  simulationBounds: [number, number, number, number];
  spawnRegion: [number, number, number, number] | null;
  viewportBuildCount: number;
  /** Crop admission/construction refusal; the previous usable field continues animating. */
  viewportFailure?: string;
  /** Whether a terrain height texture is bound for streamline Z. */
  hasAltitude: boolean;
};

/** Grid geometry shared by the CPU (with texels) and GPU (texture written by compute) paths. */
export type FlowFieldRuntimeGrid = Pick<FlowFieldGrid, 'columns' | 'rows' | 'bounds'> & {
  texels?: Float32Array;
  /** CPU grid terrain heights (NaN where none), row zero south. */
  altitude?: Float32Array;
};

/** GPU-resident grid plus the layer selection that drives it directly. */
export type FlowFieldGpuSource = {
  grid: FlowFieldGpuGrid;
  selection?: FlowFieldSelectionContext;
};

/**
 * Owns the velocity texture and GPU particle simulation of one Flow Field layer. Velocity upload
 * is the only host-to-device transfer; no particle state is ever read back.
 */
export class FlowFieldRuntime {
  readonly simulation: FlowParticleSimulation;
  readonly texture: Texture;
  /** Terrain height texture (x height in meters, z validity), when the field has an altitude. */
  readonly altitudeTexture?: Texture;
  /** Mutable copy: `bounds` follows the filtered rows on the GPU path. */
  readonly grid: FlowFieldRuntimeGrid;
  readonly gpu?: FlowFieldGpuSource;
  readonly plan: FlowFieldPlan;
  stepCount = 0;
  private lastTimestamp: number | null = null;
  private lastFrameTimestamp = -1;
  private current: FlowParticleStepResult;
  private paused = false;
  private destroyed = false;
  private builtRevision: number | null = null;
  private pendingGridBuild?: Promise<void>;
  private builtSelectionMask?: GPUVector<'uint32'>;
  private readonly destruction = new AbortController();
  private extentFailure?: string;
  private readonly viewportTracker = new FlowFieldViewportTracker();
  private readonly simulationBounds: [number, number, number, number];
  private viewportTexture?: FlowFieldViewportTexture;
  private viewportAltitude?: FlowFieldViewportTexture;
  private viewportBuildCount = 0;
  private viewportFailure?: string;
  private viewportRetryTimestamp = 0;
  /** Requests a host refresh when viewport availability changes, without interrupting redraw. */
  onViewportAvailabilityChange?: () => void;
  private viewportReservation?: KeplerMemoryReservation;
  private viewportSourceKey = '';
  private spawnRegion: [number, number, number, number] | null = null;

  private get particleTexture(): Texture {
    return this.spawnRegion ? this.viewportTexture!.texture : this.texture;
  }

  private get particleAltitude(): Texture | undefined {
    return this.spawnRegion ? this.viewportAltitude?.texture : this.altitudeTexture;
  }

  static isSupported(device: Device): boolean {
    return isFlowParticleSimulationSupported(device);
  }

  constructor(
    readonly device: Device,
    readonly id: string,
    grid: FlowFieldRuntimeGrid,
    plan: FlowFieldPlan,
    seed: number,
    gpu?: FlowFieldGpuSource,
    private readonly memoryRegistry: FlowFieldSelectionContext['memoryRegistry'] = gpu?.selection
      ?.memoryRegistry
  ) {
    this.grid = {...grid, bounds: [...grid.bounds]};
    this.simulationBounds = [...grid.bounds];
    this.plan = plan;
    this.gpu = gpu;
    let scaled: Float32Array | undefined;
    if (!gpu && grid.texels) {
      scaled = new Float32Array(grid.texels.length);
      for (let index = 0; index < scaled.length; index += 4) {
        scaled[index] = grid.texels[index] * plan.velocityScale;
        scaled[index + 1] = grid.texels[index + 1] * plan.velocityScale;
        scaled[index + 2] = grid.texels[index + 2];
        scaled[index + 3] = grid.texels[index + 3];
      }
    }
    const hasAltitude = gpu ? gpu.grid.hasAltitude : Boolean(grid.altitude);
    if (hasAltitude) {
      const altitudeTexels = grid.altitude
        ? new Float32Array(grid.columns * grid.rows * 4)
        : undefined;
      if (altitudeTexels && grid.altitude)
        for (let cell = 0; cell < grid.altitude.length; cell++)
          if (Number.isFinite(grid.altitude[cell])) {
            altitudeTexels[cell * 4] = grid.altitude[cell];
            altitudeTexels[cell * 4 + 2] = 1;
          }
      this.altitudeTexture = device.createTexture({
        id: `${id}-flow-field-altitude`,
        width: grid.columns,
        height: grid.rows,
        format: 'rgba32float',
        ...(altitudeTexels ? {data: altitudeTexels} : {}),
        usage: Texture.SAMPLE | Texture.COPY_DST | Texture.COPY_SRC,
        sampler: {minFilter: 'nearest', magFilter: 'nearest'}
      });
    }
    this.texture = device.createTexture({
      id: `${id}-flow-field-velocity`,
      width: grid.columns,
      height: grid.rows,
      format: 'rgba32float',
      ...(scaled ? {data: scaled} : {}),
      usage: Texture.SAMPLE | Texture.COPY_DST | Texture.COPY_SRC,
      sampler: {minFilter: 'nearest', magFilter: 'nearest'}
    });
    // GPU grids fill the velocity texture with compute passes before the first particle step.
    this.rebuildGpuGrid();
    this.simulation = new FlowParticleSimulation(device, {
      id: `${id}-flow-field`,
      field: {texture: this.texture, bounds: grid.bounds, coordinates: 'lnglat'},
      particleCount: plan.particleCount,
      lifetime: plan.lifetimeSeconds,
      seed
    });
    this.current = this.simulation.step(0);
  }

  get stats(): FlowFieldRuntimeStats {
    const gridReady = this.isGridReady;
    return {
      stepCount: this.stepCount,
      particleCount: this.plan.particleCount,
      byteLength:
        this.simulation.byteLength +
        (this.viewportTexture?.byteLength ?? 0) +
        (this.viewportAltitude?.byteLength ?? 0) +
        (this.gpu ? this.gpu.grid.byteLength : (this.grid.texels?.byteLength ?? 0)),
      velocityScale: this.gpu && this.plan.velocityScaleNumerator ? null : this.plan.velocityScale,
      gridMode: this.gpu ? 'gpu' : 'cpu',
      gridBuildCount: this.gpu?.grid.buildCount ?? 0,
      gridPending: !this.destroyed && !gridReady,
      gridReady,
      gridReadbackCount: this.gpu?.grid.readbackCount ?? 0,
      selectionRevision: this.builtRevision,
      bounds: [...this.grid.bounds],
      extentReadbackCount: this.gpu?.grid.extentReadbackCount ?? 0,
      ...(this.extentFailure ? {extentFailure: this.extentFailure} : {}),
      simulationBounds: [...this.simulationBounds],
      spawnRegion: this.spawnRegion,
      viewportBuildCount: this.viewportBuildCount,
      ...(this.viewportFailure ? {viewportFailure: this.viewportFailure} : {}),
      hasAltitude: Boolean(this.altitudeTexture)
    };
  }

  /** Whether the physical field matches the current mask and revision, with no build in flight. */
  get isGridReady(): boolean {
    if (this.destroyed) return false;
    if (!this.gpu) return true;
    const selection = this.gpu.selection;
    return (
      this.gpu.grid.buildCount > 0 &&
      !this.pendingGridBuild &&
      (selection?.getRevision() ?? null) === this.builtRevision &&
      selection?.mask === this.builtSelectionMask
    );
  }

  /**
   * Re-bins the GPU grid when the layer mask revision changed; the field texture is updated in
   * place. Under a filter the bounds first follow the selected rows: a GPU min/max reduction is
   * read back (20 bytes, no mask or grid readback) and the build runs once it resolves, so the old
   * field stays visible meanwhile. Returns true when a build was submitted synchronously.
   */
  rebuildGpuGrid(): boolean {
    const gpu = this.gpu;
    if (!gpu || this.destroyed || this.pendingGridBuild || this.isGridReady) return false;
    const selection = gpu.selection;
    const revision = selection?.getRevision() ?? null;
    if (!selection) {
      this.applyBounds(gpu.grid.options.extent.bounds);
      gpu.grid.emptyField = false;
      gpu.grid.build(this.texture, undefined, this.altitudeTexture);
      this.builtRevision = null;
      this.builtSelectionMask = undefined;
      return true;
    }
    // Capture the mask itself: the selection context can replace its mask during the reduction.
    // One promise serializes all reduction chains, shared by rendering and analysis callers.
    const mask = selection.mask;
    const isCurrent = () =>
      !this.destroyed && gpu.selection?.mask === mask && gpu.selection.getRevision() === revision;
    const pending = this.measureStableBounds(mask)
      .then(
        bounds => {
          // A later revision must measure its own mask before it can publish a field.
          this.finishFilteredBuild(revision, mask, bounds ?? undefined, true);
        },
        error => {
          this.finishFilteredBuild(
            revision,
            mask,
            undefined,
            false,
            error instanceof Error ? error.message : String(error)
          );
        }
      )
      .finally(() => {
        this.pendingGridBuild = undefined;
      });
    this.pendingGridBuild = pending;
    // Rendering starts builds without awaiting them. Analysis awaits this same promise.
    void pending.catch(error => {
      if (!isCurrent()) return;
      this.extentFailure = error instanceof Error ? error.message : String(error);
      gpu.selection?.requestUpdate();
    });
    return false;
  }

  /** Await actual build completion, including a newer revision that superseded an in-flight build. */
  async waitForGridReady(timeoutMilliseconds = 10000): Promise<void> {
    const deadline = performance.now() + timeoutMilliseconds;
    while (!this.isGridReady) {
      if (this.destroyed) throw new Error('Flow grid was destroyed while awaiting analysis');
      const remaining = deadline - performance.now();
      if (remaining <= 0)
        throw new Error(
          'Flow grid did not settle under the current filters before the analysis timeout'
        );
      this.rebuildGpuGrid();
      const pending = this.pendingGridBuild;
      if (!pending) {
        if (this.isGridReady) return;
        throw new Error('Flow grid has no pending build for analysis');
      }
      await new Promise<void>((resolve, reject) => {
        const finish = (error?: unknown) => {
          clearTimeout(timer);
          this.destruction.signal.removeEventListener('abort', onDestroyed);
          if (error) reject(error);
          else resolve();
        };
        const onDestroyed = () =>
          finish(new Error('Flow grid was destroyed while awaiting analysis'));
        const timer = setTimeout(
          () =>
            finish(
              new Error(
                'Flow grid did not settle under the current filters before the analysis timeout'
              )
            ),
          remaining
        );
        this.destruction.signal.addEventListener('abort', onDestroyed, {once: true});
        pending.then(
          () => finish(),
          error => finish(error)
        );
      });
    }
  }

  /**
   * The mask for a new revision can still be encoded after this frame's reduction is submitted,
   * so the reduction repeats, one frame apart, until two consecutive results agree.
   */
  private async measureStableBounds(
    mask: GPUVector<'uint32'>
  ): Promise<[number, number, number, number] | null> {
    const gpu = this.gpu;
    if (!gpu) return null;
    let previous = await gpu.grid.measureSelectionBounds(mask);
    for (let attempt = 0; attempt < 3; attempt++) {
      if (this.destroyed) return previous;
      await new Promise(resolve =>
        typeof requestAnimationFrame === 'function'
          ? requestAnimationFrame(resolve)
          : setTimeout(resolve, 16)
      );
      if (this.destroyed) return previous;
      const next = await gpu.grid.measureSelectionBounds(mask);
      const stable =
        next === previous ||
        (next && previous && next.every((value, index) => value === previous![index]));
      previous = next;
      if (stable) break;
    }
    return previous;
  }

  private finishFilteredBuild(
    revision: number | null,
    mask: GPUVector<'uint32'>,
    bounds: [number, number, number, number] | undefined,
    measured: boolean,
    failure?: string
  ): void {
    const gpu = this.gpu;
    if (!gpu || this.destroyed) return;
    const selection = gpu.selection;
    if (!selection || selection.getRevision() !== revision || selection.mask !== mask) return;
    this.extentFailure = failure;
    if (measured) {
      this.extentFailure = undefined;
      // Too few selected rows for a grid: keep the old bounds and show an invalid (empty) field.
      gpu.grid.emptyField = !bounds;
      this.applyBounds(bounds ?? this.grid.bounds);
    } else {
      gpu.grid.emptyField = false;
      this.applyBounds(gpu.grid.options.extent.bounds);
    }
    gpu.grid.build(this.texture, mask, this.altitudeTexture);
    this.builtRevision = revision;
    this.builtSelectionMask = mask;
    gpu.selection?.requestUpdate();
  }

  /** Moves the grid to new bounds: particles reset (their positions are relative to the bounds). */
  private applyBounds(bounds: readonly [number, number, number, number]): void {
    const gpu = this.gpu;
    if (!gpu || bounds.every((value, index) => value === this.grid.bounds[index])) return;
    this.grid.bounds = [...bounds];
    gpu.grid.activeBounds = [...bounds];
    if (this.plan.velocityScaleNumerator)
      gpu.grid.options.velocityScaleNumerator = getFlowFieldVelocityNumerator(
        bounds,
        this.plan.lifetimeSeconds,
        this.plan.unitPolicy
      );
    // Keep uncropped bounds available immediately (also before the first viewport is known).
    // Cropped domains are updated with their matching texture together in advance().
    if (!this.spawnRegion) {
      for (let index = 0; index < 4; index++) this.simulationBounds[index] = bounds[index];
      this.simulation?.setField({
        texture: this.texture,
        bounds: [...bounds],
        coordinates: 'lnglat'
      });
    }
  }

  /** Reserve and construct the pair before publishing either allocation to the live runtime. */
  private prepareViewportTextures(timestamp: number): boolean {
    if (this.viewportTexture) return true;
    if (timestamp < this.viewportRetryTimestamp) return false;
    let reservation: KeplerMemoryReservation | undefined;
    const textures: FlowFieldViewportTexture[] = [];
    try {
      const textureCount = this.altitudeTexture ? 2 : 1;
      reservation = this.memoryRegistry?.reserveMemory(
        textureCount * (this.grid.columns * this.grid.rows * 16 + 32),
        'gpu',
        'scratch'
      );
      textures.push(new FlowFieldViewportTexture(this.device, this.grid.columns, this.grid.rows));
      if (this.altitudeTexture)
        textures.push(new FlowFieldViewportTexture(this.device, this.grid.columns, this.grid.rows));
    } catch (error) {
      // A refused zoom must not throw from getNeedsRedraw and stop Deck's animation loop. Try all
      // cleanup actions even if one fails; no partial allocation/reservation becomes live state.
      const errors: unknown[] = [error];
      for (const resource of [...textures, ...(reservation ? [reservation] : [])]) {
        try {
          if ('release' in resource) resource.release();
          else resource.destroy();
        } catch (cleanupError) {
          errors.push(cleanupError);
        }
      }
      this.viewportRetryTimestamp = timestamp + 1000;
      this.setViewportFailure(
        `Viewport concentration unavailable: ${errors
          .map(failure => (failure instanceof Error ? failure.message : String(failure)))
          .join('; ')}`
      );
      return false;
    }
    this.viewportTexture = textures[0];
    this.viewportAltitude = textures[1];
    this.viewportReservation = reservation;
    this.viewportRetryTimestamp = 0;
    this.setViewportFailure(undefined);
    return true;
  }

  private setViewportFailure(failure: string | undefined): void {
    if (failure === this.viewportFailure) return;
    this.viewportFailure = failure;
    this.onViewportAvailabilityChange?.();
  }

  /** Public setField resets particles only after a settled domain change, never on each pan frame. */
  private updateViewport(timestamp: number, viewportBounds?: FlowFieldBounds): void {
    // No usable GPU field exists before the initial asynchronous extent reduction/build. Cropping
    // that uninitialised texture counted an early build, followed by a late recrop at a fixed camera.
    if (this.gpu && this.gpu.grid.buildCount === 0) return;
    const desired = getFlowFieldViewportBounds(viewportBounds, this.grid.bounds);
    if (desired === this.grid.bounds) this.setViewportFailure(undefined);
    const sourceKey = `${this.grid.bounds.join('|')}|${this.gpu?.grid.buildCount ?? 0}`;
    const sourceChanged = sourceKey !== this.viewportSourceKey;
    const viewportChanged = this.viewportTracker.update(desired, this.simulationBounds, timestamp);
    // Expanding back to the full field must be immediate when a second view appears.
    const restoreFullField = !viewportBounds && this.spawnRegion !== null;
    if (!sourceChanged && !viewportChanged && !restoreFullField) return;
    // A filter may change the source domain while the camera moves: crop against the new source
    // immediately. Unchanged sources wait for the camera to settle (and ignore small jitter).
    const bounds = desired;
    const region: [number, number, number, number] = [
      (bounds[0] - this.grid.bounds[0]) / (this.grid.bounds[2] - this.grid.bounds[0]),
      (bounds[1] - this.grid.bounds[1]) / (this.grid.bounds[3] - this.grid.bounds[1]),
      (bounds[2] - this.grid.bounds[0]) / (this.grid.bounds[2] - this.grid.bounds[0]),
      (bounds[3] - this.grid.bounds[1]) / (this.grid.bounds[3] - this.grid.bounds[1])
    ];
    const cropped = region.some((value, index) => value !== (index < 2 ? 0 : 1));
    if (cropped) {
      if (!this.prepareViewportTextures(timestamp)) return;
      if (!this.viewportTexture!.update(this.texture, region)) return;
      if (this.altitudeTexture) {
        if (!this.viewportAltitude!.update(this.altitudeTexture, region)) return;
      }
      this.spawnRegion = region;
      this.viewportBuildCount++;
    } else {
      this.setViewportFailure(undefined);
      this.spawnRegion = null;
    }
    for (let index = 0; index < 4; index++) this.simulationBounds[index] = bounds[index];
    this.simulation.setField({
      texture: this.particleTexture,
      bounds: [...this.simulationBounds],
      coordinates: 'lnglat'
    });
    this.viewportSourceKey = sourceKey;
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    if (paused) this.lastTimestamp = null;
  }

  /** Advances at most once per animation frame; gaps beyond eight substeps are dropped by the simulation. */
  advance(
    timestamp: number = performance.now(),
    viewportBounds?: FlowFieldBounds
  ): FlowParticleStepResult {
    if (this.destroyed || timestamp === this.lastFrameTimestamp) return this.current;
    this.lastFrameTimestamp = timestamp;
    // Filter and time-filter changes drive the field directly through the GPU mask.
    this.rebuildGpuGrid();
    this.updateViewport(timestamp, viewportBounds);
    const deltaSeconds =
      this.paused || this.lastTimestamp === null ? 0 : (timestamp - this.lastTimestamp) / 1000;
    this.lastTimestamp = timestamp;
    if (deltaSeconds > 0) {
      this.current = this.simulation.step(deltaSeconds);
      this.stepCount++;
    }
    return this.current;
  }

  createLayer(options: {
    id: string;
    color: readonly number[] | undefined;
    visible: boolean;
    parameters?: FlowParticleLayerProps['parameters'];
    /** Linear 0-1 RGB speed bands (see `getFlowFieldPalette`); empty or omitted keeps one color. */
    speedPalette?: readonly (readonly number[])[];
    /** Native elevation multiplier for streamline Z (needs an altitude); zero drapes. */
    elevationMultiplier?: number;
  }): AnimatedFlowParticleLayer {
    const runtime = this;
    const props: FlowParticleLayerProps & {
      advance: (viewportBounds?: [number, number, number, number]) => void;
    } = {
      id: options.id,
      visible: options.visible,
      pickable: false,
      particles: () => this.current,
      advance: viewportBounds => {
        if (options.visible) this.advance(performance.now(), viewportBounds);
      },
      particleCount: this.plan.particleCount,
      bounds: this.simulationBounds,
      widthPixels: this.plan.widthPixels,
      trailSeconds: this.plan.trailSeconds,
      color: normalizeFlowFieldColor(options.color, this.plan.opacity),
      ...(options.speedPalette?.length
        ? {
            speedColoring: {
              get velocityTexture() {
                return runtime.particleTexture;
              },
              palette: options.speedPalette
            }
          }
        : {}),
      ...(this.altitudeTexture && (options.elevationMultiplier ?? 1) > 0
        ? {
            altitudeField: {
              get texture() {
                return runtime.particleAltitude!;
              },
              scale: options.elevationMultiplier ?? 1
            }
          }
        : {}),
      ...(options.parameters ? {parameters: options.parameters} : {})
    };
    return new AnimatedFlowParticleLayer(props);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.destruction.abort();
    this.simulation.destroy();
    this.viewportTexture?.destroy();
    this.viewportAltitude?.destroy();
    this.viewportReservation?.release();
    this.texture.destroy();
    this.altitudeTexture?.destroy();
    this.gpu?.grid.destroy();
  }
}
