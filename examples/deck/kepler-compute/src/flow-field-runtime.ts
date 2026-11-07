// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import {
  FlowParticleSimulation,
  isFlowParticleSimulationSupported,
  type FlowParticleStepResult
} from '@luma.gl/graph/experimental/simulation';
import {FlowParticleLayer, type FlowParticleLayerProps} from '@deck.gl-community/gpu-layers';
import type {FlowFieldGrid} from './flow-field-grid';
import type {FlowFieldGpuGrid} from './flow-field-gpu';
import type {FlowFieldSelectionContext} from './flow-field-filter';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  getFlowFieldSpawnRegion,
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
    props.advance?.(
      (this.context.viewport as {getBounds?: () => number[]} | undefined)?.getBounds?.() as
        | [number, number, number, number]
        | undefined
    );
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
  /** Normalized spawn rectangle concentrating the particle budget on the viewport, if any. */
  spawnRegion: [number, number, number, number] | null;
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
  private measuringRevision: number | null = null;
  private extentFailure?: string;
  private spawnRegion: [number, number, number, number] | null = null;

  static isSupported(device: Device): boolean {
    return isFlowParticleSimulationSupported(device);
  }

  constructor(
    readonly device: Device,
    readonly id: string,
    grid: FlowFieldRuntimeGrid,
    plan: FlowFieldPlan,
    seed: number,
    gpu?: FlowFieldGpuSource
  ) {
    this.grid = {...grid, bounds: [...grid.bounds]};
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
    return {
      stepCount: this.stepCount,
      particleCount: this.plan.particleCount,
      byteLength:
        this.simulation.byteLength +
        (this.gpu ? this.gpu.grid.byteLength : (this.grid.texels?.byteLength ?? 0)),
      velocityScale: this.gpu && this.plan.velocityScaleNumerator ? null : this.plan.velocityScale,
      gridMode: this.gpu ? 'gpu' : 'cpu',
      gridBuildCount: this.gpu?.grid.buildCount ?? 0,
      gridReadbackCount: this.gpu?.grid.readbackCount ?? 0,
      selectionRevision: this.builtRevision,
      bounds: [...this.grid.bounds],
      extentReadbackCount: this.gpu?.grid.extentReadbackCount ?? 0,
      ...(this.extentFailure ? {extentFailure: this.extentFailure} : {}),
      spawnRegion: this.spawnRegion,
      hasAltitude: Boolean(this.altitudeTexture)
    };
  }

  /**
   * Re-bins the GPU grid when the layer mask revision changed; the field texture is updated in
   * place. Under a filter the bounds first follow the selected rows: a GPU min/max reduction is
   * read back (20 bytes, no mask or grid readback) and the build runs once it resolves, so the old
   * field stays visible meanwhile. Returns true when a build was submitted synchronously.
   */
  rebuildGpuGrid(): boolean {
    const gpu = this.gpu;
    if (!gpu || this.destroyed) return false;
    const revision = gpu.selection ? gpu.selection.getRevision() : null;
    if (gpu.grid.buildCount > 0 && revision === this.builtRevision) return false;
    if (!gpu.selection) {
      this.applyBounds(gpu.grid.options.extent.bounds);
      gpu.grid.emptyField = false;
      gpu.grid.build(this.texture, undefined, this.altitudeTexture);
      this.builtRevision = null;
      return true;
    }
    if (this.measuringRevision === revision) return false;
    this.measuringRevision = revision;
    const selection = gpu.selection;
    void this.measureStableBounds(selection.mask).then(
      bounds => {
        this.finishFilteredBuild(revision, bounds ?? undefined, true);
      },
      error => {
        this.extentFailure = error instanceof Error ? error.message : String(error);
        this.finishFilteredBuild(revision, undefined, false);
      }
    );
    return false;
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
    bounds: [number, number, number, number] | undefined,
    measured: boolean
  ): void {
    const gpu = this.gpu;
    if (!gpu || this.destroyed) return;
    this.measuringRevision = null;
    if (measured) {
      this.extentFailure = undefined;
      // Too few selected rows for a grid: keep the old bounds and show an invalid (empty) field.
      gpu.grid.emptyField = !bounds;
      this.applyBounds(bounds ?? this.grid.bounds);
    } else {
      gpu.grid.emptyField = false;
      this.applyBounds(gpu.grid.options.extent.bounds);
    }
    gpu.grid.build(this.texture, gpu.selection?.mask, this.altitudeTexture);
    this.builtRevision = revision;
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
    // Not yet constructed while the first build runs inside the constructor.
    this.simulation?.setField({texture: this.texture, bounds, coordinates: 'lnglat'});
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    if (paused) this.lastTimestamp = null;
  }

  /** Advances at most once per animation frame; gaps beyond eight substeps are dropped by the simulation. */
  advance(
    timestamp: number = performance.now(),
    viewportBounds?: readonly [number, number, number, number]
  ): FlowParticleStepResult {
    if (this.destroyed || timestamp === this.lastFrameTimestamp) return this.current;
    this.lastFrameTimestamp = timestamp;
    // Filter and time-filter changes drive the field directly through the GPU mask.
    this.rebuildGpuGrid();
    // The particle budget is per screen, as native lines per screen: spawn inside the viewport.
    const region = getFlowFieldSpawnRegion(viewportBounds, this.grid.bounds);
    this.simulation.setSpawnRegion(region);
    this.spawnRegion = region ?? null;
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
      bounds: this.grid.bounds,
      widthPixels: this.plan.widthPixels,
      trailSeconds: this.plan.trailSeconds,
      color: normalizeFlowFieldColor(options.color, this.plan.opacity),
      ...(options.speedPalette?.length
        ? {speedColoring: {velocityTexture: this.texture, palette: options.speedPalette}}
        : {}),
      ...(this.altitudeTexture && (options.elevationMultiplier ?? 1) > 0
        ? {
            altitudeField: {
              texture: this.altitudeTexture,
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
    this.simulation.destroy();
    this.texture.destroy();
    this.altitudeTexture?.destroy();
    this.gpu?.grid.destroy();
  }
}
