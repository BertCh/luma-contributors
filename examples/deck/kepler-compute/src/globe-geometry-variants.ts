// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import type * as arrow from 'apache-arrow';
import {
  prepareGeometrySourceIncremental,
  type PreparedGeometryGPUVectors,
  type PreparedGeometrySource
} from '@luma.gl/engine/arrow';
import type {KeplerMemoryReservation} from '@luma-contributors/kepler';

/**
 * Zoom-banded, time-sliced, single-resident globe geometry variant.
 *
 * - The variant is prepared in slices (one source batch per step, a time budget per task) so the
 *   first globe frame is never blocked; until it is ready the caller keeps drawing the planar
 *   geometry (or the previous band).
 * - Finer cut/subdivision resolution is prepared as the zoom enters a finer band (with hysteresis).
 *   Only one variant is resident at a time; a replacement is uploaded only after its CPU
 *   preparation finished, and the previous GPU variant is destroyed after a short grace period
 *   (in-flight frames may still reference it).
 * - `release()` drops the variant when leaving globe so GeoJSON geometry is not resident twice.
 * - Every upload is admitted through the memory registry; a band that exceeds the maximum is
 *   recorded as failed (not retried) and the previous variant stays in use.
 */
export type GlobeZoomBand = Readonly<{minimumZoom: number; resolutionDegrees: number}>;

export const GLOBE_ZOOM_BANDS: readonly GlobeZoomBand[] = [
  {minimumZoom: 0, resolutionDegrees: 5},
  {minimumZoom: 3, resolutionDegrees: 2},
  {minimumZoom: 5, resolutionDegrees: 1}
];
const BAND_HYSTERESIS = 0.25;

/** Band index for a zoom; moving away from `currentBand` needs to cross its edge by the hysteresis. */
export function getGlobeZoomBand(
  zoom: number | undefined,
  currentBand?: number,
  bands: readonly GlobeZoomBand[] = GLOBE_ZOOM_BANDS
): number {
  const value = zoom ?? 0;
  let band = 0;
  for (let index = 0; index < bands.length; index++)
    if (value >= bands[index].minimumZoom) band = index;
  if (currentBand === undefined || currentBand === band) return band;
  const edge = bands[Math.max(band, currentBand)].minimumZoom;
  // Moving up requires zoom >= edge + hysteresis; moving down requires zoom <= edge - hysteresis.
  if (band > currentBand) return value >= edge + BAND_HYSTERESIS ? band : currentBand;
  return value <= edge - BAND_HYSTERESIS ? band : currentBand;
}

export type GlobeVariantState = 'idle' | 'preparing' | 'ready' | 'failed';
export type GlobeVariantStatus = Readonly<{
  state: GlobeVariantState;
  /** Resolution of the resident variant, if any. */
  residentResolutionDegrees?: number;
  /** Resolution currently being prepared. */
  preparingResolutionDegrees?: number;
  failedResolutionDegrees: readonly number[];
  notice?: string;
}>;

export type GlobeGeometryVariantsProps = {
  /** Creates a sliceable preparation for a resolution; each `next()` should be one bounded step. */
  createPreparation(resolutionDegrees: number): Generator<void, PreparedGeometrySource, void>;
  /** Uploads prepared geometry; throws on device-limit failures. */
  makeVectors(prepared: PreparedGeometrySource): PreparedGeometryGPUVectors;
  /** Registry admission. `byteLength` is an upper bound; the result is released with the variant. */
  reserve(byteLength: number): KeplerMemoryReservation;
  /** Bytes of the planar geometry, to scale the admission bound. */
  getPlanarByteLength(): number;
  requestUpdate(): void;
  /** Largest allowed variant in bytes; defaults to 256 MiB. */
  maximumBytes?: number;
  /** Time budget of one slice task; defaults to 8 ms. */
  sliceMilliseconds?: number;
  /** Injected for tests. */
  schedule?(callback: () => void, milliseconds: number): unknown;
  cancel?(handle: unknown): void;
  now?(): number;
};

type Resident = {
  vectors: PreparedGeometryGPUVectors;
  reservation: KeplerMemoryReservation;
  resolutionDegrees: number;
};

const DEFAULT_MAXIMUM_BYTES = 256 * 1024 * 1024;
const DESTROY_GRACE_MILLISECONDS = 1000;

export class GlobeGeometryVariants {
  private resident?: Resident;
  private preparing?: {
    resolutionDegrees: number;
    generator: Generator<void, PreparedGeometrySource, void>;
  };
  private band?: number;
  private timer?: unknown;
  private notice?: string;
  private destroyed = false;
  private readonly failed = new Set<number>();
  private readonly retiring = new Set<unknown>();

  constructor(private readonly props: GlobeGeometryVariantsProps) {}

  get status(): GlobeVariantStatus {
    return {
      state: this.preparing
        ? 'preparing'
        : this.resident
          ? 'ready'
          : this.failed.size
            ? 'failed'
            : 'idle',
      ...(this.resident ? {residentResolutionDegrees: this.resident.resolutionDegrees} : {}),
      ...(this.preparing ? {preparingResolutionDegrees: this.preparing.resolutionDegrees} : {}),
      failedResolutionDegrees: [...this.failed],
      ...(this.notice ? {notice: this.notice} : {})
    };
  }

  /**
   * The variant to draw now, or undefined while the first one is still being prepared. Starts the
   * background preparation for the zoom's band when the resident variant is a different resolution.
   */
  getVariant(zoom: number | undefined): PreparedGeometryGPUVectors | undefined {
    if (this.destroyed) return undefined;
    this.band = getGlobeZoomBand(zoom, this.band);
    const resolutionDegrees = GLOBE_ZOOM_BANDS[this.band].resolutionDegrees;
    if (
      this.resident?.resolutionDegrees !== resolutionDegrees &&
      this.preparing?.resolutionDegrees !== resolutionDegrees &&
      !this.failed.has(resolutionDegrees)
    )
      this.startPreparation(resolutionDegrees);
    return this.resident?.vectors;
  }

  /** The resident variant without starting any preparation (planar rebuild draws it meanwhile). */
  peekVariant(): PreparedGeometryGPUVectors | undefined {
    return this.destroyed ? undefined : this.resident?.vectors;
  }

  /** Leaving globe: drop the variant so the GeoJSON geometry is not resident twice. */
  release(): void {
    this.cancelPreparation();
    this.band = undefined;
    this.failed.clear();
    this.notice = undefined;
    this.retire(this.resident);
    this.resident = undefined;
  }

  destroy(): void {
    this.release();
    this.destroyed = true;
  }

  private startPreparation(resolutionDegrees: number): void {
    this.cancelPreparation();
    this.preparing = {
      resolutionDegrees,
      generator: this.props.createPreparation(resolutionDegrees)
    };
    this.scheduleSlice(0);
  }

  private cancelPreparation(): void {
    if (this.timer !== undefined) (this.props.cancel ?? clearTimeout)(this.timer as never);
    this.timer = undefined;
    this.preparing?.generator.return(undefined as never);
    this.preparing = undefined;
  }

  private scheduleSlice(milliseconds: number): void {
    this.timer = (this.props.schedule ?? setTimeout)(() => this.runSlice(), milliseconds);
  }

  private runSlice(): void {
    this.timer = undefined;
    const preparing = this.preparing;
    if (!preparing || this.destroyed) return;
    const now = this.props.now ?? (() => performance.now());
    const deadline = now() + (this.props.sliceMilliseconds ?? 8);
    try {
      for (let step = preparing.generator.next(); ; step = preparing.generator.next()) {
        if (step.done) {
          this.preparing = undefined;
          this.install(preparing.resolutionDegrees, step.value);
          return;
        }
        if (now() >= deadline) break;
      }
      this.scheduleSlice(0);
    } catch (error) {
      this.fail(preparing.resolutionDegrees, error);
    }
  }

  private install(resolutionDegrees: number, prepared: PreparedGeometrySource): void {
    const planarBytes = this.props.getPlanarByteLength();
    // Cuts scale with the square of the refinement relative to the 5 degree baseline.
    const refinement = (5 / resolutionDegrees) ** 2;
    const maximumBytes = this.props.maximumBytes ?? DEFAULT_MAXIMUM_BYTES;
    const budget = Math.min(maximumBytes, 8 * 1024 * 1024 + planarBytes * 16 * refinement);
    let reservation: KeplerMemoryReservation | undefined;
    let vectors: PreparedGeometryGPUVectors | undefined;
    try {
      // Peak residency is bounded: the previous variant is retired before the replacement uploads.
      const previous = this.resident;
      this.resident = undefined;
      this.retire(previous);
      reservation = this.props.reserve(budget);
      vectors = this.props.makeVectors(prepared);
      if (vectors.gpuByteLength + vectors.mappingByteLength > budget)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Globe geometry exceeds its admitted budget'
        });
      this.resident = {vectors, reservation, resolutionDegrees};
      this.notice = undefined;
      this.props.requestUpdate();
    } catch (error) {
      vectors?.destroy();
      reservation?.release();
      this.fail(resolutionDegrees, error);
    }
  }

  private fail(resolutionDegrees: number, error: unknown): void {
    this.preparing = undefined;
    this.failed.add(resolutionDegrees);
    this.notice = `Globe geometry at ${resolutionDegrees} degrees unavailable: ${
      error instanceof Error ? error.message : String(error)
    }`;
    // The planar geometry (or the previous band) keeps drawing; re-render to surface the notice.
    this.props.requestUpdate();
  }

  /** In-flight frames may still bind the buffers, so destruction follows after a grace period. */
  private retire(resident: Resident | undefined): void {
    if (!resident) return;
    const handle = (this.props.schedule ?? setTimeout)(() => {
      this.retiring.delete(handle);
      resident.vectors.destroy();
      resident.reservation.release();
    }, DESTROY_GRACE_MILLISECONDS);
    this.retiring.add(handle);
  }
}

/**
 * Sliceable preparation for a retained source column. GeoJSON (no extension) rows are
 * materialized lazily per source batch inside the generator; other encodings run `prepareSync`
 * in one step because their decode is not sliceable.
 */
export function* prepareGlobeGeometryIncrementally(
  source: arrow.Table,
  name: string,
  /** Undefined prepares the planar geometry (used to rebuild the released planar copy). */
  resolutionDegrees: number | undefined,
  prepareSync: (
    source: arrow.Table,
    name: string,
    options: {globe?: {resolutionDegrees: number}}
  ) => PreparedGeometrySource
): Generator<void, PreparedGeometrySource, void> {
  const field = source.schema.fields.find(candidate => candidate.name === name);
  if (!field || field.metadata.get('ARROW:extension:name') !== undefined) {
    yield;
    return prepareSync(
      source,
      name,
      resolutionDegrees === undefined ? {} : {globe: {resolutionDegrees}}
    );
  }
  const metadataText = field.metadata.get('ARROW:extension:metadata');
  const metadata: unknown = metadataText ? JSON.parse(metadataText) : undefined;
  const crs =
    metadata && typeof metadata === 'object' && 'crs' in metadata ? metadata.crs : undefined;
  function* readBatches() {
    for (const [index, batch] of source.batches.entries())
      yield {
        id: `geometry-${index}`,
        rows: Array.from({length: batch.numRows}, (_, row): unknown =>
          batch.getChild(name)?.get(row)
        )
      };
  }
  return yield* prepareGeometrySourceIncremental(readBatches(), {
    crs,
    ...(resolutionDegrees === undefined ? {} : {globe: {resolutionDegrees}})
  });
}
