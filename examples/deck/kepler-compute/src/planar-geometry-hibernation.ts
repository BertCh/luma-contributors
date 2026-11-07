// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Releases the planar GeoJSON geometry copy (GPU vectors, source-row gathers, GPU reservation)
 * while a globe view draws from its globe variant, and rebuilds it in time slices on exit.
 *
 * - Release is deferred by a grace period after the first globe frame that has a ready variant,
 *   so frames that were still bound to planar buffers retire first.
 * - On exit the planar copy is prepared from the retained source in slices; until it is installed
 *   the caller keeps drawing the (subdivided) globe variant, so there is no blank frame.
 * - A failed restore (admission or device limit) is not retried; the globe variant keeps drawing
 *   and the notice explains why.
 * - Dependents keep working through lightweight metadata the owner caches before release
 *   (bounds, invalid row count, byte length, feature types, polygon analysis source).
 */
export type PlanarGeometryState = 'resident' | 'released' | 'restoring';

export type PlanarGeometryHibernationProps = {
  /** Destroys the planar runtime, its vectors and its GPU reservation. */
  releasePlanar(): void;
  /** One bounded step per `next()`; the return value is passed to `installPlanar`. */
  createPreparation(): Generator<void, unknown, void>;
  /** Uploads and admits the planar geometry; throws on failure. */
  installPlanar(prepared: unknown): void;
  requestUpdate(): void;
  graceMilliseconds?: number;
  sliceMilliseconds?: number;
  schedule?(callback: () => void, milliseconds: number): unknown;
  cancel?(handle: unknown): void;
  now?(): number;
};

export type PlanarGeometryStatus = Readonly<{state: PlanarGeometryState; notice?: string}>;

const DEFAULT_GRACE_MILLISECONDS = 1500;

export class PlanarGeometryHibernation {
  private state: PlanarGeometryState = 'resident';
  private releaseTimer?: unknown;
  private sliceTimer?: unknown;
  private generator?: Generator<void, unknown, void>;
  private notice?: string;
  private restoreFailed = false;
  private destroyed = false;

  constructor(private readonly props: PlanarGeometryHibernationProps) {}

  get status(): PlanarGeometryStatus {
    return {state: this.state, ...(this.notice ? {notice: this.notice} : {})};
  }

  /** A globe frame is being drawn; `variantReady` says the globe variant can stand in for planar. */
  noteGlobeFrame(variantReady: boolean): void {
    if (this.destroyed) return;
    if (this.state === 'restoring') {
      // Back in globe before the rebuild finished: the globe variant still draws, drop the rebuild.
      if (variantReady) {
        this.cancelRestore();
        this.state = 'released';
      }
      return;
    }
    // The globe variant vanished (a replacement band failed): the planar copy has to come back.
    if (this.state === 'released' && !variantReady) {
      this.noteFlatFrame();
      return;
    }
    if (this.state !== 'resident' || !variantReady || this.releaseTimer !== undefined) return;
    this.releaseTimer = this.schedule(() => {
      this.releaseTimer = undefined;
      if (this.destroyed || this.state !== 'resident') return;
      this.props.releasePlanar();
      this.state = 'released';
      this.restoreFailed = false;
      this.notice = undefined;
      this.props.requestUpdate();
    }, this.props.graceMilliseconds ?? DEFAULT_GRACE_MILLISECONDS);
  }

  /**
   * A flat (non-globe) frame is being drawn. Returns the state; only `resident` means the planar
   * geometry can be drawn now. Otherwise the caller keeps drawing the globe variant.
   */
  noteFlatFrame(): PlanarGeometryState {
    if (this.destroyed) return this.state;
    if (this.releaseTimer !== undefined) {
      this.cancelTimer(this.releaseTimer);
      this.releaseTimer = undefined;
    }
    if (this.state === 'released' && !this.restoreFailed) {
      this.state = 'restoring';
      this.generator = this.props.createPreparation();
      this.sliceTimer = this.schedule(() => this.runSlice(), 0);
    }
    return this.state;
  }

  destroy(): void {
    this.destroyed = true;
    if (this.releaseTimer !== undefined) this.cancelTimer(this.releaseTimer);
    this.releaseTimer = undefined;
    this.cancelRestore();
  }

  private runSlice(): void {
    this.sliceTimer = undefined;
    const generator = this.generator;
    if (!generator || this.destroyed || this.state !== 'restoring') return;
    const now = this.props.now ?? (() => performance.now());
    const deadline = now() + (this.props.sliceMilliseconds ?? 8);
    try {
      for (let step = generator.next(); ; step = generator.next()) {
        if (step.done) {
          this.generator = undefined;
          this.props.installPlanar(step.value);
          this.state = 'resident';
          this.notice = undefined;
          this.props.requestUpdate();
          return;
        }
        if (now() >= deadline) break;
      }
      this.sliceTimer = this.schedule(() => this.runSlice(), 0);
    } catch (error) {
      this.generator = undefined;
      this.state = 'released';
      this.restoreFailed = true;
      this.notice = `Planar geometry could not be restored: ${
        error instanceof Error ? error.message : String(error)
      }; the globe variant keeps drawing`;
      this.props.requestUpdate();
    }
  }

  private cancelRestore(): void {
    if (this.sliceTimer !== undefined) this.cancelTimer(this.sliceTimer);
    this.sliceTimer = undefined;
    this.generator?.return(undefined);
    this.generator = undefined;
  }

  private schedule(callback: () => void, milliseconds: number): unknown {
    return (this.props.schedule ?? setTimeout)(callback, milliseconds);
  }

  private cancelTimer(handle: unknown): void {
    (this.props.cancel ?? clearTimeout)(handle as never);
  }
}
