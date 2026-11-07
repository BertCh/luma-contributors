// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUKeplerSpatialSelection} from '@luma-contributors/kepler';
import {makeBrushCircleSelection} from './resident-brush-geometry';

/** Largest mask a brushed-row readback accepts; larger tables must reduce on the GPU instead. */
export const MAXIMUM_BRUSH_READBACK_ROWS = 2 ** 20;

export type ResidentBrushSelectionProps = {
  device: Device;
  datasetId: string;
  /** Borrowed resident point positions (longitude, latitude); never read back. */
  positions: GPUVector<'float32x2'>;
  admitMemory?: (byteLength: number) => void;
};

/**
 * Cursor-driven brush mask over resident points. A circle polygon around the cursor is tested on
 * the GPU against the borrowed positions and written to a source-aligned uint32 mask. The mask is
 * a selection scope that is independent of the filter mask: it never removes rows, and it carries
 * its own revision so consumers can detect a stale brushed set.
 */
export class ResidentBrushSelection {
  readonly datasetId: string;
  private readonly device: Device;
  private readonly selection: GPUKeplerSpatialSelection;
  private revision = 0;
  private active = false;
  private unsupportedReason = '';
  private destroyed = false;
  private readonly listeners = new Set<() => void>();

  constructor(props: ResidentBrushSelectionProps) {
    this.datasetId = props.datasetId;
    this.device = props.device;
    this.selection = new GPUKeplerSpatialSelection({
      device: props.device,
      id: `brush-${props.datasetId}`,
      positions: props.positions,
      selection: makeBrushCircleSelection(null, 0).selection,
      coordinateSpace: 'planar',
      admitMemory: props.admitMemory
    });
  }

  /** Source-aligned brushed flags; borrowed by consumers until destroy(). */
  get mask(): GPUVector<'uint32'> {
    return this.selection.mask;
  }
  get currentRevision(): number {
    return this.revision;
  }
  get isActive(): boolean {
    return this.active;
  }
  get unsupported(): string {
    return this.unsupportedReason;
  }

  /** Updates the circle and recomputes the mask in one submitted GPU pass. */
  setBrush(center: readonly [number, number] | null, radiusMeters: number): number {
    if (this.destroyed) return this.revision;
    const circle = makeBrushCircleSelection(center, radiusMeters);
    this.selection.updateSelection(circle.selection);
    this.active = circle.status === 'ready';
    this.unsupportedReason = circle.status === 'unsupported' ? circle.reason : '';
    const encoder = this.device.createCommandEncoder();
    this.selection.encode(encoder);
    this.device.submit(encoder.finish());
    this.revision++;
    for (const listener of this.listeners) listener();
    return this.revision;
  }

  isCurrent(revision: number): boolean {
    return !this.destroyed && revision === this.revision;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Explicit bounded readback of brushed source row indices for table and chart consumers. Returns
   * null if the brush changed during the read. Refused above MAXIMUM_BRUSH_READBACK_ROWS rows.
   */
  async readBrushedRows(limit = 1024): Promise<{rows: Uint32Array; count: number} | null> {
    const rowCount = this.mask.length;
    // Brushed readback is bounded to 1,048,576 rows; use GPU reductions for larger tables.
    if (rowCount > MAXIMUM_BRUSH_READBACK_ROWS) throw new Error('Brush readback is too large');
    const revision = this.revision;
    const rows: number[] = [];
    let count = 0;
    let rowOffset = 0;
    for (const chunk of this.mask.data) {
      if (chunk.length === 0) continue;
      const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
      if (!this.isCurrent(revision)) return null;
      const flags = new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length);
      for (let index = 0; index < flags.length; index++) {
        if (!flags[index]) continue;
        count++;
        if (rows.length < limit) rows.push(rowOffset + index);
      }
      rowOffset += chunk.length;
    }
    return {rows: Uint32Array.from(rows), count};
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.listeners.clear();
    this.selection.destroy();
  }
}
