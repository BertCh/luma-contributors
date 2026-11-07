// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import type {Texture} from '@luma.gl/core';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import type {NumericRasterBand, NumericRasterTile} from './raster-tile-ingest';

/** A GPU-resident band: opaque to this module so node tests can inject fakes. */
export type RasterBandGpuResource = Readonly<{
  byteLength: number;
  /** True when the CPU samples were converted to float32 without exact representation. */
  hasPrecisionLoss: boolean;
  /** r32float values texture; present for real GPU uploads, absent for node-test fakes. */
  valueTexture?: Texture;
  /** r8unorm validity texture (255 valid, 0 nodata); present for real GPU uploads. */
  validityTexture?: Texture;
  destroy: () => void;
}>;

export type RasterBandUploader = (
  tile: NumericRasterTile,
  band: NumericRasterBand
) => RasterBandGpuResource;

type ResidentEntry = {
  key: string;
  tileId: string;
  bandIndex: number;
  resource: RasterBandGpuResource;
  reservation?: KeplerMemoryReservation;
  lastUse: number;
};

export type RasterBandResidencyStatistics = Readonly<{
  residentBandCount: number;
  residentByteLength: number;
  budgetByteLength: number;
  uploadCount: number;
  evictionCount: number;
}>;

/** Estimated GPU bytes for one band: float32 values plus one validity byte per sample. */
export function estimateRasterBandGpuByteLength(tile: NumericRasterTile): number {
  return tile.width * tile.height * 5;
}

/**
 * Per-band GPU residency with a hard byte budget and least-recently-used eviction.
 * When a host `KeplerDatasetRegistry` is supplied, every band reserves registry memory before
 * upload, so host-wide admission control also applies; eviction releases the reservation.
 * Evicted bands remain readable from the CPU source, so identify and zonal results never change.
 */
export class RasterBandResidency {
  private readonly entries = new Map<string, ResidentEntry>();
  private residentByteLength = 0;
  private clock = 0;
  private uploadCount = 0;
  private evictionCount = 0;
  private isDestroyed = false;

  constructor(
    private readonly options: {
      budgetByteLength: number;
      upload: RasterBandUploader;
      memoryRegistry?: KeplerDatasetRegistry;
    }
  ) {}

  private static getKey(tileId: string, bandIndex: number): string {
    return `${tileId}#${bandIndex}`;
  }

  isBandResident(tileId: string, bandIndex: number): boolean {
    return this.entries.has(RasterBandResidency.getKey(tileId, bandIndex));
  }

  /** A tile counts as resident only when all requested bands are resident. */
  isTileResident(tile: NumericRasterTile, bandIndexes?: readonly number[]): boolean {
    const required = bandIndexes ?? tile.bands.map(band => band.index);
    return required.every(bandIndex => this.isBandResident(tile.id, bandIndex));
  }

  /** Returns the resident resource without touching least-recently-used order; undefined when evicted. */
  peekBandResource(tileId: string, bandIndex: number): RasterBandGpuResource | undefined {
    return this.entries.get(RasterBandResidency.getKey(tileId, bandIndex))?.resource;
  }

  ensureBandResident(tile: NumericRasterTile, bandIndex: number): RasterBandGpuResource {
    // Using a destroyed residency is a lifecycle bug in the caller.
    if (this.isDestroyed) throw new Error('Raster residency is destroyed');
    const key = RasterBandResidency.getKey(tile.id, bandIndex);
    const existing = this.entries.get(key);
    if (existing) {
      existing.lastUse = ++this.clock;
      return existing.resource;
    }
    const band = tile.bands.find(candidate => candidate.index === bandIndex);
    if (!band) throw new Error(`Raster tile ${tile.id} has no band ${bandIndex}`);
    const estimatedByteLength = estimateRasterBandGpuByteLength(tile);
    // One band larger than the whole budget can never be admitted, so evicting would be futile.
    if (estimatedByteLength > this.options.budgetByteLength)
      throw new KeplerAllocationError('memory-budget', {
        message: 'Raster band exceeds the residency budget'
      });
    while (this.residentByteLength + estimatedByteLength > this.options.budgetByteLength)
      this.evictLeastRecentlyUsed();
    const reservation = this.reserveWithEviction(estimatedByteLength);
    let resource: RasterBandGpuResource;
    try {
      resource = this.options.upload(tile, band);
    } catch (error) {
      reservation?.release();
      throw error;
    }
    if (reservation && resource.byteLength !== reservation.byteLength)
      reservation.resize(resource.byteLength);
    this.entries.set(key, {
      key,
      tileId: tile.id,
      bandIndex,
      resource,
      reservation,
      lastUse: ++this.clock
    });
    this.residentByteLength += resource.byteLength;
    this.uploadCount++;
    return resource;
  }

  private reserveWithEviction(byteLength: number): KeplerMemoryReservation | undefined {
    const registry = this.options.memoryRegistry;
    if (!registry) return undefined;
    for (;;) {
      try {
        return registry.reserveMemory(byteLength, 'gpu', 'source');
      } catch (error) {
        // Host admission refused: free our own least recently used band and retry, or give up.
        if (!this.entries.size) throw error;
        this.evictLeastRecentlyUsed();
      }
    }
  }

  private evictLeastRecentlyUsed(): void {
    let oldest: ResidentEntry | undefined;
    for (const entry of this.entries.values())
      if (!oldest || entry.lastUse < oldest.lastUse) oldest = entry;
    // The budget loop guarantees an entry exists whenever eviction is required.
    if (!oldest) throw new Error('Raster residency has nothing to evict');
    this.evictEntry(oldest);
  }

  private evictEntry(entry: ResidentEntry): void {
    this.entries.delete(entry.key);
    this.residentByteLength -= entry.resource.byteLength;
    this.evictionCount++;
    entry.resource.destroy();
    entry.reservation?.release();
  }

  evictTile(tileId: string): number {
    let evicted = 0;
    for (const entry of [...this.entries.values()])
      if (entry.tileId === tileId) {
        this.evictEntry(entry);
        evicted++;
      }
    return evicted;
  }

  getStatistics(): RasterBandResidencyStatistics {
    return {
      residentBandCount: this.entries.size,
      residentByteLength: this.residentByteLength,
      budgetByteLength: this.options.budgetByteLength,
      uploadCount: this.uploadCount,
      evictionCount: this.evictionCount
    };
  }

  /** Destroys every resident band and releases its reservation; idempotent. */
  destroy(): void {
    if (this.isDestroyed) return;
    this.isDestroyed = true;
    for (const entry of [...this.entries.values()]) this.evictEntry(entry);
    this.evictionCount = 0;
  }
}
