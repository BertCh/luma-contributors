// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUNetworkIsochrones} parameter view. */
export const GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH = 12;

/** Per-frame settings of {@link GPUNetworkIsochrones}. */
export type GPUNetworkIsochronesSettings = {
  /**
   * Number of active ascending cost breaks, at most `costBreaks.length`. `breakCount + 1` bands exist:
   * band `k` holds cost in `(costBreaks[k - 1], costBreaks[k]]`, band 0 everything at or below the
   * first break and the last band everything above the last active break (including unreached
   * pixels). A cost exactly on a break is within it, the rule of `GPUNetworkReachability` bands.
   */
  breakCount: number;
  /**
   * Raster extent `[minX, minY, maxX, maxY]` in the context's length unit. Row 0 is at `minY`.
   * The packed float32 endpoints must be finite with a positive width and height (an extent of
   * width 0.25 at 1e7 rounds to zero width and is rejected).
   */
  extent: readonly [number, number, number, number];
  /**
   * Walking buffer radius in the context's length unit. Pixels within it of an edge sample get that
   * sample's cost plus `walkCostPerUnit` times the distance. The effective radius is at least half
   * a pixel diagonal so every sample marks its own pixel. Defaults to 0.
   */
  bufferRadius?: number;
  /** Cost per context length unit walked off the network inside the buffer. Defaults to 0. */
  walkCostPerUnit?: number;
  /** First band emitted as geometry. Defaults to 0. */
  firstBand?: number;
  /** Last band (inclusive) emitted as geometry. Defaults to `breakCount - 1`, so unreached space has no triangles. */
  lastBand?: number;
  /**
   * Cost limit of the cell-outline path: nodes with cost `<=` it contribute their cell. Defaults
   * to `Infinity`, every reached node.
   */
  cellCostLimit?: number;
};

/**
 * Packs per-frame {@link GPUNetworkIsochrones} parameters.
 *
 * Layout (float32): `[breakCount, minX, minY, maxX, maxY, bufferRadius, walkCostPerUnit,
 * firstBand, lastBand, cellCostLimit, 0, 0]`. `cellCostLimit` is capped to a finite float32 so a
 * limit of `Infinity` stays well defined in WGSL.
 *
 * @param settings Break count, extent, buffer, band window and cell cost limit.
 * @param target Optional destination of at least 12 elements.
 * @throws If the target is too short or a value is invalid (the message starts with `GPUNetworkIsochrones`).
 */
export function getGPUNetworkIsochronesParameterValues(
  settings: GPUNetworkIsochronesSettings,
  target: Float32Array = new Float32Array(GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH) {
    throw new Error(
      `GPUNetworkIsochrones parameter target must hold ${GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} elements`
    );
  }
  const {breakCount} = settings;
  const firstBand = settings.firstBand ?? 0;
  const lastBand = settings.lastBand ?? Math.max(breakCount - 1, 0);
  for (const [name, count] of [
    ['breakCount', breakCount],
    ['firstBand', firstBand],
    ['lastBand', lastBand]
  ] as const) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`GPUNetworkIsochrones ${name} must be a non-negative integer`);
    }
  }
  const [minX, minY, maxX, maxY] = settings.extent;
  if (![minX, minY, maxX, maxY].every(Number.isFinite) || maxX <= minX || maxY <= minY) {
    throw new Error('GPUNetworkIsochrones extent must be finite with maxX > minX and maxY > minY');
  }
  // The shader reads float32 uniforms: a JavaScript extent that is valid can still round to a
  // zero-width span or overflow to infinity once packed, so validate the packed values too.
  const packedExtent = [minX, minY, maxX, maxY].map(Math.fround);
  const packedWidth = Math.fround(packedExtent[2] - packedExtent[0]);
  const packedHeight = Math.fround(packedExtent[3] - packedExtent[1]);
  if (
    !packedExtent.every(Number.isFinite) ||
    !(packedWidth > 0) ||
    !(packedHeight > 0) ||
    !Number.isFinite(packedWidth) ||
    !Number.isFinite(packedHeight)
  ) {
    throw new Error(
      `GPUNetworkIsochrones extent [${settings.extent.join(', ')}] becomes ` +
        `[${packedExtent.join(', ')}] as float32; it must stay finite with a positive float32 width and height`
    );
  }
  const bufferRadius = settings.bufferRadius ?? 0;
  const walkCostPerUnit = settings.walkCostPerUnit ?? 0;
  if (!(bufferRadius >= 0) || !Number.isFinite(Math.fround(bufferRadius))) {
    throw new Error(
      'GPUNetworkIsochrones bufferRadius must be a non-negative number finite as float32'
    );
  }
  if (!(walkCostPerUnit >= 0) || !Number.isFinite(Math.fround(walkCostPerUnit))) {
    throw new Error(
      'GPUNetworkIsochrones walkCostPerUnit must be a non-negative number finite as float32'
    );
  }
  target.fill(0, 0, GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH);
  target[0] = breakCount;
  target[1] = minX;
  target[2] = minY;
  target[3] = maxX;
  target[4] = maxY;
  target[5] = bufferRadius;
  target[6] = walkCostPerUnit;
  target[7] = firstBand;
  target[8] = lastBand;
  target[9] = Math.min(settings.cellCostLimit ?? Infinity, 3.0e38);
  return target;
}
