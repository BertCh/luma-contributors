// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterLiveController} from './raster-live-controller';
import type {RasterIdentifyResult} from './raster-identify';
import type {RasterGpuSampleCache} from './raster-gpu-identify';

type HoverRow = {name: string; value: string};
type HoverData = {rasterFeatureData?: HoverRow[]} | null;
type HoverLayer = {
  id: string;
  getHoverData(
    object: unknown,
    dataContainer: unknown,
    fields: unknown,
    animationConfig: unknown,
    hoverInfo?: {coordinate?: number[]}
  ): HoverData;
};

const PATCH_MARKER = Symbol.for('luma.kepler.rasterLiveTooltip');
let activeController: RasterLiveController | undefined;
let activeGpuCache: RasterGpuSampleCache | undefined;

/** Rows appended to Kepler's native raster tooltip by the numeric identify. */
export function formatRasterIdentifyRows(result: RasterIdentifyResult): HoverRow[] {
  const statusLabels: Record<RasterIdentifyResult['status'], string> = {
    value: 'Valid sample',
    nodata: 'Nodata',
    outside: 'Outside loaded tiles',
    evicted: 'Tile evicted'
  };
  const rows: HoverRow[] = [{name: 'Pixel status', value: statusLabels[result.status]}];
  if (result.tileId !== undefined && result.column !== undefined && result.row !== undefined)
    rows.push({
      name: 'Tile pixel',
      value: `${result.tileId.split('/').slice(-3).join('/')} (${result.column}, ${result.row})`
    });
  if (result.status === 'nodata' && result.bandValues?.length)
    rows.push({
      name: 'Valid bands',
      value: String(result.bandValues.filter(v => v !== null).length)
    });
  return rows;
}

/** Value read back from the resident texture; the first hover requests it, the next one shows it. */
function formatGpuSampleRows(result: RasterIdentifyResult): HoverRow[] {
  const cache = activeGpuCache;
  if (
    !cache ||
    result.tileId === undefined ||
    result.column === undefined ||
    result.row === undefined ||
    (result.status !== 'value' && result.status !== 'nodata')
  )
    return [];
  const sample = cache.getSample(result.tileId, result.column, result.row);
  if (!sample) {
    cache.request(result.tileId, result.column, result.row);
    return [];
  }
  return [{name: 'GPU sample', value: sample.isValid ? String(sample.value) : 'Nodata'}];
}

/**
 * Wraps `getHoverData` on Kepler's `RasterTileLayer` prototype (once) so the native tooltip also
 * reports the numeric identify status (valid, nodata, outside) from the live numeric source.
 * Kepler's own band rows are kept unchanged; this only appends.
 */
export function installRasterIdentifyTooltip(
  sampleLayer: object,
  controller: RasterLiveController,
  gpuCache?: RasterGpuSampleCache
): void {
  activeController = controller;
  activeGpuCache = gpuCache;
  const prototype = Object.getPrototypeOf(sampleLayer) as HoverLayer & {
    [PATCH_MARKER]?: boolean;
  };
  if (prototype[PATCH_MARKER]) return;
  const original = prototype.getHoverData;
  prototype.getHoverData = function getHoverData(
    this: HoverLayer,
    object,
    dataContainer,
    fields,
    animationConfig,
    hoverInfo
  ) {
    const data = original.call(this, object, dataContainer, fields, animationConfig, hoverInfo);
    const coordinate =
      hoverInfo?.coordinate ?? (object as {coordinate?: number[]} | undefined)?.coordinate;
    if (!data?.rasterFeatureData || !coordinate || !activeController) return data;
    // A WMS feature info object is not a pixel hover.
    if ((object as {wmsFeatureInfo?: unknown} | undefined)?.wmsFeatureInfo) return data;
    const result = activeController.identify(this.id, coordinate[0], coordinate[1]);
    return result
      ? {
          ...data,
          rasterFeatureData: [
            ...data.rasterFeatureData,
            ...formatRasterIdentifyRows(result),
            ...formatGpuSampleRows(result)
          ]
        }
      : data;
  };
  prototype[PATCH_MARKER] = true;
}
