// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RasterEmbeddingsPanel} from './raster-embeddings-panel';
import type {createRasterLiveHost} from './raster-live-host';
import {RasterTimeSeriesPanel} from './raster-timeseries-panel';
import {RasterTerrainPanel} from './raster-terrain-panel';
import {RasterZonalPanel} from './raster-zonal-panel';
import {WmsTimePicker, type WmsTimePickerLayer} from './wms-time-picker';

export type RasterToolsPanelProps = Readonly<{
  host: ReturnType<typeof createRasterLiveHost>;
  rasterLayers: readonly {id: string; label: string}[];
  wmsLayers: readonly WmsTimePickerLayer[];
  polygonDatasets: readonly {id: string; label: string}[];
  isDerivedGridAvailable?: boolean;
}>;

/** Raster and WMS tools: zonal statistics and WMS TIME. Renders nothing without a relevant layer. */
export function RasterToolsPanel({
  host,
  rasterLayers,
  wmsLayers,
  polygonDatasets,
  isDerivedGridAvailable = false
}: RasterToolsPanelProps) {
  return (
    <>
      <RasterEmbeddingsPanel host={host.embeddings} rasterLayers={rasterLayers} />
      <RasterTimeSeriesPanel
        host={host.timeSeries}
        rasterLayers={rasterLayers}
        polygonDatasets={polygonDatasets}
      />
      {rasterLayers.length > 0 ? (
        <RasterZonalPanel
          rasterLayers={rasterLayers}
          polygonDatasets={polygonDatasets}
          run={request => host.runZonalAction(request)}
        />
      ) : null}
      {rasterLayers.length > 0 || isDerivedGridAvailable ? (
        <RasterTerrainPanel
          terrain={host.terrain}
          rasterLayers={rasterLayers}
          isDerivedGridAvailable={isDerivedGridAvailable}
        />
      ) : null}
      {wmsLayers.length > 0 ? (
        <WmsTimePicker
          layers={wmsLayers}
          setTime={(layerId, time, extentText) => host.wms.setTime(layerId, time, extentText)}
          getTime={layerId => host.wms.getState(layerId).time}
        />
      ) : null}
    </>
  );
}
