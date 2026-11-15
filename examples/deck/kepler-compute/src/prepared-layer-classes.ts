// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerClassesType, Layer as KeplerLayer} from '@kepler.gl/layers';
import {DatasetType} from '@kepler.gl/constants';

const TABLELESS_DATASET_TYPES = new Set<string>([
  DatasetType.TILE_3D,
  DatasetType.BITMAP,
  DatasetType.VECTOR_TILE,
  DatasetType.RASTER_TILE,
  DatasetType.WMS_TILE
]);

export const PREPARE_DATA_MESSAGE =
  'Open Add Data → Prepare data, select the imported dataset, and publish it to show it on the map.';

export const AUTO_PREPARE_FAILURE_PREFIX = 'Automatic preparation failed:';

/** Keep native ingestion available without sending its GLSL-only extensions to WebGPU. */
export function createPreparedLayerClasses(
  classes: LayerClassesType,
  options: {
    isPrepared(datasetId: string): boolean;
    /** Automatic preparation is queued or running: draw nothing, and show no notice yet. */
    isPending?(datasetId: string): boolean;
    /** Automatic preparation failed with this reason (shown instead of the manual guidance). */
    getFailure?(datasetId: string): string | undefined;
    onPreparationRequired(layerId: string, reason: string | null): void;
  }
): LayerClassesType {
  const pendingLayers = new Set<string>();
  function guardLayer(NativeLayer: typeof KeplerLayer): typeof KeplerLayer {
    return class PreparedLayer extends NativeLayer {
      override renderLayer(
        ...parameters: Parameters<KeplerLayer['renderLayer']>
      ): ReturnType<KeplerLayer['renderLayer']> {
        const datasetId = this.config.dataId;
        // All five native streamed/image dataset types carry URLs or tile metadata rather than
        // a table to prepare. Blocking vector/raster/WMS here prevents their resident tile/WGSL
        // adapters from ever running, and cannot be remedied by publishing tabular fields.
        const datasetType = parameters[0]?.dataset?.type as string | undefined;
        const isTableless = datasetType !== undefined && TABLELESS_DATASET_TYPES.has(datasetType);
        if (datasetId && !isTableless && !options.isPrepared(datasetId)) {
          if (options.isPending?.(datasetId)) return [];
          const failure = options.getFailure?.(datasetId);
          if (!pendingLayers.has(this.id)) {
            pendingLayers.add(this.id);
            options.onPreparationRequired(
              this.id,
              failure ? `${AUTO_PREPARE_FAILURE_PREFIX} ${failure}` : PREPARE_DATA_MESSAGE
            );
          }
          return [];
        }
        if (pendingLayers.delete(this.id)) options.onPreparationRequired(this.id, null);
        return super.renderLayer(...parameters);
      }
    };
  }
  return Object.fromEntries(
    Object.entries(classes).map(([name, NativeLayer]) => [name, guardLayer(NativeLayer)])
  ) as LayerClassesType;
}
