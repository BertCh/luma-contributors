// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isSampleRecord} from './sample-map-catalog';

/**
 * Each remote sample supplies exactly one descriptor. Saved exports can retain a generated
 * dataset ID while the descriptor uses a stable public ID. addDataToMap joins them by ID, not
 * by label or type. Bind the descriptor to that one saved ID, preserving every config reference
 * (including tooltip keys and filters). Reject ambiguous/mixed configs instead of guessing.
 */
export function alignSampleRemoteDataset(
  descriptor: Record<string, unknown> & {info: Record<string, unknown>},
  saved: Record<string, unknown>
): typeof descriptor {
  const configuration = saved['config'];
  const visState = isSampleRecord(configuration) ? configuration['visState'] : undefined;
  const layers = isSampleRecord(visState) ? visState['layers'] : undefined;
  if (!Array.isArray(layers) || !layers.length)
    throw new Error('Remote sample config has no layers to bind.');
  const layerType = {
    'vector-tile': 'vectorTile',
    'raster-tile': 'rasterTile',
    'wms-tile': 'wms'
  }[String(descriptor.info['type'])];
  if (!layerType || layers.some(layer => !isSampleRecord(layer) || layer['type'] !== layerType))
    throw new Error('Remote sample config layers do not match its dataset type.');
  const datasetIds = layers.map(layer =>
    isSampleRecord(layer) && isSampleRecord(layer['config']) ? layer['config']['dataId'] : undefined
  );
  const uniqueIds = [...new Set(datasetIds)];
  if (uniqueIds.length !== 1 || typeof uniqueIds[0] !== 'string' || !uniqueIds[0])
    throw new Error('Remote sample config must reference exactly one dataset ID.');
  return {...descriptor, info: {...descriptor.info, id: uniqueIds[0]}};
}

export type SampleMapLimitation = {
  code: 'dense-aggregation-capacity' | 'wide-integer-style';
  layerId: string;
  fallbackLayerId: string;
  reason: string;
};

/**
 * Explicit presentation fallbacks for known public samples. Source rows/fields stay unchanged,
 * and original layers are retained hidden so their settings remain inspectable and exportable.
 */
export function adaptSampleMapConfig(sampleId: string, saved: Record<string, unknown>) {
  const configuration = saved['config'] as Record<string, unknown>;
  const visState = configuration['visState'] as Record<string, unknown>;
  const limitations: SampleMapLimitation[] = [];
  const layers = (visState['layers'] as unknown[]).flatMap(layer => {
    if (!isSampleRecord(layer) || !isSampleRecord(layer['config'])) return [layer];
    const config = layer['config'];
    const channels = isSampleRecord(layer['visualChannels']) ? layer['visualChannels'] : {};
    const visConfig = isSampleRecord(config['visConfig']) ? config['visConfig'] : {};
    const field = channels['sizeField'];
    const code =
      sampleId === 'sftrees' && layer['type'] === 'hexagon' && visConfig['worldUnitSize'] === 0.05
        ? 'dense-aggregation-capacity'
        : sampleId === 'la_assessorparcels' &&
            layer['type'] === 'point' &&
            isSampleRecord(field) &&
            field['name'] === 'roll_totalvalue'
          ? 'wide-integer-style'
          : undefined;
    if (!code || typeof layer['id'] !== 'string') return [layer];
    const layerId = layer['id'];
    const fallbackLayerId = `${layerId}-sample-locations`;
    limitations.push({
      code,
      layerId,
      fallbackLayerId,
      reason:
        code === 'dense-aggregation-capacity'
          ? 'The 50 m tree hexagons over the full source extent (including distant outliers) exceed the resident dense lattice limit of 65,536 cells. Native Kepler uses sparse occupied bins. The original hexagons are retained hidden; unaggregated tree locations are shown. No rows, radius or capacity limits were changed.'
          : 'roll_totalvalue contains integers outside ±2^24. The current Kepler resident numeric style binding requires exact Float32 values for both color and sqrt size; its scalar path has no split-precision integer route. The original point encoding is retained hidden; constant-style parcel locations are shown. Exact source integers and filters are preserved.'
    });
    return [
      {...layer, config: {...config, isVisible: false}},
      {
        id: fallbackLayerId,
        type: 'point',
        config: {
          dataId: config['dataId'],
          label:
            code === 'dense-aggregation-capacity'
              ? 'Tree locations (hexagons unavailable)'
              : 'Parcel locations (constant style)',
          color: config['color'],
          columns: config['columns'],
          isVisible: config['isVisible'] !== false,
          visConfig: {radius: 3, opacity: 0.6, filled: true, outline: false}
        },
        visualChannels: {colorField: null, sizeField: null}
      }
    ];
  });
  const splitMaps = Array.isArray(visState['splitMaps'])
    ? visState['splitMaps'].map(splitMap => {
        if (!isSampleRecord(splitMap) || !isSampleRecord(splitMap['layers'])) return splitMap;
        const visibility = splitMap['layers'];
        return {
          ...splitMap,
          layers: {
            ...visibility,
            ...Object.fromEntries(
              limitations.map(limitation => [
                limitation.fallbackLayerId,
                visibility[limitation.layerId]
              ])
            )
          }
        };
      })
    : visState['splitMaps'];
  return {
    config: {
      ...saved,
      version: 'v1' as const,
      config: {
        ...configuration,
        visState: {
          ...visState,
          layers,
          splitMaps,
          ...(Array.isArray(visState['layerOrder'])
            ? {
                layerOrder: [
                  ...visState['layerOrder'],
                  ...limitations.map(limitation => limitation.fallbackLayerId)
                ]
              }
            : {})
        }
      }
    },
    limitations
  };
}
