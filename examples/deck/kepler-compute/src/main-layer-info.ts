// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Layer} from '@deck.gl/core';
import {getResidentHeatmapReadiness} from './resident-heatmap-layer';

export function getRenderedLayerInfo(value: unknown): {
  id: string;
  renderId: string;
  layerClass: string;
  extruded?: boolean;
  filled?: boolean;
  wireframe?: boolean;
  elevationScale?: number;
  widthScale?: number;
  capRounded?: boolean;
  jointRounded?: boolean;
  getElevation?: ReturnType<typeof getGPUAttributeInfo>;
  getElevationOffset?: ReturnType<typeof getGPUAttributeInfo>;
  getWidth?: ReturnType<typeof getGPUAttributeInfo>;
  getPosition?: ReturnType<typeof getGPUAttributeInfo>;
  depthCompare?: unknown;
  depthWriteEnabled?: unknown;
  opacity?: number;
  visible: boolean;
  loaded: boolean;
  heatmapReadiness?: ReturnType<typeof getResidentHeatmapReadiness>;
  iconGeometryBufferId?: string;
  positionBufferIds?: string[];
  indexBufferIds?: string[];
  maskBufferIds?: string[];
  labelBufferIds?: string[];
}[] {
  if (Array.isArray(value)) return value.flatMap(getRenderedLayerInfo);
  if (
    !value ||
    typeof value !== 'object' ||
    !('id' in value) ||
    typeof value.id !== 'string' ||
    !('props' in value)
  )
    return [];
  const properties = value.props;
  return [
    {
      id:
        properties &&
        typeof properties === 'object' &&
        'nativeLayerId' in properties &&
        typeof properties.nativeLayerId === 'string'
          ? properties.nativeLayerId
          : value.id,
      renderId: value.id,
      layerClass: value.constructor.name,
      ...(properties && typeof properties === 'object' && 'extruded' in properties
        ? {extruded: Boolean(properties.extruded)}
        : {}),
      ...(properties && typeof properties === 'object'
        ? {
            ...('filled' in properties ? {filled: Boolean(properties.filled)} : {}),
            ...('getPosition' in properties
              ? {getPosition: getGPUAttributeInfo(properties.getPosition)}
              : {}),
            ...('parameters' in properties &&
            properties.parameters &&
            typeof properties.parameters === 'object'
              ? {
                  ...('depthCompare' in properties.parameters
                    ? {depthCompare: properties.parameters.depthCompare}
                    : {}),
                  ...('depthWriteEnabled' in properties.parameters
                    ? {depthWriteEnabled: properties.parameters.depthWriteEnabled}
                    : {})
                }
              : {}),
            ...('wireframe' in properties ? {wireframe: Boolean(properties.wireframe)} : {}),
            ...('opacity' in properties && typeof properties.opacity === 'number'
              ? {opacity: properties.opacity}
              : {}),
            ...('elevationScale' in properties && typeof properties.elevationScale === 'number'
              ? {elevationScale: properties.elevationScale}
              : {}),
            ...('widthScale' in properties && typeof properties.widthScale === 'number'
              ? {widthScale: properties.widthScale}
              : {}),
            ...('capRounded' in properties && typeof properties.capRounded === 'boolean'
              ? {capRounded: properties.capRounded}
              : {}),
            ...('jointRounded' in properties && typeof properties.jointRounded === 'boolean'
              ? {jointRounded: properties.jointRounded}
              : {}),
            getElevation:
              'getElevation' in properties
                ? getGPUAttributeInfo(properties.getElevation)
                : undefined,
            getElevationOffset:
              'getElevationOffset' in properties
                ? getGPUAttributeInfo(properties.getElevationOffset)
                : undefined,
            getWidth:
              'getWidth' in properties ? getGPUAttributeInfo(properties.getWidth) : undefined,
            positionBufferIds:
              'positions' in properties
                ? getGPUVectorBufferIds(properties.positions)
                : 'getPosition' in properties
                  ? getGPUVectorBufferIds(properties.getPosition)
                  : [],
            indexBufferIds:
              'indices' in properties ? getGPUVectorBufferIds(properties.indices) : [],
            maskBufferIds:
              'selectionMask' in properties ? getGPUVectorBufferIds(properties.selectionMask) : [],
            labelBufferIds: getPointLabelBuffers(properties).map(buffer => buffer.id)
          }
        : {}),
      visible: Boolean(
        properties &&
          typeof properties === 'object' &&
          'visible' in properties &&
          properties.visible
      ),
      loaded: 'isLoaded' in value && Boolean(value.isLoaded),
      heatmapReadiness: value instanceof Layer ? getResidentHeatmapReadiness(value) : undefined,
      ...(properties && typeof properties === 'object' && 'iconGeometry' in properties
        ? {
            iconGeometryBufferId: getGPUDataBufferId(properties.iconGeometry)
          }
        : {})
    }
  ];
}

export function getGPUAttributeInfo(
  value: unknown
): number | {format: string; length: number; bufferIds: string[]} | undefined {
  if (typeof value === 'number') return value;
  if (
    !value ||
    typeof value !== 'object' ||
    !('format' in value) ||
    typeof value.format !== 'string' ||
    !('length' in value) ||
    typeof value.length !== 'number'
  )
    return undefined;
  return {format: value.format, length: value.length, bufferIds: getGPUVectorBufferIds(value)};
}

export function getGPUDataBufferId(value: unknown): string | undefined {
  return getGPUDataBuffer(value)?.id;
}

export function getGPUDataBuffer(value: unknown) {
  if (!value || typeof value !== 'object' || !('buffer' in value)) return undefined;
  const buffer = value.buffer;
  return buffer && typeof buffer === 'object' && 'id' in buffer && typeof buffer.id === 'string'
    ? {id: buffer.id, resource: buffer}
    : undefined;
}

export function getPointLabelBuffers(properties: object) {
  if (!('batches' in properties) || !Array.isArray(properties.batches)) return [];
  return properties.batches.flatMap((batch: unknown) => {
    if (
      !batch ||
      typeof batch !== 'object' ||
      !('glyphLayouts' in batch) ||
      !('glyphRows' in batch)
    )
      return [];
    return [getGPUDataBuffer(batch.glyphLayouts), getGPUDataBuffer(batch.glyphRows)].filter(
      buffer => buffer !== undefined
    );
  });
}

export function getPointLabelResourceEntries(value: unknown): [string, object[]][] {
  if (Array.isArray(value)) return value.flatMap(getPointLabelResourceEntries);
  if (
    !value ||
    typeof value !== 'object' ||
    !('id' in value) ||
    typeof value.id !== 'string' ||
    !('props' in value) ||
    !value.props ||
    typeof value.props !== 'object'
  )
    return [];
  const buffers = getPointLabelBuffers(value.props);
  return buffers.length ? [[value.id, buffers.map(buffer => buffer.resource)]] : [];
}

export function getGPUVectorBufferIds(value: unknown): string[] {
  if (!value || typeof value !== 'object' || !('data' in value) || !Array.isArray(value.data))
    return [];
  return value.data.flatMap(chunk => {
    const identifier = getGPUDataBufferId(chunk);
    return identifier ? [identifier] : [];
  });
}
