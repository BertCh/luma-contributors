// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CompositeLayer, type Color, type DefaultProps, type LayersList} from '@deck.gl/core';
import {GeoJsonLayer, TextLayer, type GeoJsonLayerProps} from '@deck.gl/layers';
import type {Feature, Position} from 'geojson';
import {GlobeBasemapPolygonLayer} from './globe-basemap-polygon-layer';

type GlobeBasemapTileProps = GeoJsonLayerProps & {
  config?: {labels?: boolean; labelsColor?: Color};
  labelSizeUnits?: 'pixels' | 'meters' | 'common';
  labelBackground?: Color | null;
  billboard?: boolean;
};

type GlobeBasemapLabel = {position: Position; text: string};
const labelCache = new WeakMap<object, GlobeBasemapLabel[]>();

function getCountryLabels(data: GlobeBasemapTileProps['data']): GlobeBasemapLabel[] {
  if (!data || typeof data !== 'object') return [];
  const cached = labelCache.get(data);
  if (cached) return cached;
  const features: Feature[] = Array.isArray(data)
    ? data
    : 'features' in data
      ? (data.features as Feature[])
      : [];
  const labels: GlobeBasemapLabel[] = [];
  for (const feature of features) {
    const properties = feature.properties;
    if (
      !properties ||
      properties['class'] !== 'country' ||
      !['place', 'place_label'].includes(properties['layerName'])
    )
      continue;
    const text =
      (properties['layerName'] === 'place' && properties['name:en']) ||
      properties['name_en'] ||
      properties['name'] ||
      '';
    const geometry = feature.geometry;
    const positions =
      geometry?.type === 'Point'
        ? [geometry.coordinates]
        : geometry?.type === 'MultiPoint'
          ? geometry.coordinates
          : [];
    for (const position of positions) labels.push({position, text});
  }
  labelCache.set(data, labels);
  return labels;
}

/**
 * Keep deck 10 MVT's local CRS, model matrix, clipping and visible-tile traversal. The analysis
 * resident renderer expects lng/lat and Kepler channel styles, whereas this native basemap uses
 * tile-local coordinates and feature accessors. Deck's published WGSL glyphs already horizon-cull;
 * Kepler's EnhancedMultiIconLayer only supplies the obsolete GLSL equivalent.
 */
export class GlobeBasemapTileLayer extends CompositeLayer<GlobeBasemapTileProps> {
  static override layerName = 'GlobeBasemapTileLayer';
  static override defaultProps: DefaultProps = GeoJsonLayer.defaultProps;

  override renderLayers(): LayersList {
    const props: GlobeBasemapTileProps = this.props;
    const geometry = new GeoJsonLayer(
      props,
      this.getSubLayerProps({id: 'geojson', updateTriggers: props.updateTriggers}),
      {
        _subLayerProps: {
          ...props._subLayerProps,
          'polygons-fill': {
            ...props._subLayerProps?.['polygons-fill'],
            type: GlobeBasemapPolygonLayer
          }
        }
      }
    );
    if (!props.config?.labels) return [geometry];
    const labels = new TextLayer<GlobeBasemapLabel>(this.getSubLayerProps({id: 'text'}), {
      data: getCountryLabels(props.data),
      parameters: {cullMode: 'none', depthCompare: 'always', depthWriteEnabled: false},
      billboard: props.billboard ?? true,
      sizeUnits: props.labelSizeUnits ?? 'pixels',
      characterSet: 'auto',
      fontFamily: 'Monaco, monospace',
      background: Boolean(props.labelBackground),
      getBackgroundColor: props.labelBackground ?? [0, 0, 0, 0],
      getPosition: label => label.position as [number, number, number],
      getText: label => label.text,
      getSize: 10,
      getColor: props.config.labelsColor ?? [0, 0, 0, 255],
      updateTriggers: {getColor: props.config.labelsColor}
    });
    return [geometry, labels];
  }
}

export function renderGlobeBasemapTile(props: GlobeBasemapTileProps): GlobeBasemapTileLayer {
  return new GlobeBasemapTileLayer(props);
}

export function isGlobeBasemapVectorTile(layer: {id: string}): boolean {
  return layer.id.startsWith('globe-basemap-mvt-layer-');
}
