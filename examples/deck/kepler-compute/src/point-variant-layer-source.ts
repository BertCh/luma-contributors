// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {PreparedArrowAnalyticsTable} from '@luma-contributors/analysis/analysis-core';

import type {PointPositionVariantRuntime} from './point-position-variants';
import {
  planPointLayerSelector,
  type PointSelectorLayer,
  type PointSelectorPlan,
  type ResidentPositionSource
} from './resident-point-layer-selectors';

export type PointVariantDataset = {
  id: string;
  revision: number;
  prepared: PreparedArrowAnalyticsTable;
};

export type PointVariantLayerSource =
  | {status: 'primary'}
  | {
      status: 'variant';
      generation: number;
      positions: GPUVector<'float32x2'>;
      positions3D?: GPUVector<'float32x3'>;
      altitudeValidity?: GPUVector<'uint32'>;
      columns: {lng: string; lat: string; altitude?: string};
    }
  | {status: 'unavailable'; reason: string};

/** Layer types that render from variant positions (selectors cover every point-test type). */
const VARIANT_RENDER_LAYER_TYPES: ReadonlySet<string> = new Set(['point', 'icon']);

export function canRenderPointPositionVariant(layerType: string | null): boolean {
  return layerType !== null && VARIANT_RENDER_LAYER_TYPES.has(layerType);
}

/**
 * Resolves the positions a point-like layer should use. Layers matching the dataset's primary
 * lng/lat pair keep the shared positions (and drop any variant they held); a different pair gets a
 * shared, on-demand variant owned by `variants`.
 */
export function resolvePointVariantSource(
  variants: PointPositionVariantRuntime,
  layer: PointSelectorLayer,
  dataset: PointVariantDataset,
  primary: {longitude: string | undefined; latitude: string | undefined}
): PointVariantLayerSource {
  const longitude = layer.config.columns['lng']?.value;
  const latitude = layer.config.columns['lat']?.value;
  if (
    !longitude ||
    !latitude ||
    (longitude === primary.longitude && latitude === primary.latitude)
  ) {
    variants.releaseLayer(layer.id);
    return {status: 'primary'};
  }
  const altitude = layer.config.columns['altitude']?.value || undefined;
  const result = variants.resolve(layer.id, {
    datasetId: dataset.id,
    datasetRevision: dataset.revision,
    prepared: dataset.prepared,
    longitude,
    latitude,
    altitude
  });
  if (result.status === 'unavailable') return result;
  const vectors = result.variant.vectors;
  return {
    status: 'variant',
    generation: result.variant.generation,
    positions: vectors.positions,
    positions3D: vectors.positions3D,
    altitudeValidity: vectors.altitudeValidity,
    columns: {lng: longitude, lat: latitude, ...(altitude ? {altitude} : {})}
  };
}

/** Selector plan that falls back to a per-layer variant when the lng/lat pair is not primary. */
export function planPointLayerSelectorWithVariants(
  variants: PointPositionVariantRuntime,
  layer: PointSelectorLayer,
  dataset: PointVariantDataset,
  source: ResidentPositionSource
): PointSelectorPlan {
  const plan = planPointLayerSelector(layer, source);
  if (plan.status !== 'mismatch') {
    // Primary-mapped (or non-point) layers hold no variant.
    variants.releaseLayer(layer.id);
    return plan;
  }
  const variantSource = resolvePointVariantSource(variants, layer, dataset, source);
  if (variantSource.status === 'unavailable')
    return {status: 'mismatch', layerId: layer.id, reason: variantSource.reason};
  if (variantSource.status === 'primary') return plan;
  return planPointLayerSelector(layer, {
    longitude: variantSource.columns.lng,
    latitude: variantSource.columns.lat,
    positions: variantSource.positions,
    // Offset keeps variant revisions distinct from dataset revisions when a layer changes mapping.
    revision: 1_000_000_000 + variantSource.generation
  });
}

export type PointVariantBindingOverride = {
  point?: {
    positions: GPUVector<'float32x2' | 'float32x3'>;
    columns: {lat: string; lng: string; altitude?: string};
  };
  selectionMask?: GPUVector<'uint32'>;
  unsupportedReason?: string;
};

/**
 * Render-time override for a `point` layer mapped to a non-primary lng/lat pair: the variant
 * positions (XYZ when the layer maps an altitude column) replace the shared positions, and the
 * per-layer selection mask additionally hides rows whose variant altitude is not finite.
 */
export function makePointVariantBindingOverride(
  variants: PointPositionVariantRuntime,
  layer: PointSelectorLayer,
  dataset: PointVariantDataset,
  primary: {longitude: string | undefined; latitude: string | undefined},
  resolveSelectionMask: (altitudeValidity: GPUVector<'uint32'> | undefined) => GPUVector<'uint32'>
): PointVariantBindingOverride | undefined {
  if (!canRenderPointPositionVariant(layer.type)) return undefined;
  const source = resolvePointVariantSource(variants, layer, dataset, primary);
  if (source.status === 'primary') return undefined;
  if (source.status === 'unavailable') return {unsupportedReason: source.reason};
  const useAltitude = Boolean(source.columns.altitude && source.positions3D);
  return {
    point: {
      positions: useAltitude ? source.positions3D! : source.positions,
      columns: source.columns
    },
    selectionMask: resolveSelectionMask(useAltitude ? source.altitudeValidity : undefined)
  };
}
