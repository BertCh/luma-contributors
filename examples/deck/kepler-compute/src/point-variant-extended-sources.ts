// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerDatasetRegistry} from '@luma-contributors/kepler';

import {AggregationRuntime, type ResidentAggregationBinding} from './aggregation-runtime';
import type {PointPositionVariantRuntime} from './point-position-variants';
import {resolvePointVariantSource, type PointVariantDataset} from './point-variant-layer-source';
import type {PointSelectorLayer} from './resident-point-layer-selectors';

type PrimaryCoordinates = {longitude: string | undefined; latitude: string | undefined};
type ColumnReference = {value?: string | null} | undefined;

/** Positions a heatmap or cluster layer should read when mapped to a non-primary lng/lat pair. */
export type VariantPositionsResult =
  | {status: 'primary'}
  | {
      status: 'variant';
      positions: GPUVector<'float32x2'>;
      columns: {lat: string; lng: string};
    }
  | {status: 'unavailable'; reason: string};

/** Shared 2D variant positions for heatmap and cluster layers (altitude is never read). */
export function resolveVariantPositions(
  variants: PointPositionVariantRuntime,
  layer: PointSelectorLayer,
  dataset: PointVariantDataset,
  primary: PrimaryCoordinates
): VariantPositionsResult {
  const source = resolvePointVariantSource(variants, layer, dataset, primary);
  if (source.status !== 'variant') return source;
  return {
    status: 'variant',
    positions: source.positions,
    columns: {lat: source.columns.lat, lng: source.columns.lng}
  };
}

export type AggregationVariantDataset = PointVariantDataset & {
  rowCount: number;
  batchRowCounts: readonly number[];
  columns: ConstructorParameters<typeof AggregationRuntime>[0]['columns'];
};

export type AggregationVariantContext = {
  device: Device;
  registry: KeplerDatasetRegistry;
  requestUpdate: () => void;
  /** Overridable for tests; defaults to the real CPU projection plus GPU upload. */
  createRuntime?: (
    props: ConstructorParameters<typeof AggregationRuntime>[0]
  ) => AggregationRuntime;
};

/**
 * Grid and hexagon layers mapped to a non-primary lng/lat pair use their own projected
 * AggregationRuntime, built once per (dataset revision, pair), shared by every layer using the
 * pair and destroyed with the position variant. Returns undefined when the layer uses the
 * primary pair, so callers keep the dataset aggregation.
 */
export function resolveAggregationVariant(
  variants: PointPositionVariantRuntime,
  context: AggregationVariantContext,
  layer: Parameters<AggregationRuntime['resolve']>[0] & PointSelectorLayer,
  dataset: AggregationVariantDataset,
  primary: PrimaryCoordinates,
  selectionMask: GPUVector<'uint32'>,
  selectionRevision: number
): {binding?: ResidentAggregationBinding; reason?: string} | undefined {
  const source = resolvePointVariantSource(variants, layer, dataset, primary);
  if (source.status === 'primary') return undefined;
  if (source.status === 'unavailable') return {reason: source.reason};
  const variant = variants.getVariant(layer.id);
  if (!variant) return {reason: 'The position variant is unavailable.'};
  try {
    let runtime = variant.aggregation as AggregationRuntime | undefined;
    if (!runtime) {
      const table = dataset.prepared.table;
      const longitudeSource = table.getChild(source.columns.lng);
      const latitudeSource = table.getChild(source.columns.lat);
      if (!longitudeSource || !latitudeSource)
        return {reason: 'The mapped coordinate columns are unavailable.'};
      const longitude = new Float64Array(dataset.rowCount);
      const latitude = new Float64Array(dataset.rowCount);
      for (let row = 0; row < dataset.rowCount; row++) {
        const longitudeValue: unknown = longitudeSource.get(row);
        const latitudeValue: unknown = latitudeSource.get(row);
        longitude[row] = typeof longitudeValue === 'number' ? longitudeValue : NaN;
        latitude[row] = typeof latitudeValue === 'number' ? latitudeValue : NaN;
      }
      const create = context.createRuntime ?? (props => new AggregationRuntime(props));
      runtime = create({
        device: context.device,
        longitude,
        latitude,
        sourceChunkLengths: dataset.batchRowCounts,
        coordinateColumns: {lat: source.columns.lat, lng: source.columns.lng},
        selectionMask,
        columns: dataset.columns,
        memoryRegistry: context.registry,
        requestUpdate: context.requestUpdate
      });
      variant.aggregation = runtime;
    }
    return runtime.resolve(layer, selectionMask, selectionRevision);
  } catch (error) {
    return {reason: error instanceof Error ? error.message : String(error)};
  }
}

export type ConnectionVariantColumns = {
  lat0: string;
  lng0: string;
  lat1: string;
  lng1: string;
  alt0?: string;
  alt1?: string;
};

export type ConnectionVariantBinding = {
  sourcePositions: GPUVector<'float32x2'>;
  targetPositions: GPUVector<'float32x2'>;
  columns: ConnectionVariantColumns;
};

/**
 * Arc and line layers whose endpoint columns differ from the bound connection geometry: each
 * endpoint pair becomes a shared 2D position variant (keyed `<layerId>\u0001source|target`).
 * Altitude endpoints are not supported and keep the original geometry (and its mismatch reason).
 */
export function makeConnectionVariantBinding(
  variants: PointPositionVariantRuntime,
  layer: PointSelectorLayer,
  dataset: PointVariantDataset,
  bound: ConnectionVariantColumns | undefined
): ConnectionVariantBinding | {unsupportedReason: string} | undefined {
  const columns: Record<string, ColumnReference> = layer.config.columns;
  const lng0 = columns['lng0']?.value;
  const lat0 = columns['lat0']?.value;
  const lng1 = columns['lng1']?.value;
  const lat1 = columns['lat1']?.value;
  if (!lng0 || !lat0 || !lng1 || !lat1) return undefined;
  if (
    bound &&
    bound.lng0 === lng0 &&
    bound.lat0 === lat0 &&
    bound.lng1 === lng1 &&
    bound.lat1 === lat1
  ) {
    variants.releaseLayer(`${layer.id}\u0001source`);
    variants.releaseLayer(`${layer.id}\u0001target`);
    return undefined;
  }
  if (columns['alt0']?.value || columns['alt1']?.value) return undefined;
  const request = {
    datasetId: dataset.id,
    datasetRevision: dataset.revision,
    prepared: dataset.prepared
  };
  const source = variants.resolve(`${layer.id}\u0001source`, {
    ...request,
    longitude: lng0,
    latitude: lat0
  });
  if (source.status === 'unavailable') return {unsupportedReason: source.reason};
  const target = variants.resolve(`${layer.id}\u0001target`, {
    ...request,
    longitude: lng1,
    latitude: lat1
  });
  if (target.status === 'unavailable') return {unsupportedReason: target.reason};
  return {
    sourcePositions: source.variant.vectors.positions,
    targetPositions: target.variant.vectors.positions,
    columns: {lat0, lng0, lat1, lng1}
  };
}

/** Spreadable render-binding override for arc and line layers; empty when the bound geometry matches. */
export function makeConnectionVariantOverride(
  variants: PointPositionVariantRuntime,
  layer: PointSelectorLayer,
  dataset: PointVariantDataset,
  bound: ConnectionVariantColumns | undefined
): {connections?: ConnectionVariantBinding; unsupportedReason?: string} {
  if (layer.type !== 'arc' && layer.type !== 'line') return {};
  const binding = makeConnectionVariantBinding(variants, layer, dataset, bound);
  if (!binding) return {};
  return 'unsupportedReason' in binding
    ? {unsupportedReason: binding.unsupportedReason}
    : {connections: binding};
}
