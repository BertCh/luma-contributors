// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerSpatialSelection} from './prepare-spatial-selection';

/** Geometry predicates reduce to one Boolean per original dataset row. */
export type KeplerSelectionGeometry =
  | {kind: 'point'; positions: GPUVector<'float32x2'>; validity?: GPUVector<'uint32'>}
  | {
      kind: 'endpoint-pair';
      sourcePositions: GPUVector<'float32x2'>;
      targetPositions: GPUVector<'float32x2'>;
      sourceValidity?: GPUVector<'uint32'>;
      targetValidity?: GPUVector<'uint32'>;
    }
  | {
      kind: 'multipoint';
      /** One flattened point chunk per source row chunk. */
      positions: GPUVector<'float32x2'>;
      /** Local CSR offsets: each chunk has source-row-count + 1 elements. */
      rowOffsets: GPUVector<'uint32'>;
      pointValidity?: GPUVector<'uint32'>;
      validity?: GPUVector<'uint32'>;
    }
  | {
      kind: 'representative-point';
      positions: GPUVector<'float32x2'>;
      validity?: GPUVector<'uint32'>;
      /** Preserve upstream source-mode semantics; these recipes are not interchangeable. */
      recipe: 'json-bounds-center' | 'geoarrow-mean-center' | 'wkb-vertex-mean';
      /** Native GeoJSON alone uses inclusive bounds for a Rectangle feature with a bbox. */
      nativeRectangleBounds?: boolean;
    }
  | KeplerNoRepresentativeGeometry;

/**
 * Rows whose pinned native converter yields no centroid (unlabelled dense-union GeoArrow). Native
 * `isInPolygon` is false for every such row, so a targeting filter selects nothing until inverted.
 */
export type KeplerNoRepresentativeGeometry = {
  kind: 'no-representative';
};

export type KeplerSelectionLayer = {
  id: string;
  geometryRevision: number;
  geometry: KeplerSelectionGeometry;
  /** Visibility does not alter native targeted-layer union semantics. */
  visible?: boolean;
};

/** Borrowed independent predicates, including distinct native filters on the same column. */
export type KeplerScalarMaskFilter = {
  filterId: string;
  mask: GPUVector<'uint32'>;
  dynamicDomain: boolean;
  /** Native CPU filterRecord membership, computed before renderer GPU-filter suppression. */
  nativeTopology?: boolean;
  /** Native field name at this dataset's dataId index, used by Charts GPU-channel exclusions. */
  nativeFieldName?: string;
  /** Negate the completed predicate, including its null rejection, as native getFilterFunction does. */
  inverted?: boolean;
  enabled?: boolean;
};

/** Charts omit owned CPU predicates by id and GPU channels by axis field, independently. */
export type KeplerConsumerSelectionScope = {
  id: string;
  excludedCpuFilterIds?: readonly string[];
  excludedGpuFieldNames?: readonly string[];
};

export function planKeplerConsumerFilterIds(
  filters: readonly KeplerScalarMaskFilter[],
  scope: KeplerConsumerSelectionScope
): readonly string[] {
  const excludedCpu = new Set(scope.excludedCpuFilterIds);
  const excludedGpu = new Set(scope.excludedGpuFieldNames);
  if (
    excludedGpu.size &&
    filters.some(
      filter =>
        filter.enabled !== false && filter.nativeTopology === false && !filter.nativeFieldName
    )
  )
    throw new Error('GPU field exclusions require native field metadata');
  return filters
    .filter(
      filter =>
        filter.enabled !== false &&
        (filter.nativeTopology !== false
          ? !excludedCpu.has(filter.filterId)
          : !filter.nativeFieldName || !excludedGpu.has(filter.nativeFieldName))
    )
    .map(filter => filter.filterId);
}

export type KeplerPolygonMaskFilter = {
  id: string;
  layerIds: readonly string[];
  selection: KeplerSpatialSelection;
  inverted?: boolean;
  enabled?: boolean;
  /** Set only for native shape:'Rectangle' plus explicit bbox metadata. */
  rectangleBounds?: readonly [number, number, number, number];
};

export type KeplerSelectionScopePlan = {
  scalarFilterIds: readonly string[];
  domainFilterIds: readonly string[];
  nativeTopologyFilterIds: readonly string[];
  layerFilterIds: Readonly<Record<string, readonly string[]>>;
  targetedLayerIds: readonly string[];
};

/** Pure metadata planning: no dataset rows, geometry vertices, or selection indices are read. */
export function planKeplerSelectionScopes(
  layers: readonly KeplerSelectionLayer[],
  scalarFilters: readonly KeplerScalarMaskFilter[],
  polygonFilters: readonly KeplerPolygonMaskFilter[]
): KeplerSelectionScopePlan {
  const layerFilterIds: Record<string, string[]> = Object.create(null);
  for (const layer of layers) {
    if (Object.hasOwn(layerFilterIds, layer.id)) throw new Error('Duplicate selection layer id');
    layerFilterIds[layer.id] = [];
  }
  const identifiers = new Set<string>();
  const scalarFilterIds: string[] = [];
  const domainFilterIds: string[] = [];
  const nativeTopologyFilterIds: string[] = [];
  for (const filter of scalarFilters) {
    if (identifiers.has(filter.filterId)) throw new Error('Duplicate native filter id');
    identifiers.add(filter.filterId);
    if (filter.enabled === false) continue;
    scalarFilterIds.push(filter.filterId);
    if (filter.dynamicDomain) domainFilterIds.push(filter.filterId);
    if (filter.nativeTopology !== false) nativeTopologyFilterIds.push(filter.filterId);
  }
  for (const filter of polygonFilters) {
    if (identifiers.has(filter.id)) throw new Error('Duplicate native filter id');
    identifiers.add(filter.id);
    if (filter.enabled === false) continue;
    for (const layerId of new Set(filter.layerIds)) {
      if (Object.hasOwn(layerFilterIds, layerId)) layerFilterIds[layerId].push(filter.id);
    }
  }
  return {
    scalarFilterIds,
    domainFilterIds,
    nativeTopologyFilterIds,
    layerFilterIds,
    targetedLayerIds: layers
      .filter(layer => layerFilterIds[layer.id].length > 0)
      .map(layer => layer.id)
  };
}
