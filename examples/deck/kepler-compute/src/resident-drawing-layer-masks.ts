// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerComputeFilter} from './kepler-table-bridge';
import {validateDrawnPolygon, type DrawingIssue} from './resident-drawing-geometry';

export type DrawingMaskLayer = {id: string; type: string; dataId: string};

export type DrawingLayerMask =
  | {
      layerId: string;
      status: 'masked';
      /** Filters intersected for this layer, in filter order. */
      filterIds: readonly string[];
      /** Changes when any contributing geometry, inversion or target changes. */
      signature: string;
      warnings: readonly DrawingIssue[];
    }
  | {layerId: string; status: 'unfiltered'}
  | {
      layerId: string;
      status: 'invalid';
      filterIds: readonly string[];
      reasons: readonly (DrawingIssue & {filterId: string})[];
    }
  | {layerId: string; status: 'unsupported'; filterIds: readonly string[]; reason: string};

/** Layer types for which a resident polygon mask over point positions exists. */
const MASKABLE_LAYER_TYPES: ReadonlySet<string> = new Set(['point', 'icon', 'heatmap']);

/**
 * Per-layer view of native polygon filters: each layer gets only the filters that target it, with
 * its own signature, so two layers drawn with different polygons keep distinct masks. A filter with
 * an invalid shape never silently disables or widens a mask; it marks every targeted layer invalid
 * with the reasons. Layers a filter targets but cannot mask are reported unsupported.
 * Pure metadata planning: no dataset rows are read.
 */
export function planDrawingLayerMasks(
  layers: readonly DrawingMaskLayer[],
  filters: readonly KeplerComputeFilter[]
): DrawingLayerMask[] {
  const polygonFilters = filters.filter(
    filter => filter.type === 'polygon' && filter.enabled !== false
  );
  const validations = new Map(
    polygonFilters.map(filter => [filter.id, validateFilterShape(filter)] as const)
  );
  return layers.map((layer): DrawingLayerMask => {
    const targeting = polygonFilters.filter(filter => {
      const dataIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
      return dataIds.includes(layer.dataId) && filter.layerId?.includes(layer.id);
    });
    if (!targeting.length) return {layerId: layer.id, status: 'unfiltered'};
    const filterIds = targeting.map(filter => filter.id);
    if (!MASKABLE_LAYER_TYPES.has(layer.type))
      return {
        layerId: layer.id,
        status: 'unsupported',
        filterIds,
        reason: `A resident polygon mask is not available for ${layer.type} layers.`
      };
    const reasons = targeting.flatMap(filter => {
      const validation = validations.get(filter.id)!;
      return validation.status === 'invalid'
        ? validation.reasons.map(reason => ({...reason, filterId: filter.id}))
        : [];
    });
    if (reasons.length) return {layerId: layer.id, status: 'invalid', filterIds, reasons};
    return {
      layerId: layer.id,
      status: 'masked',
      filterIds,
      signature: targeting.map(filter => getFilterSignature(filter)).join('|'),
      warnings: targeting.flatMap(filter => {
        const validation = validations.get(filter.id)!;
        return validation.status === 'valid' ? validation.warnings : [];
      })
    };
  });
}

/** Layers whose mask changed between two plans, including layers added or removed. */
export function getChangedDrawingLayerMasks(
  previous: readonly DrawingLayerMask[],
  next: readonly DrawingLayerMask[]
): string[] {
  const describe = (mask: DrawingLayerMask) =>
    mask.status === 'masked' ? `masked:${mask.signature}` : mask.status;
  const before = new Map(previous.map(mask => [mask.layerId, describe(mask)]));
  const after = new Map(next.map(mask => [mask.layerId, describe(mask)]));
  return [...new Set([...before.keys(), ...after.keys()])].filter(
    layerId => before.get(layerId) !== after.get(layerId)
  );
}

/** One human-readable line per layer whose polygon filter cannot be applied. */
export function describeDrawingMaskProblems(masks: readonly DrawingLayerMask[]): string[] {
  return masks.flatMap(mask =>
    mask.status === 'invalid'
      ? [`Layer ${mask.layerId}: ${mask.reasons.map(reason => reason.message).join(' ')}`]
      : mask.status === 'unsupported'
        ? [`Layer ${mask.layerId}: ${mask.reason}`]
        : []
  );
}

function validateFilterShape(filter: KeplerComputeFilter) {
  const value = filter.value;
  const feature =
    value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Parameters<typeof validateDrawnPolygon>[0])
      : null;
  return validateDrawnPolygon(feature);
}

function getFilterSignature(filter: KeplerComputeFilter): string {
  const value = filter.value as {geometry?: unknown} | null | undefined;
  return `${filter.id}:${filter.inverted ? 'not' : 'in'}:${JSON.stringify(value?.geometry ?? null)}`;
}
