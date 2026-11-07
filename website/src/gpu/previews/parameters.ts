// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Slider parameters of every preview kind. Free of luma.gl imports so pages can render the controls
// in prerendered HTML; the builders read the same ids, and the static fallback uses the defaults.

import type {PreviewKind} from '../../catalog/schema.ts';
import type {GpuSceneParameter} from '../types.ts';
import {getEntryPreview} from './entries/index.ts';

export type PreviewParameterValues = Record<string, number>;

export const PREVIEW_PARAMETERS: Record<PreviewKind, GpuSceneParameter[]> = {
  points: [
    {id: 'count', label: 'Points', min: 20, max: 400, step: 10, default: 90},
    {id: 'size', label: 'Size', min: 0.8, max: 4, step: 0.1, default: 1.5, unit: 'px'}
  ],
  'point-clusters': [
    {id: 'radius', label: 'Cluster radius', min: 3, max: 14, step: 0.5, default: 7, unit: 'px'},
    {id: 'clusters', label: 'Clusters', min: 1, max: 4, step: 1, default: 4}
  ],
  hotspots: [
    {id: 'threshold', label: 'Significance', min: 0.2, max: 3, step: 0.1, default: 1.2, unit: 'σ'},
    {id: 'radius', label: 'Neighborhood', min: 6, max: 26, step: 1, default: 13, unit: 'px'}
  ],
  'density-surface': [
    {id: 'bandwidth', label: 'Bandwidth', min: 3, max: 22, step: 0.5, default: 9, unit: 'px'},
    {id: 'events', label: 'Events', min: 8, max: 70, step: 2, default: 36}
  ],
  interpolation: [
    {id: 'power', label: 'Power', min: 1, max: 6, step: 0.1, default: 2},
    {id: 'samples', label: 'Samples', min: 5, max: 26, step: 1, default: 13}
  ],
  raster: [
    {id: 'resolution', label: 'Cells across', min: 6, max: 64, step: 2, default: 32},
    {id: 'contrast', label: 'Contrast', min: 0.4, max: 2.6, step: 0.1, default: 1}
  ],
  terrain: [
    {id: 'azimuth', label: 'Sun azimuth', min: 0, max: 360, step: 5, default: 315, unit: '°'},
    {id: 'exaggeration', label: 'Exaggeration', min: 1, max: 14, step: 0.5, default: 5, unit: '×'}
  ],
  'flow-accumulation': [
    {id: 'threshold', label: 'Threshold', min: 0, max: 100, step: 5, default: 0, unit: '%'},
    {id: 'azimuth', label: 'Sun azimuth', min: 0, max: 360, step: 5, default: 315, unit: '°'}
  ],
  contours: [
    {id: 'levels', label: 'Levels', min: 3, max: 32, step: 1, default: 14},
    {id: 'roughness', label: 'Roughness', min: 0.5, max: 2.4, step: 0.1, default: 1.1}
  ],
  network: [
    {id: 'hops', label: 'Hops', min: 0, max: 8, step: 1, default: 3},
    {id: 'closures', label: 'Closed links', min: 0, max: 16, step: 1, default: 0}
  ],
  'shortest-path': [{id: 'closures', label: 'Closed roads', min: 0, max: 24, step: 1, default: 0}],
  isochrones: [
    {id: 'budget', label: 'Travel budget', min: 10, max: 100, step: 5, default: 55, unit: '%'},
    {id: 'bands', label: 'Bands', min: 2, max: 14, step: 1, default: 8}
  ],
  flows: [
    {id: 'arcs', label: 'Flows', min: 3, max: 44, step: 1, default: 18},
    {id: 'speed', label: 'Speed', min: 0.05, max: 1, step: 0.05, default: 0.3}
  ],
  lines: [
    {id: 'count', label: 'Lines', min: 2, max: 24, step: 1, default: 9},
    {id: 'width', label: 'Width', min: 0.5, max: 4, step: 0.1, default: 1.3, unit: 'px'}
  ],
  polygons: [
    {id: 'vertices', label: 'Vertices', min: 4, max: 24, step: 1, default: 10},
    {id: 'roughness', label: 'Roughness', min: 0, max: 0.7, step: 0.05, default: 0.25}
  ],
  'polygon-overlay': [
    {id: 'separation', label: 'Separation', min: 0, max: 80, step: 2, default: 36, unit: 'px'},
    {id: 'size', label: 'Size', min: 14, max: 38, step: 1, default: 30, unit: 'px'}
  ],
  buffers: [{id: 'distance', label: 'Distance', min: 2, max: 24, step: 1, default: 12, unit: 'px'}],
  tessellation: [
    {id: 'vertices', label: 'Vertices', min: 4, max: 40, step: 1, default: 12},
    {id: 'roughness', label: 'Roughness', min: 0, max: 0.7, step: 0.05, default: 0.35}
  ],
  cells: [
    {id: 'size', label: 'Cell size', min: 4, max: 16, step: 0.5, default: 7.2, unit: 'px'},
    {id: 'classes', label: 'Classes', min: 2, max: 5, step: 1, default: 5}
  ],
  'spatial-weights': [
    {id: 'neighbors', label: 'Neighbors (k)', min: 1, max: 8, step: 1, default: 3}
  ],
  regression: [
    {id: 'slope', label: 'Slope', min: -0.6, max: 0.6, step: 0.05, default: 0.4},
    {id: 'noise', label: 'Noise', min: 1, max: 20, step: 0.5, default: 7, unit: 'px'}
  ],
  histogram: [
    {id: 'bins', label: 'Bins', min: 5, max: 48, step: 1, default: 22},
    {id: 'spread', label: 'Spread', min: 0.08, max: 0.4, step: 0.01, default: 0.2}
  ],
  classification: [
    {id: 'classes', label: 'Classes', min: 2, max: 8, step: 1, default: 5},
    {id: 'skew', label: 'Skew', min: 0.5, max: 3, step: 0.1, default: 1}
  ],
  'time-series': [
    {id: 'window', label: 'Window', min: 1, max: 14, step: 1, default: 5},
    {id: 'series', label: 'Series', min: 1, max: 3, step: 1, default: 3}
  ],
  trajectories: [
    {id: 'trail', label: 'Trail length', min: 0.2, max: 3, step: 0.1, default: 1},
    {id: 'count', label: 'Tracks', min: 3, max: 32, step: 1, default: 16}
  ],
  'point-cloud': [
    {id: 'spin', label: 'Spin', min: -90, max: 90, step: 5, default: 15, unit: '°/s'},
    {id: 'depth', label: 'Perspective', min: 0, max: 8, step: 0.5, default: 4.5}
  ],
  pipeline: [
    {id: 'branches', label: 'Branches', min: 1, max: 3, step: 1, default: 2},
    {id: 'speed', label: 'Speed', min: 0.1, max: 1.2, step: 0.05, default: 0.45}
  ],
  'affine-transform': [
    {id: 'rotate', label: 'Rotate', min: -180, max: 180, step: 5, default: 30, unit: '°'},
    {id: 'scale', label: 'Scale', min: 0.4, max: 1.6, step: 0.05, default: 1, unit: '×'},
    {id: 'skew', label: 'Skew', min: -0.8, max: 0.8, step: 0.05, default: 0.3}
  ],
  'cell-cover': [
    {id: 'resolution', label: 'Resolution', min: 3, max: 8, step: 0.5, default: 6},
    {id: 'vertices', label: 'Polygon vertices', min: 5, max: 20, step: 1, default: 11}
  ]
};

/** Kinds whose picture also reacts to the pointer (source, probe, light or focus). */
export const POINTER_KINDS: ReadonlySet<PreviewKind> = new Set<PreviewKind>([
  'density-surface',
  'interpolation',
  'terrain',
  'flow-accumulation',
  'network',
  'shortest-path',
  'isochrones',
  'polygon-overlay',
  'buffers',
  'spatial-weights',
  'point-cloud'
]);

/** Slider parameters of a preview: the entry's own when it declares them, else its kind's. */
export function getPreviewParameterList(kind: PreviewKind, entryId?: string): GpuSceneParameter[] {
  return (entryId && getEntryPreview(entryId)?.parameters) || PREVIEW_PARAMETERS[kind];
}

/** How a preview reacts to the pointer: the entry's own mode when set, else its kind's. */
export function getPointerMode(kind: PreviewKind, entryId?: string): 'none' | 'hover' | 'sticky' {
  const mode = entryId ? getEntryPreview(entryId)?.pointer : undefined;
  if (mode) return mode;
  if (!POINTER_KINDS.has(kind)) return 'none';
  return STICKY_POINTER_KINDS.has(kind) ? 'sticky' : 'hover';
}

/** Kinds whose pointer choice (a source or focus node) stays after the pointer leaves. */
export const STICKY_POINTER_KINDS: ReadonlySet<PreviewKind> = new Set<PreviewKind>([
  'network',
  'shortest-path',
  'isochrones',
  'spatial-weights'
]);

export function getDefaultParameterValues(
  kind: PreviewKind,
  entryId?: string
): PreviewParameterValues {
  const values: PreviewParameterValues = {};
  for (const parameter of getPreviewParameterList(kind, entryId)) {
    values[parameter.id] = parameter.default;
  }
  return values;
}
