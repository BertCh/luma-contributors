// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PreviewKind} from '../../catalog/schema.ts';
import type {GpuSceneLoader, GpuSceneParameters} from '../types.ts';
import {POINTER_KINDS, PREVIEW_PARAMETERS} from './parameters.ts';

export {PreviewFallback, type PreviewFallbackProps} from './fallback.tsx';
export {PREVIEW_PARAMETERS} from './parameters.ts';
export {PreviewCanvas, type PreviewCanvasProps} from './preview-canvas.tsx';

export const PREVIEW_KINDS: PreviewKind[] = [
  'points',
  'point-clusters',
  'hotspots',
  'density-surface',
  'interpolation',
  'raster',
  'terrain',
  'flow-accumulation',
  'contours',
  'network',
  'shortest-path',
  'isochrones',
  'flows',
  'lines',
  'polygons',
  'polygon-overlay',
  'buffers',
  'tessellation',
  'cells',
  'spatial-weights',
  'regression',
  'histogram',
  'classification',
  'time-series',
  'trajectories',
  'point-cloud',
  'pipeline',
  'affine-transform',
  'cell-cover'
];

const loaders = new Map<PreviewKind, GpuSceneLoader>();

/** Loader for the live scene of a preview kind; luma.gl is imported only when it runs. */
export function getPreviewScene(kind: PreviewKind): GpuSceneLoader {
  let loader = loaders.get(kind);
  if (!loader) {
    loader = async () => {
      const runtime = await import('./runtime.ts');
      return {default: runtime.createPreviewScene(kind), parameters: PREVIEW_PARAMETERS[kind]};
    };
    loaders.set(kind, loader);
  }
  return loader;
}

/** Slider parameters of a preview kind, known without loading luma.gl. Defaults match the fallback. */
export function getPreviewParameters(kind: PreviewKind): GpuSceneParameters {
  return PREVIEW_PARAMETERS[kind];
}

/** True when the picture of a preview kind also reacts to the pointer. */
export function isPreviewInteractive(kind: PreviewKind): boolean {
  return POINTER_KINDS.has(kind);
}
