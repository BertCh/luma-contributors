// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PreviewKind} from '../../catalog/schema.ts';
import type {GpuSceneLoader} from '../types.ts';

export {PreviewFallback, type PreviewFallbackProps} from './fallback.tsx';
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
  'pipeline'
];

const loaders = new Map<PreviewKind, GpuSceneLoader>();

/** Loader for the live scene of a preview kind; luma.gl is imported only when it runs. */
export function getPreviewScene(kind: PreviewKind): GpuSceneLoader {
  let loader = loaders.get(kind);
  if (!loader) {
    loader = async () => {
      const runtime = await import('./runtime.ts');
      return {default: runtime.createPreviewScene(kind)};
    };
    loaders.set(kind, loader);
  }
  return loader;
}
