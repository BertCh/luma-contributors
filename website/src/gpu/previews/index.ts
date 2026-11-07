// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PreviewKind} from '../../catalog/schema.ts';
import type {GpuSceneLoader, GpuSceneParameters} from '../types.ts';
import {getPointerMode, getPreviewParameterList} from './parameters.ts';

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

const loaders = new Map<string, GpuSceneLoader>();

/**
 * Loader for the live scene of a preview; luma.gl is imported only when it runs. With `entryId`,
 * the entry's function-specific preview is drawn when it has one.
 */
export function getPreviewScene(kind: PreviewKind, entryId?: string): GpuSceneLoader {
  const key = entryId ? `${kind}:${entryId}` : kind;
  let loader = loaders.get(key);
  if (!loader) {
    loader = async () => {
      const runtime = await import('./runtime.ts');
      return {
        default: runtime.createPreviewScene(kind, entryId),
        parameters: getPreviewParameterList(kind, entryId)
      };
    };
    loaders.set(key, loader);
  }
  return loader;
}

/** Slider parameters of a preview, known without loading luma.gl. Defaults match the fallback. */
export function getPreviewParameters(kind: PreviewKind, entryId?: string): GpuSceneParameters {
  return getPreviewParameterList(kind, entryId);
}

/** True when the picture of a preview also reacts to the pointer. */
export function isPreviewInteractive(kind: PreviewKind, entryId?: string): boolean {
  return getPointerMode(kind, entryId) !== 'none';
}
