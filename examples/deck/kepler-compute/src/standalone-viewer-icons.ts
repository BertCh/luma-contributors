// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareNativeIconCatalog, type NativeIconDefinition} from './icon-source';

/**
 * Real icon catalog meshes for the offline viewer. Each native triangle mesh (already Y-flipped by
 * prepareNativeIconCatalog) is filled into one cell of a canvas atlas, so deck IconLayer draws the
 * catalog shape in pixel sizes. Custom icons travel in the bundle; the default Kepler catalog is
 * fetched only when the page is online.
 */
export const DEFAULT_ICON_CATALOG_URL =
  'https://d1a3f4spazzrp4.cloudfront.net/kepler.gl/icons/svg-icons.json';
const CELL_SIZE = 64;
const COLUMN_COUNT = 16;
const FALLBACK_ICON_ID = '__fallback_pin';

export type ViewerIconAtlas = {
  canvas: HTMLCanvasElement;
  mapping: Record<string, {x: number; y: number; width: number; height: number; mask: boolean}>;
  hasIcon(iconId: string): boolean;
};

export function createIconAtlas(
  definitions: readonly NativeIconDefinition[]
): ViewerIconAtlas | null {
  if (typeof document === 'undefined') return null;
  const meshes = new Map(
    prepareNativeIconCatalog([], definitions, {
      maximumIconCount: 4096,
      maximumVertexCount: 2_000_000
    }).meshes
  );
  const ids = [FALLBACK_ICON_ID, ...meshes.keys()];
  const canvas = document.createElement('canvas');
  canvas.width = COLUMN_COUNT * CELL_SIZE;
  canvas.height = Math.ceil(ids.length / COLUMN_COUNT) * CELL_SIZE;
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#fff';
  const mapping: ViewerIconAtlas['mapping'] = {};
  ids.forEach((id, index) => {
    const x = (index % COLUMN_COUNT) * CELL_SIZE;
    const y = Math.floor(index / COLUMN_COUNT) * CELL_SIZE;
    mapping[id] = {x, y, width: CELL_SIZE, height: CELL_SIZE, mask: true};
    if (id === FALLBACK_ICON_ID) {
      context.beginPath();
      context.arc(x + 32, y + 24, 18, 0, Math.PI * 2);
      context.moveTo(x + 32, y + 60);
      context.lineTo(x + 18, y + 34);
      context.lineTo(x + 46, y + 34);
      context.fill();
      return;
    }
    fillMesh(context, meshes.get(id)!, x, y);
  });
  return {canvas, mapping, hasIcon: id => id in mapping && id !== FALLBACK_ICON_ID};
}

function fillMesh(
  context: CanvasRenderingContext2D,
  vertices: Float32Array,
  originX: number,
  originY: number
): void {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < vertices.length; index += 3) {
    minimumX = Math.min(minimumX, vertices[index]);
    maximumX = Math.max(maximumX, vertices[index]);
    minimumY = Math.min(minimumY, vertices[index + 1]);
    maximumY = Math.max(maximumY, vertices[index + 1]);
  }
  const scale = (CELL_SIZE - 4) / Math.max(maximumX - minimumX, maximumY - minimumY, 1e-9);
  const offsetX = originX + (CELL_SIZE - (maximumX - minimumX) * scale) / 2 - minimumX * scale;
  const offsetY = originY + (CELL_SIZE - (maximumY - minimumY) * scale) / 2 - minimumY * scale;
  context.beginPath();
  for (let index = 0; index < vertices.length; index += 9) {
    context.moveTo(offsetX + vertices[index] * scale, offsetY + vertices[index + 1] * scale);
    context.lineTo(offsetX + vertices[index + 3] * scale, offsetY + vertices[index + 4] * scale);
    context.lineTo(offsetX + vertices[index + 6] * scale, offsetY + vertices[index + 7] * scale);
    context.closePath();
  }
  // Nonzero fill unions adjacent triangles of either winding only if they agree; use evenodd-free fill.
  context.fill('nonzero');
}

/** Default catalog from the CDN; resolves to [] when offline or on any failure. */
export async function fetchDefaultIconDefinitions(): Promise<NativeIconDefinition[]> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return [];
  try {
    const response = await fetch(DEFAULT_ICON_CATALOG_URL, {mode: 'cors'});
    const parsed = await response.json();
    return Array.isArray(parsed?.svgIcons) ? parsed.svgIcons : [];
  } catch {
    return [];
  }
}
