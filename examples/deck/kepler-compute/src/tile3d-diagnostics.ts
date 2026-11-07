// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Runtime diagnostics for 3D tile and scenegraph layers, read through the dev test hook.
 *
 * They only read Deck state (rendered sublayers, per-tile model matrices, picking results) and the
 * luma resource counters; nothing here changes rendering. Browser scripts use them to assert tile
 * transforms, feature picking and that GPU resources drop after a layer is removed.
 */

type LayerLike = {
  id: string;
  constructor: {name: string};
  props?: Record<string, unknown>;
  state?: Record<string, unknown> | null;
  parent?: LayerLike | null;
};

type DeckLike = {
  layerManager?: {getLayers(): unknown[]};
  getViewports?(): {width: number; height: number; project(position: number[]): number[]}[];
  pickObjectAsync?(options: {x: number; y: number; radius?: number; layerIds?: string[]}): Promise<{
    index: number;
    layer?: LayerLike;
    coordinate?: number[];
    object?: unknown;
  } | null>;
};

export type MatrixSummary = {
  allFinite: boolean;
  isIdentity: boolean;
  /** Absolute determinant of the upper-left 3x3 (scale cubed); zero means degenerate. */
  scaleDeterminant: number;
  translation: [number, number, number];
};

/** Summarizes a 4x4 column-major matrix so scripts can assert a tile transform is usable. */
export function describeModelMatrix(
  matrix: ArrayLike<number> | null | undefined
): MatrixSummary | null {
  if (!matrix || matrix.length !== 16) return null;
  const values = Array.from(matrix);
  const allFinite = values.every(Number.isFinite);
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const [xx, xy, xz, , yx, yy, yz, , zx, zy, zz] = values;
  const determinant =
    xx * (yy * zz - yz * zy) - yx * (xy * zz - xz * zy) + zx * (xy * yz - xz * yy);
  return {
    allFinite,
    isIdentity: values.every((value, index) => value === identity[index]),
    scaleDeterminant: Math.abs(determinant),
    translation: [values[12], values[13], values[14]]
  };
}

export type Tile3DLayerDiagnostic = {
  id: string;
  className: string;
  pickable: boolean;
  isLoaded: boolean | undefined;
  modelMatrix: MatrixSummary | null;
  /** Tileset statistics when the layer is the tileset owner (Tile3DLayer state). */
  tileset: {tileCount: number; selectedTileCount: number; loaded: boolean} | null;
};

const TILE3D_LAYER_PATTERN = /Tile3D|Scenegraph|SimpleMesh|PointCloud|MeshLayer|Tile-?3d/i;

/** Lists rendered 3D-tile related layers with transforms and tileset state. */
export function getTile3DDiagnostics(deck: DeckLike | undefined): Tile3DLayerDiagnostic[] {
  const layers = (deck?.layerManager?.getLayers() ?? []) as LayerLike[];
  return layers
    .filter(
      layer => TILE3D_LAYER_PATTERN.test(layer.constructor.name) || /tile3d|3d-tile/i.test(layer.id)
    )
    .map(layer => {
      const tileset = layer.state?.['tileset3d'] as
        | {tiles?: unknown[]; selectedTiles?: unknown[]; isLoaded?: () => boolean}
        | null
        | undefined;
      return {
        id: layer.id,
        className: layer.constructor.name,
        pickable: Boolean(layer.props?.['pickable']),
        isLoaded: (layer as unknown as {isLoaded?: boolean}).isLoaded,
        modelMatrix: describeModelMatrix(
          (layer.props?.['modelMatrix'] ?? layer.props?.['_modelMatrix']) as
            | ArrayLike<number>
            | undefined
        ),
        tileset: tileset
          ? {
              tileCount: tileset.tiles?.length ?? 0,
              selectedTileCount: tileset.selectedTiles?.length ?? 0,
              loaded: Boolean(tileset.isLoaded?.())
            }
          : null
      };
    });
}

export type Tile3DPickResult = {
  hit: boolean;
  layerId: string | null;
  className: string | null;
  index: number | null;
  /** Geographic coordinate Deck reports for the hit, when available. */
  coordinate: number[] | null;
  hasObject: boolean;
};

/**
 * Picks at a screen position (defaults to the viewport center) among 3D tile/scenegraph layers.
 * Only layers that rendered and are pickable can answer; the result names the sublayer hit.
 */
export async function pickTile3D(
  deck: DeckLike | undefined,
  position?: {x: number; y: number; radius?: number}
): Promise<Tile3DPickResult> {
  const viewport = deck?.getViewports?.()[0];
  const x = position?.x ?? (viewport ? viewport.width / 2 : 0);
  const y = position?.y ?? (viewport ? viewport.height / 2 : 0);
  const layerIds = getTile3DDiagnostics(deck)
    .filter(layer => layer.pickable)
    .map(layer => layer.id);
  const picked = layerIds.length
    ? await deck?.pickObjectAsync?.({x, y, radius: position?.radius ?? 4, layerIds})
    : null;
  return {
    hit: Boolean(picked && picked.index >= 0),
    layerId: picked?.layer?.id ?? null,
    className: picked?.layer?.constructor.name ?? null,
    index: picked ? picked.index : null,
    coordinate: picked?.coordinate ?? null,
    hasObject: Boolean(picked?.object)
  };
}

type StatsLike = {get(name: string): {count: number}};
type DeviceLike = {statsManager?: {getStats(name: string): StatsLike}};

/** Live GPU resource counts from luma's counters, for before/after layer removal comparisons. */
export function getDeviceResourceCounts(device: DeviceLike | undefined): Record<string, number> {
  const stats = device?.statsManager?.getStats('GPU Resource Counts');
  const result: Record<string, number> = {};
  if (!stats) return result;
  for (const name of [
    'Resources',
    'Buffers',
    'Textures',
    'TextureViews',
    'Framebuffers',
    'Samplers'
  ]) {
    result[name] = stats.get(`${name} Active`).count;
  }
  return result;
}
