// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure per-view planning for native split-map and swipe comparison on ONE shared Deck device.
 *
 * Contract: every view renders from the same resident source/style/selection buffers owned by the
 * single runtime; a view contributes only its layer visibility. Style plans, domain masks and
 * selection masks are layer- and dataset-scoped and are never duplicated per view, so adding a view
 * allocates no source, style or mask memory. Per-view GPU resources are limited to the Deck
 * instance's own framebuffers/picking targets. A view that cannot borrow the shared device is
 * refused here rather than creating a second device.
 */
export type ResidentViewLayer = {id: string; isVisible: boolean};
export type ResidentViewPlan = {
  /** Kepler map index; 0 is the primary view. */
  index: number;
  /** Layer ids that Kepler renders in this view, in source order. */
  visibleLayerIds: readonly string[];
};
export type ResidentViewPlanInput = {
  layers: readonly ResidentViewLayer[];
  /** Kepler `visState.splitMaps`; empty or absent means one view. */
  splitMaps?: readonly {layers?: Readonly<Record<string, boolean>>}[];
  /** Kepler `mapState.mapSplitMode`. */
  mapSplitMode?: string | null;
  /** Number of views the host can attach to the shared device; defaults to one. */
  supportedViewCount?: number;
};

const MAXIMUM_VIEW_COUNT = 2;

/** Saved maps predating mapSplitMode still use isSplit; the installed reducer keeps both. */
export function getResidentMapSplitMode(mapState: {
  isSplit?: boolean;
  mapSplitMode?: string | null;
}): string {
  return mapState.isSplit && (!mapState.mapSplitMode || mapState.mapSplitMode === 'SINGLE_MAP')
    ? 'DUAL_MAP'
    : (mapState.mapSplitMode ?? 'SINGLE_MAP');
}

/** One shared Deck instance must retain a visible instance from either native map. */
export function mergeResidentViewLayers<Layer extends {id: string; props: {visible?: boolean}}>(
  primary: readonly Layer[],
  secondary: readonly Layer[]
): Layer[] {
  const merged = primary.map(layer =>
    layer.props.visible === false
      ? (secondary.find(
          candidate => candidate.id === layer.id && candidate.props.visible !== false
        ) ?? layer)
      : layer
  );
  const known = new Set(merged.map(layer => layer.id));
  return [...merged, ...secondary.filter(layer => !known.has(layer.id))];
}

/** Mirrors native `isLayerVisible`: a layer must be enabled and, in split mode, flagged for that map. */
export function planResidentViews(
  input: ResidentViewPlanInput
): {supported: true; views: readonly ResidentViewPlan[]} | {supported: false; reason: string} {
  const splitMaps = input.splitMaps ?? [];
  const viewCount = Math.max(1, splitMaps.length);
  const supportedViewCount = Math.min(input.supportedViewCount ?? 1, MAXIMUM_VIEW_COUNT);
  const mode = input.mapSplitMode ?? 'SINGLE_MAP';
  if (
    viewCount > supportedViewCount ||
    (viewCount > 1 && mode !== 'DUAL_MAP' && mode !== 'SWIPE_COMPARE') ||
    (viewCount === 1 && mode !== 'SINGLE_MAP')
  )
    return {
      supported: false,
      reason: `This resident workspace supports ${supportedViewCount} map view(s) on its shared Deck device; requested ${viewCount} (${mode}).`
    };
  const views = Array.from({length: viewCount}, (_, index) => {
    const mapLayers = splitMaps[index]?.layers;
    return {
      index,
      visibleLayerIds: input.layers
        .filter(layer => layer.isVisible && (!mapLayers || mapLayers[layer.id]))
        .map(layer => layer.id)
    };
  });
  return {supported: true, views};
}

/**
 * Deck `layerFilter` for the single-Deck, multi-viewport design: viewport ids are `view-<index>`,
 * and a layer draws only in views whose native per-map visibility includes its Kepler layer id.
 * Layers unknown to every view (editor, analysis overlays) draw in all views.
 */
export function createResidentViewLayerFilter(
  views: readonly ResidentViewPlan[],
  getKeplerLayerId: (deckLayerId: string) => string
): (options: {layer: {id: string}; viewport: {id: string}}) => boolean {
  const known = new Set(views.flatMap(view => view.visibleLayerIds));
  return ({layer, viewport}) => {
    const keplerLayerId = getKeplerLayerId(layer.id);
    if (!known.has(keplerLayerId)) return true;
    return views.some(
      view => viewport.id === `view-${view.index}` && view.visibleLayerIds.includes(keplerLayerId)
    );
  };
}

/**
 * Swipe compare draws two full-size maps and shows each only on its side of the divider. A Deck
 * viewport clips to its own rectangle, so each view gets the rectangle `[x, x + width]` of the
 * canvas and a projection that still maps geography exactly where the full-size map would. Given
 * the full-canvas column-major projection matrix, clip x' = canvasWidth / width * clip x + shift * clip w.
 */
export function getResidentSubViewProjection(
  fullProjectionMatrix: ArrayLike<number>,
  canvasWidth: number,
  x: number,
  width: number
): number[] {
  const scale = canvasWidth / width;
  const shift = (canvasWidth - 2 * x) / width - 1;
  const matrix = Array.from(fullProjectionMatrix);
  for (let column = 0; column < 4; column++)
    matrix[column * 4] = scale * matrix[column * 4] + shift * matrix[column * 4 + 3];
  return matrix;
}
