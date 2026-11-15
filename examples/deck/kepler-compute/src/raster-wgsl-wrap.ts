// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterLiveController} from './raster-live-controller';
import {
  RasterWgslLayer,
  type RasterWgslImages,
  type RasterWgslLayerProps
} from './raster-wgsl-layer';
import {
  planRasterRender,
  type RasterWgslSubLayerProps,
  type RasterTerrainMesh
} from './raster-wgsl-shader';

type DeckLayerLike = {
  id: string;
  constructor: {layerName?: string};
  props: Record<string, unknown> & {
    renderSubLayers?: (props: Record<string, unknown>) => unknown;
  };
  clone(props: Record<string, unknown>): DeckLayerLike;
};

type TileLike = {
  index?: {x: number; y: number; z: number};
  boundingBox?: [[number, number], [number, number]];
  bbox?: {west: number; south: number; east: number; north: number};
};

const WRAPPED = Symbol.for('luma.kepler.rasterWgslWrapped');

function isDeckLayerLike(value: unknown): value is DeckLayerLike {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as DeckLayerLike).clone === 'function' &&
    typeof (value as DeckLayerLike).props === 'object'
  );
}

/**
 * Builds the sublayer props `planRasterRender` reads from the Kepler `renderSubLayers` arguments
 * and the tile's image descriptor.
 */
export function getRasterSubLayerPlanProps(
  subLayerProps: Record<string, unknown>,
  images: RasterWgslImages
): RasterWgslSubLayerProps {
  // Native renderSubLayers aliases props.minPixelValue/maxPixelValue to local "global" names;
  // those aliases are not props. Finite overrides must win over each tile's own statistics.
  const tileData = subLayerProps['data'] as
    | {minPixelValue?: number; maxPixelValue?: number}
    | undefined;
  const resolveRange = (name: 'minPixelValue' | 'maxPixelValue') => {
    const globalValue = subLayerProps[name];
    return Number.isFinite(globalValue) ? (globalValue as number) : tileData?.[name];
  };
  return {
    renderBandIndexes: subLayerProps['renderBandIndexes'] as readonly number[] | undefined,
    bandCombination: subLayerProps['bandCombination'] as string | undefined,
    bandMathExpression: subLayerProps['bandMathExpression'] as string | undefined,
    minPixelValue: resolveRange('minPixelValue'),
    maxPixelValue: resolveRange('maxPixelValue'),
    linearRescalingFactor: subLayerProps['linearRescalingFactor'] as
      | readonly [number, number]
      | undefined,
    nonLinearRescaling: Boolean(subLayerProps['nonLinearRescaling']),
    filterEnabled: Boolean(subLayerProps['filterEnabled']),
    filterRange: subLayerProps['filterRange'] as readonly [number, number] | undefined,
    gammaContrastFactor: subLayerProps['gammaContrastFactor'] as number | undefined,
    sigmoidalContrastFactor: subLayerProps['sigmoidalContrastFactor'] as number | undefined,
    sigmoidalBiasFactor: subLayerProps['sigmoidalBiasFactor'] as number | undefined,
    saturationValue: subLayerProps['saturationValue'] as number | undefined,
    hasCategoricalColorMap: Boolean(subLayerProps['hasCategoricalColorMap']),
    dataType: subLayerProps['dataType'] as string | undefined,
    minCategoricalBandValue: subLayerProps['minCategoricalBandValue'] as number | undefined,
    maxCategoricalBandValue: subLayerProps['maxCategoricalBandValue'] as number | undefined,
    imageBandCount: images.imageBands?.length ?? 0,
    hasColormap: Boolean(images.imageColormap),
    hasRgba: Boolean(images.imageRgba),
    hasMask: Boolean(images.imageMask)
  };
}

/**
 * Replaces Kepler's GLSL `RasterLayer` tile sublayers with the WGSL `RasterWgslLayer`. The host
 * calls `wrapLayer` for every top-level Deck layer; only tile layers that render Kepler raster
 * sublayers are cloned, with a memoized `renderSubLayers` so the prop identity stays stable.
 * Only unparsable custom expressions and expressions past b8 keep Kepler's own
 * sublayer; unknown presets that parse as band math draw in WGSL with up to eight bands. Rescale (linear/non-linear), gamma, sigmoidal contrast, saturation and
 * categorical colormap stepping are reproduced in WGSL.
 */
export class RasterWgslWrapper {
  private readonly wrappedRenderers = new WeakMap<
    object,
    Map<string, (props: Record<string, unknown>) => unknown>
  >();
  /** Counts replaced and kept sublayers, for test hooks. */
  readonly statistics = {replacedCount: 0, keptCount: 0};

  constructor(private readonly controller: RasterLiveController) {}

  wrapLayer<LayerType>(layer: LayerType): LayerType {
    if (!isDeckLayerLike(layer) || layer.constructor.layerName !== 'TileLayer') return layer;
    const original = layer.props.renderSubLayers;
    if (typeof original !== 'function' || (original as {[WRAPPED]?: boolean})[WRAPPED])
      return layer;
    let renderers = this.wrappedRenderers.get(original);
    if (!renderers) {
      renderers = new Map();
      this.wrappedRenderers.set(original, renderers);
    }
    let wrapped = renderers.get(layer.id);
    if (!wrapped) {
      const tileLayerId = layer.id;
      wrapped = (subLayerProps: Record<string, unknown>) =>
        this.renderSubLayers(tileLayerId, original, subLayerProps);
      (wrapped as unknown as {[WRAPPED]: boolean})[WRAPPED] = true;
      // Native raster layers share the same module-level renderer. Its resident-texture lookup
      // also needs the owning layer id, or a second raster can sample the first raster's bands.
      renderers.set(layer.id, wrapped);
    }
    return layer.clone({renderSubLayers: wrapped}) as LayerType;
  }

  private renderSubLayers(
    tileLayerId: string,
    original: (props: Record<string, unknown>) => unknown,
    subLayerProps: Record<string, unknown>
  ): unknown {
    const result = original(subLayerProps);
    const tileIndex = (subLayerProps['tile'] as TileLike | undefined)?.index;
    const replaceSublayer = (sublayer: unknown): unknown => {
      if (Array.isArray(sublayer)) return sublayer.map(replaceSublayer);
      if (
        !isDeckLayerLike(sublayer) ||
        !['RasterLayer', 'RasterMeshLayer'].includes(sublayer.constructor.layerName ?? '')
      )
        return sublayer;
      const props = sublayer.props as unknown as {
        id: string;
        images?: RasterWgslImages;
        bounds?: [number, number, number, number];
        mesh?: RasterTerrainMesh;
        opacity?: number;
        visible?: boolean;
      };
      const tile = subLayerProps['tile'] as TileLike | undefined;
      const tileBounds = tile?.boundingBox;
      const tileBox = tile?.bbox;
      const bounds =
        props.bounds ??
        (tileBounds
          ? ([...tileBounds[0], ...tileBounds[1]] as [number, number, number, number])
          : tileBox
            ? ([tileBox.west, tileBox.south, tileBox.east, tileBox.north] as [
                number,
                number,
                number,
                number
              ])
            : undefined);
      const plan =
        props.images && bounds
          ? planRasterRender(getRasterSubLayerPlanProps(subLayerProps, props.images))
          : null;
      if (!plan || !props.images || !bounds) {
        this.statistics.keptCount++;
        return sublayer;
      }
      if (tileIndex)
        this.controller.ensureTileBandsResident(tileLayerId, tileIndex, plan.bandPositions);
      this.statistics.replacedCount++;
      const layerProps: Partial<RasterWgslLayerProps> = {
        id: `${props.id}-wgsl`,
        bounds,
        mesh: props.mesh,
        images: props.images,
        plan,
        opacity: props.opacity,
        visible: props.visible,
        pickable: false,
        getResidentBands: tileIndex
          ? positions => this.controller.peekTileBandTextures(tileLayerId, tileIndex, positions)
          : undefined
      };
      // Inherit the tile sublayer props (coordinate system, model matrix, extensions) from Kepler's layer.
      return new RasterWgslLayer(sublayer.props as unknown as RasterWgslLayerProps, layerProps);
    };
    // Deck accepts a layer, a nested layer array, or null from renderSubLayers. Every RasterLayer
    // must be replaced before initialization: its deck 9 ShaderAssembler/inject hooks do not
    // exist in luma 10, even though constructing the unused native layer is harmless.
    return replaceSublayer(result);
  }
}
