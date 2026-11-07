// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterLiveController} from './raster-live-controller';
import {
  RasterWgslLayer,
  type RasterWgslImages,
  type RasterWgslLayerProps
} from './raster-wgsl-layer';
import {planRasterRender, type RasterWgslSubLayerProps} from './raster-wgsl-shader';

type DeckLayerLike = {
  id: string;
  constructor: {layerName?: string};
  props: Record<string, unknown> & {
    renderSubLayers?: (props: Record<string, unknown>) => unknown;
  };
  clone(props: Record<string, unknown>): DeckLayerLike;
};

type TileLike = {index?: {x: number; y: number; z: number}};

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
  return {
    renderBandIndexes: subLayerProps['renderBandIndexes'] as readonly number[] | undefined,
    bandCombination: subLayerProps['bandCombination'] as string | undefined,
    bandMathExpression: subLayerProps['bandMathExpression'] as string | undefined,
    minPixelValue: subLayerProps['minPixelValue'] as number | undefined,
    maxPixelValue: subLayerProps['maxPixelValue'] as number | undefined,
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
 * Only unparsable custom expressions, expressions past b8 and terrain meshes keep Kepler's own
 * sublayer; unknown presets that parse as band math draw in WGSL with up to eight bands. Rescale (linear/non-linear), gamma, sigmoidal contrast, saturation and
 * categorical colormap stepping are reproduced in WGSL.
 */
export class RasterWgslWrapper {
  private readonly wrappedRenderers = new WeakMap<
    object,
    (props: Record<string, unknown>) => unknown
  >();
  /** Counts replaced and kept sublayers, for test hooks. */
  readonly statistics = {replacedCount: 0, keptCount: 0};

  constructor(private readonly controller: RasterLiveController) {}

  wrapLayer<LayerType>(layer: LayerType): LayerType {
    if (!isDeckLayerLike(layer) || layer.constructor.layerName !== 'TileLayer') return layer;
    const original = layer.props.renderSubLayers;
    if (typeof original !== 'function' || (original as {[WRAPPED]?: boolean})[WRAPPED])
      return layer;
    let wrapped = this.wrappedRenderers.get(original);
    if (!wrapped) {
      const tileLayerId = layer.id;
      wrapped = (subLayerProps: Record<string, unknown>) =>
        this.renderSubLayers(tileLayerId, original, subLayerProps);
      (wrapped as unknown as {[WRAPPED]: boolean})[WRAPPED] = true;
      this.wrappedRenderers.set(original, wrapped);
    }
    return layer.clone({renderSubLayers: wrapped}) as LayerType;
  }

  private renderSubLayers(
    tileLayerId: string,
    original: (props: Record<string, unknown>) => unknown,
    subLayerProps: Record<string, unknown>
  ): unknown {
    const result = original(subLayerProps);
    if (!Array.isArray(result)) return result;
    const tileIndex = (subLayerProps['tile'] as TileLike | undefined)?.index;
    return result.map(sublayer => {
      if (!isDeckLayerLike(sublayer) || sublayer.constructor.layerName !== 'RasterLayer')
        return sublayer;
      const props = sublayer.props as unknown as {
        id: string;
        images?: RasterWgslImages;
        bounds?: [number, number, number, number];
        opacity?: number;
        visible?: boolean;
      };
      const plan =
        props.images && props.bounds
          ? planRasterRender(getRasterSubLayerPlanProps(subLayerProps, props.images))
          : null;
      if (!plan || !props.images || !props.bounds) {
        this.statistics.keptCount++;
        return sublayer;
      }
      if (tileIndex)
        this.controller.ensureTileBandsResident(tileLayerId, tileIndex, plan.bandPositions);
      this.statistics.replacedCount++;
      const layerProps: Partial<RasterWgslLayerProps> = {
        id: `${props.id}-wgsl`,
        bounds: props.bounds,
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
    });
  }
}
