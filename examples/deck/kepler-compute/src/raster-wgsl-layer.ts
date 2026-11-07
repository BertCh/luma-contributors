// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Layer, color, project32, type LayerContext, type LayerProps} from '@deck.gl/core';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';
import {
  buildRasterTileWgsl,
  getRasterRescaleFlags,
  createRasterTileMesh,
  RASTER_WGSL_BAND_SLOT_COUNT,
  RASTER_WGSL_EXTENDED_SLOT_COUNT,
  RASTER_WGSL_MODE_CODES,
  RASTER_WGSL_MODULE_SOURCE,
  RASTER_WGSL_TEXTURE_NAMES,
  RASTER_WGSL_UNFILTERABLE_TEXTURE_NAMES,
  type RasterWgslRenderPlan
} from './raster-wgsl-shader';

/** Structural view of one Kepler tile band descriptor (`images.imageBands[n]`). */
type KeplerBandDescriptor = {data?: ArrayLike<number>; width?: number; height?: number};

/** Structural view of the per-tile `images` object Kepler's `renderSubLayers` receives. */
export type RasterWgslImages = Readonly<{
  imageBands?: readonly KeplerBandDescriptor[];
  imageMask?: KeplerBandDescriptor | null;
  imageRgba?: KeplerBandDescriptor | null;
  imageColormap?: unknown;
}>;

/** Resident texture pair for one band (see raster-band-gpu.ts). */
export type RasterWgslBandTextures = Readonly<{valueTexture: Texture; validityTexture: Texture}>;

export type RasterWgslLayerProps = LayerProps & {
  bounds: readonly [number, number, number, number];
  images: RasterWgslImages;
  plan: RasterWgslRenderPlan;
  /** Source nodata sentinel written to the shader for the first band, when known. */
  nodataValue?: number | null;
  /**
   * Looks up the resident textures of the `imageBands` positions. Called every draw so an evicted
   * band is never bound; returning undefined for any needed position falls back to private uploads.
   */
  getResidentBands?: (
    positions: readonly number[]
  ) => readonly (RasterWgslBandTextures | undefined)[] | undefined;
};

const rasterWgslModule = {
  name: 'rasterWgsl',
  source: RASTER_WGSL_MODULE_SOURCE,
  bindingLayout: [
    {name: 'rasterWgsl', group: 3},
    ...RASTER_WGSL_TEXTURE_NAMES.map(name => ({name, group: 3, visibility: 2}))
  ],
  uniformTypes: {
    domain: 'vec4<f32>',
    settings: 'vec4<f32>',
    bands: 'vec4<f32>',
    rescale: 'vec4<f32>',
    tone: 'vec4<f32>'
  }
} as unknown as ShaderModule;

type OwnedTextures = {
  bands: (RasterWgslBandTextures | undefined)[];
  rgba?: Texture;
  colormap?: Texture;
  images?: RasterWgslImages;
};

type RasterWgslState = {
  model?: Model;
  positions?: Buffer;
  texCoords?: Buffer;
  expression?: string;
  dummies?: {value: Texture; validity: Texture; color: Texture};
  owned: OwnedTextures;
};

/**
 * WebGPU draw of one Kepler raster tile: samples r32float band textures with textureLoad, applies
 * validity and nodata masking, band math, value filter and a colormap or RGB composite in WGSL.
 * Replaces Kepler's GLSL-only `RasterLayer`, which has no WebGPU draw.
 */
export class RasterWgslLayer extends Layer<RasterWgslLayerProps> {
  static override layerName = 'RasterWgslLayer';

  declare state: RasterWgslState;

  override getShaders() {
    return super.getShaders({
      source: buildRasterTileWgsl(this.props.plan.expression),
      modules: [project32, color, rasterWgslModule]
    });
  }

  override initializeState({device}: LayerContext): void {
    this.getAttributeManager()?.remove(['instancePickingColors']);
    this.setState({
      owned: {bands: []},
      dummies: {
        value: device.createTexture({
          format: 'r32float',
          width: 1,
          height: 1,
          usage: Texture.SAMPLE | Texture.COPY_DST,
          data: new Float32Array([0])
        }),
        validity: device.createTexture({
          format: 'r8unorm',
          width: 1,
          height: 1,
          usage: Texture.SAMPLE | Texture.COPY_DST,
          data: new Uint8Array([255])
        }),
        color: device.createTexture({
          format: 'rgba8unorm',
          width: 1,
          height: 1,
          usage: Texture.SAMPLE | Texture.COPY_DST,
          data: new Uint8Array([255, 255, 255, 255])
        })
      }
    });
    this.createModel(device);
  }

  private createModel(device: Device): void {
    this.state.model?.destroy();
    this.state.positions?.destroy();
    this.state.texCoords?.destroy();
    const mesh = createRasterTileMesh(this.props.bounds);
    const positions = device.createBuffer({
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: mesh.positions
    });
    const texCoords = device.createBuffer({
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: mesh.texCoords
    });
    const model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      attributes: {positions, texCoords},
      bufferLayout: [
        {name: 'positions', format: 'float32x3'},
        {name: 'texCoords', format: 'float32x2'}
      ],
      shaderLayout: {
        attributes: [],
        bindings: RASTER_WGSL_TEXTURE_NAMES.map((name, index) => ({
          name,
          type: 'texture' as const,
          group: 3,
          location: index + 1,
          sampleType: RASTER_WGSL_UNFILTERABLE_TEXTURE_NAMES.includes(name)
            ? ('unfilterable-float' as const)
            : ('float' as const)
        }))
      },
      topology: 'triangle-list',
      vertexCount: mesh.vertexCount
    });
    this.setState({model, positions, texCoords, expression: this.props.plan.expression});
  }

  override updateState({
    props,
    oldProps
  }: {
    props: RasterWgslLayerProps;
    oldProps: RasterWgslLayerProps;
  }): void {
    if (props.bounds !== oldProps.bounds || props.plan.expression !== this.state.expression)
      this.createModel(this.context.device);
  }

  override finalizeState(context: LayerContext): void {
    super.finalizeState(context);
    this.state.model?.destroy();
    this.state.positions?.destroy();
    this.state.texCoords?.destroy();
    this.destroyOwnedTextures();
    this.state.dummies?.value.destroy();
    this.state.dummies?.validity.destroy();
    this.state.dummies?.color.destroy();
  }

  private destroyOwnedTextures(): void {
    const owned = this.state.owned;
    for (const pair of owned.bands) {
      pair?.valueTexture.destroy();
      pair?.validityTexture.destroy();
    }
    owned.bands = [];
    owned.rgba?.destroy();
    owned.colormap?.destroy();
    owned.rgba = undefined;
    owned.colormap = undefined;
  }

  /** Private uploads from the Kepler band arrays; rebuilt only when the images object changes. */
  private ensureOwnedTextures(): OwnedTextures {
    const {images} = this.props;
    const owned = this.state.owned;
    if (owned.images === images) return owned;
    this.destroyOwnedTextures();
    owned.images = images;
    const device = this.context.device;
    const usage = Texture.SAMPLE | Texture.COPY_DST;
    const mask = images.imageMask?.data;
    images.imageBands?.forEach((band, position) => {
      if (!band.data || !band.width || !band.height) return;
      const sampleCount = band.width * band.height;
      const values = band.data instanceof Float32Array ? band.data : Float32Array.from(band.data);
      const validity = new Uint8Array(sampleCount);
      for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        validity[sampleIndex] = !mask || mask[sampleIndex] >= 1 ? 255 : 0;
      owned.bands[position] = {
        valueTexture: device.createTexture({
          format: 'r32float',
          width: band.width,
          height: band.height,
          usage,
          data: values
        }),
        validityTexture: device.createTexture({
          format: 'r8unorm',
          width: band.width,
          height: band.height,
          usage,
          data: validity
        })
      };
    });
    const rgba = images.imageRgba;
    if (rgba?.data && rgba.width && rgba.height)
      owned.rgba = device.createTexture({
        format: 'rgba8unorm',
        width: rgba.width,
        height: rgba.height,
        usage,
        data: new Uint8Array(Uint8Array.from(rgba.data as ArrayLike<number>))
      });
    const colormap = images.imageColormap as
      | {data?: ArrayLike<number>; width?: number; height?: number}
      | undefined;
    if (colormap?.data && colormap.width && colormap.height)
      owned.colormap = device.createTexture({
        format: 'rgba8unorm',
        width: colormap.width,
        height: colormap.height,
        usage,
        data: Uint8Array.from(colormap.data)
      });
    return owned;
  }

  override draw(): void {
    const {model, dummies} = this.state;
    if (!model || !dummies) return;
    const {plan, getResidentBands, nodataValue} = this.props;
    const {rescale} = plan;
    const owned = this.ensureOwnedTextures();
    const resident = getResidentBands?.(plan.bandPositions);
    const bindings: Record<string, Texture> = {};
    for (let slot = 0; slot < RASTER_WGSL_EXTENDED_SLOT_COUNT; slot++) {
      const position = plan.bandPositions[slot];
      const pair = position === undefined ? undefined : (resident?.[slot] ?? owned.bands[position]);
      bindings[`valueTexture${slot + 1}`] = pair?.valueTexture ?? dummies.value;
      // Slots beyond the fourth carry no validity texture (see RASTER_WGSL_EXTENDED_SLOT_COUNT).
      if (slot < RASTER_WGSL_BAND_SLOT_COUNT)
        bindings[`validityTexture${slot + 1}`] = pair?.validityTexture ?? dummies.validity;
    }
    // A band that is needed but has no texture at all means the tile is not decoded yet.
    for (let slot = 0; slot < RASTER_WGSL_EXTENDED_SLOT_COUNT; slot++)
      if (
        plan.usedBandMask & (1 << slot) &&
        bindings[`valueTexture${slot + 1}`] === dummies.value &&
        plan.mode !== 'rgba'
      )
        return;
    const colormapTexture = owned.colormap ?? dummies.color;
    bindings['colormapTexture'] = colormapTexture;
    bindings['rgbaTexture'] = owned.rgba ?? dummies.color;
    if (plan.mode === 'rgba' && !owned.rgba) return;
    model.setBindings(bindings);
    model.shaderInputs.setProps({
      rasterWgsl: {
        domain: [plan.domain[0], plan.domain[1], plan.filterRange[0], plan.filterRange[1]],
        settings: [
          RASTER_WGSL_MODE_CODES[plan.mode],
          colormapTexture.width,
          nodataValue === null || nodataValue === undefined ? 0 : 1,
          nodataValue ?? 0
        ],
        bands: [plan.usedBandMask, plan.isFilterEnabled ? 1 : 0, plan.isSignedColormap ? 1 : 0, 0],
        rescale: [
          rescale.scaler,
          rescale.offset,
          rescale.saturation,
          getRasterRescaleFlags(rescale)
        ],
        tone: [
          rescale.gamma,
          rescale.sigmoidalContrast,
          rescale.sigmoidalBias,
          rescale.categoricalStep
        ]
      }
    });
    model.draw(this.context.renderPass);
  }
}
