// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  color,
  project32,
  phongMaterial,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {
  Buffer,
  Texture,
  getExternalImageSize,
  isExternalImage,
  type Device,
  type ExternalImage
} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';
import {
  buildRasterTileWgsl,
  getRasterRescaleFlags,
  createRasterTileMesh,
  createRasterTerrainMesh,
  RASTER_WGSL_BAND_SLOT_COUNT,
  RASTER_WGSL_EXTENDED_SLOT_COUNT,
  RASTER_WGSL_MODE_CODES,
  RASTER_WGSL_MODULE_SOURCE,
  RASTER_WGSL_TEXTURE_NAMES,
  RASTER_WGSL_UNFILTERABLE_TEXTURE_NAMES,
  type RasterWgslRenderPlan,
  type RasterTerrainMesh
} from './raster-wgsl-shader';

/** Structural view of one Kepler tile band descriptor (`images.imageBands[n]`). */
type KeplerBandDescriptor = {data?: ArrayLike<number>; width?: number; height?: number};
type KeplerColorDescriptor = {
  data?: ArrayLike<number> | ExternalImage;
  width?: number;
  height?: number;
};

/** Structural view of the per-tile `images` object Kepler's `renderSubLayers` receives. */
export type RasterWgslImages = Readonly<{
  imageBands?: readonly KeplerBandDescriptor[];
  imageMask?: KeplerBandDescriptor | null;
  imageRgba?: KeplerColorDescriptor | null;
  imageColormap?: KeplerColorDescriptor | null;
}>;

/** Resident texture pair for one band (see raster-band-gpu.ts). */
export type RasterWgslBandTextures = Readonly<{valueTexture: Texture; validityTexture: Texture}>;

export type RasterWgslLayerProps = LayerProps & {
  bounds: readonly [number, number, number, number];
  images: RasterWgslImages;
  plan: RasterWgslRenderPlan;
  mesh?: RasterTerrainMesh;
  flatShading?: boolean;
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
  normals?: Buffer;
  expression?: string;
  subdivisionStep?: number;
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
    const mesh = this.props.mesh;
    // Native RasterMeshLayer enables face normals when quantized-mesh has no NORMAL attribute.
    // Placeholder vertex normals keep the buffer layout stable, but must not flatten the lighting.
    const hasNormals = Boolean(mesh?.attributes['NORMAL'] ?? mesh?.attributes['normals']);
    return super.getShaders({
      source: buildRasterTileWgsl(
        this.props.plan.expression,
        Boolean(this.props.mesh),
        this.props.flatShading || Boolean(mesh && !hasNormals)
      ),
      modules: [project32, color, rasterWgslModule, ...(this.props.mesh ? [phongMaterial] : [])]
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
    this.state.normals?.destroy();
    // Deck 10 resolution counts intervals around the earth. Keep planar raster rows for the
    // Mercator UV mapping, and subdivide BOTH axes on a globe so low-zoom tiles follow its shell.
    const subdivisionStep = this.getSubdivisionStep();
    const [west, south, east, north] = this.props.bounds;
    const mesh = this.props.mesh
      ? createRasterTerrainMesh(this.props.mesh)
      : createRasterTileMesh(
          this.props.bounds,
          Math.max(16, subdivisionStep ? Math.ceil((north - south) / subdivisionStep) : 1),
          subdivisionStep ? Math.max(1, Math.ceil((east - west) / subdivisionStep)) : 1
        );
    const positions = device.createBuffer({
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: mesh.positions
    });
    const texCoords = device.createBuffer({
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: mesh.texCoords
    });
    const normals =
      'normals' in mesh && mesh.normals instanceof Float32Array
        ? device.createBuffer({
            usage: Buffer.VERTEX | Buffer.COPY_DST,
            data: mesh.normals
          })
        : undefined;
    const model = new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      attributes: {positions, texCoords, ...(normals ? {normals} : {})},
      bufferLayout: [
        {name: 'positions', format: 'float32x3'},
        {name: 'texCoords', format: 'float32x2'},
        ...(normals ? [{name: 'normals', format: 'float32x3' as const}] : [])
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
    this.setState({
      model,
      positions,
      texCoords,
      normals,
      expression: this.props.plan.expression,
      subdivisionStep
    });
  }

  override updateState({props, oldProps}: UpdateParameters<this>): void {
    if (
      props.bounds !== oldProps.bounds ||
      props.mesh !== oldProps.mesh ||
      props.flatShading !== oldProps.flatShading ||
      props.plan.expression !== this.state.expression ||
      this.getSubdivisionStep() !== this.state.subdivisionStep
    )
      this.createModel(this.context.device);
  }

  override finalizeState(context: LayerContext): void {
    super.finalizeState(context);
    this.state.model?.destroy();
    this.state.positions?.destroy();
    this.state.texCoords?.destroy();
    this.state.normals?.destroy();
    this.destroyOwnedTextures();
    this.state.dummies?.value.destroy();
    this.state.dummies?.validity.destroy();
    this.state.dummies?.color.destroy();
  }

  override shouldUpdateState(
    parameters: Parameters<Layer<RasterWgslLayerProps>['shouldUpdateState']>[0]
  ): boolean {
    return super.shouldUpdateState(parameters) || parameters.changeFlags.viewportChanged;
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
    owned.rgba = this.createColorTexture(images.imageRgba);
    owned.colormap = this.createColorTexture(images.imageColormap);
    return owned;
  }

  private createColorTexture(
    descriptor: KeplerColorDescriptor | null | undefined
  ): Texture | undefined {
    if (!descriptor?.data) return undefined;
    // Native loadImage returns {data: ImageBitmap}, without descriptor width/height. Treating it
    // as an array left PMTiles imagery invisible and every continuous colormap solid white.
    const external = isExternalImage(descriptor.data);
    const {width, height} = external
      ? getExternalImageSize(descriptor.data as ExternalImage)
      : descriptor;
    if (!width || !height) return undefined;
    return this.context.device.createTexture({
      format: 'rgba8unorm',
      width,
      height,
      usage: Texture.SAMPLE | Texture.COPY_DST | Texture.RENDER,
      data: external
        ? (descriptor.data as ExternalImage)
        : Uint8Array.from(descriptor.data as ArrayLike<number>)
    });
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
