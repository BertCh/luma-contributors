// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LightingEffect, _LayersPass as LayersPass} from '@deck.gl/core';
import type {EffectContext, Layer, LayersPassRenderOptions, Viewport} from '@deck.gl/core';
import {Texture, type Device, type Framebuffer} from '@luma.gl/core';
import {Matrix4, Vector3} from '@math.gl/core';
import {getWGSLShadowFallbackTexture, WGSL_SHADOW_MODULE_NAME} from './wgsl-shadow';
import {
  getWGSLShadowFramePoints,
  makeWGSLShadowProjection,
  WGSL_SHADOW_GLOBE_METERS_PER_UNIT,
  mergeWGSLShadowLayerBounds,
  type WGSLShadowFrustumViewport,
  type WGSLShadowProjection,
  type WGSLShadowSceneBounds
} from './wgsl-shadow-frustum';

/** Deck's globe projection mode; shadow maps need a planar light-space projection. */
const PROJECTION_MODE_GLOBE = 2;
const DEFAULT_SHADOW_COLOR: [number, number, number, number] = [0, 0, 0, 200 / 255];
const MAX_SHADOW_LIGHTS = 2;
/** Default longest side of a globe shadow map, in texels. */
export const WGSL_SHADOW_DEFAULT_GLOBE_MAP_SIZE = 2048;
/** Default texel budget of one globe shadow map (one map per light). */
export const WGSL_SHADOW_DEFAULT_GLOBE_TEXEL_BUDGET = 2048 * 2048;

/** Shadow resolution of one viewport in the last frame; read it with `getViewportShadowStats`. */
export type WGSLShadowViewportStats = {
  /** Shadow map size in texels (per light). */
  shadowMapSize: [number, number];
  /** Shadow map texels per viewport device pixel along each axis (1 = the legacy viewport-sized map). */
  resolutionScale: number;
  /** Texels per surface meter of the first light's map (coarsest axis); globe viewports only. */
  texelsPerMeter: number | null;
  /** What the legacy viewport-sized, unrotated map would give for the same view. */
  baselineTexelsPerMeter: number | null;
};

/** A directional light; `getProjectedLight` (Deck SunLight) refreshes `direction` for the viewport. */
export type WGSLShadowLight = {
  direction: ArrayLike<number>;
  getProjectedLight?: (options: {layer: {context: {viewport: unknown}}}) => unknown;
};

export type WGSLShadowLightingEffectOptions = {
  /** Directional lights that cast shadows (at most two are used), in Deck's light direction convention. */
  lights: readonly WGSLShadowLight[];
  /** RGB in 0..1 or 0..255 plus the maximum shadow opacity in 0..1. */
  shadowColor?: ArrayLike<number>;
  /** Kepler night time: shadow every receiver uniformly (re-read every frame). */
  outputUniformShadow?: boolean;
  /**
   * Layer id prefixes that neither cast nor receive shadows and are not lit (Kepler's editor
   * layer, `kepler_editor_layer`, and its sublayers).
   */
  excludedLayerIdPrefixes?: readonly string[];
  /**
   * Scene bounds in world coordinates for non-geospatial (orbit, orthographic) viewports. Without
   * it the participating layers' `getBounds()` are used, then the view frustum with a clamped
   * far plane.
   */
  sceneBounds?: WGSLShadowSceneBounds | null;
  /**
   * Globe viewports: longest side of the shadow map in texels (default 2048, clamped to the device
   * limit; the map is never smaller than the viewport). The light-space extent is fitted to the
   * visible cap of the sphere and rotated to the tightest axes, so the larger map directly raises
   * the texels per surface meter. 0 keeps the map viewport-sized (the rotated fit still applies).
   */
  globeShadowMapSize?: number;
  /**
   * Globe viewports: texel budget of one shadow map (default 2048 x 2048). Each light uses one
   * RGBA8 color plus one depth24 map, so memory is about 8 bytes per texel per light.
   */
  globeShadowTexelBudget?: number;
};

/** Per-viewport shadow state: one packed-depth map per light plus the matrices used to draw it. */
type ViewportShadowState = {
  passes: WGSLShadowPass[];
  shadowMatrices: Matrix4[];
  /** Set when Deck's frustum fitting does not apply to the viewport (globe, orbit, ortho). */
  projections?: WGSLShadowProjection[];
  active: boolean;
  fitting: 'deck-ground-plane' | 'scene-bounds' | 'view-frustum' | 'globe';
  stats?: WGSLShadowViewportStats;
};

type ModelHolder = {shaderInputs?: {modules?: Record<string, unknown>}};

/** Whether a layer drew at least one model that includes the WGSL shadow module. */
export function isWGSLShadowParticipant(layer: Layer): boolean {
  return layer
    .getModels()
    .some(model =>
      Boolean((model as unknown as ModelHolder).shaderInputs?.modules?.[WGSL_SHADOW_MODULE_NAME])
    );
}

/** PBR layers (3D tile sublayers): scenegraph `_lighting: 'pbr'` or a `pbrMaterial` prop. */
function isPbrLayer(layer: Layer): boolean {
  const props = layer.props as {_lighting?: string; pbrMaterial?: unknown};
  return props._lighting === 'pbr' || Boolean(props.pbrMaterial);
}

/** Normalizes Kepler/Deck shadow colors (0..255 RGB or 0..1 RGB) to 0..1 RGB plus opacity. */
export function normalizeWGSLShadowColor(
  shadowColor: ArrayLike<number> | undefined
): [number, number, number, number] {
  const color = shadowColor ?? DEFAULT_SHADOW_COLOR;
  const scale = Math.max(color[0] ?? 0, color[1] ?? 0, color[2] ?? 0) > 1 ? 255 : 1;
  return [
    (color[0] ?? 0) / scale,
    (color[1] ?? 0) / scale,
    (color[2] ?? 0) / scale,
    color[3] ?? DEFAULT_SHADOW_COLOR[3]
  ];
}

/** Light view matrices exactly as Deck's LightingEffect computes them (eye at the negated direction). */
export function makeWGSLShadowMatrices(lights: readonly WGSLShadowLight[]): Matrix4[] {
  return lights
    .slice(0, MAX_SHADOW_LIGHTS)
    .map(light => new Matrix4().lookAt({eye: new Vector3(Array.from(light.direction)).negate()}));
}

/**
 * Renders participating layers into an RGBA8 packed-depth map from the light, like Deck's
 * ShadowPass, but with explicit WebGPU blend state so the packed depth is stored unmodified.
 */
class WGSLShadowPass extends LayersPass {
  private framebuffer: Framebuffer | null;
  private isExcluded: (layer: Layer) => boolean;
  /** Shadow map texels per viewport device pixel (uniform, so pixel-sized geometry keeps its world size). */
  resolutionScale = 1;

  constructor(device: Device, isExcluded: (layer: Layer) => boolean) {
    super(device, {id: 'wgsl-shadow-pass'});
    this.isExcluded = isExcluded;
    const shadowMap = device.createTexture({
      id: 'wgsl-shadow-map',
      format: 'rgba8unorm',
      usage: Texture.SAMPLE | Texture.RENDER | Texture.COPY_SRC,
      width: 1,
      height: 1,
      sampler: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    const depthBuffer = device.createTexture({
      id: 'wgsl-shadow-depth',
      format: 'depth24plus',
      width: 1,
      height: 1
    });
    this.framebuffer = device.createFramebuffer({
      id: 'wgsl-shadow-map',
      width: 1,
      height: 1,
      colorAttachments: [shadowMap],
      depthStencilAttachment: depthBuffer
    });
  }

  delete(): void {
    this.framebuffer?.destroy();
    this.framebuffer = null;
  }

  getShadowMap(): Texture {
    return this.framebuffer!.colorAttachments[0].texture;
  }

  override render(params: LayersPassRenderOptions): void {
    const target = this.framebuffer!;
    const canvasContext = (params as {canvasContext?: {cssToDeviceRatio(): number}}).canvasContext;
    const pixelRatio = canvasContext?.cssToDeviceRatio() ?? 1;
    const viewport = params.viewports[0];
    const scale = this.resolutionScale;
    const width = Math.max(1, Math.round(viewport.width * pixelRatio * scale));
    const height = Math.max(1, Math.round(viewport.height * pixelRatio * scale));
    if (width !== target.width || height !== target.height) target.resize({width, height});
    // A larger map is drawn with a scaled device pixel ratio: world-space sizes stay the same.
    const baseContext =
      canvasContext ?? (this.device.canvasContext as unknown as typeof canvasContext);
    const scaledContext =
      scale === 1 || !baseContext
        ? canvasContext
        : new Proxy(baseContext, {
            get(contextTarget, key) {
              if (key === 'cssToDeviceRatio') return () => pixelRatio * scale;
              if (key === 'getDrawingBufferSize') return () => [width, height];
              const value = (contextTarget as any)[key];
              return typeof value === 'function' ? value.bind(contextTarget) : value;
            }
          });
    super.render({
      ...params,
      ...(scaledContext ? {canvasContext: scaledContext} : {}),
      clearColor: [1, 1, 1, 1],
      target,
      pass: 'shadow'
    } as never);
  }

  override getLayerParameters(layer: Layer): Record<string, unknown> {
    return {
      ...(layer.props.parameters as Record<string, unknown> | undefined),
      blend: false,
      blendColorOperation: 'add',
      blendColorSrcFactor: 'one',
      blendColorDstFactor: 'zero',
      blendAlphaOperation: 'add',
      blendAlphaSrcFactor: 'one',
      blendAlphaDstFactor: 'zero',
      depthWriteEnabled: true,
      depthCompare: 'less-equal',
      cullMode: 'none'
    };
  }

  override shouldDrawLayer(layer: Layer): boolean {
    return (
      (layer.props as {shadowEnabled?: boolean}).shadowEnabled !== false &&
      !this.isExcluded(layer) &&
      isWGSLShadowParticipant(layer)
    );
  }

  override getShaderModuleProps(
    _layer: Layer,
    _effects: unknown,
    otherShaderModuleProps: Record<string, any>
  ): Record<string, unknown> {
    return {
      [WGSL_SHADOW_MODULE_NAME]: {project: otherShaderModuleProps['project'], drawToShadowMap: true}
    };
  }
}

type PreRenderOptions = {
  layers: Layer[];
  layerFilter?: unknown;
  viewports: Viewport[];
  onViewportActive?: unknown;
  views?: unknown;
};

type ShadowViewport = Viewport & WGSLShadowFrustumViewport;

function hasFiniteDirection(light: WGSLShadowLight): boolean {
  return Array.from(light.direction)
    .slice(0, 3)
    .every(value => Number.isFinite(value));
}

/**
 * WebGPU Light and Shadow: a Deck `LightingEffect` (shadowless lights) that adds shadow mapping
 * compatible with Deck's through the WGSL `shadow` module.
 *
 * Every viewport (split maps, minimaps) gets its own shadow maps, fitted to that viewport. Map
 * views use Deck's ground-plane fitting; orbit/orthographic views fit the light frustum to the
 * scene bounds and globe views to the visible sphere (see `wgsl-shadow-frustum.ts`).
 *
 * Shadow state (`isShadowActive`, `skipReason`) is exposed for diagnostics. Scenes with no
 * participating layers fall back to no shadow map; the host decides whether a screen-space
 * fallback pass runs in that case.
 */
export class WGSLShadowLightingEffect extends LightingEffect {
  /** True when the last frame drew a shadow map for at least one viewport. */
  isShadowActive = false;
  /** Why the last frame drew no shadow map, or null. */
  skipReason: string | null = null;
  /** Layers that drew into the shadow map during the last frame. */
  participantLayerCount = 0;
  renderedFrameCount = 0;
  /** Number of viewports with a shadow map in the last frame. */
  activeViewportCount = 0;
  private shadowOptions: WGSLShadowLightingEffectOptions;
  private viewportStates = new Map<string, ViewportShadowState>();

  /** `lightingProps` are plain Deck lights (shadow flags must be unset); shadows come from `options`. */
  constructor(lightingProps: Record<string, unknown>, options: WGSLShadowLightingEffectOptions) {
    super(lightingProps as never);
    this.shadowOptions = options;
  }

  /** The shadow map textures of the last frame, viewport by viewport and light by light. */
  getShadowMaps(viewportId?: string): Texture[] {
    const states =
      viewportId === undefined
        ? [...this.viewportStates.values()]
        : [this.viewportStates.get(viewportId)].filter(Boolean);
    return states.flatMap(state =>
      state!.active ? state!.passes.map(pass => pass.getShadowMap()) : []
    );
  }

  /** How the light frustum of each viewport was fitted in the last frame, by viewport id. */
  getViewportFitting(): Record<string, ViewportShadowState['fitting'] | 'inactive'> {
    return Object.fromEntries(
      [...this.viewportStates].map(([id, state]) => [id, state.active ? state.fitting : 'inactive'])
    );
  }

  /** Shadow map size and texel density of each viewport in the last frame, by viewport id. */
  getViewportShadowStats(): Record<string, WGSLShadowViewportStats> {
    return Object.fromEntries(
      [...this.viewportStates]
        .filter(([, state]) => state.active && state.stats)
        .map(([id, state]) => [id, state.stats!])
    );
  }

  setShadowOptions(options: WGSLShadowLightingEffectOptions): void {
    this.shadowOptions = options;
  }

  /** Whether a layer is excluded from lighting and shadows (editor layers). */
  isLayerExcluded(layer: Layer): boolean {
    const prefixes = this.shadowOptions.excludedLayerIdPrefixes;
    return Boolean(prefixes?.some(prefix => layer.id.startsWith(prefix)));
  }

  override preRender(options: PreRenderOptions): void {
    const device = this.context?.device;
    const lights = this.shadowOptions.lights.slice(0, MAX_SHADOW_LIGHTS);
    this.isShadowActive = false;
    this.activeViewportCount = 0;
    for (const state of this.viewportStates.values()) state.active = false;
    if (!device || !lights.length) {
      this.skipReason = 'No shadow-casting directional light.';
      this.releaseViewportStates();
      return;
    }
    if (device.type !== 'webgpu') {
      this.skipReason = 'Shadow maps need the WGSL shadow module (WebGPU only).';
      this.releaseViewportStates();
      return;
    }
    const layers = options.layers.filter(layer => !this.isLayerExcluded(layer));
    const participants = layers.filter(isWGSLShadowParticipant);
    this.participantLayerCount = participants.length;
    if (!participants.length) {
      this.skipReason = 'No visible layer includes the WGSL shadow module.';
      this.releaseViewportStates();
      return;
    }
    const dummyShadowMap = getWGSLShadowFallbackTexture(device);
    const skipReasons: string[] = [];
    const seenViewportIds = new Set<string>();
    for (const rawViewport of options.viewports) {
      const viewport = rawViewport as ShadowViewport;
      seenViewportIds.add(viewport.id);
      let reason: string | null;
      try {
        reason = this.renderViewportShadow(device, viewport, lights, participants, dummyShadowMap, {
          ...options,
          layers
        });
      } catch (error) {
        // A viewport whose shadow pass fails draws unshadowed instead of breaking the frame.
        reason = `Shadow pass failed: ${(error as Error).message}`;
        const state = this.viewportStates.get(viewport.id);
        if (state) state.active = false;
      }
      if (reason) skipReasons.push(`${viewport.id}: ${reason}`);
    }
    for (const [id, state] of this.viewportStates) {
      if (seenViewportIds.has(id)) continue;
      state.passes.forEach(pass => pass.delete());
      this.viewportStates.delete(id);
    }
    this.isShadowActive = this.activeViewportCount > 0;
    this.skipReason = this.isShadowActive ? null : skipReasons.join('; ') || 'No viewport.';
    if (this.isShadowActive) this.renderedFrameCount++;
  }

  /** Frees every shadow map and depth buffer; the effect is inactive until the next shadowed frame. */
  private releaseViewportStates(): void {
    for (const state of this.viewportStates.values()) state.passes.forEach(pass => pass.delete());
    this.viewportStates.clear();
  }

  /** Texels per device pixel for a globe shadow map: the budget only ever enlarges the map. */
  private getGlobeResolutionScale(device: Device, width: number, height: number): number {
    const maximumSize = this.shadowOptions.globeShadowMapSize ?? WGSL_SHADOW_DEFAULT_GLOBE_MAP_SIZE;
    if (!maximumSize) return 1;
    const budget =
      this.shadowOptions.globeShadowTexelBudget ?? WGSL_SHADOW_DEFAULT_GLOBE_TEXEL_BUDGET;
    const limit = device.limits.maxTextureDimension2D;
    const wanted = Math.min(
      maximumSize / Math.max(width, height),
      Math.sqrt(budget / (width * height))
    );
    return Math.min(Math.max(1, wanted), limit / Math.max(width, height));
  }

  /** Draws the shadow maps of one viewport; returns the reason when it could not. */
  private renderViewportShadow(
    device: Device,
    viewport: ShadowViewport,
    lights: readonly WGSLShadowLight[],
    participants: Layer[],
    dummyShadowMap: Texture,
    options: PreRenderOptions
  ): string | null {
    const isGlobe =
      (viewport as {projectionMode?: number}).projectionMode === PROJECTION_MODE_GLOBE;
    // Time-of-day suns compute their direction from the viewport location.
    for (const light of lights) light.getProjectedLight?.({layer: {context: {viewport}}});
    if (!lights.every(hasFiniteDirection)) {
      return 'The light direction is not defined for this view (no latitude/longitude).';
    }
    const shadowMatrices = makeWGSLShadowMatrices(lights);
    let projections: WGSLShadowProjection[] | undefined;
    let fitting: ViewportShadowState['fitting'] = 'deck-ground-plane';
    let resolutionScale = 1;
    let texelsPerMeter: number | null = null;
    let baselineTexelsPerMeter: number | null = null;
    const pixelRatio = this.context?.device.canvasContext?.cssToDeviceRatio() ?? 1;
    const deviceWidth = Math.max(1, Math.round(viewport.width * pixelRatio));
    const deviceHeight = Math.max(1, Math.round(viewport.height * pixelRatio));
    if (isGlobe) {
      resolutionScale = this.getGlobeResolutionScale(device, deviceWidth, deviceHeight);
    }
    if (isGlobe || !viewport.isGeospatial) {
      const sceneBounds =
        this.shadowOptions.sceneBounds ?? mergeWGSLShadowLayerBounds(participants);
      const points = getWGSLShadowFramePoints(viewport, {globe: isGlobe, sceneBounds});
      if (!points) return 'The globe is not visible in this view.';
      const aspectRatio = deviceWidth / deviceHeight;
      const mapSize: [number, number] = [
        Math.max(1, Math.round(deviceWidth * resolutionScale)),
        Math.max(1, Math.round(deviceHeight * resolutionScale))
      ];
      projections = shadowMatrices.map(matrix =>
        makeWGSLShadowProjection(
          matrix,
          points,
          isGlobe
            ? {
                aspectRatio,
                mapSize,
                // The globe frustum spans the whole shell, so the bias is a fixed 2 m.
                depthBias: 2 / WGSL_SHADOW_GLOBE_METERS_PER_UNIT
              }
            : {mapSize}
        )
      );
      if (isGlobe) {
        const legacy = makeWGSLShadowProjection(shadowMatrices[0], points);
        const getDensity = (extentSize: [number, number], width: number, height: number) =>
          Math.min(width / extentSize[0], height / extentSize[1]) /
          WGSL_SHADOW_GLOBE_METERS_PER_UNIT;
        texelsPerMeter = getDensity(
          projections[0].extentSize,
          deviceWidth * resolutionScale,
          deviceHeight * resolutionScale
        );
        baselineTexelsPerMeter = getDensity(legacy.extentSize, deviceWidth, deviceHeight);
      }
      fitting = points.source === 'globe' ? 'globe' : points.source;
    }
    let state = this.viewportStates.get(viewport.id);
    if (!state) {
      state = {passes: [], shadowMatrices, active: false, fitting};
      this.viewportStates.set(viewport.id, state);
    }
    while (state.passes.length > lights.length) state.passes.pop()?.delete();
    while (state.passes.length < lights.length) {
      state.passes.push(new WGSLShadowPass(device, layer => this.isLayerExcluded(layer)));
    }
    state.shadowMatrices = shadowMatrices;
    state.projections = projections;
    state.fitting = fitting;
    state.stats = {
      shadowMapSize: [
        Math.max(1, Math.round(deviceWidth * resolutionScale)),
        Math.max(1, Math.round(deviceHeight * resolutionScale))
      ],
      resolutionScale,
      texelsPerMeter,
      baselineTexelsPerMeter
    };
    for (const pass of state.passes) pass.resolutionScale = resolutionScale;
    // The shadow map is a standalone texture: draw it from the viewport's origin, not its split-view offset.
    const passViewport = Object.create(viewport, {x: {value: 0}, y: {value: 0}}) as Viewport;
    state.passes.forEach((pass, lightIndex) => {
      pass.render({
        layers: options.layers,
        layerFilter: options.layerFilter,
        viewports: [passViewport],
        onViewportActive: options.onViewportActive,
        views: options.views,
        shaderModuleProps: {
          [WGSL_SHADOW_MODULE_NAME]: {
            shadowLightId: lightIndex,
            dummyShadowMap,
            shadowMatrices,
            shadowProjections: projections
          }
        }
      } as never);
    });
    state.active = true;
    this.activeViewportCount++;
    return null;
  }

  override getShaderModuleProps(
    layer: Layer,
    otherShaderModuleProps: Record<string, unknown>
  ): ReturnType<LightingEffect['getShaderModuleProps']> {
    const device = this.context?.device;
    const dummyShadowMap = device ? getWGSLShadowFallbackTexture(device) : undefined;
    if (this.isLayerExcluded(layer)) {
      // Kepler: no lighting and no shadows for the editor layer and its sublayers.
      return {
        [WGSL_SHADOW_MODULE_NAME]: {shadowEnabled: false, dummyShadowMap},
        lighting: {enabled: false},
        phongMaterial: null,
        gouraudMaterial: null
      } as never;
    }
    const lightingProps = super.getShaderModuleProps(layer, otherShaderModuleProps as never);
    // Kepler replaces PBR with simple phong whenever the effect has shadow-casting lights.
    const simplePhong = this.shadowOptions.lights.length > 0 && isPbrLayer(layer);
    const viewportId = (otherShaderModuleProps['project'] as {viewport?: {id: string}} | undefined)
      ?.viewport?.id;
    const state = viewportId === undefined ? undefined : this.viewportStates.get(viewportId);
    return {
      ...lightingProps,
      [WGSL_SHADOW_MODULE_NAME]: state?.active
        ? {
            project: otherShaderModuleProps['project'],
            shadowMaps: state.passes.map(pass => pass.getShadowMap()),
            dummyShadowMap,
            shadowColor: normalizeWGSLShadowColor(this.shadowOptions.shadowColor),
            shadowMatrices: state.shadowMatrices,
            shadowProjections: state.projections,
            outputUniformShadow: Boolean(this.shadowOptions.outputUniformShadow),
            useSimplePhong: simplePhong
          }
        : {shadowEnabled: false, dummyShadowMap, useSimplePhong: simplePhong},
      ...(simplePhong && isPbrLayer(layer) ? {pbrMaterial: {unlit: 1}} : {})
    } as never;
  }

  override cleanup(context: EffectContext): void {
    this.releaseViewportStates();
    this.isShadowActive = false;
    // Layers keep their last shadow uniforms, so switch the shadow lookup off explicitly.
    const dummyShadowMap = getWGSLShadowFallbackTexture(context.device);
    const layers = ((
      context.deck as unknown as {layerManager?: {getLayers(): Layer[]}} | undefined
    )?.layerManager?.getLayers() ?? []) as Layer[];
    for (const layer of layers) {
      layer.setShaderModuleProps({
        [WGSL_SHADOW_MODULE_NAME]: {shadowEnabled: false, dummyShadowMap}
      } as never);
    }
    super.cleanup(context);
  }
}
