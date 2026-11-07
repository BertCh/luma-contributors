// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL replacements for Kepler's GLSL-only effects on the WebGPU host.
 *
 * - Distance fog and surface fog become `WgslFogEffect`: a Deck post-process effect that runs the
 *   ported WGSL passes of `effects-wgsl-fog-passes.ts` against Deck's scene color and depth.
 *   It reads the original Kepler effect's `props` every frame, so slider changes apply live and
 *   the Kepler effect state is never modified (saved settings round trip).
 * - Light & Shadow (`CustomDeckLightingEffect`) becomes a `WGSLShadowLightingEffect`: a Deck
 *   `LightingEffect` with the same ambient, directional and point lights (shadow flags cleared for
 *   Deck's GLSL shadow module) plus real WGSL shadow mapping (light-space packed-depth pass over
 *   participating layers, Deck matrices, `shadowColor`, time-of-day sun direction, night-time
 *   uniform shadow, simple-phong for PBR layers). The accompanying `WgslFogEffect` of kind
 *   'shadow' (screen-space ray march of `effects-wgsl-shadow-pass.ts`) is only the fallback: it
 *   stands down whenever the shadow map drew, and runs only for scenes with no
 *   shadow-participating layer (globe, orbit and split views get their own shadow maps) (`shadowSkipReason` in `getReplacements`).
 *
 * Replacements are cached per Kepler effect object so Deck's effect manager keeps one instance.
 */

import {
  WGSLShadowLightingEffect,
  type WGSLShadowLightingEffectOptions
} from '@deck.gl-community/gpu-layers';
import type {Effect, EffectContext, PostRenderOptions} from '@deck.gl/core';
import type {Device, Framebuffer, Texture} from '@luma.gl/core';
import {BackgroundTextureModel, ShaderPassRenderer} from '@luma.gl/engine';
import {
  type FogViewportLike,
  getFogCameraUniforms,
  keplerDistanceFogPass,
  keplerSurfaceFogPass
} from './effects-wgsl-fog-passes';
import {getShadowPassUniforms, keplerShadowPass} from './effects-wgsl-shadow-pass';

export type WgslFogKind = 'distance' | 'surface' | 'shadow';

type KeplerFogProps = {
  density?: number;
  fogStart?: number;
  fogRange?: number;
  height?: number;
  thickness?: number;
  fogColor?: number[];
  pattern?: boolean;
};

type SourceEffectLike = {
  id?: string;
  props?: KeplerFogProps;
  ambientLight?: unknown;
  directionalLights?: {shadow?: boolean; direction?: ArrayLike<number>}[];
  shadowColor?: ArrayLike<number>;
  pointLights?: unknown[];
  constructor: {name: string};
};

/** Returns which fog a Kepler deck effect implements, or null. */
export function getWgslFogKind(effect: SourceEffectLike): WgslFogKind | null {
  const name = effect.constructor?.name ?? '';
  if (/DistanceFog/.test(name) || effect.id === 'distance-fog-effect') return 'distance';
  if (/SurfaceFog/.test(name) || effect.id === 'surface-fog-effect') return 'surface';
  return null;
}

/** Whether the effect is Kepler's Light and Shadow deck effect. */
export function isKeplerLightingEffect(effect: SourceEffectLike): boolean {
  return effect.constructor?.name === 'CustomDeckLightingEffect';
}

/** Computes the pass uniforms for one frame from the Kepler effect props and the Deck viewport. */
export function getFogPassUniforms(
  kind: 'distance' | 'surface',
  props: KeplerFogProps,
  viewport: FogViewportLike | undefined
): Record<string, unknown> {
  const camera = getFogCameraUniforms(viewport);
  const fogColor = (
    props.fogColor ?? (kind === 'distance' ? [217, 222, 230] : [230, 235, 242])
  ).slice(0, 3);
  if (kind === 'distance') {
    return {
      inverseMatrix: camera.inverseMatrix,
      fogColor,
      density: props.density ?? 0.5,
      fogStart: props.fogStart ?? 0.3,
      fogRange: props.fogRange ?? 0.5,
      nearPlane: camera.nearPlane,
      farPlane: camera.farPlane,
      viewportSize: camera.viewportSize
    };
  }
  return {
    inverseMatrix: camera.inverseMatrix,
    fogColor,
    density: props.density ?? 0.6,
    fogHeight: (props.height ?? 50) * camera.unitsPerMeter,
    fogThickness: (props.thickness ?? 50) * camera.unitsPerMeter,
    unitsPerMeter: camera.unitsPerMeter,
    pattern: props.pattern ? 1 : 0,
    viewportSize: camera.viewportSize
  };
}

/** Deck post-process effect running a ported Kepler fog pass in WGSL. */
export class WgslFogEffect implements Effect {
  readonly id: string;
  readonly props: KeplerFogProps;
  readonly useInPicking = false;
  /** Frames where the pass could not run (no depth on the input buffer, for example). */
  skippedFrameCount = 0;
  renderedFrameCount = 0;
  /** Frames where the screen-space shadow pass stood down because the shadow map drew. */
  shadowMapFrameCount = 0;
  /** Set on the screen-space shadow fallback: the real shadow map effect it defers to. */
  shadowMapEffect?: WGSLShadowLightingEffect;
  private device?: Device;
  private renderer?: ShaderPassRenderer;
  private presenter?: BackgroundTextureModel;
  private width = 0;
  private height = 0;

  constructor(
    readonly kind: WgslFogKind,
    readonly source: SourceEffectLike
  ) {
    this.id =
      kind === 'shadow'
        ? `${source.id ?? 'lighting-effect'}-shadow`
        : (source.id ?? `${kind}-fog-effect`);
    this.props = source.props ?? {};
    this.postRender = this.runPostRender;
  }

  setup({device}: Pick<EffectContext, 'device'>): void {
    this.device = device;
    this.renderer = new ShaderPassRenderer(device, {
      shaderPasses: [
        this.kind === 'distance'
          ? keplerDistanceFogPass
          : this.kind === 'surface'
            ? keplerSurfaceFogPass
            : keplerShadowPass
      ]
    });
  }

  /**
   * Deck only presents the scene to the canvas through the last post effect, and decides which effect
   * is last from `postRender` being defined when `preRender` returns. A shadow fallback that stands
   * down must therefore drop `postRender`, otherwise the canvas would keep the last frame drawn
   * before the effect existed (the whole map freezes).
   */
  postRender?: (options: PostRenderOptions) => Framebuffer;

  preRender(): void {
    const isStandingDown = this.kind === 'shadow' && Boolean(this.shadowMapEffect?.isShadowActive);
    if (isStandingDown) this.shadowMapFrameCount++;
    this.postRender = isStandingDown ? undefined : this.runPostRender;
  }

  /** Copies the scene to the canvas when this is the last post effect and the pass could not run. */
  private presentInput(options: PostRenderOptions): Framebuffer {
    const {inputBuffer} = options;
    const sourceTexture = inputBuffer.colorAttachments[0]?.texture;
    if (!options.target || !this.device || !sourceTexture) return inputBuffer;
    return this.presentTexture(sourceTexture, options);
  }

  private presentTexture(texture: Texture, options: PostRenderOptions): Framebuffer {
    const device = this.device!;
    this.presenter ??= new BackgroundTextureModel(device, {
      id: `${this.id}-presenter`,
      backgroundTexture: texture,
      flipY: device.type === 'webgpu'
    });
    this.presenter.setProps({backgroundTexture: texture});
    const lastEffect = options.effects?.filter(effect => effect.postRender).at(-1);
    const target =
      options.target ??
      (!lastEffect || lastEffect.id === this.id
        ? (options.canvasContext || device.getCanvasContext()).getCurrentFramebuffer()
        : options.swapBuffer);
    const commandEncoder = device.commandEncoder;
    this.presenter.predraw(commandEncoder);
    const pass = commandEncoder.beginRenderPass({
      id: `${this.id}-present`,
      framebuffer: target,
      clearColor: false,
      clearDepth: false,
      parameters: {viewport: [0, 0, target.width, target.height]}
    });
    this.presenter.draw(pass);
    pass.end();
    // Deck's layer pass submits before postRender; present this effect in the same frame.
    device.submit();
    this.renderedFrameCount++;
    return target;
  }

  private readonly runPostRender = (options: PostRenderOptions): Framebuffer => {
    const {inputBuffer} = options;
    const depthTexture = inputBuffer.depthStencilAttachment?.texture;
    const sourceTexture = inputBuffer.colorAttachments[0]?.texture;
    if (!this.device || !this.renderer || !depthTexture || !sourceTexture) {
      // Same as Kepler: only the first post effect of the chain still owns the depth attachment.
      this.skippedFrameCount++;
      return this.presentInput(options);
    }
    const {width, height} = sourceTexture;
    if (width !== this.width || height !== this.height) {
      this.renderer.resize([width, height]);
      this.width = width;
      this.height = height;
    }
    const passName =
      this.kind === 'distance'
        ? 'keplerDistanceFog'
        : this.kind === 'surface'
          ? 'keplerSurfaceFog'
          : 'keplerShadow';
    const viewport = options.viewports?.[0] as FogViewportLike | undefined;
    const output = this.renderer.renderToTexture({
      sourceTexture,
      bindings: {depthTexture},
      uniforms: {
        [passName]:
          this.kind === 'shadow'
            ? getShadowPassUniforms(
                this.source.directionalLights,
                this.source.shadowColor,
                viewport as never,
                getFogCameraUniforms(viewport)
              )
            : getFogPassUniforms(this.kind, this.source.props ?? this.props, viewport)
      }
    });
    if (!output) {
      this.skippedFrameCount++;
      return this.presentInput(options);
    }
    return this.presentTexture(output, options);
  };

  cleanup(): void {
    this.renderer?.destroy();
    this.presenter?.destroy();
    this.renderer = undefined;
    this.presenter = undefined;
    this.device = undefined;
    this.width = this.height = 0;
  }
}

type LightLike = {type?: string; shadow?: boolean; [key: string]: unknown};

function makeLightingSignature(source: SourceEffectLike): string {
  return JSON.stringify([source.ambientLight, source.directionalLights, source.pointLights]);
}

/** Builds Deck LightingEffect props from Kepler's lights with every shadow flag cleared. */
export function makeLightingEffectProps(source: SourceEffectLike): Record<string, LightLike> {
  const props: Record<string, LightLike> = {};
  if (source.ambientLight) props['ambientLight'] = source.ambientLight as LightLike;
  (source.directionalLights ?? []).forEach((light, index) => {
    // A shallow copy keeps the Kepler light untouched; Deck reads `shadow` from the copy.
    const copy = Object.assign(Object.create(Object.getPrototypeOf(light)), light, {shadow: false});
    props[`directionalLight${index}`] = copy as LightLike;
  });
  ((source.pointLights ?? []) as LightLike[]).forEach((light, index) => {
    props[`pointLight${index}`] = light;
  });
  return props;
}

/** Shadow-casting directional lights and shadow color of a Kepler Light and Shadow effect. */
export function makeShadowOptions(source: SourceEffectLike): WGSLShadowLightingEffectOptions {
  return {
    // The Kepler light objects themselves: a SunLight refreshes its direction for every viewport.
    lights: (source.directionalLights ?? []).filter(light => light?.shadow) as never,
    shadowColor: source.shadowColor,
    outputUniformShadow: Boolean((source as {outputUniformShadow?: boolean}).outputUniformShadow),
    // Kepler's CustomDeckLightingEffect skips lighting and shadows for the editor layer (EDITOR_LAYER_ID).
    excludedLayerIdPrefixes: ['kepler_editor_layer']
  };
}

export type WgslEffectReplacer = {
  /** Returns the WGSL replacement for a Kepler GLSL effect, or null when none exists. */
  replace(effect: unknown): Effect | null;
  /** Returns the extra WGSL shadow effect that accompanies a Light and Shadow replacement. */
  replaceShadow(effect: unknown): Effect | null;
  /** Live replacement effects, for diagnostics and the test hook. */
  getReplacements(): {
    id: string;
    kind: string;
    renderedFrameCount?: number;
    skippedFrameCount?: number;
    shadowMapFrameCount?: number;
    shadowMode?: 'shadow-map' | 'screen-space';
    shadowSkipReason?: string;
    shadowViewportCount?: number;
    shadowViewportFitting?: Record<string, string>;
    shadowParticipantLayerCount?: number;
    /** Per viewport shadow map size and texel density of the last frame (see getViewportShadowStats). */
    shadowViewportStats?: ReturnType<WGSLShadowLightingEffect['getViewportShadowStats']>;
  }[];
};

/** Creates the per-host replacement cache; `report` receives one visible message per condition. */
export function makeWgslEffectReplacer(report: (message: string) => void): WgslEffectReplacer {
  const fogCache = new WeakMap<object, WgslFogEffect>();
  const shadowCache = new WeakMap<object, WgslFogEffect>();
  const lightingCache = new WeakMap<
    object,
    {effect: WGSLShadowLightingEffect; signature: string}
  >();
  const live = new Set<WgslFogEffect | WGSLShadowLightingEffect>();
  return {
    replace(effect) {
      if (!effect || typeof effect !== 'object') return null;
      const source = effect as SourceEffectLike;
      const kind = getWgslFogKind(source);
      if (kind) {
        let replacement = fogCache.get(source);
        if (!replacement) {
          replacement = new WgslFogEffect(kind, source);
          fogCache.set(source, replacement);
          live.add(replacement);
        }
        return replacement;
      }
      if (isKeplerLightingEffect(source)) {
        const signature = makeLightingSignature(source);
        let entry = lightingCache.get(source);
        if (!entry) {
          entry = {
            effect: new WGSLShadowLightingEffect(
              makeLightingEffectProps(source),
              makeShadowOptions(source)
            ),
            signature
          };
          lightingCache.set(source, entry);
          live.add(entry.effect);
        } else {
          entry.effect.setShadowOptions(makeShadowOptions(source));
          if (entry.signature !== signature) {
            entry.effect.setProps(makeLightingEffectProps(source) as never);
            entry.signature = signature;
          }
        }
        return entry.effect as unknown as Effect;
      }
      return null;
    },
    replaceShadow(effect) {
      if (!effect || typeof effect !== 'object') return null;
      const source = effect as SourceEffectLike;
      if (
        !isKeplerLightingEffect(source) ||
        !source.directionalLights?.some(light => light?.shadow)
      )
        return null;
      let shadowEffect = shadowCache.get(source);
      if (!shadowEffect) {
        shadowEffect = new WgslFogEffect('shadow', source);
        shadowEffect.shadowMapEffect = lightingCache.get(source)?.effect;
        shadowCache.set(source, shadowEffect);
        live.add(shadowEffect);
      }
      return shadowEffect;
    },
    getReplacements() {
      return [...live].map(effect =>
        effect instanceof WgslFogEffect
          ? {
              id: effect.id,
              kind: effect.kind === 'shadow' ? 'shadow' : `${effect.kind}-fog`,
              renderedFrameCount: effect.renderedFrameCount,
              skippedFrameCount: effect.skippedFrameCount,
              shadowMapFrameCount: effect.shadowMapFrameCount
            }
          : {
              id: effect.id,
              kind: 'lighting',
              shadowMode: effect.isShadowActive ? 'shadow-map' : 'screen-space',
              shadowSkipReason: effect.skipReason ?? undefined,
              shadowViewportCount: effect.activeViewportCount,
              shadowViewportFitting: effect.getViewportFitting(),
              shadowParticipantLayerCount: effect.participantLayerCount,
              shadowViewportStats: effect.getViewportShadowStats(),
              renderedFrameCount: effect.renderedFrameCount
            }
      );
    }
  };
}

/**
 * Wraps the base WebGPU effect filter: replaceable Kepler effects become WGSL effects in place,
 * everything else keeps the existing skip-with-reason behavior.
 */
export function createWebGPUEffectsPipeline(
  report: (message: string) => void,
  filterRemaining: (effects: unknown) => unknown[]
): {filter: <EffectT>(effects: unknown) => EffectT[]; replacer: WgslEffectReplacer} {
  const replacer = makeWgslEffectReplacer(report);
  const filter = <EffectT>(effects: unknown): EffectT[] => {
    const input = Array.isArray(effects) ? (effects as EffectT[]) : [];
    const replaced = new Map<EffectT, Effect>();
    for (const effect of input) {
      const replacement = replacer.replace(effect);
      if (replacement) replaced.set(effect, replacement);
    }
    const remaining = filterRemaining(input.filter(effect => !replaced.has(effect)));
    const result: EffectT[] = [];
    for (const effect of input) {
      const replacement = replaced.get(effect);
      if (replacement) {
        result.push(replacement as unknown as EffectT);
        const shadowEffect = replacer.replaceShadow(effect);
        if (shadowEffect) result.push(shadowEffect as unknown as EffectT);
      } else if (remaining.includes(effect)) result.push(effect);
    }
    return result;
  };
  return {filter, replacer};
}
