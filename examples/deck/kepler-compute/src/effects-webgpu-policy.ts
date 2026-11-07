// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Decides which Kepler (deck.gl) effects can run on the WebGPU host.
 *
 * Audit result (Kepler 3.3.0-alpha.18, luma 10, Deck 9.4):
 * - The 15 image post-processing effects come from `@luma.gl/effects` and all carry WGSL `source`.
 * - Kepler's own Light & Shadow (`CustomDeckLightingEffect`), distance fog and surface fog passes
 *   are GLSL only: `#version 300 es` fragment strings, `shadow` module string surgery and direct
 *   `device.gl` access. They cannot compile on a WebGPU device.
 * Disabled effects are skipped at render time only. The Kepler effect state is not modified, so
 * saved effect settings, ordering and enabled flags round trip through the share/save path.
 */

export type EffectLike = {
  id?: string;
  module?: {name?: string; source?: unknown; fs?: unknown; vs?: unknown};
  directionalLights?: {shadow?: boolean}[];
  constructor: {name: string};
};

export type DisabledEffect = {id: string; reason: string};

const GLSL_ONLY_REASON =
  'GLSL-only shader pass; no WGSL implementation exists, so it is disabled on the WebGPU host.';
const SHADOW_REASON =
  "Kepler's GLSL shadow module cannot compile on WebGPU; it is replaced by the WGSL shadow map effect, so only an unreplaced instance is disabled.";

/** Returns a reason when the effect cannot run on WebGPU, otherwise null. */
export function getEffectDisabledReason(effect: EffectLike): string | null {
  const name = effect.constructor?.name ?? '';
  if (
    name === 'CustomDeckLightingEffect' ||
    effect.directionalLights?.some(light => light?.shadow)
  ) {
    return SHADOW_REASON;
  }
  const module = effect.module;
  if (module && typeof module.source !== 'string' && (module.fs || module.vs)) {
    return GLSL_ONLY_REASON;
  }
  if (/DistanceFog|SurfaceFog/.test(name) || /fog/i.test(effect.id ?? '')) {
    // Kepler fog effects expose a GLSL module with no `source`; the name check covers lazy modules.
    return module && typeof module.source === 'string' ? null : GLSL_ONLY_REASON;
  }
  return null;
}

/** Splits effects into those safe to render and those disabled with a reason. */
export function selectWebGPUEffects<EffectT extends EffectLike>(
  effects: readonly EffectT[]
): {effects: EffectT[]; disabled: DisabledEffect[]} {
  const enabled: EffectT[] = [];
  const disabled: DisabledEffect[] = [];
  for (const effect of effects) {
    const reason = effect ? getEffectDisabledReason(effect) : null;
    if (reason) disabled.push({id: effect.id ?? effect.constructor.name, reason});
    else if (effect) enabled.push(effect);
  }
  return {effects: enabled, disabled};
}

/**
 * Filters Kepler effects for the WebGPU host and reports each newly disabled effect once.
 * Report callbacks receive the user-visible message; repeats for the same id and reason are
 * suppressed until the effect is re-enabled or removed.
 */
export function createEffectsFilter(
  report: (message: string) => void
): <EffectT extends EffectLike>(effects: unknown) => EffectT[] {
  const reported = new Map<string, string>();
  return <EffectT extends EffectLike>(effects: unknown): EffectT[] => {
    const input = Array.isArray(effects) ? (effects as EffectT[]) : [];
    const result = selectWebGPUEffects(input);
    const present = new Set<string>();
    for (const {id, reason} of result.disabled) {
      present.add(id);
      if (reported.get(id) !== reason) {
        reported.set(id, reason);
        report(`Effect "${id}" is disabled: ${reason}`);
      }
    }
    for (const id of [...reported.keys()]) if (!present.has(id)) reported.delete(id);
    return result.effects;
  };
}
