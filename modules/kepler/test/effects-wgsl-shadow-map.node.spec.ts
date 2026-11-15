// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AmbientLight, DirectionalLight, LightingEffect} from '@deck.gl/core';
import {WGSLShadowLightingEffect} from '../../../examples/deck/kepler-compute/gpu-layers/index';
import {expect, test} from 'vitest';
import {
  createWebGPUEffectsPipeline,
  makeShadowOptions
} from '../../../examples/deck/kepler-compute/src/effects-wgsl-effects';

class CustomDeckLightingEffect {
  id = 'kepler-light-and-shadow';
  ambientLight = new AmbientLight({color: [255, 255, 255], intensity: 1});
  directionalLights = [
    new DirectionalLight({
      color: [255, 255, 255],
      intensity: 1,
      direction: [-1, 0, -1],
      _shadow: true
    })
  ];
  pointLights: unknown[] = [];
  shadowColor = [0, 0, 0, 0.6];
  outputUniformShadow = false;
}

test('Kepler Light and Shadow becomes a WGSL shadow map effect with a screen-space fallback', () => {
  const sourceEffect = new CustomDeckLightingEffect();
  const {filter, replacer} = createWebGPUEffectsPipeline(
    () => {},
    effects => effects as unknown[]
  );
  const effects = filter<object>([sourceEffect]);
  expect(effects).toHaveLength(2);
  const [lighting, fallback] = effects;
  expect(lighting).toBeInstanceOf(WGSLShadowLightingEffect);
  expect(lighting).toBeInstanceOf(LightingEffect);
  expect((fallback as {kind: string}).kind).toBe('shadow');
  expect((fallback as {shadowMapEffect?: unknown}).shadowMapEffect).toBe(lighting);
  // Stable instances across frames so Deck keeps one effect per Kepler effect.
  expect(filter<object>([sourceEffect])).toEqual(effects);
  const replacements = replacer.getReplacements();
  expect(replacements.find(item => item.kind === 'lighting')).toMatchObject({
    shadowMode: 'screen-space'
  });
  expect(replacements.find(item => item.kind === 'shadow')).toBeDefined();
});

test('shadow options carry the live shadow-casting lights, color and night flag', () => {
  const sourceEffect = new CustomDeckLightingEffect();
  sourceEffect.outputUniformShadow = true;
  const options = makeShadowOptions(sourceEffect as never);
  expect(options.lights).toEqual([sourceEffect.directionalLights[0]]);
  expect(options.shadowColor).toBe(sourceEffect.shadowColor);
  expect(options.outputUniformShadow).toBe(true);
  sourceEffect.directionalLights[0].shadow = false;
  expect(makeShadowOptions(sourceEffect as never).lights).toHaveLength(0);
});
