// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {PathStyleExtension} from '@deck.gl/extensions';
import {afterEach, expect, test} from 'vitest';
import {
  disableEditorDashExtension,
  getEditorDashStatus,
  makeEditorDashExtensions,
  resetEditorDashStatusForTest
} from '../../../examples/deck/kepler-compute/src/editor-dash-extension';
import {
  createEffectsFilter,
  getEffectDisabledReason
} from '../../../examples/deck/kepler-compute/src/effects-webgpu-policy';
import {
  makeBasemapContext,
  readStyleAttributions
} from '../../../examples/deck/kepler-compute/src/basemap-context';

afterEach(() => resetEditorDashStatusForTest());

test('dash extension can be disabled and falls back to solid strokes', () => {
  expect(makeEditorDashExtensions()[0]).toBeInstanceOf(PathStyleExtension);
  disableEditorDashExtension('shader failed');
  expect(getEditorDashStatus()).toEqual({enabled: false, reason: 'shader failed'});
  expect(makeEditorDashExtensions()).toEqual([]);
});

test('only GLSL-capable effects are disabled with a reason', () => {
  class CustomDeckLightingEffect {
    id = 'light-and-shadow';
  }
  class PostProcessEffect {
    id = 'ink';
    module = {name: 'ink', source: 'fn x() {}', fs: 'void main(){}'};
  }
  class DeckDistanceFogEffect {
    id = 'distance-fog-effect';
    module = {name: 'distanceFog', fs: 'void main(){}'};
  }
  expect(getEffectDisabledReason(new CustomDeckLightingEffect())).toMatch(/shadow/i);
  expect(getEffectDisabledReason(new PostProcessEffect())).toBeNull();
  expect(getEffectDisabledReason(new DeckDistanceFogEffect())).toMatch(/GLSL-only/);

  const messages: string[] = [];
  const filter = createEffectsFilter(message => messages.push(message));
  const effects = [new PostProcessEffect(), new DeckDistanceFogEffect()];
  expect(filter(effects)).toEqual([effects[0]]);
  filter(effects);
  expect(messages).toHaveLength(1);
});

test('basemap context keeps local background and explains failures', async () => {
  const states: string[] = [];
  const failing = makeBasemapContext({
    fetchStyle: async () => {
      throw new Error('HTTP 404');
    },
    onChange: state => states.push(state.status)
  });
  const failed = await failing.setStyleUrl('https://example.test/style.json');
  expect(failed.status).toBe('failed');
  expect(failed.activeStyleUrl).toBeNull();
  expect(failed.errorMessage).toMatch(/HTTP 404.*data is unchanged/);
  expect(states).toEqual(['loading', 'failed']);

  const working = makeBasemapContext({
    fetchStyle: async () => ({
      version: 8,
      sources: {a: {attribution: '© Provider'}, b: {attribution: '© Provider'}, c: {}},
      layers: []
    })
  });
  const external = await working.setStyleUrl('https://example.test/ok.json');
  expect(external).toMatchObject({status: 'external', attributions: ['© Provider']});
  expect((await working.setStyleUrl(null)).status).toBe('local');
  expect(readStyleAttributions({sources: {x: {attribution: ' A '}}})).toEqual(['A']);

  const invalid = makeBasemapContext({fetchStyle: async () => ({version: 8})});
  expect((await invalid.setStyleUrl('https://example.test/bad.json')).status).toBe('failed');
});

test('Deck tile and scenegraph paths carry WGSL and no GLSL-only injections', () => {
  const require = createRequire(import.meta.url);
  const read = (packageName: string, relativePath: string): string | null => {
    try {
      const root = dirname(require.resolve(`${packageName}/package.json`));
      return readFileSync(join(root, relativePath), 'utf8');
    } catch {
      return null;
    }
  };
  for (const [packageName, relativePath] of [
    ['@deck.gl/mesh-layers', 'src/scenegraph-layer/scenegraph-layer.wgsl.ts'],
    ['@deck.gl/mesh-layers', 'src/simple-mesh-layer/simple-mesh-layer.wgsl.ts']
  ]) {
    const source = read(packageName, relativePath);
    if (source) expect(source).toContain('@fragment');
  }
  const tileLayer = read('@deck.gl/geo-layers', 'src/tile-3d-layer/tile-3d-layer.ts');
  if (tileLayer) expect(tileLayer).not.toMatch(/'(vs|fs):#?(decl|main-start|main-end)'/);
});
