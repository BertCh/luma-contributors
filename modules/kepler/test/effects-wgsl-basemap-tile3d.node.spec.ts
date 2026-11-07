// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getActiveBasemapAttributions,
  makeAttributionPlainText
} from '../../../examples/deck/kepler-compute/src/basemap-attribution';
import {makeExternalStyleId} from '../../../examples/deck/kepler-compute/src/basemap-picker';
import {createEffectsFilter} from '../../../examples/deck/kepler-compute/src/effects-webgpu-policy';
import {
  createWebGPUEffectsPipeline,
  getFogPassUniforms,
  getWgslFogKind,
  makeLightingEffectProps
} from '../../../examples/deck/kepler-compute/src/effects-wgsl-effects';
import {
  getFogCameraUniforms,
  keplerDistanceFogPass,
  keplerSurfaceFogPass
} from '../../../examples/deck/kepler-compute/src/effects-wgsl-fog-passes';
import {
  describeModelMatrix,
  getDeviceResourceCounts,
  getTile3DDiagnostics
} from '../../../examples/deck/kepler-compute/src/tile3d-diagnostics';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function getStructFieldNames(source: string, structName: string): string[] {
  const body = new RegExp(`struct ${structName} \\{([^}]*)\\}`).exec(source)?.[1] ?? '';
  return body
    .split(',')
    .map(field => field.split(':')[0].trim())
    .filter(Boolean);
}

test('fog WGSL uniform structs match the declared uniform types in order', () => {
  expect(getStructFieldNames(keplerDistanceFogPass.source, 'keplerDistanceFogUniforms')).toEqual(
    Object.keys(keplerDistanceFogPass.uniformTypes)
  );
  expect(getStructFieldNames(keplerSurfaceFogPass.source, 'keplerSurfaceFogUniforms')).toEqual(
    Object.keys(keplerSurfaceFogPass.uniformTypes)
  );
  expect(keplerDistanceFogPass.source).toContain('fn keplerDistanceFog_sampleColor');
  expect(keplerSurfaceFogPass.source).toContain('fn keplerSurfaceFog_sampleColor');
  expect(keplerDistanceFogPass.source).not.toMatch(/#version|gl_FragCoord|sampler2D/);
});

test('fog camera uniforms follow the Kepler viewport reads', () => {
  expect(getFogCameraUniforms(undefined).inverseMatrix).toEqual(IDENTITY);
  const projection = new Array(16).fill(0);
  // near = 14 / (10 - 1) style extraction: n = pm14 / (pm10 - 1), f = pm14 / (pm10 + 1)
  projection[10] = -1.002;
  projection[14] = -0.2002;
  const camera = getFogCameraUniforms({
    width: 800,
    height: 600,
    projectionMatrix: projection,
    pixelUnprojectionMatrix: IDENTITY,
    distanceScales: {unitsPerMeter: [1, 1, 0.5]}
  });
  expect(camera.viewportSize).toEqual([800, 600]);
  expect(camera.nearPlane).toBeCloseTo(0.1, 3);
  expect(camera.farPlane).toBeGreaterThan(camera.nearPlane);
  expect(camera.unitsPerMeter).toBe(0.5);
});

test('surface fog uniforms convert metres to common units and pass the pattern flag', () => {
  const uniforms = getFogPassUniforms(
    'surface',
    {density: 0.7, height: 10, thickness: 20, fogColor: [1, 2, 3], pattern: true},
    {width: 10, height: 10, distanceScales: {unitsPerMeter: [1, 1, 2]}}
  );
  expect(uniforms).toMatchObject({
    fogHeight: 20,
    fogThickness: 40,
    pattern: 1,
    fogColor: [1, 2, 3],
    density: 0.7
  });
});

test('effects pipeline replaces fog and lighting in place and still skips unknown GLSL effects', () => {
  class DeckDistanceFogEffect {
    id = 'distance-fog-effect';
    props = {density: 0.4};
  }
  class CustomDeckLightingEffect {
    id = 'lighting-effect';
    props = {};
    directionalLights = [{shadow: true, type: 'directional', color: [255, 255, 255]}];
    ambientLight = {type: 'ambient', color: [255, 255, 255], intensity: 1};
    pointLights = [];
  }
  class MysteryShaderEffect {
    id = 'mystery';
    module = {name: 'mystery', fs: 'glsl only'};
  }
  const messages: string[] = [];
  const pipeline = createWebGPUEffectsPipeline(
    message => messages.push(message),
    createEffectsFilter(message => messages.push(message))
  );
  const fog = new DeckDistanceFogEffect();
  const lighting = new CustomDeckLightingEffect();
  const result = pipeline.filter([fog, new MysteryShaderEffect(), lighting]) as {id: string}[];
  expect(result.map(effect => effect.id)).toEqual([
    'distance-fog-effect',
    'lighting-effect',
    'lighting-effect-shadow'
  ]);
  expect(result[0]).not.toBe(fog);
  // Stable replacement objects keep Deck's effect manager from re-running setup every frame.
  expect(pipeline.filter([fog, lighting])[0]).toBe(result[0]);
  expect(messages.some(message => /shadow maps are not rendered/.test(message))).toBe(false);
  expect(messages.some(message => /mystery/.test(message))).toBe(true);
  expect(getWgslFogKind(fog)).toBe('distance');
});

test('lighting replacement clears shadow flags without mutating Kepler lights', () => {
  const light = {shadow: true, type: 'directional'};
  const props = makeLightingEffectProps({
    constructor: {name: 'CustomDeckLightingEffect'},
    directionalLights: [light],
    ambientLight: {type: 'ambient'}
  } as never);
  expect((props['directionalLight0'] as {shadow: boolean}).shadow).toBe(false);
  expect(light.shadow).toBe(true);
  expect(props['ambientLight']).toBeDefined();
});

test('basemap attribution is reduced to safe plain text', () => {
  expect(
    makeAttributionPlainText(
      '<a href="x" onclick="evil()">&copy; Maps &amp; Co</a><script>x</script>'
    )
  ).toBe('© Maps & Cox');
  const attributions = getActiveBasemapAttributions({
    styleType: 'external-a',
    mapStyles: {
      'external-a': {style: {sources: {one: {attribution: '<b>A</b>'}, two: {attribution: 'A'}}}}
    }
  });
  expect(attributions).toEqual(['A']);
  expect(getActiveBasemapAttributions({styleType: 'compute-background', mapStyles: {}})).toEqual(
    []
  );
  expect(makeExternalStyleId('https://a/b.json')).toBe(makeExternalStyleId('https://a/b.json'));
  expect(makeExternalStyleId('https://a/b.json')).not.toBe(makeExternalStyleId('https://a/c.json'));
});

test('tile3d diagnostics summarize transforms, tilesets and resource counters', () => {
  expect(describeModelMatrix(null)).toBeNull();
  expect(describeModelMatrix(IDENTITY)?.isIdentity).toBe(true);
  const moved = describeModelMatrix([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 5, 6, 7, 1]);
  expect(moved).toMatchObject({allFinite: true, isIdentity: false, scaleDeterminant: 8});
  expect(moved?.translation).toEqual([5, 6, 7]);
  expect(describeModelMatrix([NaN, ...IDENTITY.slice(1)])?.allFinite).toBe(false);
  class ScenegraphLayer {}
  const sublayer = Object.assign(new ScenegraphLayer(), {
    id: 'tile3d-1-scenegraph',
    props: {pickable: true, modelMatrix: IDENTITY},
    state: null
  });
  const unrelated = {id: 'points', props: {}, constructor: {name: 'ScatterplotLayer'}};
  const diagnostics = getTile3DDiagnostics({
    layerManager: {getLayers: () => [sublayer, unrelated]}
  });
  expect(diagnostics.map(layer => layer.id)).toEqual(['tile3d-1-scenegraph']);
  expect(diagnostics[0].pickable).toBe(true);
  const counts = getDeviceResourceCounts({
    statsManager: {getStats: () => ({get: (name: string) => ({count: name.length})})}
  });
  expect(counts['Buffers']).toBe('Buffers Active'.length);
});

test('shadow pass uniforms point toward casting lights only', async () => {
  const {getShadowPassUniforms} = await import(
    '../../../examples/deck/kepler-compute/src/effects-wgsl-shadow-pass'
  );
  const uniforms = getShadowPassUniforms(
    [
      {shadow: false, direction: [0, 0, -1]},
      {shadow: true, direction: [0, 0, -2]}
    ],
    undefined,
    {width: 100, height: 50},
    {nearPlane: 1, farPlane: 10}
  );
  expect(uniforms.lightDirection0).toEqual([-0, -0, 1, 1]);
  expect(uniforms.lightDirection1[3]).toBe(0);
  expect(uniforms.viewportSize).toEqual([100, 50]);
});
