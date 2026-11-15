// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getShaderAssembler, picking} from '@deck.gl/core';
import {PathLayer} from '@deck.gl/layers';
import {PathStyleExtension} from '@deck.gl/extensions';
import {
  mergeShaderPluginModules,
  resolveShaderPlugins,
  type WGSLShaderAssembler
} from '@luma.gl/shaders';
import {Call, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {expect, test} from 'vitest';
import {
  RESIDENT_EDITOR_PICKING_PLUGIN,
  ResidentEditorPickingExtension
} from '../../../examples/deck/kepler-compute/src/resident-editor-picking-extension';

function assembleEditorStroke(adapted: boolean) {
  const assembler = getShaderAssembler('wgsl') as WGSLShaderAssembler;
  const layer = {
    id: 'kepler_editor_layer-geojson-polygons-stroke',
    props: {
      extensions: [
        new PathStyleExtension({dash: true}),
        ...(adapted ? [new ResidentEditorPickingExtension()] : [])
      ]
    },
    state: {pathTesselator: {}},
    context: {
      device: {type: 'webgpu'},
      viewport: {projectionMode: 1},
      defaultShaderModules: [],
      defaultShaderPlugins: [],
      shaderAssembler: assembler
    }
  };
  const shaders = PathLayer.prototype.getShaders.call(layer as unknown as PathLayer);
  const plugins = resolveShaderPlugins(shaders.plugins, 'wgsl');
  return assembler.assembleWGSLShader({
    ...shaders,
    modules: mergeShaderPluginModules(shaders.modules, plugins.modules),
    defines: {...shaders.defines, ...plugins.defines},
    pluginInjectionTargets: plugins.injectionTargets,
    pluginInjections: plugins.injections,
    pluginVertexInputs: plugins.vertexInputs,
    pluginInterStageVariables: plugins.interStageVariables,
    vertexEntryPoint: 'vertexMain',
    fragmentEntryPoint: 'fragmentMain',
    platformInfo: {
      type: 'webgpu',
      shaderLanguage: 'wgsl',
      shaderLanguageVersion: 300,
      gpu: 'apple',
      features: new Set()
    }
  });
}

test('published PathLayer WGSL has no picking-color output; editor adaptation restores it without changing uniform bindings', () => {
  const original = assembleEditorStroke(false);
  expect(original.source).not.toContain('picking_filterPickingColor');
  expect(original.source).not.toContain('residentEditorPickingColor');
  const adapted = assembleEditorStroke(true);
  const reflection = new WgslReflect(adapted.source);
  expect(reflection.entry.vertex.map(entry => entry.name)).toContain('vertexMain');
  expect(reflection.entry.fragment.map(entry => entry.name)).toContain('fragmentMain');
  expect(adapted.source).toContain('residentEditorPickingColor = geometry.pickingColor');
  expect(adapted.source).toContain('*color = vec4<f32>(residentEditorPickingColor, 1.0)');
  expect(adapted.source).toContain('if (!picking_isColorValid(residentEditorPickingColor))');
  expect(adapted.source).toContain('if (picking.isAttribute > 0.5)');
  expect(adapted.source).toContain('@interpolate(flat) residentEditorPickingColor');
  expect(adapted.bindingAssignments).toEqual(original.bindingAssignments);
});

test('editor picking extension leaves WebGL and unrelated primitive layers on their published path', () => {
  const extension = new ResidentEditorPickingExtension();
  for (const layer of [
    {
      id: 'kepler_editor_layer-geojson-polygons-stroke',
      state: {pathTesselator: {}},
      context: {device: {type: 'webgl'}}
    },
    {id: 'ordinary-path', state: {pathTesselator: {}}, context: {device: {type: 'webgpu'}}},
    {
      id: 'kepler_editor_layer-geojson-polygons-fill',
      state: {},
      context: {device: {type: 'webgpu'}}
    }
  ]) {
    expect(extension.getShaders.call(layer as never)).toBeNull();
  }
});

function runStrokeColorFilter({
  color = [0.2, 0.4, 0.6, 0.25],
  highlightColor = [1, 1, 0, 0.5],
  highlightedObjectColor = [8, 0, 0],
  useByteColors = 1,
  isHighlightActive = 1,
  isActive = 0,
  isAttribute = 0
} = {}): number[] {
  const injections = RESIDENT_EDITOR_PICKING_PLUGIN.injections;
  if (!Array.isArray(injections)) throw new Error('Expected structured editor shader injections');
  const fragment = injections.find(
    entry => entry.stage === 'fragment' && entry.target === 'DECKGL_FILTER_COLOR'
  )?.wgsl;
  if (!fragment) throw new Error('Expected an editor fragment color injection');
  const injection = fragment.replaceAll('(*color)', 'color').replaceAll('*color =', 'color =');
  // Execute the actual shipped fragment injection and public Deck picking helpers on the CPU.
  // Private storage replaces the GPU uniform binding. The interpreter cannot pass function
  // pointers, so only the pointer syntax is lowered to a local color variable, inline.
  const helpers = picking.source!.replace('@group(0) @binding(auto) var<uniform>', 'var<private>');
  const source = `${helpers}
var<private> residentEditorPickingColor: vec3<f32>;
var<private> residentEditorPickingDepth: f32;
var<private> probeResult: vec4<f32>;
fn probe() {
  picking.isActive = ${isActive}.0;
  picking.isAttribute = ${isAttribute}.0;
  picking.isHighlightActive = ${isHighlightActive}.0;
  picking.useByteColors = ${useByteColors}.0;
  picking.highlightedObjectColor = vec3<f32>(${highlightedObjectColor.join(', ')});
  picking.highlightColor = vec4<f32>(${highlightColor.join(', ')});
  residentEditorPickingColor = vec3<f32>(8.0 / 255.0, 0.0, 0.0);
  residentEditorPickingDepth = 0.25;
  var color = vec4<f32>(${color.join(', ')});
  ${injection}
  probeResult = color;
}`;
  const interpreter = new WgslExec(new WgslParser().parse(source));
  interpreter.execute();
  interpreter.execStatement(new Call('probe', []), interpreter.context);
  return interpreter.getVariableValue('probeResult') as number[];
}

for (const useByteColors of [0, 1]) {
  test(`editor hover blends the matching stroke with straight-alpha highlight (byte colors=${useByteColors})`, () => {
    const result = runStrokeColorFilter({
      useByteColors,
      highlightedObjectColor: [useByteColors ? 8 : 8 / 255, 0, 0]
    });
    // Half-opaque yellow over a quarter-opaque stroke: alpha=.625, highlight weight=.8.
    for (const [index, expected] of [0.84, 0.88, 0.12, 0.625].entries()) {
      expect(result[index]).toBeCloseTo(expected, 6);
    }
  });
}

test('editor hover leaves unmatched strokes and inactive highlights unchanged', () => {
  for (const options of [{highlightedObjectColor: [9, 0, 0]}, {isHighlightActive: 0}]) {
    const result = runStrokeColorFilter(options);
    for (const [index, expected] of [0.2, 0.4, 0.6, 0.25].entries()) {
      expect(result[index]).toBeCloseTo(expected, 6);
    }
  }
});

test('editor hover keeps zero-alpha highlight blending finite', () => {
  expect(runStrokeColorFilter({color: [0, 0, 0, 0], highlightColor: [1, 1, 0, 0]})).toEqual([
    0, 0, 0, 0
  ]);
});

test('editor picking identity and depth override a matching hover highlight', () => {
  const identity = runStrokeColorFilter({isActive: 1});
  expect(identity[0]).toBeCloseTo(8 / 255, 6);
  expect(identity.slice(1)).toEqual([0, 0, 1]);
  expect(runStrokeColorFilter({isActive: 1, isAttribute: 1})).toEqual([0.25, 0, 0, 1]);
});
