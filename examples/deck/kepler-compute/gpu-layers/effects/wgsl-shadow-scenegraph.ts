// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScenegraphLayer} from '@deck.gl/mesh-layers';
import type {Model} from '@luma.gl/engine';
import {
  applyWGSLShadow,
  ensureWGSLShadowInputs,
  replaceWGSLShadowAnchor,
  wgslShadow
} from './wgsl-shadow';

let isInstalled = false;

/** glTF pins depth24plus, but deck picking uses depth16unorm. Let Model infer the pass format. */
export function prepareScenegraphPassAttachments(
  models: readonly Pick<Model, 'parameters' | 'setParameters'>[]
): void {
  for (const model of models) {
    if (model.parameters.depthFormat !== undefined) {
      const {depthFormat: _depthFormat, ...parameters} = model.parameters;
      model.setParameters(parameters);
    }
  }
}

const SIMPLE_PHONG_ANCHOR = 'fragColor = pbr_filterColor(fragColor);';
/**
 * Kepler's simple-phong override for PBR layers (3D tiles): ambient plus Lambertian diffuse from
 * screen-space derivative normals, applied when the effect sets \`useSimplePhong\` and PBR is unlit.
 */
const SIMPLE_PHONG_WGSL = /* wgsl */ `  if (shadow.useSimplePhong > 0.5) {
    let simpleNormal = normalize(cross(dpdx(inputs.pbrPosition), dpdy(inputs.pbrPosition)));
    var simpleLit = lighting.ambientColor;
    for (var lightIndex = 0; lightIndex < lighting.directionalLightCount; lightIndex++) {
      let simpleLight = lighting_getDirectionalLight(lightIndex);
      simpleLit += simpleLight.color * max(dot(simpleNormal, -normalize(simpleLight.direction)), 0.0);
    }
    fragColor = vec4<f32>(fragColor.rgb * simpleLit, fragColor.a);
  }`;

/** Adds WGSL shadow map casting and receiving to a Deck 9.4 scenegraph WGSL shader source. */
export function applyScenegraphWGSLShadow(source: string): string {
  const shaded = applyWGSLShadow(source, {
    varyingAnchor: '@location(5) pickingColor: vec3<f32>,',
    firstLocation: 6,
    positionAssignment:
      'outputs.position = project_common_position_to_clipspace(geometry.position);',
    commonPosition: 'geometry.position',
    clipPosition: 'project_common_position_to_clipspace(geometry.position)',
    finalReturn: 'return deckgl_premultiplied_alpha(fragColor);',
    finalColorExpression: 'fragColor',
    outputName: 'outputs',
    inputName: 'inputs'
  });
  return replaceWGSLShadowAnchor(
    shaded,
    SIMPLE_PHONG_ANCHOR,
    `${SIMPLE_PHONG_ANCHOR}\n${SIMPLE_PHONG_WGSL}`
  );
}

/**
 * Makes every `ScenegraphLayer` (glTF 3D tiles, trip markers, scenegraph layers) cast and receive
 * WebGPU shadow maps by wrapping `getShaders` and `draw`. Idempotent; WebGL devices are untouched.
 */
export function installScenegraphShadowPatch(): void {
  if (isInstalled) return;
  isInstalled = true;
  const originalGetShaders = ScenegraphLayer.prototype.getShaders;
  ScenegraphLayer.prototype.getShaders = function getShaders(this: ScenegraphLayer) {
    const shaders = originalGetShaders.call(this) as {
      modules?: {name: string}[];
      source?: string;
    };
    if (
      this.context.device.type === 'webgpu' &&
      shaders.modules &&
      typeof shaders.source === 'string' &&
      !shaders.modules.some(module => module.name === wgslShadow.name)
    ) {
      shaders.modules = [...shaders.modules, wgslShadow];
      shaders.source = applyScenegraphWGSLShadow(shaders.source);
    }
    return shaders;
  } as typeof ScenegraphLayer.prototype.getShaders;
  const originalDraw = ScenegraphLayer.prototype.draw;
  ScenegraphLayer.prototype.draw = function draw(
    this: ScenegraphLayer,
    parameters: Parameters<ScenegraphLayer['draw']>[0]
  ) {
    if (this.context.device.type === 'webgpu') {
      prepareScenegraphPassAttachments(this.getModels());
      ensureWGSLShadowInputs(this.getModels() as never, this.context.device);
    }
    return originalDraw.call(this, parameters);
  } as typeof ScenegraphLayer.prototype.draw;
}
