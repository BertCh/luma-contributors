// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {
  GLSLShaderAssembler,
  WGSLShaderAssembler,
  resolveShaderPlugins,
  type ShaderPlugin,
  type ResolvedShaderInjection,
  type ShaderInjectionTarget,
  type PlatformInfo
} from '@luma.gl/shaders';
import {WgslReflect} from 'wgsl_reflect';
import {patchKeplerGlobeBackfaceCullSource} from '../../../examples/deck/kepler-compute/kepler-compute-transform';

type LabelShaders = {
  inject: Record<string, string | ResolvedShaderInjection>;
  plugins: ShaderPlugin[];
};

function loadPatchedHelper(): (shaders: LabelShaders) => LabelShaders {
  const source = readFileSync(
    new URL(
      '../../../node_modules/@kepler.gl/deckgl-layers/dist/esm/globe/globe-backface-cull.js',
      import.meta.url
    ),
    'utf8'
  );
  const patched = patchKeplerGlobeBackfaceCullSource(source);
  expect(patchKeplerGlobeBackfaceCullSource(patched)).toBe(patched);
  // Evaluate this small installed helper only; Babel's now-unused spread helpers do not run.
  return new Function(
    patched.replace(/^import .*;\n/gm, '').replace('export function', 'function') +
      '\nreturn injectGlobeBackfaceCull;'
  )() as (shaders: LabelShaders) => LabelShaders;
}

test('installed Kepler globe label helper preserves base hooks and selects GLSL/WGSL through public plugins', () => {
  const inject = loadPatchedHelper();
  const existing = {
    inject: {'vs:DECKGL_FILTER_GL_POSITION': '/* base hook */', 'vs:#main-start': '/* start */'},
    plugins: [{name: 'base-plugin'}]
  };
  const shaders = inject(existing);
  expect(existing.inject['vs:DECKGL_FILTER_GL_POSITION']).toBe('/* base hook */');
  expect(shaders.inject).toEqual({'vs:#main-start': '/* start */'});
  expect(shaders.plugins[0]).toBe(existing.plugins[0]);
  const glsl = resolveShaderPlugins(shaders.plugins, 'glsl').injections[
    'vs:DECKGL_FILTER_GL_POSITION'
  ].at(-1)!.injection;
  const wgsl = resolveShaderPlugins(shaders.plugins, 'wgsl').injections[
    'vs:DECKGL_FILTER_GL_POSITION'
  ].at(-1)!.injection;
  expect(glsl).toContain('vec3 anchorCommon');
  expect(glsl).toContain('radius / camDist + 0.02');
  expect(wgsl).toContain('radius / cameraDistance + 0.02');
  expect(wgsl).toContain('*position = vec4<f32>');
  expect(wgsl).not.toContain('vec3 anchorCommon');
  expect(
    () =>
      new WgslReflect(`
    const PROJECTION_MODE_GLOBE: i32 = 2;
    struct Project {projectionMode: i32, cameraPosition: vec3<f32>};
    struct VertexGeometry {position: vec4<f32>};
    var<private> project: Project;
    fn cull(position: ptr<function, vec4<f32>>, geometry: VertexGeometry) {${wgsl}}
  `)
  ).not.toThrow();
});

const POSITION_TARGET: ShaderInjectionTarget = {
  name: 'DECKGL_FILTER_GL_POSITION',
  stage: 'vertex',
  glsl: {parameters: 'inout vec4 position, VertexGeometry geometry'},
  wgsl: {parameters: 'position: ptr<function, vec4<f32>>, geometry: VertexGeometry'}
};

for (const language of ['glsl', 'wgsl'] as const) {
  for (const order of [undefined, 41, Number.POSITIVE_INFINITY]) {
    test(`${language} final position hook culls after inherited assignment with order ${order ?? 'string'}`, () => {
      const assignment =
        language === 'glsl'
          ? 'position = vec4(0.0, 0.0, 0.0, 1.0);'
          : '*position = vec4<f32>(0.0, 0.0, 0.0, 1.0);';
      const inherited = `/* inherited position */ ${assignment}`;
      const shaders = loadPatchedHelper()({
        inject: {
          'vs:DECKGL_FILTER_GL_POSITION':
            order === undefined ? inherited : {injection: inherited, order}
        },
        // Even an already-final plugin must precede the appended cull, including ties at infinity.
        plugins: [
          {
            name: 'existing-position',
            injections: [
              {
                stage: 'vertex',
                target: 'DECKGL_FILTER_GL_POSITION',
                order: Number.POSITIVE_INFINITY,
                glsl: '/* plugin position */ position = vec4(1.0);',
                wgsl: '/* plugin position */ *position = vec4<f32>(1.0);'
              }
            ]
          }
        ]
      });
      const resolved = resolveShaderPlugins(shaders.plugins, language);
      const platformInfo: PlatformInfo = {
        type: language === 'glsl' ? 'webgl' : 'webgpu',
        shaderLanguage: language,
        shaderLanguageVersion: 300,
        gpu: 'test',
        features: new Set()
      };
      const options = {
        platformInfo,
        inject: shaders.inject,
        pluginInjections: resolved.injections,
        injectionTargets: [POSITION_TARGET],
        modules: [
          {
            name: 'test-geometry',
            vs: `
          const int PROJECTION_MODE_GLOBE = 2;
          struct VertexGeometry {vec4 position;};
          struct Project {int projectionMode; vec3 cameraPosition;};
          Project project;
        `,
            source: `
          const PROJECTION_MODE_GLOBE: i32 = 2;
          struct VertexGeometry {position: vec4<f32>};
          struct Project {projectionMode: i32, cameraPosition: vec3<f32>};
          var<private> project: Project;
        `
          }
        ]
      };
      const assembled =
        language === 'glsl'
          ? new GLSLShaderAssembler().assembleGLSLShaderPair({
              ...options,
              vs: `#version 300 es
            void main() {
              vec4 position = vec4(1.0);
              VertexGeometry geometry;
              geometry.position = position;
              DECKGL_FILTER_GL_POSITION(position, geometry);
              gl_Position = position;
            }`,
              fs: '#version 300 es\nprecision highp float; out vec4 color; void main() {color = vec4(1.0);}'
            }).vs
          : new WGSLShaderAssembler().assembleWGSLShader({
              ...options,
              source: `
          @vertex fn vertexMain() -> @builtin(position) vec4<f32> {
            var position = vec4<f32>(1.0);
            var geometry: VertexGeometry;
            geometry.position = position;
            DECKGL_FILTER_GL_POSITION(&position, geometry);
            return position;
          }
          @fragment fn fragmentMain() -> @location(0) vec4<f32> {return vec4<f32>(1.0);}
        `
            }).source;
      const cullAssignment =
        language === 'glsl'
          ? 'position = vec4(0.0, 0.0, 2.0, 1.0);'
          : '*position = vec4<f32>(0.0, 0.0, 2.0, 1.0);';
      expect(assembled).toContain(inherited);
      expect(assembled).toContain('/* plugin position */');
      expect(assembled).toContain(cullAssignment);
      expect(assembled.indexOf(inherited)).toBeLessThan(assembled.indexOf(cullAssignment));
      expect(assembled.indexOf('/* plugin position */')).toBeLessThan(
        assembled.indexOf(cullAssignment)
      );
      if (order === Number.POSITIVE_INFINITY) {
        expect(assembled.indexOf('/* plugin position */')).toBeLessThan(
          assembled.indexOf(inherited)
        );
      } else {
        expect(assembled.indexOf(inherited)).toBeLessThan(
          assembled.indexOf('/* plugin position */')
        );
      }
      if (language === 'wgsl') expect(() => new WgslReflect(assembled)).not.toThrow();
    });
  }
}
