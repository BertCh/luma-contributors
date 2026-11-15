// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';
import {GPU_GLOBE_WGSL} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-globe';

const readLayer = (name: string) =>
  readFileSync(
    new URL(
      `../../../../examples/deck/kepler-compute/gpu-layers/layers/${name}.ts`,
      import.meta.url
    ),
    'utf8'
  );

describe('GPU GlobeView support', () => {
  it('uses the shared project module for horizon tests and great-circle interpolation', () => {
    expect(GPU_GLOBE_WGSL).toContain('PROJECTION_MODE_GLOBE_METER_OFFSETS');
    expect(GPU_GLOBE_WGSL).toContain('if (!gpuGlobeIsActive()) { return false; }');
    expect(GPU_GLOBE_WGSL).toContain(
      'dot(commonPosition, project.cameraPosition - commonPosition)'
    );
    expect(GPU_GLOBE_WGSL).toContain('abs(angularDistance - PI) < 0.001');
    expect(GPU_GLOBE_WGSL).toContain('.yxz');
  });

  it('culls scatterplot points and splits arcs at the antimeridian or horizon', () => {
    expect(readLayer('gpu-scatterplot-layer')).toContain('gpuGlobeIsHidden(position)');
    const arc = readLayer('gpu-arc-layer');
    expect(arc).toContain('if (gpuGlobeUsesGreatCircle())');
    expect(arc).toContain('abs(startGlobe.x - endGlobe.x) > 180.0');
    expect(arc).toContain('if (globeHidden)');
  });

  it('clips paths and polygons per fragment at the horizon with the interpolated common position', () => {
    expect(GPU_GLOBE_WGSL).toContain('fn gpuGlobeIsCommonHidden');
    for (const name of ['gpu-path-layer', 'gpu-solid-polygon-layer']) {
      const source = readLayer(name);
      expect(source).toContain('globePosition: vec3<f32>');
      expect(source).toContain('output.globePosition = geometry.position.xyz;');
      expect(source).toContain('gpuGlobeIsCommonHidden(input.globePosition)');
    }
  });

  it('orients arrowheads to the great-circle tangent and culls them past the horizon', () => {
    const head = readLayer('gpu-arrow-head-layer');
    expect(head).toContain('arrowStyle.mode == 1u && gpuGlobeUsesGreatCircle()');
    expect(head).toContain('gpuGlobeInterpolate(sourcePositions, targetPositions');
    expect(head).toContain('if (globeHidden)');
  });
});
