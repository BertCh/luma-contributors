// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  FRAGMENT_SHADER,
  SOURCE,
  VERTEX_SHADER,
  flowStreak
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/flow-particle-layer-shaders';

describe('flow particle layer shader module', () => {
  const uniformNames = Object.keys(flowStreak.uniformTypes);

  test('WGSL struct, GLSL blocks, uniform types and defaults declare the same fields in order', () => {
    const wgsl = flowStreak.source.match(/struct FlowStreakUniforms \{([^}]*)\}/)![1];
    const wgslNames = [...wgsl.matchAll(/(\w+): vec4<f32>/g)].map(match => match[1]);
    expect(wgslNames).toEqual(uniformNames);
    for (const glsl of [flowStreak.vs, flowStreak.fs]) {
      const block = glsl.match(/uniform flowStreakUniforms \{([^}]*)\}/)![1];
      expect([...block.matchAll(/vec4 (\w+);/g)].map(match => match[1])).toEqual(uniformNames);
    }
    expect(Object.keys(flowStreak.defaultUniforms)).toEqual(uniformNames);
  });

  test('textures are declared consistently for WGSL, GLSL and the binding layout', () => {
    const textures = ['flowState', 'flowPreviousState', 'flowVelocity', 'flowAltitude'];
    expect(flowStreak.bindingLayout.map(binding => binding.name)).toEqual([
      'flowStreak',
      ...textures
    ]);
    textures.forEach((name, index) => {
      expect(flowStreak.source).toContain(`@binding(${index + 1}) var ${name}: texture_2d<f32>`);
      expect(flowStreak.vs).toContain(`sampler2D ${name}`);
    });
  });

  test('both shader languages tint streaks from the speed band and lift them by the altitude', () => {
    expect(SOURCE).toContain('output.tint');
    expect(SOURCE).toContain('input.tint');
    expect(VERTEX_SHADER).toContain('tint =');
    expect(FRAGMENT_SHADER).toContain('in vec4 tint');
    for (const source of [SOURCE, VERTEX_SHADER]) {
      expect(source).toContain('sampleAltitude(position)');
      expect(source).toContain('sampleSpeed(particle.xy)');
    }
  });
});
