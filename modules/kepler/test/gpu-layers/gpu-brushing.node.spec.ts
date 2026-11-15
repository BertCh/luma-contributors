// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  applyGPUBrushingToShader,
  applyGPUBrushingVisibility,
  gpuBrushing
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-brushing';

describe('GPU hover brushing shader attachment', () => {
  const source = 'output.selected = selectionMask;';

  it('leaves unbrushed shaders byte-identical', () => {
    expect(applyGPUBrushingToShader(source, false, 'brushing_isPointInRange(position.xy)')).toBe(
      source
    );
  });

  it('folds visibility into the existing selection output', () => {
    expect(applyGPUBrushingToShader(source, true, 'brushing_isPointInRange(position.xy)')).toBe(
      'output.selected = select(0u, selectionMask, brushing_isPointInRange(position.xy));'
    );
    expect(() => applyGPUBrushingToShader('', true, 'true')).toThrow();
  });

  it('matches native either-endpoint and meters-in-common-space semantics', () => {
    expect(gpuBrushing.source).toContain('brushing_isPointInRange(sourcePosition) ||');
    expect(gpuBrushing.source).toContain('project.commonUnitsPerMeter.xy');
  });

  it('attaches per-vertex visibility only when enabled', () => {
    const source = '/*BRUSHING_VISIBILITY*/ output.brushed = brushingVisibility(position.xy);';
    expect(applyGPUBrushingVisibility(source, false)).toContain('return 1.0;');
    expect(applyGPUBrushingVisibility(source, false)).not.toContain('brushing_isPointInRange');
    expect(applyGPUBrushingVisibility(source, true)).toContain('brushing_isPointInRange(position)');
  });
});
