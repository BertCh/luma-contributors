// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';
import {applyGPUBrushingToShader} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-brushing';
import {GPU_ARROW_HEAD_SHADER} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-arrow-head-layer';
import {
  GPU_HIGHLIGHT_RANGE_WGSL,
  GPU_HIGHLIGHT_WGSL,
  setGPUHighlightRangeProps
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-highlight';

const HIGHLIGHTED_LAYER_FILES = [
  'gpu-line-layer',
  'gpu-arc-layer',
  'gpu-path-layer',
  'gpu-solid-polygon-layer',
  'gpu-scatterplot-layer',
  'gpu-mesh-column-layer',
  'gpu-arrow-head-layer'
];

const RANGE_LAYER_FILES = ['gpu-path-layer', 'gpu-solid-polygon-layer', 'gpu-scatterplot-layer'];

describe('GPU hover highlight', () => {
  it('compares picking ids in the fragment stage using Deck picking uniforms', () => {
    expect(GPU_HIGHLIGHT_WGSL).toContain('picking.isHighlightActive');
    expect(GPU_HIGHLIGHT_WGSL).toContain('picking.highlightedObjectColor');
    expect(GPU_HIGHLIGHT_WGSL).toContain('picking.highlightColor');
  });

  it('is composed into every highlighted layer shader after the picking pass exit', () => {
    for (const name of HIGHLIGHTED_LAYER_FILES) {
      const source = readFileSync(
        new URL(
          `../../../../examples/deck/kepler-compute/gpu-layers/layers/${name}.ts`,
          import.meta.url
        ),
        'utf8'
      );
      const ranged = RANGE_LAYER_FILES.includes(name);
      expect(source, name).toContain(
        ranged ? '${GPU_HIGHLIGHT_RANGE_WGSL}' : '${GPU_HIGHLIGHT_WGSL}'
      );
      expect(source, name).toContain(ranged ? 'gpuHighlightRows(' : 'gpuHighlight(');
      if (ranged) expect(source, name).toContain('gpuHighlightRange');
    }
  });

  it('keeps arrowhead brushing on target endpoint semantics', () => {
    expect(GPU_ARROW_HEAD_SHADER).toContain('gpuHighlight(input.color, input.pickingColor)');
    const brushed = applyGPUBrushingToShader(
      GPU_ARROW_HEAD_SHADER,
      true,
      'brushing_arePointsInRange(sourcePositions, targetPositions)'
    );
    expect(brushed).toContain(
      'output.selected = select(0u, selectionMask, brushing_arePointsInRange(sourcePositions, targetPositions));'
    );
    expect(applyGPUBrushingToShader(GPU_ARROW_HEAD_SHADER, false, 'true')).toBe(
      GPU_ARROW_HEAD_SHADER
    );
  });

  it('defines every mesh shape with six vertices so one draw covers all shapes', () => {
    for (const shape of ['chevron', 'triangle', 'stealth'])
      expect(GPU_ARROW_HEAD_SHADER).toContain(`var ${shape} = array<vec2<f32>, 6>`);
  });

  it('highlights a whole feature by primitive row range and falls back to the picked id', () => {
    expect(GPU_HIGHLIGHT_RANGE_WGSL).toContain('row < highlightRange.firstRow');
    expect(GPU_HIGHLIGHT_RANGE_WGSL).toContain('return gpuHighlight(color, pickingColor)');
    const updates: unknown[] = [];
    const model = {
      userData: {} as Record<string, unknown>,
      shaderInputs: {setProps: (value: unknown) => updates.push(value)}
    };
    setGPUHighlightRangeProps([model as never], {highlightedPrimitiveRange: [4, 7]});
    setGPUHighlightRangeProps([model as never], {});
    expect(updates).toEqual([
      {highlightRange: {enabled: 1, firstRow: 4, lastRow: 7}},
      {highlightRange: {enabled: 0, firstRow: 0, lastRow: 0}}
    ]);
  });
});
