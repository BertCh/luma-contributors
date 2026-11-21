// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getStudyWindowBinding,
  getStudyWindowPackedLayout,
  getStudyWindowWGSL,
  getStudyWindowWGSLOptions,
  getStudyWindowSampleCellWGSL,
  STUDY_WINDOW_BOUNDARY_CELL_BIT
} from '../../src/analysis-core/kernels/study-window-wgsl';
import {getPolygonEdgeSlabNodes} from '../../src/analysis-core/kernels/polygon-edge-slabs';
import {EXACT_GEOMETRY_WGSL} from '../../src/analysis-core/kernels/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../src/analysis-core/kernels/exact-orientation-wgsl';
import {IS_FINITE_BITS_WGSL} from '../../src/analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from './gpu-contributor-test-utils';

it('preserves the coordinator packed word layout and derives sizes from views', () => {
  const layout = getStudyWindowPackedLayout({
    vertexCount: 12,
    slabOffsetCount: 5,
    slabEntryCount: 60,
    sampleCellCapacity: 20
  });
  expect(layout).toEqual({
    summary: 0,
    edges: 8,
    slabParams: 56,
    slabOffsets: 62,
    slabEntries: 67,
    sampleCellCount: 187,
    sampleCells: 188,
    length: 208
  });
  expect(STUDY_WINDOW_BOUNDARY_CELL_BIT).toBe(0x80000000);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {slabs} = getPolygonEdgeSlabNodes(graph, {
    id: 'window-slabs',
    operation: 'StudyWindowTest',
    owner: 'polygon',
    positions: createTransientView(graph, 'positions', 'float32x2', 12),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    entryCapacity: 48
  });
  const views = {
    edges: createTransientView(graph, 'edges', 'float32', 48),
    slabs,
    sampleCells: createTransientView(graph, 'cells', 'uint32', 20),
    sampleGrid: [5, 4] as const
  };
  expect(getStudyWindowWGSLOptions(views, 'quartic')).toEqual({
    layout,
    vertexCount: 12,
    entryCapacity: 48,
    sampleGrid: [5, 4],
    kernel: 'quartic'
  });
  const packed = createTransientView(graph, 'packed', 'uint32', layout.length);
  expect(getStudyWindowBinding({packed})).toEqual({
    name: 'studyWindow',
    view: packed,
    type: 'u32',
    access: 'read'
  });
});

for (const kernel of [undefined, 'gaussian', 'quartic', 'epanechnikov'] as const) {
  it(`study-window WGSL parses, audits and has one exact path (${kernel ?? 'no kernel'})`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const layout = getStudyWindowPackedLayout({
      vertexCount: 12,
      slabOffsetCount: 5,
      slabEntryCount: 60,
      sampleCellCapacity: 20
    });
    const declarations = getStudyWindowWGSL({
      layout,
      vertexCount: 12,
      entryCapacity: 48,
      kernel,
      sampleGrid: [5, 4]
    });
    for (const name of [
      'studyWindowSummaryValue',
      'studyWindowEdge',
      'isStudyWindowUsable',
      'isInsideStudyWindow',
      'getStudyWindowBoundaryDistance',
      'getStudyWindowCircleFraction',
      'getStudyWindowSampleCell'
    ]) {
      expect(declarations).toContain(`fn ${name}(`);
    }
    expect(declarations.match(/exactProbeCrossing\(/g)).toHaveLength(1);
    // All slabbed/walked loops and the boundary fan's full walk reject unused NaN edges.
    for (const match of declarations.matchAll(/let edge = studyWindowEdge\([^;]+;/g)) {
      expect(declarations.slice(match.index! + match[0].length).trimStart()).toMatch(
        /^if \(!studyWindowFinitePoint\(edge.xy\) \|\| !studyWindowFinitePoint\(edge.zw\)\) \{ continue; \}/
      );
    }
    expect(declarations.includes('fn getStudyWindowKernelMass(')).toBe(Boolean(kernel));
    expect(declarations).toContain('(studyWindowOffset + 56u)');
    expect(declarations).toContain('(studyWindowOffset + 62u)');
    expect(declarations).toContain('(studyWindowOffset + 67u)');
    expect(declarations).toContain(getStudyWindowSampleCellWGSL([5, 4], 'studyWindowSummaryValue'));
    expect(declarations.match(/edgeSlabIndex != max/g)).toHaveLength(kernel ? 2 : 1);
    const query = createWGSLKernelNode(graph, {
      id: 'window-query',
      operation: 'StudyWindowTest',
      invocationCount: 1,
      bindings: [
        getStudyWindowBinding({
          packed: createTransientView(graph, 'packed', 'uint32', layout.length)
        }),
        {
          name: 'result',
          view: createTransientView(graph, 'result', 'float32', 1),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `${IS_FINITE_BITS_WGSL}${EXACT_ORIENTATION_WGSL}${EXACT_GEOMETRY_WGSL}${declarations}`,
      body: `let point = vec2f(400.0, 500.0);
        result[resultOffset] = select(0.0, 1.0, isInsideStudyWindow(point)) +
          getStudyWindowBoundaryDistance(point, 1000.0) + getStudyWindowCircleFraction(point, 800.0) +
          getStudyWindowSampleCell(0x80000001u).x ${kernel ? '+ getStudyWindowKernelMass(point, 500.0)' : ''};`
    });
    const sources = getParsedKernelSources([query]);
    const source = sources.get('window-query')!;
    expect(source.match(/\borientSign\(/g)).toHaveLength(2); // declaration + one call
    expect(source.match(/\bexactOrientSigns\(/g)).toHaveLength(3); // declaration + unused segment helper + crossing helper
    const configurations = sources.getConfigurations('window-query');
    expect(configurations.length).toBeGreaterThan(0);
    for (const kernel of configurations) {
      expect(getStorageBindingCount(kernel.source)).toBe(2);
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  });
}

it('CORE-105 the boundary fan sums its angles with the guarded atan2', () => {
  // float32 transcription of the fan's dot product at the right-angle vertex of (0,0), (-1,0),
  // (0,-1): start (-1,0), end (0,-1), each term -1 * 0 = -0.0, and -0.0 + -0.0 = -0.0. The
  // builtin atan2(1, -0.0) is -pi/2 on the test GPU (GUARDED_ATAN2_WGSL), which made the angular
  // baseline -0.25 (clamped to 0) instead of 0.25. The headless case is in study-window-wgsl.spec.ts.
  const dot = Math.fround(Math.fround(-1 * 0) + Math.fround(0 * -1));
  expect(Object.is(dot, -0)).toBe(true);
  const layout = getStudyWindowPackedLayout({
    vertexCount: 3,
    slabOffsetCount: 2,
    slabEntryCount: 3,
    sampleCellCapacity: 1
  });
  const declarations = getStudyWindowWGSL({layout, vertexCount: 3, entryCapacity: 3});
  expect(declarations).toContain('fn studyWindowAtan2(y: f32, x: f32) -> f32 {');
  expect(declarations).toContain('angle += studyWindowAtan2(cross, dot(start, end));');
  expect(declarations).not.toContain('angle += atan2(');
});
