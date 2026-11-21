// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  addBuildingTessellationRecipe,
  addBuildingTessellationReferenceRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {addLocalBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-local';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('RCP-107 named routes and legacy dispatch preserve result types and explicit reference precedence', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const base = {
      segments: createTransientView(graph, 'segments', 'float32x4', 4),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 4),
      buildingCount: 1,
      enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
      sampleCapacity: 8,
      sampleSpacing: 1,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 16, nodedSegments: 32, vertices: 32, rings: 4}
    };
    const reference = addBuildingTessellationReferenceRecipe(graph, {
      ...base,
      id: 'named-reference'
    });
    const local = addLocalBuildingTessellationRecipe(graph, {
      ...base,
      id: 'named-local',
      candidateRadius: 10
    });
    const explicitReference = addBuildingTessellationRecipe(graph, {
      ...base,
      id: 'explicit-reference',
      mode: 'reference',
      candidateRadius: 10
    });
    const explicitLocal = addBuildingTessellationRecipe(graph, {
      ...base,
      id: 'explicit-local',
      mode: 'local',
      candidateRadius: 10
    });
    const inferredLocal = addBuildingTessellationRecipe(graph, {
      ...base,
      id: 'inferred-local',
      candidateRadius: 10
    });
    const inferredReference = addBuildingTessellationRecipe(graph, {
      ...base,
      id: 'inferred-reference'
    });
    expectTypeOf(reference.provenance.mode).toEqualTypeOf<'reference'>();
    expectTypeOf(local.provenance.mode).toEqualTypeOf<'local'>();
    expectTypeOf(explicitReference).toEqualTypeOf(reference);
    expectTypeOf(inferredReference).toEqualTypeOf(reference);
    expectTypeOf(explicitLocal).toEqualTypeOf(local);
    expectTypeOf(inferredLocal).toEqualTypeOf(local);
    for (const result of [reference, explicitReference, inferredReference]) {
      expect(result.provenance.mode).toBe('reference');
      expect(Array.isArray(result.outputs)).toBe(true);
      expect(result.outputs).toHaveLength(1);
    }
    for (const result of [local, explicitLocal, inferredLocal]) {
      expect(result.provenance.mode).toBe('local');
      expect(result.outputs.cells.format).toBe('uint32x4');
      expect(result.provenance.representation).toBe('sample-cell-union');
    }
    const nodes = (graph as unknown as {nodes: {id: string}[]}).nodes;
    // Overlay expands to primitive nodes; pin one route-specific node on each path.
    expect(nodes.some(node => node.id === 'explicit-reference-building-0-pack')).toBe(true);
    expect(nodes.some(node => node.id.startsWith('explicit-local-local-cells'))).toBe(true);
  } finally {
    device.destroy();
  }
});
