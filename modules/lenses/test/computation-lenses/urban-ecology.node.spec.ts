// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '@luma-contributors/analysis/testing';
import {buildUrbanEcologyScenario} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/urban-ecology-examples';
import {URBAN_ECOLOGY_OPERATORS} from '../../src/gpu-spatial-analysis/urban-morphology/urban-ecology-operators';
import type {S2Uploads} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {addBuildingTessellationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {runExample as runBuildingReference} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/building-tessellation-recipe.example';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import {createViewContext} from '../../src/computation-views/registry/view-registry';
import {buildResultViewModel} from '../../src/computation-views/views/result-view';
import {getDeclaredVariableKind} from '../../src/computation-views/views/value-strip';
import {
  createSpatialElementPackData,
  createSpatialPackData
} from '../../src/gpu-spatial-analysis/computation-pack/pack';
import {makeScanRecord} from '../computation-views/fixtures/scan-record';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/urban-ecology.lens';

it('RCP-107: reference example and explicit, legacy and untagged local dispatch keep their route', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    segments: createTransientView(graph, 'route-segments', 'float32x4', 8),
    segmentBuildingIds: createTransientView(graph, 'route-owners', 'uint32', 8),
    enclosure: createTransientView(graph, 'route-enclosure', 'float32x2', 4),
    buildingCount: 2,
    sampleSpacing: 1,
    sampleCapacity: 8,
    vertexTolerance: 1 / 1024,
    capacity: {intersections: 64, nodedSegments: 32, vertices: 16, rings: 4},
    candidateRadius: 100,
    maximumSamplesPerBuilding: 4,
    candidateCapacity: 8,
    cellVertexCapacity: 16
  };
  try {
    // Extra fields on a variable used to make this reference-typed example dispatch locally.
    const example = await runBuildingReference(graph, {...props, id: 'reference-example'});
    expect(Array.isArray(example.outputs)).toBe(true);
    expect(example.outputs.map(output => output.geometry.kind)).toEqual(['polygons', 'polygons']);
    const explicit = addBuildingTessellationRecipe(graph, {
      ...props,
      id: 'explicit-reference',
      mode: 'reference'
    });
    expect(Array.isArray(explicit.outputs)).toBe(true);
    const {candidateRadius, ...legacyProps} = props;
    expect(candidateRadius).toBe(100);
    const legacy = addBuildingTessellationRecipe(graph, {...legacyProps, id: 'legacy-reference'});
    expect(Array.isArray(legacy.outputs)).toBe(true);
    const local = addBuildingTessellationRecipe(graph, {...props, id: 'untagged-local'});
    expect(Array.isArray(local.outputs)).toBe(false);
    expect(local.outputs.cells.format).toBe('uint32x4');
    expect(local.outputs.buildingStatus.length).toBe(2);
  } finally {
    device.destroy();
  }
});

it('RCP-107: both named tessellation entries have route-specific pack evidence', () => {
  const pack = createSpatialPackData();
  for (const name of [
    'addBuildingTessellationReferenceRecipe',
    'addLocalBuildingTessellationRecipe'
  ]) {
    expect(URBAN_ECOLOGY_OPERATORS[name]?.formula).toBeDefined();
    expect(URBAN_ECOLOGY_OPERATORS[name]?.variables).toBeDefined();
    expect(pack.operators?.[name]).toMatchObject({scope: 'pairwise', reduction: 'select'});
    expect(Object.keys(pack.operators?.[name].variableKinds ?? {}).length).toBeGreaterThan(0);
    expect(pack.formulas?.[name]).toMatchObject({formula: 'none', reason: expect.any(String)});
  }
});

it('S1: local variable kinds resolve by their produced capture names', () => {
  const context = createViewContext(makeScanRecord(), {
    packs: [createSpatialElementPackData('addLocalBuildingTessellationRecipe')]
  });
  expect(getDeclaredVariableKind(context, 'output-positions')).toEqual({
    level: 'interval',
    extent: 'intensive'
  });
  for (const name of ['output-cells', 'output-buildingStatus', 'status-sampling'])
    expect(getDeclaredVariableKind(context, name)).toEqual({level: 'nominal'});
});

it.each(
  Object.keys(URBAN_ECOLOGY_OPERATORS)
)('%s builds its real capturable graph within core binding limits', async element => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const owned: {destroy(): void}[] = [];
  const uploads: S2Uploads = {
    own(resource) {
      owned.push(resource);
      return resource;
    },
    view(id, data, format, length) {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
      return graph.createDataView(handle, {format, length: length ?? data.length / components});
    }
  };
  try {
    const captures = await buildUrbanEcologyScenario(element, graph, uploads);
    expect(Object.keys(captures).length).toBeGreaterThan(0);
    if (element === 'addLocalBuildingTessellationRecipe') {
      expect(captures['output-cells'].view.format).toBe('uint32x4');
      expect(captures['output-buildingStatus'].view.length).toBe(1);
      expect(captures['status-candidateOverflow'].view.length).toBe(1);
      expect(captures['output-0-geometry-positions']).toBeUndefined();
      const scenario = getLensScenario('gpu-spatial-analysis.local-building-tessellation-recipe')!;
      const record = makeScanRecord();
      record.captures = {
        ...record.captures!,
        snapshots: Object.entries(captures).map(([name, spec]) => {
          const components = Number(spec.view.format.match(/x([234])$/)?.[1] ?? 1);
          return {
            id: `gpu-spatial-analysis-local-building-tessellation-recipe-snapshots-${name}-0`,
            name,
            role: spec.role,
            components,
            elementType: spec.view.format.startsWith('uint') ? ('u32' as const) : ('f32' as const),
            values: spec.view.format.startsWith('uint')
              ? new Uint32Array(spec.view.length * components)
              : new Float32Array(spec.view.length * components),
            provenance: 'gpu-readback' as const
          };
        })
      };
      const context = createViewContext(record, {
        options: scenario.presentation,
        packs: [createSpatialElementPackData(element)]
      });
      // Actual builder order ends in status-sampling; presentation must explicitly select cells.
      expect(buildResultViewModel(context).output.capture.name).toBe('output-cells');
      expect(record.captures.snapshots.some(capture => capture.name === 'status-sampling')).toBe(
        true
      );
      for (const name of [
        'output-positions',
        'output-cells',
        'output-buildingStatus',
        'status-sampling'
      ])
        expect(getDeclaredVariableKind(context, name), name).toBeDefined();
    } else if (
      element === 'addBuildingTessellationRecipe' ||
      element === 'addBuildingTessellationReferenceRecipe'
    ) {
      expect(captures['output-0-geometry-positions'].view.format).toBe('float32x2');
      expect(captures['status-complete'].view.length).toBe(2);
      expect(captures['output-cells']).toBeUndefined();
    }
    expect(Object.keys(captures).some(name => /^output-\d+$/.test(name))).toBe(false);
    expect(new Set(Object.values(captures).map(capture => capture.view)).size).toBe(
      Object.keys(captures).length
    );
    for (const capture of Object.values(captures)) expect(capture.view.length).toBeGreaterThan(0);
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    for (const resource of owned) resource.destroy();
    device.destroy();
  }
});
