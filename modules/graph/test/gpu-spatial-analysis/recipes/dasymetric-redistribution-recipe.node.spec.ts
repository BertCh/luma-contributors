// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {addDasymetricRedistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `dasymetric-test-${serial++}`, format, length);
  const props = {
    grid: {
      width: 6,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number],
      pixelInterpretation: 'area' as const
    },
    source: {
      kind: 'coverage' as const,
      coverage: {
        zoneIds: view('uint32', 8),
        cellIds: view('uint32', 8),
        fractions: view('float32', 8),
        zoneValidity: view('uint32', 3),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    },
    sourceTotals: view('float32', 3),
    targetZones: view('uint32', 6),
    targetZoneCount: 3,
    ancillary: {
      id: 'ancillary',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: view('float32', 6)}
    },
    mask: view('uint32', 6),
    pairCapacity: 8
  };
  return {device, graph, view, props};
}

it.each([
  'zone',
  'overlap'
] as const)('dasymetric %s composition keeps provenance and core binding limits', denominator => {
  const {device, graph, props} = createContext();
  try {
    const result = addDasymetricRedistributionRecipe(graph, {...props, denominator});
    expect(result.intermediates.coverage).toBe(props.source.coverage);
    expect(result.intermediates.targetZones).toBe(props.targetZones);
    expect(result.intermediates.sourceTotals).toBe(props.sourceTotals);
    expect(result.intermediates.ancillary).toBe(props.ancillary);
    expect(result.intermediates.mask).toBe(props.mask);
    expect(result.status.stages[0].status).toBe(props.source.coverage.status);
    expect(result.outputs.sourceAllocatedTotals.length).toBe(3);
    expect(result.outputs.targetTotals.length).toBe(3);
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(10);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    device.destroy();
  }
});

it.each([
  'output alias',
  'target length',
  'source validity length',
  'mask length',
  'shape bound'
])('dasymetric rejects %s', boundary => {
  const {device, graph, view, props} = createContext();
  try {
    expect(() =>
      addDasymetricRedistributionRecipe(graph, {
        ...props,
        ...(boundary === 'output alias' ? {outputs: {targetTotals: props.sourceTotals}} : {}),
        ...(boundary === 'target length' ? {targetZones: view('uint32', 5)} : {}),
        ...(boundary === 'source validity length'
          ? {
              source: {
                ...props.source,
                coverage: {...props.source.coverage, zoneValidity: view('uint32', 2)}
              }
            }
          : {}),
        ...(boundary === 'mask length' ? {mask: view('uint32', 5)} : {}),
        ...(boundary === 'shape bound' ? {targetZoneCount: 524289} : {})
      })
    ).toThrow();
  } finally {
    device.destroy();
  }
});
