// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {getSpatialSURWeightValidationNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-sur-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {
  auditWGSLModule,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

function createFixture(rows = 64) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `f-reg-${serial++}`, format, length);
  const sur = {
    id: 'sur-improvement',
    equationCount: 2,
    predictorCount: 1,
    rowCount: rows,
    predictors: view('float32', 2 * rows),
    responses: view('float32', 2 * rows),
    output: {
      coefficients: view('float32', 4),
      covariance: view('float32', 16),
      residualCovariance: view('float32', 4),
      residuals: view('float32', 2 * rows),
      status: view('uint32', 1)
    }
  };
  const panel = {
    id: 'panel-improvement',
    periodCount: 4,
    predictorCount: 1,
    predictors: view('float32', 16),
    response: view('float32', 16),
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    output: {
      coefficients: view('float32', 4),
      covariance: view('float32', 9),
      summary: view('float32', 6),
      status: view('uint32', 1),
      fitted: view('float32', 16),
      residuals: view('float32', 16)
    }
  };
  return {device, graph, view, sur, panel};
}

it('SUR accepts a million rows with bounded shader and pipeline counts as tile counts grow', () => {
  const counts: number[] = [];
  for (const rows of [64, 65537, 1048576]) {
    const fixture = createFixture(rows);
    try {
      const nodes = new GPUSeeminglyUnrelatedRegression(fixture.sur).getCommandNodes(fixture.graph);
      counts.push(new Set(getGPUCommandNodePipelineKeys(nodes)).size);
      expect(nodes).toHaveLength(9);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
    } finally {
      fixture.device.destroy();
    }
  }
  expect(counts).toEqual([9, 9, 9]);
});

it('SUR rejects row counts beyond exact float32 counting at construction', () => {
  const fixture = createFixture(2 ** 24 + 1);
  try {
    expect(() => new GPUSeeminglyUnrelatedRegression(fixture.sur)).toThrow(
      /^sur-improvement rowCount.*16777216/
    );
  } finally {
    fixture.device.destroy();
  }
});

for (const failure of ['aligned-input', 'scratch'] as const) {
  it(`SUR preflights ${failure} before declaring any resource, view or node`, () => {
    const fixture = createFixture();
    try {
      const predictors =
        failure === 'aligned-input'
          ? fixture.graph.createDataView(fixture.view('float32', 129).buffer, {
              format: 'float32',
              byteOffset: 4,
              length: 128
            })
          : fixture.sur.predictors;
      const contributor = new GPUSeeminglyUnrelatedRegression({...fixture.sur, predictors});
      // One row with a full 16-coefficient system isolates scratch from the caller bindings.
      const actualContributor =
        failure === 'scratch'
          ? new GPUSeeminglyUnrelatedRegression({
              ...fixture.sur,
              equationCount: 2,
              predictorCount: 7,
              rowCount: 1,
              predictors: fixture.view('float32', 14),
              responses: fixture.view('float32', 2),
              output: {
                coefficients: fixture.view('float32', 16),
                covariance: fixture.view('float32', 256),
                residualCovariance: fixture.view('float32', 4),
                residuals: fixture.view('float32', 2),
                status: fixture.view('uint32', 1)
              }
            })
          : contributor;
      Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {
        value: failure === 'aligned-input' ? 512 : 1024
      });
      const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
      const createView = vi.spyOn(fixture.graph, 'createDataView');
      const addNode = vi.spyOn(fixture.graph, 'add');
      expect(() => actualContributor.getCommandNodes(fixture.graph)).toThrow(
        failure === 'aligned-input' ? /predictors.*516 bytes/ : /statistics scratch/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(addNode).not.toHaveBeenCalled();
    } finally {
      fixture.device.destroy();
    }
  });
}

it('SUR rejects logical separation whose co-bound aligned writable ranges overlap', () => {
  const fixture = createFixture();
  try {
    const shared = fixture.view('float32', 20);
    const covariance = fixture.graph.createDataView(shared.buffer, {
      format: 'float32',
      byteOffset: 16,
      length: 16
    });
    const coefficients = fixture.graph.createDataView(shared.buffer, {
      format: 'float32',
      length: 4
    });
    expect(
      () =>
        new GPUSeeminglyUnrelatedRegression({
          ...fixture.sur,
          output: {...fixture.sur.output, coefficients, covariance}
        })
    ).toThrow(/sur-improvement.*aligned binding.*overlap/);
  } finally {
    fixture.device.destroy();
  }
});

it('SUR residuals use the centered solution and a runtime rounding barrier', () => {
  const fixture = createFixture();
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPUSeeminglyUnrelatedRegression(fixture.sur).getCommandNodes(fixture.graph)
    );
    const residuals = kernels.find(kernel => kernel.id.endsWith('-residuals'))!;
    expect(residuals.source).toContain('CENTERED_SOLUTION_START');
    expect(residuals.source).toContain('getRoundedDifference(');
    expect(residuals.source).not.toContain('coefficients[');
    expect(residuals.source).toContain('status[statusOffset] == 0u');
  } finally {
    fixture.device.destroy();
  }
});

it('spatial SUR accepts rows beyond 65,536 and preflights nested equation packing', () => {
  for (const rows of [64, 65537]) {
    const fixture = createFixture(rows);
    try {
      const props = {
        ...fixture.sur,
        spatial: {
          weights: {
            offsets: fixture.view('uint32', rows + 1),
            neighbors: fixture.view('uint32', rows),
            weights: fixture.view('float32', rows)
          },
          output: {parameters: fixture.view('float32', 2), status: fixture.view('uint32', 2)}
        }
      };
      const contributor = new GPUSeeminglyUnrelatedRegression(props);
      if (rows === 64) {
        Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {value: 512});
        const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
        const createView = vi.spyOn(fixture.graph, 'createDataView');
        expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(
          /SUR unfiltered equation.*768 bytes/
        );
        expect(allocate).not.toHaveBeenCalled();
        expect(createView).not.toHaveBeenCalled();
      }
    } finally {
      fixture.device.destroy();
    }
  }
});

it('spatial SUR marks upstream weight overflow invalid even when its CSR values are normalized', () => {
  const fixture = createFixture(4);
  try {
    const weights = {
      offsets: fixture.view('uint32', 5),
      neighbors: fixture.view('uint32', 4),
      weights: fixture.view('float32', 4),
      status: {overflow: fixture.view('uint32', 1)}
    };
    const output = fixture.view('float32', 1);
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, Uint32Array.of(0, 1, 2, 3, 4));
    memory.write(weights.neighbors, Uint32Array.of(1, 0, 3, 2));
    memory.write(weights.weights, Float32Array.of(1, 1, 1, 1));
    memory.write(weights.status.overflow, Uint32Array.of(1));
    executeCommandNodesOnCPU(
      getSpatialSURWeightValidationNodes(fixture.graph, {
        id: 'spatial-overflow',
        weights,
        rowCount: 4,
        output
      }),
      memory
    );
    expect(Array.from(memory.readFloat32(output))).toEqual([1]);
  } finally {
    fixture.device.destroy();
  }
});
