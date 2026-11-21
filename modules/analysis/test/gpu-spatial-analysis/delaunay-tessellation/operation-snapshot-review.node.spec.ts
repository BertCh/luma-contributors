// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUDelaunayTessellationOperation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {
  GPUDelaunayTessellationOperation,
  type DelaunayTessellationOperationProps
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(
  status: DelaunayTessellationOperationProps['output']['status']
): DelaunayTessellationOperationProps {
  return {positions: 'positions', capacity: 1, output: {triangles: 'triangles', status}};
}

for (const field of ['complete', 'unresolvedCount'] as const) {
  it(`B2 CPU rejects structurally wider unsupported ${field} before advertising outputs`, () => {
    const status = {
      count: 'count',
      requiredCount: 'required',
      overflow: 'overflow',
      [field]: 'unwritten'
    };
    expect(() => new CPUDelaunayTessellationOperation(createProps(status))).toThrow(
      new RegExp(`output.status.${field}.*unsupported`)
    );
  });
}

it('B2 CPU preserves a supported inherited convergence destination and writes it', async () => {
  let reads = 0;
  const status = Object.setPrototypeOf(
    {count: 'count', requiredCount: 'required', overflow: 'overflow'},
    {
      get converged() {
        reads++;
        return 'solver';
      }
    }
  );
  const operation = new CPUDelaunayTessellationOperation(createProps(status));
  const positions = Float32Array.of(0, 0, 2, 0, 0, 2);
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 3,
        chunks: [{values: positions, length: 3}],
        pack: () => positions
      }
    }
  });
  // The three sites form the unique CCW triangle and the solve is complete.
  expect(Array.from(result['triangles'] as Uint32Array)).toEqual([0, 1, 2]);
  expect(Array.from(result['solver'] as Uint32Array)).toEqual([1]);
  expect(reads).toBe(1);
});

for (const Operation of [CPUDelaunayTessellationOperation, GPUDelaunayTessellationOperation]) {
  it(`B2 constructor preserves inherited named ports for ${Operation.name}`, () => {
    let reads = 0;
    const status = Object.setPrototypeOf(
      {count: 'count', requiredCount: 'required', overflow: 'overflow'},
      {
        get converged() {
          reads++;
          return 'solver';
        }
      }
    );
    const operation = new Operation(createProps(status));
    expect(Object.keys(operation.outputs)).toContain(
      Operation === CPUDelaunayTessellationOperation ? 'solver' : 'converged'
    );
    if (operation instanceof GPUDelaunayTessellationOperation)
      expect(operation.outputs['converged']).toMatchObject({value: 'solver'});
    expect(reads).toBe(1);
    for (const field of ['complete', 'unresolvedCount'] as const) {
      const wideStatus = {
        count: 'count',
        requiredCount: 'required',
        overflow: 'overflow',
        [field]: 'unwritten'
      };
      expect(() => new Operation(createProps(wideStatus))).toThrow(
        new RegExp(`output.status.${field}.*unsupported`)
      );
    }
  });
}

it('S1 the shared GPU operation snapshot retains the original frame through deferred lowering', () => {
  const frame = {
    id: 'frame-at-construction',
    kind: 'local-azimuthal-equidistant' as const,
    origin: [0, 0] as [number, number]
  };
  const caller = {
    ...createProps({count: 'count', requiredCount: 'required', overflow: 'overflow'}),
    spatialContext: {
      coordinateSpace: 'planar' as const,
      metric: 'native' as const,
      units: 'native' as const,
      frame
    }
  };
  const operation = new GPUDelaunayTessellationOperation(caller);
  frame.id = '';
  frame.origin[0] = 99;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const outputs = Object.fromEntries(
    Object.entries(operation.outputs).map(([port, descriptor]) => [
      port,
      createTransientView(graph, port, descriptor.format, descriptor.length)
    ])
  );
  try {
    const contributor = operation.createContributor(
      port => (port === 'positions' ? positions : outputs[port]) as never
    );
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
