// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuilder} from '../../../../../website/src/gpu/runs/define-run';
import {vi} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

// Load the real website adapter lazily; headless runs its original Computation passes.
export async function createBufferDrawingOracle(device: Device, cpu = false) {
  const {addPolygonDrawing} = await import(
    '../../../../../website/src/gpu/runs/entries/geometry-construction-overlay/shared/polygons'
  );
  const fixture = createOutputOracleFixture(device);
  const {graph, column, output} = fixture;
  type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2';
  function wrap<StorageFormat extends Format>(
    name: string,
    format: StorageFormat,
    view: GraphDataView<StorageFormat>
  ) {
    const entry = fixture.columns.find(entry => entry.view === view)!;
    return {
      id: name,
      format,
      length: view.length,
      view,
      buffer: entry.buffer,
      write: (values: Float32Array | Uint32Array) => entry.buffer.write(values)
    };
  }
  const builder = {
    graph,
    device,
    allocate: <StorageFormat extends Format>(name: string, format: StorageFormat, length: number) =>
      wrap(name, format, output(name, format, length)),
    upload: <StorageFormat extends Format>(
      name: string,
      format: StorageFormat,
      values: Float32Array | Uint32Array
    ) => wrap(name, format, column(name, format, values))
  };
  const polygons = {
    positions: builder.upload(
      'retained-positions',
      'float32x2',
      new Float32Array([0, 0, 4, 0, 4, 4, 0, 4])
    ),
    featureOffsets: builder.upload('retained-features', 'uint32', new Uint32Array([0, 1])),
    polygonOffsets: builder.upload('retained-polygons', 'uint32', new Uint32Array([0, 1])),
    ringOffsets: builder.upload('retained-rings', 'uint32', new Uint32Array([0, 4]))
  };
  const count = builder.upload('retained-count', 'uint32', new Uint32Array([1]));
  const overflow = builder.upload('retained-overflow', 'uint32', new Uint32Array([0]));
  const candidateOverflow = builder.upload('retained-candidate', 'uint32', new Uint32Array([1]));
  const add = vi.spyOn(graph, 'addComputePass');
  const replayed: GPUCommandNode[] = [];
  const kernel = await import(
    '../../../../../website/src/gpu/runs/entries/geometry-construction-overlay/shared/kernel'
  );
  // The shared interpreter supports Kernel dispatch but not the website's Computation wrapper.
  // Replay the REAL scalar ring-count body/bindings through the supported kernel factory,
  // without changing the shader or substituting a CPU result. Headless also executes drawing.
  const replay = cpu
    ? vi.spyOn(kernel, 'addKernelPass').mockImplementation((owner, pass) => {
        if (pass.id !== 'drawing-ring-count-pass') return;
        const node = createWGSLKernelNode(owner.graph, {
          id: pass.id,
          operation: 'RunKernel',
          invocationCount: pass.invocationCount,
          declarations: pass.declarations,
          body: pass.body,
          bindings: pass.bindings.map(binding => ({
            name: binding.name,
            view: binding.buffer.view,
            access: binding.access,
            type: 'u32' as const
          }))
        });
        owner.graph.add(node);
        replayed.push(node);
      })
    : undefined;
  // This small adapter implements exactly the buffer formats used by this drawing, not the
  // unrelated scene/resource lifecycle methods of a full RunBuilder.
  addPolygonDrawing(builder as unknown as RunBuilder, 'drawing', polygons, count, {
    status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
  });
  const nodes = cpu
    ? replayed
    : add.mock.calls.map(([node]) => ({...node, type: 'compute' as const}));
  replay?.mockRestore();
  add.mockRestore();
  return {...fixture, nodes, overflow, candidateOverflow};
}
