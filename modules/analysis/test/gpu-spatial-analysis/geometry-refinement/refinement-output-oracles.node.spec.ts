// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {it} from 'vitest';
import {
  ORACLE_CASES,
  expectOracleRows,
  getVectorOracleCase
} from './refinement-output-oracles-fixtures';
import {executeCPUOracleOperation} from './refinement-output-oracle-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const createCase of ORACLE_CASES) {
  const fixture = createCase();
  it(`independent CPU output oracle: ${fixture.name}`, async () => {
    const result = await executeCPUOracleOperation(fixture.cpu(), fixture.inputs);
    expectOracleRows(result, fixture.expected);
  });
}

it('independent vector oracle respects link-masked inactive coordinates in poisoned contributor buffers', () => {
  const fixture = getVectorOracleCase();
  const operation = fixture.gpu();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const views: Record<string, GraphDataView> = {};
  for (const [name, input] of Object.entries(fixture.inputs)) {
    const components = input.format === 'float32x2' ? 2 : 1;
    views[name] = createTransientView(graph, name, input.format, input.values.length / components);
    memory.write(views[name], input.values);
  }
  for (const [name, output] of Object.entries(operation.outputs)) {
    views[name] = createTransientView(graph, name, output.format, output.length);
    const ArrayType = output.format.startsWith('float') ? Float32Array : Uint32Array;
    memory.write(
      views[name],
      new ArrayType((output.length * views[name].rowByteLength) / 4).fill(913)
    );
  }
  try {
    const contributor = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
    );
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    const actual = Object.fromEntries(
      Object.entries(operation.outputs).map(([name, output]) => [
        name,
        output.format.startsWith('float')
          ? memory.readFloat32(views[name])
          : memory.readUint32(views[name])
      ])
    );
    // Active slot coordinates and every link/status/provenance word still have independent
    // rectangle expectations. Snapshot writes do not initialize unused coordinate storage.
    expectOracleRows(actual, fixture.expected);
  } finally {
    device.destroy();
  }
});
