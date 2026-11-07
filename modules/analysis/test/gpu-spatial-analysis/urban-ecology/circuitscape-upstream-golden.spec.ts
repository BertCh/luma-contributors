// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addSparseResistanceConnectivityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/circuitscape-upstream-golden.json';

function publish(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<'float32' | 'float32x4' | 'uint32' | 'uint32x4'>
) {
  const output = fixture.output(name, source.format, source.length);
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'circuitscape-golden-readback',
      invocationCount: source.length * (source.format.endsWith('x4') ? 4 : 1),
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}
for (const pair of reference.pairs)
  it(`sparse resistance matches Circuitscape upstream golden pair ${pair.source}-${pair.sink}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new RecipeTestFixture(device, `circuitscape-${pair.source}-${pair.sink}`);
    try {
      const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
        edges: fixture.input(
          'edges',
          Uint32Array.from(reference.edges.flatMap(row => row.slice(0, 2))),
          'uint32x2',
          reference.edges.length
        ),
        conductances: fixture.input(
          'conductances',
          Float32Array.from(reference.edges.map(row => row[2])),
          'float32',
          reference.edges.length
        ),
        vertexCount: reference.vertexCount,
        source: pair.source,
        sink: pair.sink,
        maximumIterations: 32,
        tolerance: 1e-5
      });
      const potentials = publish(fixture, 'potentials', result.outputs.potentials);
      const currents = publish(fixture, 'currents', result.outputs.currents);
      const diagnostics = publish(fixture, 'diagnostics', result.outputs.diagnostics);
      const status = publish(fixture, 'status', result.status);
      fixture.run();
      const flags = await fixture.readUint32(status, 4);
      expect(flags[0]).toBe(1);
      expect(flags[1]).toBe(32);
      expect(flags[2]).toBe(1);
      expect(flags[3]).toBe(0);
      const actualVoltages = await fixture.readFloat32(potentials, reference.vertexCount);
      const actualCurrents = await fixture.readFloat32(currents, reference.edges.length);
      const actualDiagnostics = await fixture.readFloat32(diagnostics, 4);
      const origin = pair.voltages.find(row => row[0] === pair.sink)![1];
      for (const [vertex, value] of pair.voltages)
        expect(Math.abs(actualVoltages[vertex] - (value - origin))).toBeLessThan(1e-4);
      for (let edge = 0; edge < reference.edges.length; edge++) {
        const [first, second] = reference.edges[edge];
        const expected =
          pair.absoluteCurrents.find(row => row[0] === first && row[1] === second)?.[2] ?? 0;
        expect(Math.abs(Math.abs(actualCurrents[edge]) - expected)).toBeLessThan(1e-4);
      }
      const resistance = pair.voltages.find(row => row[0] === pair.source)![1] - origin;
      expect(Math.abs(actualDiagnostics[0] - resistance)).toBeLessThan(1e-4);
      expect(actualDiagnostics[1]).toBeLessThan(1e-4);
      expect(Math.abs(actualDiagnostics[2] - resistance)).toBeLessThan(1e-4);
    } finally {
      fixture.destroy();
    }
  });
