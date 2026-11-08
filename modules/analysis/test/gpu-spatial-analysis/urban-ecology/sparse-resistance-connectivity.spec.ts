// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addResistanceConnectivityRecipe,
  addSparseResistanceConnectivityRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import circuits from '../../oracles/fixtures/circuitscape-upstream-golden.json';

type Format = 'float32' | 'uint32' | 'float32x4' | 'uint32x4';

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

function publish(fixture: RecipeTestFixture, name: string, source: GraphDataView<Format>) {
  const output = fixture.output(name, source.format, source.length);
  const components = source.format.endsWith('x4') ? 4 : 1;
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'sparse-resistance-connectivity-readback',
      invocationCount: source.length * components,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}

function publishRecipe(
  fixture: RecipeTestFixture,
  prefix: string,
  result: ReturnType<typeof addSparseResistanceConnectivityRecipe>
) {
  return {
    potentials: publish(fixture, `${prefix}-potentials-copy`, result.outputs.potentials),
    currents: publish(fixture, `${prefix}-currents-copy`, result.outputs.currents),
    vertexCurrents: publish(
      fixture,
      `${prefix}-vertex-currents-copy`,
      result.outputs.vertexCurrents
    ),
    diagnostics: publish(fixture, `${prefix}-diagnostics-copy`, result.outputs.diagnostics),
    status: publish(fixture, `${prefix}-status-copy`, result.status)
  };
}

function createLCG(seed: number) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

it('matches the Circuitscape 7-vertex golden for all pairs', async () => {
  const fixture = await createFixture('sparse-resistance-circuitscape');
  try {
    const edgeValues = new Uint32Array(
      circuits.edges.flatMap(([first, second]) => [first, second])
    );
    const conductanceValues = Float32Array.from(circuits.edges.map(edge => edge[2]!));
    const edges = fixture.input('edges', edgeValues, 'uint32x2', circuits.edges.length);
    const conductances = fixture.input(
      'conductances',
      conductanceValues,
      'float32',
      circuits.edges.length
    );
    const outputs = circuits.pairs.map(pair =>
      publishRecipe(
        fixture,
        `pair-${pair.source}-${pair.sink}`,
        addSparseResistanceConnectivityRecipe(fixture.graph, {
          id: `circuitscape-${pair.source}-${pair.sink}`,
          edges,
          conductances,
          vertexCount: circuits.vertexCount,
          source: pair.source,
          sink: pair.sink,
          tolerance: 1e-6,
          maximumIterations: 64
        })
      )
    );
    fixture.run();
    const maxima = {voltage: 0, current: 0, nodeCurrent: 0, resistance: 0, energy: 0};
    for (let pairIndex = 0; pairIndex < circuits.pairs.length; pairIndex++) {
      const pair = circuits.pairs[pairIndex]!;
      const output = outputs[pairIndex]!;
      const status = await fixture.readUint32(output.status, 4);
      expect(status[0], 'converged').toBe(1);
      expect(status[2], 'reachable').toBe(1);
      const potentials = await fixture.readFloat32(output.potentials, circuits.vertexCount);
      for (const [vertex, voltage] of pair.voltages) {
        const error = Math.abs(potentials[vertex!]! - voltage!);
        maxima.voltage = Math.max(maxima.voltage, error);
        expect(error).toBeLessThanOrEqual(1e-5 * (1 + Math.abs(voltage!)));
      }
      const currents = await fixture.readFloat32(output.currents, circuits.edges.length);
      const listed = new Map(
        pair.absoluteCurrents.map(([first, second, absolute]) => [
          `${Math.min(first!, second!)}-${Math.max(first!, second!)}`,
          absolute!
        ])
      );
      circuits.edges.forEach(([first, second], edge) => {
        const expected =
          listed.get(`${Math.min(first!, second!)}-${Math.max(first!, second!)}`) ?? 0;
        const error = Math.abs(Math.abs(currents[edge]!) - expected);
        maxima.current = Math.max(maxima.current, error);
        expect(error).toBeLessThanOrEqual(1e-5);
      });
      const diagnostics = await fixture.readFloat32(output.diagnostics, 4);
      const sourceVoltage = pair.voltages.find(([vertex]) => vertex === pair.source)![1]!;
      maxima.resistance = Math.max(
        maxima.resistance,
        Math.abs(diagnostics[0]! - sourceVoltage) / (1 + Math.abs(sourceVoltage))
      );
      expect(Math.abs(diagnostics[0]! - sourceVoltage)).toBeLessThanOrEqual(
        1e-5 * (1 + Math.abs(sourceVoltage))
      );
      maxima.energy = Math.max(maxima.energy, Math.abs(diagnostics[2]! - diagnostics[0]!));
      expect(Math.abs(diagnostics[2]! - diagnostics[0]!)).toBeLessThanOrEqual(1e-4);
      const vertexCurrents = await fixture.readFloat32(output.vertexCurrents, circuits.vertexCount);
      for (const vertex of [pair.source, pair.sink]) {
        maxima.nodeCurrent = Math.max(maxima.nodeCurrent, Math.abs(vertexCurrents[vertex]! - 1));
        expect(Math.abs(vertexCurrents[vertex]! - 1)).toBeLessThanOrEqual(1e-5);
      }
      // Circuitscape node current is (sum |branch| + |injection|) / 2 around each vertex.
      for (let vertex = 0; vertex < circuits.vertexCount; vertex++) {
        let total = 0;
        circuits.edges.forEach(([first, second], edge) => {
          if (first === vertex || second === vertex) total += Math.abs(currents[edge]!);
        });
        const injection = vertex === pair.source || vertex === pair.sink ? 1 : 0;
        const expected = (total + injection) / 2;
        maxima.nodeCurrent = Math.max(
          maxima.nodeCurrent,
          Math.abs(vertexCurrents[vertex]! - expected)
        );
        expect(Math.abs(vertexCurrents[vertex]! - expected)).toBeLessThanOrEqual(1e-5);
      }
    }
    expect(maxima.voltage).toBeLessThanOrEqual(1e-5);
  } finally {
    fixture.destroy();
  }
});

it('agrees with the small recipe on 200 vertices with a disconnected component', async () => {
  const fixture = await createFixture('sparse-resistance-small-agreement');
  try {
    const next = createLCG(7);
    const edgeList: number[] = [];
    const conductanceList: number[] = [];
    const addEdge = (first: number, second: number) => {
      edgeList.push(first, second);
      conductanceList.push(0.25 + 3.75 * next());
    };
    for (let vertex = 0; vertex < 190; vertex++) addEdge(vertex, (vertex + 1) % 190);
    for (let chord = 0; chord < 400; chord++) {
      const first = Math.floor(next() * 190);
      let second = Math.floor(next() * 190);
      if (second === first) second = (second + 1) % 190;
      addEdge(first, second);
    }
    for (let vertex = 190; vertex < 199; vertex++) addEdge(vertex, vertex + 1);
    const edgeCount = conductanceList.length;
    const edges = fixture.input('edges', Uint32Array.from(edgeList), 'uint32x2', edgeCount);
    const conductances = fixture.input(
      'conductances',
      Float32Array.from(conductanceList),
      'float32',
      edgeCount
    );
    const common = {
      edges,
      conductances,
      vertexCount: 200,
      source: 3,
      sink: 101,
      tolerance: 1e-5,
      maximumIterations: 4096
    };
    const sparse = publishRecipe(
      fixture,
      'sparse',
      addSparseResistanceConnectivityRecipe(fixture.graph, {id: 'sparse-200', ...common})
    );
    const small = addResistanceConnectivityRecipe(fixture.graph, {id: 'small-200', ...common});
    const smallPotentials = publish(fixture, 'small-potentials-copy', small.outputs.potentials);
    const smallCurrents = publish(fixture, 'small-currents-copy', small.outputs.currents);
    const smallDiagnostics = publish(fixture, 'small-diagnostics-copy', small.outputs.diagnostics);
    const smallStatus = publish(fixture, 'small-status-copy', small.status);
    fixture.run();
    const sparseStatus = await fixture.readUint32(sparse.status, 4);
    const smallStatusValues = await fixture.readUint32(smallStatus, 4);
    expect(sparseStatus[0]).toBe(1);
    expect(smallStatusValues[0]).toBe(1);
    expect(sparseStatus[2]).toBe(smallStatusValues[2]);
    const sparsePotentials = await fixture.readFloat32(sparse.potentials, 200);
    const smallPotentialValues = await fixture.readFloat32(smallPotentials, 200);
    for (let vertex = 0; vertex < 200; vertex++) {
      if (vertex >= 190) {
        // Masking: the floating path component is NaN in both recipes.
        expect(sparsePotentials[vertex]).toBeNaN();
        expect(smallPotentialValues[vertex]).toBeNaN();
        continue;
      }
      const error = Math.abs(sparsePotentials[vertex]! - smallPotentialValues[vertex]!);
      expect(error).toBeLessThanOrEqual(1e-4 * (1 + Math.abs(smallPotentialValues[vertex]!)));
    }
    const sparseCurrents = await fixture.readFloat32(sparse.currents, edgeCount);
    const smallCurrentValues = await fixture.readFloat32(smallCurrents, edgeCount);
    for (let edge = 0; edge < 590; edge++) {
      expect(Math.abs(sparseCurrents[edge]! - smallCurrentValues[edge]!)).toBeLessThanOrEqual(1e-4);
    }
    const sparseDiagnostics = await fixture.readFloat32(sparse.diagnostics, 4);
    const smallDiagnosticValues = await fixture.readFloat32(smallDiagnostics, 4);
    const resistanceError =
      Math.abs(sparseDiagnostics[0]! - smallDiagnosticValues[0]!) /
      (1 + Math.abs(smallDiagnosticValues[0]!));
    expect(resistanceError).toBeLessThanOrEqual(1e-4);
    const vertexCurrents = await fixture.readFloat32(sparse.vertexCurrents, 200);
    for (let vertex = 190; vertex < 200; vertex++) expect(vertexCurrents[vertex]).toBeNaN();
    expect(vertexCurrents[3]).toBeGreaterThan(0.99);
  } finally {
    fixture.destroy();
  }
});

it.each([
  {conductance: 0, invalid: 0},
  {conductance: -1, invalid: 1}
])('rejects disconnected and invalid conductance $conductance', async ({conductance, invalid}) => {
  const fixture = await createFixture('sparse-resistance-invalid');
  try {
    const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
      edges: fixture.input('edges', new Uint32Array([0, 1]), 'uint32x2', 1),
      conductances: fixture.input('conductance', new Float32Array([conductance]), 'float32', 1),
      vertexCount: 2,
      source: 0,
      sink: 1,
      maximumIterations: 8,
      connectivityIterations: 8
    });
    const output = publishRecipe(fixture, 'invalid', result);
    fixture.run();
    expect(await fixture.readUint32(output.status, 4)).toEqual([0, 0, 0, invalid]);
    expect((await fixture.readFloat32(output.diagnostics, 4))[0]).toBeNaN();
  } finally {
    fixture.destroy();
  }
});

it('reports iteration exhaustion without converging', async () => {
  const fixture = await createFixture('sparse-resistance-exhausted');
  try {
    const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
      edges: fixture.input('edges', new Uint32Array([0, 1, 1, 2, 2, 3]), 'uint32x2', 3),
      conductances: fixture.input('conductance', new Float32Array([1, 1, 1]), 'float32', 3),
      vertexCount: 4,
      source: 0,
      sink: 3,
      maximumIterations: 1,
      connectivityIterations: 8
    });
    const output = publishRecipe(fixture, 'exhausted', result);
    fixture.run();
    expect(await fixture.readUint32(output.status, 4)).toEqual([0, 1, 1, 0]);
    const diagnostics = await fixture.readFloat32(output.diagnostics, 4);
    expect(diagnostics[0]).toBeNaN();
    expect(diagnostics[1]).toBeGreaterThan(0.1);
  } finally {
    fixture.destroy();
  }
});

it('reports undetermined reachability when label propagation has too few rounds', async () => {
  const fixture = await createFixture('sparse-resistance-undetermined');
  try {
    const vertexCount = 64;
    const edgeValues: number[] = [];
    // Path with the smallest label at the far end so one round cannot connect source and sink.
    for (let vertex = 0; vertex + 1 < vertexCount; vertex++) edgeValues.push(vertex, vertex + 1);
    const edgeCount = vertexCount - 1;
    const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
      edges: fixture.input('edges', Uint32Array.from(edgeValues), 'uint32x2', edgeCount),
      conductances: fixture.input(
        'conductance',
        new Float32Array(edgeCount).fill(1),
        'float32',
        edgeCount
      ),
      vertexCount,
      source: 63,
      sink: 40,
      maximumIterations: 256,
      connectivityIterations: 1
    });
    const output = publishRecipe(fixture, 'undetermined', result);
    fixture.run();
    const status = await fixture.readUint32(output.status, 4);
    // Labels differ and did not converge; the solve still runs and converges.
    expect(status[2]).toBe(2);
    expect(status[0]).toBe(1);
  } finally {
    fixture.destroy();
  }
});

it('does not certify convergence when labels are undetermined and the sink is disconnected', async () => {
  const fixture = await createFixture('sparse-resistance-uncertified');
  try {
    // A 4,096-vertex path holds the source; the sink sits in a separate two-vertex component.
    const pathLength = 4096;
    const vertexCount = pathLength + 2;
    const edgeValues: number[] = [];
    for (let vertex = 0; vertex + 1 < pathLength; vertex++) edgeValues.push(vertex, vertex + 1);
    edgeValues.push(pathLength, pathLength + 1);
    const edgeCount = edgeValues.length / 2;
    const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
      edges: fixture.input('edges', Uint32Array.from(edgeValues), 'uint32x2', edgeCount),
      conductances: fixture.input(
        'conductance',
        new Float32Array(edgeCount).fill(1),
        'float32',
        edgeCount
      ),
      vertexCount,
      source: 2000,
      sink: pathLength + 1,
      maximumIterations: 256,
      tolerance: 1e-2,
      connectivityIterations: 1
    });
    const output = publishRecipe(fixture, 'uncertified', result);
    fixture.run();
    const status = await fixture.readUint32(output.status, 4);
    expect(status[0], 'converged').toBe(0);
    expect(status[1], 'iterations').toBeGreaterThan(0);
    expect(status[2], 'reachable').toBe(2);
    expect(status[3], 'invalid').toBe(0);
    const diagnostics = await fixture.readFloat32(output.diagnostics, 4);
    expect(diagnostics[0]).toBeNaN();
  } finally {
    fixture.destroy();
  }
});
