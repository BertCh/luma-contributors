// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  addSerialSparseResistanceComponents,
  addSparseResistanceComponents
} from '../../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-components';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Shape =
  | 'chain'
  | 'reverse-chain'
  | 'star'
  | 'disconnected'
  | 'duplicates'
  | 'invalid'
  | 'shuffled';

function findExpected(vertexCount: number, edges: Uint32Array, weights: Float32Array) {
  const neighbors: number[][] = Array.from({length: vertexCount}, () => []);
  let invalid = 0;
  for (let edge = 0; edge < weights.length; edge++) {
    const first = edges[2 * edge];
    const second = edges[2 * edge + 1];
    const weight = weights[edge];
    if (first >= vertexCount || second >= vertexCount || !Number.isFinite(weight) || weight < 0) {
      invalid++;
    } else if (weight > 0) {
      neighbors[first].push(second);
      neighbors[second].push(first);
    }
  }
  const labels = new Array<number>(vertexCount).fill(-1);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (labels[vertex] !== -1) continue;
    labels[vertex] = vertex;
    const pending = [vertex];
    while (pending.length) {
      const current = pending.pop()!;
      for (const neighbor of neighbors[current]) {
        if (labels[neighbor] === -1) {
          labels[neighbor] = vertex;
          pending.push(neighbor);
        }
      }
    }
  }
  return {labels, invalid};
}

async function checkComponents(
  shape: Shape,
  maximumUnionAttempts: number,
  maximumRootSteps: number,
  featureLevel: 'core' | 'max' = 'max',
  boundedDispatch = false,
  interiorTerminals = false,
  useSerialComponents = false
) {
  const device = await getWebGPUTestDevice(featureLevel);
  requireTestDevice(device);
  const descriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  if (boundedDispatch) {
    const limits = device.limits;
    Object.defineProperty(device, 'limits', {
      configurable: true,
      value: new Proxy(limits, {
        get: (target, property) =>
          property === 'maxComputeWorkgroupsPerDimension'
            ? 2
            : Reflect.get(target, property, target)
      })
    });
  }
  const fixture = makeSpatialFixture(device);
  try {
    const vertexCount = useSerialComponents ? 256 : 1031;
    const sourceVertex = interiorTerminals ? 12 : 0;
    const sinkVertex = interiorTerminals ? 20 : vertexCount - 1;
    let pairs: [number, number][] = Array.from({length: vertexCount - 1}, (_, vertex) =>
      shape === 'star' ? [vertexCount - 1, vertex] : [vertex, vertex + 1]
    );
    if (shape === 'reverse-chain') pairs.reverse();
    if (shape === 'disconnected') pairs = pairs.filter(([first]) => first % 11 !== 0);
    if (shape === 'duplicates') pairs.push(...pairs.slice(0, 40), [0, 0], [17, 17]);
    if (shape === 'invalid') pairs.push([0xffffffff, 0], [vertexCount, vertexCount - 1]);
    if (shape === 'shuffled') {
      let state = 498231;
      for (let index = pairs.length - 1; index > 0; index--) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        const selected = state % (index + 1);
        [pairs[index], pairs[selected]] = [pairs[selected], pairs[index]];
      }
    }
    const edgeValues = Uint32Array.from(pairs.flat());
    const weightValues = new Float32Array(pairs.length).fill(1);
    if (shape === 'invalid') weightValues.set([0, -1, NaN, Infinity], 3);
    const byteOffset = 16;
    const edges = fixture.view('uint32', edgeValues, byteOffset);
    const weights = fixture.view('float32', weightValues, byteOffset);
    const labels = fixture.view('uint32', new Uint32Array(vertexCount), byteOffset);
    const status = fixture.view('uint32', new Uint32Array(4), byteOffset);
    const control = fixture.view('uint32', new Uint32Array(4), byteOffset);
    const componentProps = {
      id: 'components',
      edges: fixture.graph.createDataView(edges.view.buffer, {
        format: 'uint32x2',
        length: pairs.length,
        byteOffset
      }),
      conductances: weights.view,
      components: labels.view,
      status: fixture.graph.createDataView(status.view.buffer, {
        format: 'uint32x4',
        length: 1,
        byteOffset
      }),
      vertexCount,
      source: sourceVertex,
      sink: sinkVertex,
      maximumIterations: 8,
      maximumUnionAttempts,
      maximumRootSteps
    };
    if (useSerialComponents) addSerialSparseResistanceComponents(fixture.graph, componentProps);
    const result = useSerialComponents
      ? undefined
      : addSparseResistanceComponents(fixture.graph, componentProps);
    if (result)
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: 'publish-control',
          operation: 'component-test-readback',
          invocationCount: 4,
          bindings: [
            {name: 'source', view: result.control, type: 'u32', access: 'read'},
            {name: 'output', view: control.view, type: 'u32', access: 'read_write'}
          ],
          body: 'output[outputOffset + index] = source[sourceOffset + index];'
        })
      );
    for (const mode of ['original', 'zero', 'invalid', 'changed', 'restored'] as const) {
      const nextEdges = edgeValues.slice();
      const nextWeights = weightValues.slice();
      if (mode === 'zero') nextWeights.fill(0);
      if (mode === 'invalid') nextWeights[0] = -1;
      if (mode === 'changed') {
        for (let edge = 0; edge < nextWeights.length; edge++) {
          nextEdges[edge * 2] = (nextEdges[edge * 2] + 7) % vertexCount;
          nextEdges[edge * 2 + 1] = (nextEdges[edge * 2 + 1] + 7) % vertexCount;
          nextWeights[edge] = edge % 5 ? 1 : 0;
        }
      }
      edges.buffer.write(nextEdges, byteOffset);
      weights.buffer.write(nextWeights, byteOffset);
      await fixture.run();
      const expected = findExpected(vertexCount, nextEdges, nextWeights);
      expect(await readIntegers(labels.buffer, byteOffset)).toEqual(expected.labels);
      expect(await readIntegers(status.buffer, byteOffset)).toEqual([
        0,
        8,
        Number(
          expected.invalid === 0 && expected.labels[sourceVertex] === expected.labels[sinkVertex]
        ),
        expected.invalid
      ]);
      const actualControl = await readIntegers(control.buffer, byteOffset);
      expect(actualControl.every(value => value === 0 || value === 1)).toBe(true);
      expect(actualControl[3]).toBe(Number(actualControl.slice(0, 3).some(Boolean)));
      if (maximumRootSteps === 0) expect(actualControl[3]).toBe(1);
      expect(await labels.buffer.readAsync(0, byteOffset)).toEqual(new Uint8Array(byteOffset));
    }
  } finally {
    if (boundedDispatch) {
      if (descriptor) Object.defineProperty(device, 'limits', descriptor);
      else Reflect.deleteProperty(device, 'limits');
    }
    fixture.destroy();
  }
}

for (const shape of [
  'chain',
  'reverse-chain',
  'star',
  'disconnected',
  'duplicates',
  'invalid',
  'shuffled'
] as const) {
  it(`serial resistance components preserve ${shape} on small inputs`, async () => {
    await checkComponents(shape, 64, 64, 'max', false, false, true);
  });
  for (const [unionBudget, rootBudget] of [
    [64, 64],
    [0, 64],
    [64, 0],
    [1, 1]
  ]) {
    it(`bounded resistance components preserve ${shape}, union budget ${unionBudget}, root budget ${rootBudget}`, async () => {
      await checkComponents(shape, unionBudget, rootBudget);
    });
  }
}
it('bounded resistance components support padded 3D dispatches', async () => {
  await checkComponents('shuffled', 64, 64, 'max', true);
});
it('bounded resistance components fit core device limits', async () => {
  await checkComponents('star', 64, 64, 'core');
});

it('bounded resistance components connect terminals in a component whose minimum ID is nonzero', async () => {
  await checkComponents('disconnected', 64, 64, 'max', false, true);
});
