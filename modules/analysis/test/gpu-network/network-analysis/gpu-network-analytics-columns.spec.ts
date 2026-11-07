// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkAnalyticsColumns,
  type GPUNetworkAnalyticsColumnsProps
} from '../../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createRandomNetwork,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {
  buildReverseCSR,
  componentsOracle,
  createGridEdges,
  createSymmetricEdges,
  degreeOracle,
  extentOracle,
  normalizeOracle
} from './network-analytics-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type RunOptions = {
  nodeCount: number;
  edges: readonly NetworkEdge[];
  directed?: boolean;
  iterations?: number;
};

async function runColumns(device: Device, options: RunOptions) {
  const {nodeCount, edges, directed, iterations} = options;
  const graph = new GPUCommandGraph(device, {id: 'analytics'});
  const csr = buildCSR(nodeCount, edges);
  const reverse = directed ? buildReverseCSR(nodeCount, csr) : undefined;
  const buffers: Buffer[] = [];
  const capacityOf = (values: Uint32Array) => (values.length ? values : new Uint32Array(1));
  const input = (name: string, values: Uint32Array, length: number) => {
    const buffer = createInputBuffer(device, capacityOf(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', length);
  };
  const outputs: Record<string, Buffer> = {};
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs[name] = buffer;
    return importGraphBuffer(graph, `analytics-${name}`, buffer, format, length);
  };
  const props: GPUNetworkAnalyticsColumnsProps = {
    id: 'analytics',
    topology: {
      offsets: input('offsets', csr.offsets, nodeCount + 1),
      neighbors: input('neighbors', csr.neighbors, csr.neighbors.length)
    },
    degree: {
      output: output('degree', 'uint32', nodeCount),
      normalized: output('degree-normalized', 'float32', nodeCount),
      extent: output('degree-extent', 'uint32', 2)
    },
    components: {
      output: output('components', 'uint32', nodeCount),
      maximumIterations: iterations,
      status: {converged: output('components-converged', 'uint32', 1)}
    }
  };
  if (reverse) {
    props.reverseTopology = {
      offsets: input('reverse-offsets', reverse.offsets, nodeCount + 1),
      neighbors: input('reverse-neighbors', reverse.neighbors, reverse.neighbors.length)
    };
    props.inDegree = {output: output('inDegree', 'uint32', nodeCount)};
  }
  graph.add(new GPUNetworkAnalyticsColumns(props));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      csr,
      reverse,
      degree: Array.from(await readUint32(outputs.degree, nodeCount)),
      degreeNormalized: Array.from(await readFloat32(outputs['degree-normalized'], nodeCount)),
      degreeExtent: Array.from(await readUint32(outputs['degree-extent'], 2)),
      inDegree: outputs.inDegree
        ? Array.from(await readUint32(outputs.inDegree, nodeCount))
        : undefined,
      components: Array.from(await readUint32(outputs.components, nodeCount)),
      converged: (await readUint32(outputs['components-converged'], 1))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('GPUNetworkAnalyticsColumns matches the oracle on an undirected grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 36;
  const edges = createSymmetricEdges(createGridEdges(6, 6));
  const result = await runColumns(device, {nodeCount, edges});
  const expectedDegree = Array.from(degreeOracle(result.csr.offsets));
  expect(result.degree).toEqual(expectedDegree);
  expect(result.degreeExtent).toEqual(extentOracle(expectedDegree));
  expect(result.degreeNormalized).toEqual(Array.from(normalizeOracle(expectedDegree)));
  expect(result.components).toEqual(Array.from(componentsOracle(nodeCount, result.csr)));
  expect(result.converged).toBe(1);
});

it('GPUNetworkAnalyticsColumns handles a directed network with a reverse CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 40;
  const result = await runColumns(device, {
    nodeCount,
    edges: createRandomNetwork(7, nodeCount, 90),
    directed: true
  });
  expect(result.degree).toEqual(Array.from(degreeOracle(result.csr.offsets)));
  expect(result.inDegree).toEqual(Array.from(degreeOracle(result.reverse!.offsets)));
  expect(result.components).toEqual(Array.from(componentsOracle(nodeCount, result.csr)));
});

it('GPUNetworkAnalyticsColumns reports an exhausted component budget', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = 200;
  const path: NetworkEdge[] = [];
  for (let node = 0; node + 1 < nodeCount; node++) path.push([node, node + 1, 1]);
  const edges = createSymmetricEdges(path);
  const limited = await runColumns(device, {nodeCount, edges, iterations: 1});
  expect(limited.converged).toBe(0);
  const full = await runColumns(device, {nodeCount, edges, iterations: 64});
  expect(full.converged).toBe(1);
  expect(full.components).toEqual(new Array(nodeCount).fill(0));
});
