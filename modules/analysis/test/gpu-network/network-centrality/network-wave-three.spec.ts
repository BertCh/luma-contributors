// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNetworkCentralityContinuation} from '../../../src/gpu-network/network-centrality/network-centrality-continuation';
import {GPUNetworkCentrality} from '../../../src/gpu-network/network-centrality/gpu-network-centrality';
import {GPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('recovers rational-access completeness after truncated costs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const complete of [0, 1]) {
    const result = await runOperationGraph(device, {
      inputs: {
        costs: {format: 'float32', values: Float32Array.of(complete ? 2 : Infinity)},
        demand: {format: 'float32', values: Float32Array.of(1)},
        supply: {format: 'float32', values: Float32Array.of(1)},
        parameters: {format: 'float32', values: Float32Array.of(1, 1, 0.5, 0, 10, 0.01)},
        costsComplete: {format: 'uint32', values: Uint32Array.of(complete)}
      },
      operations: [
        new GPUNetworkRationalAccessOperation({
          originCount: 1,
          facilityCount: 1,
          costs: 'costs',
          demand: 'demand',
          supply: 'supply',
          parameters: 'parameters',
          costsComplete: 'costsComplete',
          output: {
            assignments: 'assignments',
            scores: 'scores',
            unservedDemand: 'unserved',
            summary: 'summary',
            status: {complete: 'complete'}
          }
        })
      ],
      outputs: ['complete', 'assignments', 'unserved']
    });
    expect(Array.from(result['complete'])).toEqual([complete]);
    expect(Array.from(result['assignments'])).toEqual([complete]);
    expect(Array.from(result['unserved'])).toEqual([1 - complete]);
  }
});

it('checks coverage and unbounded-objective horizons without replacing enumeration diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['backup-coverage', 'capacitated-p-median'] as const) {
    for (const horizon of [1, 2, Infinity]) {
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const input = <Format extends 'uint32' | 'float32'>(
        format: Format,
        values: Uint32Array | Float32Array
      ) => {
        const buffer = createInputBuffer(device, values);
        buffers.push(buffer);
        return importGraphBuffer(graph, `input-${buffers.length}`, buffer, format, values.length);
      };
      const output = <Format extends 'uint32' | 'float32'>(format: Format, length: number) => {
        const buffer = createOutputBuffer(device, length);
        buffers.push(buffer);
        return {
          buffer,
          view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, length)
        };
      };
      const summary = output('uint32', 4),
        complete = output('uint32', 1),
        objective = output('float32', 1);
      graph.add(
        new GPUNetworkFacilityOptimization({
          operation,
          selectCount: operation === 'capacitated-p-median' ? 1 : undefined,
          facilityCount: 1,
          costs: input('float32', Float32Array.of(horizon === 1 ? Infinity : 2)),
          costsComplete: input('uint32', Uint32Array.of(1)),
          costLimit: input('float32', Float32Array.of(horizon)),
          demand: input('float32', Float32Array.of(1)),
          capacities: input('float32', Float32Array.of(1)),
          threshold: input('float32', Float32Array.of(2)),
          output: {
            selectedFacilities: output('uint32', 1).view,
            assignments: output('uint32', 1).view,
            objective: objective.view,
            summary: summary.view,
            status: {complete: complete.view}
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled);
        const words = await complete.buffer.readAsync();
        expect(Array.from(new Uint32Array(words.buffer, words.byteOffset, 1))).toEqual([
          (operation === 'backup-coverage' ? horizon >= 2 : horizon === Infinity) ? 1 : 0
        ]);
        const diagnostics = await summary.buffer.readAsync();
        expect(new Uint32Array(diagnostics.buffer, diagnostics.byteOffset, 4)[2]).toBe(1);
        if (operation === 'capacitated-p-median' && horizon >= 2) {
          expect(
            Array.from(new Uint32Array(diagnostics.buffer, diagnostics.byteOffset, 3))
          ).toEqual([1, 1, 1]);
          const objectiveBytes = await objective.buffer.readAsync();
          const objectiveValue = new Float32Array(
            objectiveBytes.buffer,
            objectiveBytes.byteOffset,
            1
          )[0];
          expect(Number.isFinite(objectiveValue)).toBe(true);
          expect(objectiveValue).toBe(2); // One unit assigned to the only facility at cost two.
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }
  }
});

it('grows dynamic pivots between batches and agrees with a fresh distance-only runner', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      values instanceof Float32Array ? 'float32' : 'uint32',
      values.length
    );
  };
  const pivots = new DynamicBuffer(device, {
    byteLength: 32,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  pivots.write(Uint32Array.of(0), 4);
  const handle = graph.importBuffer(
    {id: 'pivots', byteLength: 8, usage: Buffer.STORAGE | Buffer.COPY_DST},
    pivots
  );
  const closenessBuffer = createOutputBuffer(device, 1);
  buffers.push(closenessBuffer);
  const closeness = importGraphBuffer(graph, 'closeness', closenessBuffer, 'float32', 1);
  const props = {
    topology: {
      offsets: input(Uint32Array.of(0, 1, 2, 2)) as never,
      neighbors: input(Uint32Array.of(1, 2)) as never,
      weights: input(Float32Array.of(2, 3)) as never
    },
    pivotNodes: graph.createDataView(handle, {format: 'uint32', length: 1, byteOffset: 4}),
    output: {closeness},
    roundsPerStep: 1,
    maximumIterations: 4,
    localIterations: 1
  };
  const runner = createNetworkCentralityContinuation(graph, props);
  try {
    await runner.runBatch();
    pivots.resize({byteLength: 8});
    pivots.write(Uint32Array.of(1), 4);
    await runner.runBatch();
    const bytes = await closenessBuffer.readAsync();
    const actual = new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeCloseTo(1 / 3); // Pivot 1 reaches only vertex 2 at distance 3.
    const fresh = createNetworkCentralityContinuation(graph, props);
    try {
      await fresh.runBatch();
    } finally {
      fresh.destroy();
    }
    const freshBytes = await closenessBuffer.readAsync();
    expect(new Float32Array(freshBytes.buffer, freshBytes.byteOffset, 1)[0]).toBe(actual);
    pivots.resize({byteLength: 16});
    pivots.write(Uint32Array.of(1), 4);
    await runner.runBatch();
    const grownBytes = await closenessBuffer.readAsync();
    expect(new Float32Array(grownBytes.buffer, grownBytes.byteOffset, 1)[0]).toBe(actual);
    // The ordinary graph route also accepts distance metrics without a transpose.
    graph.add(new GPUNetworkCentrality(props));
  } finally {
    runner.destroy();
    pivots.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('publishes empty query outputs and defined K-shortest batch status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {GPUKShortestPathsOperation} = await import(
    '../../../src/gpu-network/k-shortest-paths/gpu-k-shortest-paths-operation'
  );
  const {GPUAlternativeRoutesOperation} = await import(
    '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation'
  );
  const props = {
    nodeCount: 1,
    queryCount: 0,
    routeCount: 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    output: {
      vertices: 'vertices',
      edges: 'edges',
      lengths: 'lengths',
      costs: 'costs',
      counts: 'counts',
      status: {
        converged: 'converged',
        iterationLimitReached: 'limit',
        candidateOverflow: 'overflow',
        iterationCount: 'iterations'
      }
    }
  };
  for (const operation of [
    new GPUKShortestPathsOperation(props),
    new GPUAlternativeRoutesOperation({
      ...props,
      output: {
        vertices: 'vertices',
        edges: 'edges',
        lengths: 'lengths',
        costs: 'costs',
        counts: 'counts',
        complete: 'complete'
      }
    })
  ]) {
    const outputs = Object.values(operation.outputs).map(output => output.value);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 0)},
        neighbors: {format: 'uint32', values: new Uint32Array()},
        weights: {format: 'float32', values: new Float32Array()},
        sources: {format: 'uint32', values: new Uint32Array()},
        targets: {format: 'uint32', values: new Uint32Array()}
      },
      operations: [operation],
      outputs
    });
    expect(result['counts'].length).toBe(0);
    expect(result['costs'].length).toBe(0);
    if (operation instanceof GPUKShortestPathsOperation) {
      expect(Array.from(result['converged'])).toEqual([1]);
      expect(Array.from(result['limit'])).toEqual([0]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['iterations'])).toEqual([0]);
    } else expect(result['complete'].length).toBe(0);
  }
});
