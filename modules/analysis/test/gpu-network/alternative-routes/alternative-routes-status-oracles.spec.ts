// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUAlternativeRoutes} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes';
import {
  GPUAlternativeRoutesOperation,
  type AlternativeRoutesOperationProps
} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import {CPUAlternativeRoutesOperation} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {poisonOperationOutputs} from '../adjacency-matrix/network-output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const getterBudget of [false, true])
  it(`R7/R10 both operations retain the captured chain refusal (getter ${getterBudget})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUAlternativeRoutesOperation, CPUAlternativeRoutesOperation]) {
      let reads = 0;
      const props: AlternativeRoutesOperationProps = {
        nodeCount: 3,
        queryCount: 1,
        routeCount: 2,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        sources: 'sources',
        targets: 'targets',
        maximumExpansions: 1,
        maximumFrontierSize: 2,
        output: {
          vertices: 'vertices',
          edges: 'edges',
          lengths: 'lengths',
          costs: 'costs',
          counts: 'counts',
          complete: 'complete'
        }
      };
      if (getterBudget)
        Object.defineProperty(props, 'maximumExpansions', {
          configurable: true,
          get() {
            if (++reads !== 1) throw new Error('Expansion budget read twice');
            return 1;
          }
        });
      let operation = new Operation(props);
      if (getterBudget) expect(reads).toBe(1);
      Object.defineProperty(props, 'maximumExpansions', {value: 2, configurable: true});
      props.maximumFrontierSize = 3;
      if (operation instanceof GPUAlternativeRoutesOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          sources: {format: 'uint32', values: Uint32Array.of(0)},
          targets: {format: 'uint32', values: Uint32Array.of(2)}
        },
        operations: [operation],
        outputs: ['counts', 'complete']
      });
      // Construction budget one leaves edge 1->2 unexamined, regardless of the later mutation.
      expect([rows['counts'][0], rows['complete'][0]]).toEqual([0, 0]);
      if (getterBudget) expect(reads).toBe(1);
    }
  });

for (const scene of [
  'chain-limited',
  'chain-complete',
  'diamond-limited',
  'diamond-complete'
] as const) {
  it(`writes independently derived counts and complete flags on poisoned outputs: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const diamond = scene.startsWith('diamond');
    const limited = scene.endsWith('limited');
    const queryCount = diamond ? 1 : 4;
    const nodeCount = diamond ? 4 : 3;
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createOutputBuffer>[] = [];
    let nextView = 0;
    const input = <Format extends 'uint32' | 'float32'>(
      values: Uint32Array | Float32Array,
      format: Format
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `input-${nextView++}`, buffer, format, values.length)
      };
    };
    const output = <Format extends 'uint32' | 'float32' = 'uint32'>(
      length: number,
      format: Format = 'uint32' as Format
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `output-${nextView++}`, buffer, format, length)
      };
    };
    const counts = output(queryCount);
    const complete = output(queryCount);
    const weights = input(diamond ? Float32Array.of(1, 1, 1, 1) : Float32Array.of(1, 1), 'float32');
    graph.add(
      new GPUAlternativeRoutes({
        topology: {
          offsets: input(
            diamond ? Uint32Array.of(0, 2, 3, 4, 4) : Uint32Array.of(0, 1, 2, 2),
            'uint32'
          ).view,
          neighbors: input(diamond ? Uint32Array.of(1, 2, 3, 3) : Uint32Array.of(1, 2), 'uint32')
            .view,
          weights: weights.view
        },
        sources: input(diamond ? Uint32Array.of(0) : Uint32Array.of(0, 1, 2, 99), 'uint32').view,
        targets: input(diamond ? Uint32Array.of(3) : Uint32Array.of(2, 2, 2, 2), 'uint32').view,
        routeCount: 2,
        maximumFrontierSize: diamond && limited ? 1 : 2,
        maximumExpansions: diamond ? 4 : limited ? 1 : 2,
        output: {
          vertices: output(queryCount * 2 * nodeCount).view,
          edges: output(queryCount * 2 * nodeCount).view,
          lengths: output(queryCount * 2).view,
          costs: output(queryCount * 2, 'float32').view,
          counts: counts.view,
          complete: complete.view
        }
      })
    );
    const compiled = graph.compile();
    try {
      // Hand enumeration: the chain needs two edge examinations from source 0;
      // after one, edge 1->2 remains unresolved. The diamond needs two simultaneous
      // prefixes [0,1] and [0,2]; one frontier slot stops before reaching either target.
      // With both slots and four examinations, exactly two diamond paths exist.
      counts.buffer.write(new Uint32Array(queryCount).fill(0xdeadbeef));
      complete.buffer.write(new Uint32Array(queryCount).fill(0xdeadbeef));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(counts.buffer, queryCount)).toEqual(
        diamond ? (limited ? [0] : [2]) : limited ? [0, 1, 1, 0] : [1, 1, 1, 0]
      );
      expect(await readUint32(complete.buffer, queryCount)).toEqual(
        diamond ? (limited ? [0] : [1]) : limited ? [0, 1, 1, 1] : [1, 1, 1, 1]
      );
      // Reuse must clear previously discovered routes and any incomplete certificate.
      weights.buffer.write(new Float32Array(diamond ? 4 : 2).fill(-1));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(counts.buffer, queryCount)).toEqual(diamond ? [0] : [0, 0, 1, 0]);
      expect(await readUint32(complete.buffer, queryCount)).toEqual(diamond ? [1] : [1, 1, 1, 1]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
