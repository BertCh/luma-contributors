// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUPairGather,
  GPUSpatialJoinCandidates,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {readUint32, submitGraph} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {
  runExample as gatherExample,
  type PairGatherExampleInput
} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example';
import {runExample as predicateExample} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-predicate-join.example';
import {
  runExample as preparedExample,
  type SpatialJoinPreparedExampleInput
} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-join-prepared.example';

for (const [name, example] of [
  ['gather', gatherExample],
  ['predicate', predicateExample],
  ['prepared', preparedExample]
] as const) {
  test(`rr-round4-landing-R8: ${name} actual mask and histogram clear a shrinking pair prefix`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph<undefined>(device);
    const buffers: Buffer[] = [];
    const upload = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      id: string,
      data: Uint32Array | Float32Array,
      format: Format
    ) => {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const view = graph.createDataView(
        graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
        {format, length: data.length / (format === 'float32x2' ? 2 : 1)}
      );
      return {buffer, view};
    };
    const count = upload('count', Uint32Array.of(1), 'uint32');
    const pairs = (prefix: string) => ({
      leftRows: upload(`${prefix}-left`, Uint32Array.of(0, 0), 'uint32').view,
      rightRows: upload(`${prefix}-right`, Uint32Array.of(0, 1), 'uint32').view,
      status: {
        count:
          prefix === 'pairs'
            ? count.view
            : upload(`${prefix}-count`, Uint32Array.of(0), 'uint32').view,
        overflow: upload(`${prefix}-overflow`, Uint32Array.of(0), 'uint32').view,
        candidateOverflow: upload(`${prefix}-candidate-overflow`, Uint32Array.of(0), 'uint32').view
      }
    });
    const polygons = (prefix: string, features: number) => ({
      kind: 'polygons' as const,
      positions: upload(
        `${prefix}-positions`,
        new Float32Array(
          Array.from({length: features}, () => [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]).flat()
        ),
        'float32x2'
      ).view,
      ringOffsets: upload(
        `${prefix}-rings`,
        Uint32Array.from({length: features + 1}, (_, row) => row * 5),
        'uint32'
      ).view,
      polygonOffsets: upload(
        `${prefix}-polygons`,
        Uint32Array.from({length: features + 1}, (_, row) => row),
        'uint32'
      ).view,
      featureOffsets: upload(
        `${prefix}-features`,
        Uint32Array.from({length: features + 1}, (_, row) => row),
        'uint32'
      ).view
    });
    const input: PairGatherExampleInput & SpatialJoinPreparedExampleInput = {
      parks: polygons('parks', 1),
      tracts: polygons('tracts', 2),
      pairs: pairs('pairs'),
      candidatePairs: pairs('candidates'),
      parkAreas: upload('park-areas', Float32Array.of(1), 'float32').view,
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'}
    };
    // The fixture publishes pairs directly. Keep the examples' actual membership kernel and
    // GPUHistogram; omit their upstream joins/gather so they cannot overwrite our count/rows.
    const originalAdd = graph.add;
    const add = vi.spyOn(graph, 'add').mockImplementation(node => {
      if (
        node instanceof GPUSpatialPredicateJoin ||
        node instanceof GPUPairGather ||
        node instanceof GPUSpatialJoinPrepared ||
        node instanceof GPUSpatialJoinCandidates
      )
        return;
      originalAdd.call(graph, node);
    });
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      const result = await example(graph, input);
      add.mockRestore();
      const bins = upload('bins-readback', Uint32Array.of(99, 99), 'uint32');
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'copy-histogram',
          operation: 'lens-prefix-test-readback',
          invocationCount: 2,
          bindings: [
            {name: 'input', view: result.tractMatches, type: 'u32', access: 'read'},
            {name: 'output', view: bins.view, type: 'u32', access: 'read_write'}
          ],
          body: 'output[outputOffset + index] = input[inputOffset + index];'
        })
      );
      compiled = graph.compile();
      // Hand-counted: only rightRows[0]=0 is published, then no rows are published.
      submitGraph(device, compiled);
      expect(await readUint32(bins.buffer, 2)).toEqual([1, 0]);
      count.buffer.write(Uint32Array.of(0));
      submitGraph(device, compiled);
      expect(await readUint32(bins.buffer, 2)).toEqual([0, 0]);
    } finally {
      add.mockRestore();
      compiled?.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}
