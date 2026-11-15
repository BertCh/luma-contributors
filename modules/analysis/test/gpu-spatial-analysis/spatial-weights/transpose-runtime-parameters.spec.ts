// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsTransposeParameterValues,
  GPUSpatialWeightsTranspose,
  GPUChunkedSpatialWeightsTranspose
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from './spatial-weights-harness';

for (const layout of ['packed', 'chunked-input', 'chunked-output'] as const) {
  it(`${layout} transpose rereads symmetry tolerance on the same compiled graph, including ties and one-way zero slots`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const setting of ['runtime', 'numeric'] as const) {
      const rig = new WeightsRig(device);
      const parameters = new GPUParameterBuffer(device, {
        id: 'transpose-tolerance',
        format: 'float32',
        length: 3,
        values: new Float32Array([99, 0, 99])
      });
      const imported = parameters.importToGraph(rig.graph);
      const symmetryTolerance =
        setting === 'runtime'
          ? rig.graph.createDataView(imported.buffer, {format: 'float32', length: 1, byteOffset: 4})
          : 0.125;
      // One unequal reverse pair, one one-way ZERO slot, one equal pair, and two empty rows.
      const whole = rig.uploadWeights(
        {
          offsets: [0, 2, 3, 3, 4, 5, 5],
          neighbors: [1, 2, 0, 4, 3],
          weights: [1, 0, 1.125, 2, 2],
          distances: []
        },
        6
      );
      const chunked = {
        rowCount: 6,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: rig.uploadWeights(
              {offsets: [0, 2, 3], neighbors: [1, 2, 0], weights: [1, 0, 1.125], distances: []},
              2
            )
          },
          {
            rowStart: 2,
            rowCount: 1,
            weights: rig.uploadWeights(
              {offsets: [0, 0], neighbors: [], weights: [], distances: []},
              2
            )
          },
          {
            rowStart: 3,
            rowCount: 3,
            weights: rig.uploadWeights(
              {offsets: [0, 1, 2, 2], neighbors: [4, 3], weights: [2, 2], distances: []},
              2
            )
          }
        ]
      };
      const diagnostic = rig.output('uint32', 3);
      const asymmetricSlots = rig.graph.createDataView(diagnostic.view.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 4
      });
      const createOutput = (rows: number, capacity: number) => {
        const offsets = rig.output('uint32', rows + 1);
        const neighbors = rig.output('uint32', capacity);
        const values = rig.output('float32', capacity);
        return {
          offsets,
          neighbors,
          values,
          rows,
          capacity,
          spatialWeights: {offsets: offsets.view, neighbors: neighbors.view, weights: values.view}
        };
      };
      const checkOutput = async (
        output: ReturnType<typeof createOutput>,
        expected: {offsets: number[]; neighbors: number[]; weights: number[]}
      ) => {
        const padding = new Array(output.capacity - expected.neighbors.length).fill(0);
        expect(await readUint32(output.offsets.buffer, output.rows + 1)).toEqual(expected.offsets);
        expect(await readUint32(output.neighbors.buffer, output.capacity)).toEqual([
          ...expected.neighbors,
          ...padding
        ]);
        expect(await readFloat32(output.values.buffer, output.capacity)).toEqual([
          ...expected.weights,
          ...padding
        ]);
      };
      const packedOutput = createOutput(6, 11);
      const outputRows = [1, 3, 2];
      const outputStarts = [0, 1, 4];
      const capacities = [1, 1, 4];
      const chunks = outputRows.map((rowCount, index) => {
        const data = createOutput(rowCount, capacities[index]);
        const count = rig.output('uint32', 1);
        const requiredCount = rig.output('uint32', 1);
        const overflow = rig.output('uint32', 1);
        return {data, count, requiredCount, overflow, rowStart: outputStarts[index], rowCount};
      });
      if (layout === 'packed') {
        rig.graph.add(
          new GPUSpatialWeightsTranspose({
            weights: whole,
            symmetryTolerance,
            output: {...packedOutput.spatialWeights, asymmetricSlots}
          })
        );
      } else if (layout === 'chunked-input') {
        rig.graph.add(
          new GPUChunkedSpatialWeightsTranspose({
            weights: chunked,
            symmetryTolerance,
            output: {...packedOutput.spatialWeights, asymmetricSlots}
          })
        );
      } else {
        rig.graph.add(
          new GPUChunkedSpatialWeightsTranspose({
            weights: chunked,
            symmetryTolerance,
            output: {
              asymmetricSlots,
              chunks: chunks.map(chunk => ({
                rowStart: chunk.rowStart,
                rowCount: chunk.rowCount,
                weights: {
                  ...chunk.data.spatialWeights,
                  status: {
                    count: chunk.count.view,
                    requiredCount: chunk.requiredCount.view,
                    overflow: chunk.overflow.view
                  }
                }
              }))
            }
          })
        );
      }
      const compiled = rig.graph.compile();
      try {
        for (const tolerance of setting === 'runtime' ? [0, 0.125, 0.0625, 1, 0] : [0.125]) {
          parameters.write(
            getGPUSpatialWeightsTransposeParameterValues({symmetryTolerance: tolerance}),
            1
          );
          submitGraph(device, compiled, undefined);
          // Missing reverse slots count in BOTH directions even when their present weight is zero.
          expect(await readUint32(diagnostic.buffer, 3)).toEqual([
            0x7f7f7f7f,
            tolerance < 0.125 ? 4 : 2,
            0x7f7f7f7f
          ]);
          if (layout !== 'chunked-output') {
            await checkOutput(packedOutput, {
              offsets: [0, 1, 2, 3, 4, 5, 5],
              neighbors: [1, 0, 0, 4, 3],
              weights: [1.125, 1, 0, 2, 2]
            });
          } else {
            const expected = [
              {offsets: [0, 1], neighbors: [1], weights: [1.125], distances: []},
              {offsets: [0, 1, 1, 1], neighbors: [0], weights: [1], distances: []},
              {offsets: [0, 1, 1], neighbors: [3], weights: [2], distances: []}
            ];
            for (const [index, chunk] of chunks.entries()) {
              await checkOutput(chunk.data, expected[index]);
              expect(await readUint32(chunk.count.buffer, 1)).toEqual([1]);
              expect(await readUint32(chunk.requiredCount.buffer, 1)).toEqual([
                index === 1 ? 3 : 1
              ]);
              expect(await readUint32(chunk.overflow.buffer, 1)).toEqual([index === 1 ? 1 : 0]);
            }
          }
        }
      } finally {
        compiled.destroy();
        parameters.destroy();
        rig.destroy();
      }
    }
  });
}
