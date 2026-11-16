// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUForestInference} from '../../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUForestInference predicts more than the former 2^24 row cap using per-batch scratch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const lengths = [6_000_000, 5_000_000, 5_777_217];
  const features = lengths.map(rows => createOutputBuffer(device, rows));
  const predictions = lengths.map(rows => createOutputBuffer(device, rows));
  const validity = lengths.map(rows => createOutputBuffer(device, rows));
  // Nine stumps count thresholds below a feature. Marker values 1..9 predict themselves;
  // untouched zero features predict zero. Every chunk boundary and interior has a distinct marker.
  const floatBits = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0];
  const nodeWords: number[] = [],
    treeWords: number[] = [],
    leafScores: number[] = [];
  for (let tree = 0; tree < 9; tree++) {
    const root = tree * 3;
    nodeWords.push(
      0,
      root + 1,
      root + 2,
      floatBits(tree + 0.5),
      0,
      0,
      0xffffffff,
      0,
      0,
      0,
      0,
      root + 1,
      0xffffffff,
      0,
      0,
      0,
      0,
      root + 2
    );
    treeWords.push(root, root + 3, floatBits(1), 0);
    leafScores.push(0, 0, 1);
  }
  const markers = lengths.map(
    (rows, chunkIndex) =>
      new Map(
        [0, Math.floor(rows / 2), rows - 1].map((row, markerIndex) => {
          const value = chunkIndex * 3 + markerIndex + 1;
          features[chunkIndex].write(Float32Array.of(value), row * 4);
          return [row, value];
        })
      )
  );
  const nodes = createInputBuffer(device, Uint32Array.from(nodeWords));
  const trees = createInputBuffer(device, Uint32Array.from(treeWords));
  const leaves = createInputBuffer(device, Float32Array.from(leafScores));
  const parameters = createInputBuffer(device, Float32Array.of(1, 0));
  graph.add(
    new GPUForestInference({
      model: {
        featureCount: 1,
        treeCount: 9,
        nodeCount: 27,
        outputCount: 1,
        maximumDepth: 1,
        task: 'regression',
        transform: 'identity'
      },
      rowCount: 16_777_217,
      nodes: importGraphBuffer(graph, 'nodes', nodes, 'uint32', 27 * 6),
      trees: importGraphBuffer(graph, 'trees', trees, 'uint32', 9 * 4),
      leafValues: importGraphBuffer(graph, 'leaves', leaves, 'float32', 27),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 2),
      features: createVectorView(
        'features',
        'float32',
        lengths.map((rows, index) =>
          importGraphBuffer(graph, `features-${index}`, features[index], 'float32', rows)
        )
      ),
      output: {
        predictions: createVectorView(
          'predictions',
          'float32',
          lengths.map((rows, index) =>
            importGraphBuffer(graph, `predictions-${index}`, predictions[index], 'float32', rows)
          )
        ),
        validity: createVectorView(
          'validity',
          'uint32',
          lengths.map((rows, index) =>
            importGraphBuffer(graph, `validity-${index}`, validity[index], 'uint32', rows)
          )
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const [index] of lengths.entries()) {
      for (const [row, expected] of markers[index]) {
        const prediction = await predictions[index].readAsync(row * 4, 4);
        const valid = await validity[index].readAsync(row * 4, 4);
        expect(new Float32Array(prediction.buffer, prediction.byteOffset, 1)[0]).toBe(expected);
        expect(new Uint32Array(valid.buffer, valid.byteOffset, 1)[0]).toBe(1);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of [
      ...features,
      ...predictions,
      ...validity,
      nodes,
      trees,
      leaves,
      parameters
    ])
      buffer.destroy();
  }
});
