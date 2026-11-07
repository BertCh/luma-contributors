// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsValidation over an uploaded CSR with injected defects (a self slot, a neighbor
// id out of range, a negative weight, a duplicate neighbor). Points are colored by the violation
// flags the operation wrote per row; clean rows stay dark.

import {GPUSpatialWeightsValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import {parameters} from './spatial-weights-validation.parameters';
import {addKernelPass} from './shared/kernel';
import {addLinkSegments, uploadWeightsBuffers} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Rows are colored by the violation flags the GPU wrote.',
  createInputs: ({random, world}) => createUniformPoints(random, world, 140, 0.1),
  build(builder, points) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const csr = createKnnWeights(points.positions, 4);
    const weights = uploadWeightsBuffers(builder, 'uploaded', csr);
    const flags = builder.allocate('flags', 'uint32', points.count);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    builder.graph.add(
      new GPUSpatialWeightsValidation({
        weights: weights.input,
        output: {
          flags: flags.view,
          status: {invalidCount: invalidCount.view}
        }
      })
    );
    // Violation kind per row: 0 for a clean row, 2 plus the lowest violated bit otherwise.
    const kinds = builder.allocate('kinds', 'float32', points.count);
    addKernelPass(builder, {
      id: 'violation-kinds',
      invocationCount: points.count,
      bindings: [
        {name: 'flags', buffer: flags, access: 'read'},
        {name: 'kinds', buffer: kinds, access: 'read_write'}
      ],
      body: `let flag = flags[flagsOffset + index];
  kinds[kindsOffset + index] = select(0.0, 2.0 + f32(firstTrailingBit(flag)), flag != 0u);`
    });
    return {
      outputs: {
        positions,
        kinds,
        links: addLinkSegments(builder, 'uploaded', weights, positions)
      },
      update: current => {
        const neighbors = Uint32Array.from(csr.neighbors);
        const slotWeights = Float32Array.from(csr.weights);
        for (let row = 0; row < points.count; row++) {
          const hash = Math.sin(row * 78.233) * 43758.5453;
          if (hash - Math.floor(hash) >= current.defects) continue;
          const first = csr.offsets[row];
          const kind = row % 4;
          if (kind === 0) neighbors[first] = row;
          else if (kind === 1) neighbors[first] = points.count + 3;
          else if (kind === 2) slotWeights[first] = -1;
          else neighbors[first + 1] = neighbors[first];
        }
        weights.neighbors.write(neighbors);
        weights.weights.write(slotWeights);
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.links},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.kinds,
      ramp: 'sequential',
      range: [0, 7],
      fade: true,
      radius: 4.2
    }
  ]
});
