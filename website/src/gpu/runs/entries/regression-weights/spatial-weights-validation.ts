// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsValidation over an uploaded CSR with injected defects (a self slot, a neighbor
// id out of range, a negative weight, a duplicate neighbor). Points are colored by the violation
// flags the operation wrote per row; clean rows stay dark.

import {GPUSpatialWeightsValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {createLegend} from '../../text';
import {parameters} from './spatial-weights-validation.parameters';
import {addKernelPass} from './shared/kernel';
import {describeViolation, findNearestFlagged} from './shared/probes';
import {
  addLinkSegments,
  getSegmentTargets,
  uploadWeightsBuffers,
  stretchPoints
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Rows are colored by the violation flags the GPU wrote.',
  createInputs: ({random, world}) =>
    stretchPoints(createUniformPoints(random, world, 140, 0.1), world),
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
    // Lowest violated bit per row (a category), 0xffffffff for a clean row.
    const kinds = builder.allocate('kinds', 'uint32', points.count);
    addKernelPass(builder, {
      id: 'violation-kinds',
      invocationCount: points.count,
      bindings: [
        {name: 'flags', buffer: flags, access: 'read'},
        {name: 'kinds', buffer: kinds, access: 'read_write'}
      ],
      body: `let flag = flags[flagsOffset + index];
  kinds[kindsOffset + index] = select(0xffffffffu, firstTrailingBit(flag), flag != 0u);`
    });
    return {
      outputs: {
        positions,
        kinds,
        flags,
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
  // The pointer finds the nearest flagged row; its ring, links and violation are named.
  interaction: {
    kind: 'probe',
    hint: 'hover a flagged point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const flags = peek(outputs.flags);
      if (!flags) return {};
      const row = findNearestFlagged(flags, inputs.positions, pointer.x, pointer.y);
      if (row < 0) return {readout: 'no broken rows'};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: from, radius: 0.04},
        {shape: 'crosshair', at: from, size: 16}
      ];
      for (const links of [outputs.links]) {
        for (const to of getSegmentTargets(peek(links), from)) {
          glyphs.push({shape: 'line', from, to, dashed: true});
        }
      }
      return {focus: row, glyphs, readout: `row ${row} · ${describeViolation(flags[row])}`};
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.links},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.kinds,
      radius: 5.5
    },
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        title: 'violation flag',
        entries: [
          {label: 'neighbor out of range', swatch: 1},
          {label: 'duplicate neighbor', swatch: 2},
          {label: 'self neighbor', swatch: 3},
          {label: 'negative weight', swatch: 4}
        ]
      })
    }
  ]
});
