// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsSummary of a distance-band graph whose weights carry noise. Left: points colored
// by the per-row cardinality the operation wrote. Right: the six counts (slots, asymmetric slots,
// isolates, minimum and maximum cardinality, empty rows) as bars of log(1 + count).

import {
  GPUSpatialWeightsSummary,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './spatial-weights-summary.parameters';
import {
  addLinkSegments,
  addLogCounts,
  getRowPositions,
  uploadWeightsBuffers,
  stretchPoints
} from './shared/weights';

/** Independent noise on every slot, so w_ij and w_ji differ. */
function getSlotWeight(slot: number, noise: number): number {
  const hash = Math.sin(slot * 12.9898) * 43758.5453;
  return 1 + (hash - Math.floor(hash) - 0.5) * noise;
}

export default defineOperationRun({
  parameters,
  caption: 'Cardinality per point (left) and the summary counts (right, log(1 + count)).',
  createInputs: ({random, world}) => {
    const mapWorld = {width: world.width * 0.55, height: world.height};
    const points = stretchPoints(createUniformPoints(random, mapWorld, 200, 0.1), mapWorld);
    return {
      ...points,
      csr: createDistanceBandWeights(points.positions, 0.09),
      index: createPointIndex(points.positions),
      focus: {row: -1}
    };
  },
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const {csr} = points;
    const band = uploadWeightsBuffers(builder, 'band', csr);
    const statistics = builder.allocate(
      'statistics',
      'float32',
      GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length
    );
    const counts = builder.allocate(
      'counts',
      'uint32',
      GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts.length
    );
    const cardinality = builder.allocate('cardinality', 'uint32', points.count);
    builder.graph.add(
      new GPUSpatialWeightsSummary({
        weights: band.input,
        symmetryTolerance: values.symmetryTolerance,
        output: {
          statistics: statistics.view,
          counts: counts.view,
          cardinality: cardinality.view
        }
      })
    );
    return {
      outputs: {
        positions,
        loggedCounts: addLogCounts(builder, 'summary', counts),
        cardinality,
        links: addLinkSegments(builder, 'band', band, positions)
      },
      // Independent noise on every slot, so w_ij and w_ji differ.
      update: current =>
        band.weights.write(
          Float32Array.from({length: band.capacity}, (_, slot) =>
            getSlotWeight(slot, current.noise)
          )
        )
    };
  },
  rebuildOn: ['symmetryTolerance'],
  // The pointer picks a point: its band links, solid where the reverse weight differs by more than symmetryTolerance.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, values}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const {csr} = inputs;
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [{shape: 'ring', at: from, radius: 0.012}];
      let asymmetric = 0;
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        const other = csr.neighbors[slot];
        let reverse = csr.offsets[other];
        while (reverse < csr.offsets[other + 1] && csr.neighbors[reverse] !== row) reverse++;
        const isAsymmetric =
          Math.abs(getSlotWeight(slot, values.noise) - getSlotWeight(reverse, values.noise)) >
          values.symmetryTolerance;
        if (isAsymmetric) asymmetric++;
        const [to] = getRowPositions(inputs.positions, [other]);
        glyphs.push({shape: 'line', from, to, dashed: !isAsymmetric}, {shape: 'dot', at: to});
      }
      const cardinality = csr.offsets[row + 1] - csr.offsets[row];
      return {
        focus: row,
        glyphs,
        readout: `cardinality ${cardinality} · ${asymmetric} asymmetric at tolerance ${values.symmetryTolerance}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.links},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.cardinality,
      ramp: 'sequential',
      radius: 3.6
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.loggedCounts,
      color: [0.55, 0.65, 0.95, 0.9],
      rectangle: [0.64, 0.14, 0.96, 0.86]
    }
  ]
});
