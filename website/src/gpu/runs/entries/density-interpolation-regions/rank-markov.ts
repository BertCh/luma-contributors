// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURankMarkov replaces values by their rank in each period and counts rank-to-rank moves.
// Left: the rank of each tract in one period. Right: the transition probabilities between ranks
// (a bright diagonal means stable ranks) and, below, the rank mobility of each period step.

import {GPURankMarkov} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {parameters} from './rank-markov.parameters';
import {addSlice} from './support/gather';
import {createDynamics, createMatrixMesh, PERIOD_COUNT, uploadMatrixMesh} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Rank transitions between periods and the mobility of each step.',
  createInputs: ({random, world}) => ({...createDynamics(random, world, 5), world}),
  build(builder, {tracts, values, world}, current) {
    const placeCount = Math.min(tracts.count, 64);
    // GPURankMarkov handles at most 64 places; keep the first 64 (the mesh draws them all).
    const kept = Float32Array.from(
      {length: placeCount * PERIOD_COUNT},
      (_, index) => values[Math.floor(index / placeCount) * tracts.count + (index % placeCount)]
    );
    const ranks = builder.allocate('ranks', 'uint32', kept.length);
    const probabilities = builder.allocate('probabilities', 'float32', placeCount * placeCount);
    const mobility = builder.allocate('mobility', 'float32', PERIOD_COUNT - 1);
    builder.graph.add(
      new GPURankMarkov({
        values: builder.upload('values', 'float32', kept).view,
        periodCount: PERIOD_COUNT,
        mode: current.mode === 0 ? 'full-rank' : 'geographic-rank',
        output: {
          ranks: ranks.view,
          mobility: mobility.view,
          counts: builder.allocate('counts', 'uint32', placeCount * placeCount).view,
          probabilities: probabilities.view
        }
      })
    );
    const matrix = createMatrixMesh(placeCount, placeCount, [
      world.width * 0.58,
      0.27,
      world.width * 0.97,
      0.92
    ]);
    const mesh = uploadTractMesh(builder, tracts);
    return {
      // The period is per-frame: it only changes which period's ranks color the map.
      update: values =>
        mesh.owners.write(
          Uint32Array.from(
            tracts.triangleOwners,
            owner => Math.min(owner, placeCount - 1) + values.period * placeCount
          )
        ),
      outputs: {
        ranks: addSlice(builder, ranks, 0, ranks.length, 'rank-values'),
        probabilities,
        mobility,
        placeCount,
        matrixMesh: uploadMatrixMesh(builder, matrix, 'matrix'),
        ...mesh
      }
    };
  },
  rebuildOn: ['mode'],
  // x plays the periods: the map shows the ranks of that period.
  interaction: bindAxes(parameters, {
    kind: 'time',
    hint: 'play the periods',
    x: 'period',
    decorate: values => ({
      readout: `period ${values.period + 1} of ${PERIOD_COUNT} · ranks of ${values.mode === 0 ? 'full' : 'geographic'} mode`
    })
  }),
  draw: ({inputs, outputs, values, world}) => {
    const mobilityRectangle = [0.58, 0.05, 0.97, 0.2] as const;
    const axes = createChartAxes({
      rectangle: getChartRectangle(world, mobilityRectangle),
      x: {
        categories: Array.from({length: PERIOD_COUNT - 1}, (_, step) => `${step + 1}-${step + 2}`)
      }
    });
    return [
      {
        layer: 'polygons',
        positions: outputs.vertices,
        indices: outputs.indices,
        owners: outputs.owners,
        values: outputs.ranks,
        triangleCount: inputs.tracts.triangleCount,
        ramp: 'sequential',
        range: [0, outputs.placeCount - 1]
      },
      {
        layer: 'polygons',
        positions: outputs.matrixMesh.vertices,
        indices: outputs.matrixMesh.indices,
        owners: outputs.matrixMesh.owners,
        values: outputs.probabilities,
        triangleCount: outputs.matrixMesh.triangleCount,
        range: [0, 0.5]
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.mobility,
        range: 'auto',
        rectangle: mobilityRectangle,
        ramp: 'sequential'
      },
      {
        layer: 'text',
        ...mergeText(axes, {
          labels: [
            {
              text: `rank in period ${values.period + 1}`,
              at: [0.02 * world.width, 0.97],
              anchor: 'top-left',
              role: 'title'
            },
            {
              text: 'rank now (down) to rank next (across)',
              at: [0.58 * world.width, 0.93],
              anchor: 'bottom-left',
              role: 'label'
            },
            {
              text: 'rank mobility, period step',
              at: [0.58 * world.width, 0.2],
              anchor: 'bottom-left',
              offset: [0, 2],
              role: 'label'
            }
          ]
        })
      }
    ];
  }
});
