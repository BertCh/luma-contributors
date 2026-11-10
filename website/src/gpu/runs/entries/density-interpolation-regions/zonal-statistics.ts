// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUZonalStatistics summarizes the point values that fall inside each polygon: count, mean,
// extremes, spread and sum per tract. The tracts are filled with the chosen output buffer.

import {GPUZonalStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {isInsideRing} from '../../pick';
import {createTracts} from '../../inputs';
import workplaceBlocks from '../../../data/chicago-workplace-blocks.json';
import {parameters} from './zonal-statistics.parameters';
import {addSlice} from './support/gather';
import {
  createTractGlyph,
  findMiddleTract,
  findTractAt,
  getPackedPolygons,
  uploadPackedPolygons,
  uploadTractMesh
} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Workplace block jobs by Chicago tract, shown on a logarithmic color scale.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 8);
    const positions = Float32Array.from(
      workplaceBlocks.flatMap(([east, north]) => tracts.projectMeters(east, north))
    );
    const values = Float32Array.from(workplaceBlocks, block => block[2]);
    return {
      tracts,
      world,
      points: {positions, values},
      focus: {tract: findMiddleTract(tracts, world)}
    };
  },
  build(builder, {tracts, points}, current) {
    const count = Math.min(points.values.length, current.pointCount);
    const featureCount = tracts.count;
    const statistics = {
      counts: builder.allocate('counts', 'uint32', featureCount),
      means: builder.allocate('means', 'float32', featureCount),
      minima: builder.allocate('minima', 'float32', featureCount),
      maxima: builder.allocate('maxima', 'float32', featureCount),
      standardDeviations: builder.allocate('standard-deviations', 'float32', featureCount),
      sums: builder.allocate('sums', 'float32', featureCount)
    };
    builder.graph.add(
      new GPUZonalStatistics({
        features: {
          kind: 'polygons',
          ...uploadPackedPolygons(builder, getPackedPolygons(tracts)),
          candidateCapacity: count * 8
        },
        positions: builder.upload('positions', 'float32x2', points.positions.slice(0, count * 2))
          .view,
        values: builder.upload('values', 'float32', points.values.slice(0, count)).view,
        output: {
          counts: statistics.counts.view,
          means: statistics.means.view,
          minima: statistics.minima.view,
          maxima: statistics.maxima.view,
          standardDeviations: statistics.standardDeviations.view,
          sums: statistics.sums.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    return {
      outputs: {
        ...statistics,
        countValues: addSlice(builder, statistics.counts, 0, featureCount, 'count-values'),
        ...uploadTractMesh(builder, tracts),
        dots: builder.upload('dots', 'float32x2', points.positions),
        dotCount: count,
        // The blocks inside the hovered tract; unused rows stay NaN and are not drawn.
        inside: builder.upload('inside', 'float32x2', new Float32Array(count * 2).fill(Number.NaN))
      }
    };
  },
  // The hovered tract is outlined and the workplace blocks inside it are lit; the readout is the
  // operation's count and sum for that tract.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.tracts.centers[inputs.focus.tract * 2],
        inputs.tracts.centers[inputs.focus.tract * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      const tract = findTractAt(inputs.tracts, pointer.x, pointer.y);
      inputs.focus.tract = tract;
      const start = inputs.tracts.ringOffsets[tract];
      const ringLength = inputs.tracts.ringOffsets[tract + 1] - start;
      const lit = new Float32Array(outputs.dotCount * 2).fill(Number.NaN);
      let blockCount = 0;
      for (let row = 0; row < outputs.dotCount; row++) {
        const x = inputs.points.positions[row * 2];
        const y = inputs.points.positions[row * 2 + 1];
        if (!isInsideRing(inputs.tracts.vertices, x, y, start, ringLength)) continue;
        lit.set([x, y], blockCount * 2);
        blockCount++;
      }
      outputs.inside.write(lit);
      const sum = peek(outputs.sums)?.[tract];
      return {
        focus: tract,
        glyphs: [createTractGlyph(inputs.tracts, tract)],
        readout: `tract ${tract} · ${blockCount} blocks · ${sum === undefined ? '...' : `${Math.round(sum).toLocaleString('en-US')} jobs`}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: [
        outputs.countValues,
        outputs.means,
        outputs.minima,
        outputs.maxima,
        outputs.standardDeviations,
        outputs.sums
      ][values.statistic],
      triangleCount: inputs.tracts.triangleCount,
      // Workplace counts span orders of magnitude between the Loop and residential tracts.
      logarithmic: true,
      range: [0, [8, 13, 6, 16, 14, 20][values.statistic]],
      ramp: 'sequential'
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.tracts.outlineCount,
      alpha: 0.5
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.dots,
      count: outputs.dotCount,
      radius: 1.2,
      alpha: 0.6
    },
    {layer: 'dots', positions: outputs.inside, radius: 3.4, color: [1, 0.9, 0.35, 1]}
  ]
});
