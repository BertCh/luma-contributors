// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCanonicalPolygonEquality compares two polygon sets feature by feature after each is
// normalized. The right set restarts every ring at another vertex and winds some the other way,
// and displaces one vertex of each polygon by a growing amount; polygons are filled by the
// equality flag the operation wrote, so a larger tolerance accepts more of them.

import {
  GPUCanonicalPolygonEquality,
  GPUNormalizePolygons
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, restartRings, uploadPolygonSet} from './shared/polygon-set';
import {parameters} from './canonical-polygon-equality.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Polygons equal within the tolerance, after canonical normalization.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {rows: 3, columns: 5})
  }),
  build(builder, {set}, values) {
    const left = uploadPolygonSet(builder, set);
    const right = builder.upload('right-input', 'float32x2', set.positions);
    const vertexCount = set.positions.length / 2;
    const normalize = (id: string, positions: typeof right) => {
      const output = {
        positions: builder.allocate(`${id}-positions`, 'float32x2', vertexCount),
        ringOffsets: builder.allocate(`${id}-rings`, 'uint32', set.ringOffsets.length),
        polygonOffsets: builder.allocate(`${id}-polygons`, 'uint32', set.polygonOffsets.length)
      };
      builder.graph.add(
        new GPUNormalizePolygons({
          id,
          positions: positions.view,
          featureOffsets: left.featureOffsets.view,
          polygonOffsets: left.polygonOffsets.view,
          ringOffsets: left.ringOffsets.view,
          output: {
            positions: output.positions.view,
            ringOffsets: output.ringOffsets.view,
            polygonOffsets: output.polygonOffsets.view
          }
        })
      );
      return {
        positions: output.positions.view,
        ringOffsets: output.ringOffsets.view,
        polygonOffsets: output.polygonOffsets.view,
        featureOffsets: left.featureOffsets.view
      };
    };
    const equal = builder.allocate('equal', 'uint32', set.count);
    builder.graph.add(
      new GPUCanonicalPolygonEquality({
        left: normalize('left', left.positions),
        right: normalize('right', right),
        tolerance: values.tolerance,
        output: equal.view
      })
    );
    return {
      outputs: {...left, equal},
      update: current => {
        const restarted = restartRings(
          set,
          ring => ring * 3 + 2,
          ring => ring % 2 === 1
        );
        // Polygon f has its first vertex moved f * 0.0008 * displacement (the ring layout is unchanged).
        for (let feature = 0; feature < set.count; feature++) {
          const vertex = set.ringOffsets[set.polygonOffsets[set.featureOffsets[feature]]];
          restarted[vertex * 2] += feature * 0.0008 * current.displacement;
        }
        right.write(restarted);
      }
    };
  },
  rebuildOn: ['tolerance'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.equal,
      triangleCount: inputs.set.fillTriangleCount,
      ramp: 'sequential',
      range: [0, 1]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount}
  ]
});
