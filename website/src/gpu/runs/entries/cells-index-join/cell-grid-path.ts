// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellGridPath finds the H3 grid path between origin and destination cells. The cells of every
// path (origin first, destination last) are decoded with GPUCellGeometry and filled; the white dots
// are the origin and destination points that were keyed.

import {GPUCellGridPath} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEventSites, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill, addPointKeys} from './support/cells';
import {getWindowCenter} from './support/pointer';
import {parameters} from './cell-grid-path.parameters';

const RESOLUTION = 5;
const LONGITUDE_SCALE = Math.cos((((HONSHU_WINDOW[1] + HONSHU_WINDOW[3]) / 2) * Math.PI) / 180);

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'H3 paths from four Honshu epicenters to the epicenters around the pointer.',
  createInputs: () => ({
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number],
    origins: Float32Array.from(
      getHonshuEventSites([
        [140.8, 36.45],
        [142.1, 38.05],
        [141.0, 37.35],
        [143.4, 37.85]
      ]).flat()
    )
  }),
  build(builder, inputs, values) {
    const pairCount = inputs.origins.length / 2;
    const origins = builder.upload('origins', 'float32x2', inputs.origins);
    const destinations = builder.allocate('destinations', 'float32x2', pairCount);
    const originCells = addPointKeys(builder, 'h3', RESOLUTION, origins, 'origin-keys');
    const destinationCells = addPointKeys(
      builder,
      'h3',
      RESOLUTION,
      destinations,
      'destination-keys'
    );
    const length = values.maximumPathLength;
    const pathCells = builder.allocate('path-cells', 'uint32x2', pairCount * length);
    const counts = builder.allocate('counts', 'uint32', pairCount);
    builder.graph.add(
      new GPUCellGridPath({
        family: 'h3',
        origins: originCells.view,
        destinations: destinationCells.view,
        maximumPathLength: length,
        output: {cells: pathCells.view, counts: counts.view}
      })
    );
    return {
      outputs: {
        counts,
        origins,
        destinations,
        pairs: builder.upload(
          'pairs',
          'uint32',
          Uint32Array.from({length: pairCount * length}, (_, row) => Math.floor(row / length))
        ),
        fill: addCellFill(builder, 'h3', pathCells)
      },
      update: current => {
        // The pointer is the destination region: each pair ends at the measured epicenter nearest
        // its own spot on a fan of radius `spread` around the pointer, so both endpoints are observations.
        const targets: [number, number][] = [];
        for (let pair = 0; pair < pairCount; pair++) {
          const angle = ((pair + 0.5) / pairCount) * 2 * Math.PI;
          targets.push([
            inputs.pointer[0] + (Math.cos(angle) * current.spread) / LONGITUDE_SCALE,
            inputs.pointer[1] + Math.sin(angle) * current.spread
          ]);
        }
        destinations.write(Float32Array.from(getHonshuEventSites(targets).flat()));
      }
    };
  },
  rebuildOn: ['maximumPathLength'],
  // The pointer is where the paths go: every origin routes to the epicenters around it.
  interaction: {
    kind: 'place',
    hint: 'choose a destination',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const counts = peek(outputs.counts);
      const lengths = counts ? [...counts].filter(length => length > 0) : [];
      const longest = lengths.length > 0 ? Math.max(...lengths) : 0;
      return {
        rerun: true,
        glyphs: [
          {shape: 'crosshair', at: inputs.pointer},
          {shape: 'ring', at: inputs.pointer, radius: values.spread, dashed: true}
        ],
        readout: `paths up to ${longest} cells (max ${values.maximumPathLength})`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.pairs,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.85
    },
    {layer: 'dots', positions: outputs.origins, radius: 4, color: [1, 1, 1, 1]},
    {layer: 'dots', positions: outputs.destinations, radius: 4, color: [1, 1, 1, 1]}
  ]
});
