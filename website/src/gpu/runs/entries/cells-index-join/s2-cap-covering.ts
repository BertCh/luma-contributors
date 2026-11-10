// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2CapCovering covers spherical caps (a center on the unit sphere and an angular radius) with
// uniform-level S2 cells: all cells that touch the cap (exterior) or lie inside it (interior).

import {GPUS2CapCovering} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {getHonshuEventSites, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill} from './support/cells';
import {createCircleOutline, getUnitVector} from './support/polygons';
import {parameters} from './s2-cap-covering.parameters';

const CAPACITY = 256;
const RADIUS_PARAMETER = findParameter(parameters, 'radius');
const LONGITUDE_SCALE = Math.cos((((HONSHU_WINDOW[1] + HONSHU_WINDOW[3]) / 2) * Math.PI) / 180);
const DEGREES_TO_RADIANS = Math.PI / 180;

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'S2 covers for hypothetical search radii around three observed Honshu epicenters.',
  createInputs: () => {
    const centers = getHonshuEventSites([
      [141.2, 36.6],
      [142.3, 37.5],
      [141.6, 38.1]
    ]);
    return {
      centers,
      index: createPointIndex(centers.flat()),
      held: -1
    };
  },
  build(builder, inputs, values) {
    const capCount = inputs.centers.length;
    const caps = builder.allocate('caps', 'float32x4', capCount);
    const outline = builder.allocate('cap-outline', 'float32x4', capCount * 64);
    const cells = builder.allocate('cells', 'uint32x2', capCount * CAPACITY);
    const counts = builder.allocate('counts', 'uint32x2', capCount);
    const status = builder.allocate('cap-status', 'uint32', capCount);
    builder.graph.add(
      new GPUS2CapCovering({
        caps: caps.view,
        level: values.level,
        mode: values.mode === 0 ? 'exterior' : 'interior',
        capacity: CAPACITY,
        output: {cells: cells.view, counts: counts.view, status: status.view}
      })
    );
    return {
      outputs: {
        outline,
        counts,
        sources: builder.upload(
          'sources',
          'uint32',
          Uint32Array.from({length: capCount * CAPACITY}, (_, row) => Math.floor(row / CAPACITY))
        ),
        fill: addCellFill(builder, 's2', cells)
      },
      update: current => {
        caps.write(
          Float32Array.from(
            inputs.centers.flatMap(([longitude, latitude]) => [
              ...getUnitVector(longitude, latitude),
              current.radius * DEGREES_TO_RADIANS
            ])
          )
        );
        outline.write(
          Float32Array.from(
            inputs.centers.flatMap(center => [...createCircleOutline(center, current.radius)])
          )
        );
      }
    };
  },
  rebuildOn: ['level', 'mode'],
  // Reach: the distance from the pointer to the nearest cap center is the radius of the caps.
  interaction: {
    kind: 'reach',
    hint: 'set the cap radius',
    rest: ({toUnit, inputs}) => {
      const [longitude, latitude] = inputs.centers[0];
      return toUnit([longitude + RADIUS_PARAMETER.default / LONGITUDE_SCALE, latitude]);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.held = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.held);
      const [longitude, latitude] = inputs.centers[inputs.held];
      // Degrees of arc on the ground: longitude shrinks with the cosine of the latitude.
      const arc = Math.hypot((pointer.x - longitude) * LONGITUDE_SCALE, pointer.y - latitude);
      const radius = snapToParameter(RADIUS_PARAMETER, arc);
      const counts = peek(outputs.counts);
      return {
        values: {radius},
        glyphs: [
          {shape: 'line', from: [longitude, latitude], to: [pointer.x, pointer.y]},
          {shape: 'dot', at: [longitude, latitude]},
          {shape: 'crosshair', at: [pointer.x, pointer.y], size: 6}
        ],
        readout: `cap radius ${radius.toFixed(2)}° · ${counts ? counts[inputs.held * 2] : '…'} cells at level ${values.level}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.sources,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.7
    },
    {
      layer: 'triangles',
      positions: outputs.fill.boundaries,
      triangles: outputs.fill.wireframe,
      count: outputs.fill.wireframeCount,
      radius: 1,
      color: [1, 1, 1, 0.5]
    },
    {layer: 'lines', context: true, segments: outputs.outline, radius: 1.8}
  ]
});
