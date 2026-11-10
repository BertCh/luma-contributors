// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellCover fills two polygons with cells (polyfill). Cells that are provably inside the polygon
// (core) are drawn in one color, border cells in another; the outline is the input polygon.

import {createShoreRing} from '../../../data/hydrography';
import {GPUCellCover} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {addCellFill, getCellFamily} from './support/cells';
import {packPolygonRings} from './support/polygons';
import {parameters} from './cell-cover.parameters';

const OUTPUT_CAPACITY = 2048;

export default defineOperationRun({
  parameters,
  caption: 'Cells covering two polygons; core cells (inside) and border cells differ in color.',
  createInputs: ({world}) => {
    const polygons = packPolygonRings([
      createShoreRing(0, world.width * 0.3, 0.5, 0.42, 32),
      createShoreRing(1, world.width * 0.7, 0.5, 0.42, 32)
    ]);
    // `centers` are where the two polygons sit now: the pointer drags the nearer one.
    return {
      ...polygons,
      origin: polygons.centers,
      centers: Float32Array.from(polygons.centers),
      index: createPointIndex(polygons.centers),
      held: -1
    };
  },
  build(builder, polygons, values) {
    const family = getCellFamily(values.family);
    const resolution = values.resolution + (family === 'h3' ? 0 : 6);
    const positions = builder.upload('positions', 'float32x2', polygons.positions);
    const outline = builder.upload('outline', 'float32x4', polygons.outline);
    const cells = builder.allocate('cells', 'uint32x2', OUTPUT_CAPACITY);
    const featureIds = builder.allocate('feature-ids', 'uint32', OUTPUT_CAPACITY);
    const core = builder.allocate('core', 'uint32', OUTPUT_CAPACITY);
    const status = {
      count: builder.allocate('count', 'uint32', 1),
      overflow: builder.allocate('overflow', 'uint32', 1),
      candidateOverflow: builder.allocate('candidate-overflow', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellCover({
        family,
        resolution,
        candidateCapacity: 1 << 16,
        positions: positions.view,
        featureOffsets: builder.upload('feature-offsets', 'uint32', polygons.featureOffsets).view,
        polygonOffsets: builder.upload('polygon-offsets', 'uint32', polygons.polygonOffsets).view,
        ringOffsets: builder.upload('ring-offsets', 'uint32', polygons.ringOffsets).view,
        output: {
          cells: cells.view,
          featureIds: featureIds.view,
          core: core.view,
          status: {
            count: status.count.view,
            overflow: status.overflow.view,
            candidateOverflow: status.candidateOverflow.view
          }
        }
      })
    );
    return {
      update: current => {
        // The cover appends, so clear rows left over from a larger polygon.
        cells.write(new Uint32Array(OUTPUT_CAPACITY * 2));
        core.write(new Uint32Array(OUTPUT_CAPACITY));
        const scale = (array: Float32Array, stride: number, ownerOf: (index: number) => number) =>
          Float32Array.from(array, (value, index) => {
            const ring = ownerOf(Math.floor(index / stride));
            const axis = index % 2;
            const origin = polygons.origin[ring * 2 + axis];
            return polygons.centers[ring * 2 + axis] + (value - origin) * current.size;
          });
        const ringOf = (vertex: number) => (vertex < polygons.ringOffsets[1] ? 0 : 1);
        positions.write(scale(polygons.positions, 2, ringOf));
        // Outline segments carry two vertices each; both belong to the same ring.
        outline.write(scale(polygons.outline, 4, ringOf));
      },
      outputs: {
        core,
        count: status.count,
        outline,
        fill: addCellFill(builder, family, cells)
      }
    };
  },
  rebuildOn: ['family', 'resolution'],
  // The pointer drags the nearer polygon; the cover follows it on the GPU every frame.
  interaction: {
    kind: 'drag',
    hint: 'drag a polygon',
    rest: ({toUnit, inputs}) => toUnit([inputs.origin[0], inputs.origin[1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.held = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.held);
      inputs.centers.set(inputs.origin);
      inputs.centers[inputs.held * 2] = pointer.x;
      inputs.centers[inputs.held * 2 + 1] = pointer.y;
      const core = peek(outputs.core);
      const count = peek(outputs.count);
      let coreCount = 0;
      if (core && count) for (let row = 0; row < count[0]; row++) coreCount += core[row];
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: count ? `core ${coreCount} · border ${count[0] - coreCount} cells` : 'cover …'
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.core,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.8
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.outlineCount,
      radius: 1.8
    }
  ]
});
