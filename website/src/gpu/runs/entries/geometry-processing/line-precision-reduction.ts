// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULinePrecisionReduction rounds a line to a grid. The output vertex count is the last line
// offset the operation wrote, copied into a count buffer so the path draws exactly what was kept.

import {GPULinePrecisionReduction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findNearestOnPolyline} from '../../pick';
import {parameters} from './line-precision-reduction.parameters';
import {createIdentityIds, createSingleLine} from './shared/polygon-set';
import {
  createStaleGuard,
  findPointAtDistance,
  getBinCenter,
  getBinIndex,
  getLatticeGlyphs,
  readFresh
} from './shared/pointer-support';
import {addKernel, allocateLattice} from './shared/shape-segments';

/** gridSize is a constructor prop: the pointer settles on one of these prebuilt values. */
const GRID_SIZES = [0.02, 0.04, 0.06, 0.1, 0.16];
/** Distance from the line that spans the five gridSize bins. */
const REACH = 0.34;

export default defineOperationRun({
  parameters,
  caption: 'The line rounded to the precision grid; collapsed lines follow the chosen mode.',
  buildCache: 5,
  createInputs: ({random, world}) => {
    const line = createSingleLine(random, world, 120);
    return {
      ...line,
      world,
      guard: createStaleGuard(),
      restPoint: findPointAtDistance(
        (x, y) => findNearestOnPolyline(line.positions, x, y)?.distance ?? null,
        getBinCenter(GRID_SIZES.indexOf(0.06), REACH, GRID_SIZES.length),
        world
      )
    };
  },
  build(builder, line, values) {
    const positions = builder.upload('positions', 'float32x2', line.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const reduced = builder.allocate('reduced', 'float32x2', line.count);
    const reducedOffsets = builder.allocate('reduced-offsets', 'uint32', 2);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPULinePrecisionReduction({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        gridSize: values.gridSize,
        mode: 'valid-output',
        output: {positions: reduced.view, lineOffsets: reducedOffsets.view, status: status.view}
      })
    );
    const count = builder.allocate('count', 'uint32', 1);
    addKernel(builder, {
      id: 'copy-count',
      invocationCount: 1,
      bindings: [
        {name: 'offsets', buffer: reducedOffsets, access: 'read'},
        {name: 'count', buffer: count, access: 'read_write'}
      ],
      body: 'count[countOffset] = offsets[offsetsOffset + 1u];'
    });
    return {
      outputs: {
        positions,
        reduced,
        count,
        lattice: allocateLattice(builder, 'lattice', values.gridSize, line.world),
        ids: builder.upload('ids', 'uint32', createIdentityIds(line.count))
      }
    };
  },
  // The pointer's distance from the line picks the lattice; the glyph shows where the pointer rounds to.
  interaction: {
    kind: 'reach',
    hint: 'move away from the line',
    rest: ({inputs, toUnit}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, values, peek, outputs}) {
      const hit = findNearestOnPolyline(inputs.positions, pointer.x, pointer.y);
      const gridSize = GRID_SIZES[getBinIndex(hit?.distance ?? 0, REACH, GRID_SIZES.length)];
      const kept = readFresh(inputs.guard, String(values.gridSize), peek(outputs.count))?.[0];
      const point = [pointer.x, pointer.y] as const;
      return {
        settle: {gridSize},
        glyphs: [
          ...getLatticeGlyphs(point, gridSize),
          ...(hit ? [{shape: 'line' as const, from: point, to: hit.point, dashed: true}] : [])
        ],
        readout:
          gridSize === values.gridSize
            ? `gridSize ${gridSize} · ${inputs.count} -> ${kept ?? '...'} vertices`
            : `gridSize ${values.gridSize} -> ${gridSize}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.lattice, radius: 0.6, alpha: 0.3},
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: inputs.count
    },
    {layer: 'paths', nodes: outputs.reduced, ids: outputs.ids, count: outputs.count, radius: 2}
  ]
});
