// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonPrecisionReduction rounds polygons to a grid and rebuilds them valid. The muted
// outlines are the input stars; the rings the operation wrote are drawn one color per ring.

import {GPUPolygonPrecisionReduction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {findNearestSegment} from '../../pick';
import {createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {
  createStaleGuard,
  findPointAtDistance,
  getBinCenter,
  getBinIndex,
  getLatticeGlyphs,
  readFresh
} from './shared/pointer-support';
import {addOutputEdges, addRingCategories, allocateLattice} from './shared/shape-segments';
import {createTopologyBuffers} from './shared/topology-output';
import {parameters} from './polygon-precision-reduction.parameters';

/** gridSize is a constructor prop: the pointer settles on one of these prebuilt values. */
const GRID_SIZES = [0.015, 0.03, 0.06, 0.09, 0.12];
/** Distance from the nearest polygon edge that spans the five bins. */
const REACH = 0.25;

export default defineOperationRun({
  parameters,
  caption: 'Polygons rounded to the precision grid, rebuilt valid.',
  buildCache: 5,
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {
      rows: 2,
      columns: 4,
      vertexCount: 14,
      irregularity: 0.4,
      closeRings: true
    });
    return {
      set,
      world,
      guard: createStaleGuard(),
      restPoint: findPointAtDistance(
        (x, y) => findNearestSegment(set.outline, x, y, set.outlineCount)?.distance ?? null,
        getBinCenter(GRID_SIZES.indexOf(0.06), REACH, GRID_SIZES.length),
        world
      )
    };
  },
  build(builder, {set, world}, values) {
    const polygons = uploadPolygonSet(builder, set);
    const topology = createTopologyBuffers(builder, {segments: 1024, rings: 64, vertices: 2048});
    builder.graph.add(
      new GPUPolygonPrecisionReduction({
        polygons: {
          kind: 'polygons',
          positions: polygons.positions.view,
          featureOffsets: polygons.featureOffsets.view,
          polygonOffsets: polygons.polygonOffsets.view,
          ringOffsets: polygons.ringOffsets.view
        },
        gridSize: values.gridSize,
        mode: 'valid-output',
        lattice: builder.allocate('lattice', 'float32', 4).view,
        intersectionCapacity: 512,
        output: topology.output
      })
    );
    return {
      outputs: {
        ...polygons,
        ringOffsets: topology.ringOffsets,
        lattice: allocateLattice(builder, 'grid-lattice', values.gridSize, world),
        reduced: topology.positions,
        reducedEdges: addOutputEdges(
          builder,
          'reduced',
          topology.ringOffsets,
          topology.vertexCapacity,
          true
        ),
        categories: addRingCategories(
          builder,
          'reduced',
          topology.ringOffsets,
          topology.vertexCapacity
        )
      }
    };
  },
  // The pointer's distance from the nearest polygon edge picks the lattice; the glyph shows where the pointer rounds to.
  interaction: {
    kind: 'reach',
    hint: 'move away from the polygons',
    rest: ({inputs, toUnit}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const hit = findNearestSegment(
        inputs.set.outline,
        pointer.x,
        pointer.y,
        inputs.set.outlineCount
      );
      const gridSize = GRID_SIZES[getBinIndex(hit?.distance ?? 0, REACH, GRID_SIZES.length)];
      const offsets = readFresh(inputs.guard, String(values.gridSize), peek(outputs.ringOffsets));
      let ringCount = 0;
      while (offsets && offsets[ringCount + 1] > offsets[ringCount]) ringCount++;
      const point = [pointer.x, pointer.y] as const;
      return {
        settle: {gridSize},
        glyphs: getLatticeGlyphs(point, gridSize),
        readout:
          gridSize === values.gridSize
            ? `gridSize ${gridSize} · ${offsets ? ringCount : '...'} rings`
            : `gridSize ${values.gridSize} -> ${gridSize}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.lattice, radius: 0.6, alpha: 0.3},
    {layer: 'segments', context: true, nodes: outputs.positions, edges: outputs.edges},
    {
      layer: 'segments',
      nodes: outputs.reduced,
      edges: outputs.reducedEdges,
      values: outputs.categories,
      radius: 2.4
    }
  ]
});
