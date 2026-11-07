// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSnapshotTap placed inside a real relaxation. The tap follows the last node of iteration
// `afterNode` and copies `views` of the graph into an arena. The top shows the described graph:
// the tap is the red copy node with its explicit afterNode edge. The row of ticks is the frame's
// own `iteration` coordinate, which does not have to match where the tap sits, and below are the
// views the tap captured, read back from the arena.

import {
  describeGPUCommandGraph,
  GPUComputationRecorder,
  GPUSnapshotArena
} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {
  buildRelaxation,
  createDiagram,
  createRecordTracts,
  getGridCell,
  groupCapturesBySlot,
  isObservationTap,
  RELAXATION_SLOT_NAMES,
  ShapeBatch
} from '../../record-demo';
import {parameters} from './snapshot-tap.parameters';

const COORDINATE_COUNT = 10;

export default defineOperationRun({
  parameters,
  caption: 'The tap node in the graph, and the views it copied into the arena.',
  createInputs: ({random, world}) => ({tracts: createRecordTracts(random), world}),
  build(builder, {tracts}, values) {
    const recorder = new GPUComputationRecorder();
    const slotNames = RELAXATION_SLOT_NAMES.slice(0, values.views);
    const arena = builder.own(
      new GPUSnapshotArena(builder.device, {
        id: 'tap-arena',
        slots: slotNames.map(name => ({name, byteLength: tracts.count * 4, elementType: 'f32'})),
        capacity: 1
      })
    );
    const tapped = Math.min(values.afterNode, values.iterations);
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      recorder,
      deviation: values.views >= 3,
      scope: iteration => ({namespace: `iteration-${iteration}`}),
      tap: iteration =>
        iteration.iteration === tapped
          ? {
              arena,
              views: Object.fromEntries(
                slotNames.map(name => [name, iteration[name] ?? iteration.state])
              ),
              label: `after iteration ${tapped}`,
              iteration: values.iteration
            }
          : undefined
    });
    return {outputs: {arena, recorder}};
  },
  async afterRun({compiled, outputs, inputs, values, upload}) {
    const {tracts, world} = inputs;
    const described = describeGPUCommandGraph(compiled, {recorder: outputs.recorder}).graph;
    const tapIds = new Set(described.nodes.filter(isObservationTap).map(node => node.id));
    const diagram = createDiagram(
      upload,
      'tap-graph',
      described,
      {x0: 0.05 * world.width, y0: 0.54, x1: 0.95 * world.width, y1: 0.97},
      node => (tapIds.has(node.id) ? 1 : 0),
      edge => !tapIds.has(edge.to.node.id)
    );
    // The dependency edge into the tap, highlighted.
    const highlight = new ShapeBatch();
    for (const edge of diagram.layout.edges) {
      if (!tapIds.has(edge.to.node.id)) continue;
      highlight.addArrow(
        edge.from.x1,
        (edge.from.y0 + edge.from.y1) / 2,
        edge.to.x0,
        (edge.to.y0 + edge.to.y1) / 2,
        0.014
      );
    }

    const captures = await outputs.arena.read();
    const bySlot = groupCapturesBySlot(captures);
    const slotNames = RELAXATION_SLOT_NAMES.slice(0, values.views);
    const maps = RELAXATION_SLOT_NAMES.map(() => new ShapeBatch());
    const mapArea = {x0: 0.05 * world.width, x1: 0.95 * world.width, y0: 0.03, y1: 0.43};
    let maximumResidual = 1e-6;
    for (const frame of bySlot.get('residual') ?? []) {
      for (const value of Array.from(frame.values))
        maximumResidual = Math.max(maximumResidual, value);
    }
    slotNames.forEach((name, column) => {
      const frame = bySlot.get(name)?.[0];
      if (frame) {
        maps[column].addTractMap(
          tracts,
          getGridCell(mapArea, slotNames.length, 1, column, 0, 0.02),
          frame.values
        );
      }
    });
    // The iteration coordinate axis: one tick per coordinate, the frame's own tick taller.
    const ticks = new ShapeBatch();
    const marker = new ShapeBatch();
    const frameIteration = captures[0]?.iteration ?? 0;
    const axis = {x0: 0.05 * world.width, x1: 0.95 * world.width};
    for (let coordinate = 0; coordinate < COORDINATE_COUNT; coordinate++) {
      const x = axis.x0 + ((coordinate + 0.5) / COORDINATE_COUNT) * (axis.x1 - axis.x0);
      ticks.addSegment(x, 0.455, x, 0.505);
      // A bar from the start of the axis to the frame's own coordinate.
      if (coordinate === frameIteration) {
        marker.addRectangle({x0: axis.x0, x1: x, y0: 0.468, y1: 0.492}, 1);
      }
    }
    return {
      diagram,
      highlight: highlight.uploadSegments(upload, 'tap-edge'),
      highlightCount: highlight.segmentCount,
      ticks: ticks.uploadSegments(upload, 'coordinate-ticks'),
      tickCount: ticks.segmentCount,
      marker: marker.uploadShapes(upload, 'coordinate-marker'),
      maps: maps.map((batch, index) => batch.uploadShapes(upload, `map-${index}`)),
      maximumResidual
    };
  },
  draw: ({extra, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: extra.diagram.edges,
      count: extra.diagram.edgeCount,
      radius: 1
    },
    {
      layer: 'polygons',
      positions: extra.diagram.shapes.positions,
      indices: extra.diagram.shapes.indices,
      owners: extra.diagram.shapes.owners,
      values: extra.diagram.shapes.categories,
      triangleCount: extra.diagram.shapes.triangleCount,
      categories: [theme.muted, theme.categories[2]]
    },
    {
      layer: 'lines',
      segments: extra.highlight,
      count: extra.highlightCount,
      radius: 2.2,
      color: [...theme.categories[2], 1]
    },
    {layer: 'lines', context: true, segments: extra.ticks, count: extra.tickCount, radius: 1.4},
    {
      layer: 'polygons',
      positions: extra.marker.positions,
      indices: extra.marker.indices,
      owners: extra.marker.owners,
      values: extra.marker.categories,
      triangleCount: extra.marker.triangleCount,
      categories: [theme.muted, theme.categories[1]]
    },
    ...extra.maps.map((map, index): import('../../define-run').RunLayer => ({
      layer: 'polygons',
      positions: map.positions,
      indices: map.indices,
      owners: map.owners,
      values: map.values,
      triangleCount: map.triangleCount,
      range: index === 0 ? [0, 1] : index === 1 ? [0, extra.maximumResidual] : [-0.4, 0.4],
      ramp: index === 2 ? 'diverging' : 'sequential'
    }))
  ]
});
