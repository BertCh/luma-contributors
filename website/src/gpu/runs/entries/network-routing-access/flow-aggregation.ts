// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFlowAggregation: bin origin-destination records into grid zones and sum the flow between
// zone pairs. The grid shows each zone's outgoing weight; lines join zone centers of the heaviest
// flows.

import {GPUFlowAggregation} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getRasterCell} from '../../pick';
import {addEdgePairs} from './shared/display';
import {fitPositions} from './shared/graphs';
import {parameters} from './flow-aggregation.parameters';
import {COMMUTE_FLOWS, getCommutePositions} from '../../../data/commute-flows';

const TOP_COUNT = 160;
const COLUMN_OPTIONS = [8, 12, 16];

export default defineOperationRun({
  parameters,
  caption: 'Chicago home-to-work job counts, summed by zone and linked to workplace zones.',
  createInputs: ({world}) => {
    const positions = fitPositions(getCommutePositions(world), world);
    // Raster row zero draws at the north edge. Convert north-up positions to raster y.
    const endpoint = (tract: number) => [positions[tract * 2], 1 - positions[tract * 2 + 1]];
    return {
      world,
      zoneMaximums: new Map<string, number>(),
      origins: Float32Array.from(COMMUTE_FLOWS.flatMap(([origin]) => endpoint(origin))),
      destinations: Float32Array.from(
        COMMUTE_FLOWS.flatMap(([, destination]) => endpoint(destination))
      )
    };
  },
  build(builder, inputs, values) {
    const columns = COLUMN_OPTIONS[values.gridSize];
    const rows = Math.max(2, Math.round((columns * 9) / 16));
    const zoneCount = columns * rows;
    const {world} = inputs;
    const origins = builder.upload('origins', 'float32x2', inputs.origins);
    const destinations = builder.upload('destinations', 'float32x2', inputs.destinations);
    const weights = builder.allocate('workers', 'float32', COMMUTE_FLOWS.length);
    const ids = builder.allocate('ids', 'uint32', TOP_COUNT);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const pairOverflow = builder.allocate('pair-overflow', 'uint32', 1);
    const originZones = builder.allocate('flow-origin-zones', 'uint32', TOP_COUNT);
    const destinationZones = builder.allocate('flow-destination-zones', 'uint32', TOP_COUNT);
    const flowWeights = builder.allocate('flow-weights', 'float32', TOP_COUNT);
    const zoneOutWeights = builder.allocate('zone-out-weights', 'float32', zoneCount);
    const zoneInWeights = builder.allocate('zone-in-weights', 'float32', zoneCount);
    builder.graph.add(
      new GPUFlowAggregation({
        zones: {kind: 'grid', bounds: [0, 0, world.width, 1], gridSize: [columns, rows]},
        origins: origins.view,
        destinations: destinations.view,
        weights: weights.view,
        excludeSelfFlows: values.excludeSelfFlows === 1,
        pairCapacity: 4096,
        output: {
          ids: ids.view,
          status: {count: count.view, overflow: overflow.view, pairOverflow: pairOverflow.view},
          flowOriginZoneIds: originZones.view,
          flowDestinationZoneIds: destinationZones.view,
          flowWeights: flowWeights.view,
          zoneOutWeights: zoneOutWeights.view,
          zoneInWeights: zoneInWeights.view
        }
      })
    );
    // Zone centers, drawn with row 0 at the top like the grid; one spare node stays offscreen.
    const centers = new Float32Array((zoneCount + 1) * 2).fill(-100);
    for (let zone = 0; zone < zoneCount; zone++) {
      centers[zone * 2] = ((zone % columns) + 0.5) * (world.width / columns);
      centers[zone * 2 + 1] = 1 - (Math.floor(zone / columns) + 0.5) / rows;
    }
    return {
      outputs: {
        zoneOutWeights,
        flowWeights,
        centers: builder.upload('centers', 'float32x2', centers),
        flows: addEdgePairs(builder, {
          id: 'flows',
          from: originZones,
          to: destinationZones,
          fallbackNode: zoneCount
        }),
        originZones,
        destinationZones
      },
      update: current => {
        weights.write(
          Float32Array.from(COMMUTE_FLOWS, ([, , jobs]) => (jobs >= current.minimumJobs ? jobs : 0))
        );
        originZones.write(new Uint32Array(TOP_COUNT).fill(0xffffffff));
        destinationZones.write(new Uint32Array(TOP_COUNT).fill(0xffffffff));
      }
    };
  },
  rebuildOn: ['gridSize', 'excludeSelfFlows'],
  // Probe: the hovered zone, the flows leaving it and the zones they reach.
  interaction: {
    kind: 'probe',
    hint: 'hover a zone',
    rest: [0.5, 0.5],
    resolve(pointer, {outputs, values, world, peek}) {
      const columns = COLUMN_OPTIONS[values.gridSize];
      const rows = Math.max(2, Math.round((columns * 9) / 16));
      const cell = getRasterCell(columns, rows, [0, 0, world.width, 1], pointer.x, pointer.y);
      const zone = cell.index;
      const center = (index: number): [number, number] => [
        ((index % columns) + 0.5) * (world.width / columns),
        1 - (Math.floor(index / columns) + 0.5) / rows
      ];
      const origins = peek(outputs.originZones);
      const destinations = peek(outputs.destinationZones);
      const weights = peek(outputs.flowWeights);
      const totals = peek(outputs.zoneOutWeights);
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [cell.column * (world.width / columns), 1 - (cell.row + 1) / rows],
          maximum: [(cell.column + 1) * (world.width / columns), 1 - cell.row / rows]
        }
      ];
      let flowCount = 0;
      let strongest = 0;
      if (origins && destinations) {
        for (let flow = 0; flow < origins.length; flow++) {
          if (origins[flow] !== zone) continue;
          flowCount++;
          if (weights) strongest = Math.max(strongest, weights[flow]);
          glyphs.push({shape: 'arrow', from: center(zone), to: center(destinations[flow])});
        }
      }
      return {
        spotlight: null,
        glyphs,
        readout: `zone ${zone} · out ${totals ? Math.round(totals[zone]) : 0} jobs · ${flowCount} flows${strongest ? ` · top ${Math.round(strongest)}` : ''}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const columns = COLUMN_OPTIONS[values.gridSize];
    const rows = Math.max(2, Math.round((columns * 9) / 16));
    return [
      {
        layer: 'raster',
        values: outputs.zoneOutWeights,
        columns,
        rows,
        ramp: 'sequential',
        range: [0, getZoneMaximum(inputs, columns, rows, values.excludeSelfFlows === 1)]
      },
      {
        layer: 'segments',
        nodes: outputs.centers,
        edges: outputs.flows,
        radius: 1.2,
        color: [1, 1, 1, 0.25]
      }
    ];
  }
});

/** Largest zone outflow before the minimum-workers filter, so the filter visibly lowers the totals. */
function getZoneMaximum(
  inputs: {
    origins: Float32Array;
    destinations: Float32Array;
    world: {width: number};
    zoneMaximums: Map<string, number>;
  },
  columns: number,
  rows: number,
  excludeSelf: boolean
): number {
  const key = `${columns}:${excludeSelf}`;
  const known = inputs.zoneMaximums.get(key);
  if (known !== undefined) return known;
  const totals = new Float64Array(columns * rows);
  const getZone = (x: number, y: number) =>
    Math.min(columns - 1, Math.floor((x / inputs.world.width) * columns)) +
    Math.min(rows - 1, Math.floor(y * rows)) * columns;
  COMMUTE_FLOWS.forEach(([, , jobs], flow) => {
    const origin = getZone(inputs.origins[flow * 2], inputs.origins[flow * 2 + 1]);
    const destination = getZone(inputs.destinations[flow * 2], inputs.destinations[flow * 2 + 1]);
    if (!excludeSelf || origin !== destination) totals[origin] += jobs;
  });
  const maximum = Math.max(...totals);
  inputs.zoneMaximums.set(key, maximum);
  return maximum;
}
