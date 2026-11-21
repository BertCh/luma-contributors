// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBoundedFacilityAllocation: exhaustive p-center and capacitated p-median over a small candidate
// set. Demand dots take the color of the facility that serves them; capacities bound how many can.

import {GPUBoundedFacilityAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUNetworkCostMatrix} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {createAccessScene, getFacilityMarkers} from './shared/access';
import {addFlaggedIds} from './shared/display';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './bounded-facility-allocation.parameters';

const DEMAND_COUNT = 48;
const CANDIDATE_COUNT = 8;

export default defineOperationRun({
  parameters,
  caption:
    'The chosen facilities (bright) and the demand each serves. Capacitated mode assigns demand in row order.',
  createInputs: ({random, world}) =>
    createAccessScene(random, world, CANDIDATE_COUNT, DEMAND_COUNT),
  build(builder, scene, values) {
    const {network} = scene;
    const {topology} = uploadNetworkTopology(builder, network);
    const matrix = builder.allocate('cost-matrix', 'float32', DEMAND_COUNT * CANDIDATE_COUNT);
    const costsComplete = builder.allocate('costs-complete', 'uint32', 1);
    builder.graph.add(
      new GPUNetworkCostMatrix({
        topology,
        startNodes: builder.upload('start-nodes', 'uint32', scene.demandNodes).view,
        targetNodes: builder.upload('target-nodes', 'uint32', scene.facilityNodes).view,
        maximumIterations: 48,
        output: {costs: matrix.view, status: {converged: costsComplete.view}}
      })
    );
    const capacities = builder.allocate('capacities', 'float32', CANDIDATE_COUNT);
    const selected = builder.allocate('selected', 'uint32', CANDIDATE_COUNT);
    const assignments = builder.allocate('assignments', 'uint32', DEMAND_COUNT);
    const objective = builder.allocate('objective', 'float32', 1);
    const status = builder.allocate('status', 'uint32', 3);
    const complete = builder.allocate('allocation-complete', 'uint32', 1);
    const selectCount = Math.round(values.selectCount);
    builder.graph.add(
      new GPUBoundedFacilityAllocation({
        operation: values.operation === 0 ? 'p-center' : 'capacitated-p-median',
        facilityCount: CANDIDATE_COUNT,
        selectCount,
        costs: matrix.view,
        costsComplete: costsComplete.view,
        demand: builder.upload(
          'demand',
          'float32',
          Float32Array.from(scene.population, value => Math.round(value))
        ).view,
        capacities: capacities.view,
        output: {
          selectedFacilities: selected.view,
          assignments: assignments.view,
          objective: objective.view,
          status: status.view,
          ports: {status: {complete: complete.view}}
        }
      })
    );
    const totalDemand = scene.population.reduce((sum, value) => sum + Math.round(value), 0);
    return {
      outputs: {
        ...uploadRoads(builder, network),
        assignments,
        selected,
        complete,
        selectedIds: addFlaggedIds(builder, {
          id: 'selected',
          flags: selected,
          count: CANDIDATE_COUNT
        }),
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        candidates: builder.upload('candidates', 'float32x2', scene.facilityPositions)
      },
      update: current => {
        // Capacity per facility as a share of the average load for the selected count.
        capacities.write(
          new Float32Array(CANDIDATE_COUNT).fill((totalDemand / selectCount) * current.capacity)
        );
      }
    };
  },
  async afterRun({outputs}) {
    const bytes = await outputs.complete.buffer.readAsync();
    return {complete: new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0] !== 0};
  },
  rebuildOn: ['operation', 'selectCount'],
  // Probe: the hovered chosen facility, the demand it serves and its load against its capacity.
  interaction: {
    kind: 'probe',
    hint: 'hover a facility',
    rest: ({inputs, toUnit}) => toUnit([inputs.facilityPositions[0], inputs.facilityPositions[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const complete = peek(outputs.complete);
      const selected = peek(outputs.selected);
      const assignments = peek(outputs.assignments);
      if (!complete) return {};
      if (!complete[0])
        return {readout: 'Assignments unavailable due to incomplete costs or invalid inputs'};
      if (!selected || !assignments) return {};
      let facility = 0;
      let nearestDistance = Infinity;
      for (let site = 0; site < CANDIDATE_COUNT; site++) {
        if (selected[site] !== 1) continue;
        const distance = Math.hypot(
          inputs.facilityPositions[site * 2] - pointer.x,
          inputs.facilityPositions[site * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          facility = site;
        }
      }
      const at: [number, number] = [
        inputs.facilityPositions[facility * 2],
        inputs.facilityPositions[facility * 2 + 1]
      ];
      const glyphs: RunGlyph[] = [{shape: 'ring', at, radius: 0.04}];
      let served = 0;
      let load = 0;
      for (let demand = 0; demand < assignments.length; demand++) {
        if (assignments[demand] !== facility) continue;
        served++;
        load += Math.round(inputs.population[demand]);
        if (glyphs.length < 200) {
          glyphs.push({
            shape: 'line',
            from: at,
            to: [inputs.demandPositions[demand * 2], inputs.demandPositions[demand * 2 + 1]]
          });
        }
      }
      const total = inputs.population.reduce((sum, value) => sum + Math.round(value), 0);
      const capacity = (total / Math.round(values.selectCount)) * values.capacity;
      const loadText = values.operation === 0 ? `${load}` : `${load} of ${capacity.toFixed(0)}`;
      return {
        spotlight: null,
        glyphs,
        readout: `facility ${facility + 1} · serves ${served} · load ${loadText}`
      };
    }
  },
  draw: ({outputs, extra, theme}) => [
    getRoadContext(outputs),
    ...(extra?.complete
      ? [
          {
            layer: 'dots' as const,
            positions: outputs.demands,
            values: outputs.assignments,
            radius: 7
          },
          ...getFacilityMarkers(outputs.candidates, theme, {ids: outputs.selectedIds, radius: 11})
        ]
      : [{layer: 'dots' as const, positions: outputs.demands, context: true, radius: 7}])
  ]
});
