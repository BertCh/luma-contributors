// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkFacilityOptimization (backup coverage): pick `selectCount` of the candidate sites so the
// most demand has a backup within `threshold`. Demand dots take the color of their assigned site.

import {GPUNetworkFacilityOptimization} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  addCostMatrix,
  createAccessScene,
  createFacilityDrag,
  getFacilityMarkers,
  moveFacility,
  uploadSearch
} from './shared/access';
import type {RunGlyph} from '../../interaction';
import {addFlaggedIds} from './shared/display';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './network-facility-optimization.parameters';

const DEMAND_COUNT = 40;
const CANDIDATE_COUNT = 8;

export default defineOperationRun({
  parameters,
  caption: 'Candidate sites chosen for backup coverage (bright) among the demand locations.',
  createInputs: ({random, world}) => {
    const scene = createAccessScene(random, world, CANDIDATE_COUNT, DEMAND_COUNT);
    // Demand clusters around the candidate sites, so a backup-coverage selection is feasible.
    scene.demandNodes.forEach((_, demand) => {
      const site = demand % CANDIDATE_COUNT;
      const node = scene.network.findNearestNode(
        scene.facilityPositions[site * 2] + (random() - 0.5) * 0.4,
        scene.facilityPositions[site * 2 + 1] + (random() - 0.5) * 0.4
      );
      scene.demandNodes[demand] = node;
      scene.demandPositions[demand * 2] = scene.network.nodes[node * 2];
      scene.demandPositions[demand * 2 + 1] = scene.network.nodes[node * 2 + 1];
    });
    return {...scene, drag: createFacilityDrag(scene)};
  },
  build(builder, scene, values) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const candidateNodes = builder.upload('candidate-nodes', 'uint32', scene.facilityNodes);
    const candidates = builder.upload('candidates', 'float32x2', scene.facilityPositions);
    const matrix = addCostMatrix(
      builder,
      search,
      scene.demandNodes,
      network.nodeCount,
      candidateNodes
    );
    const threshold = builder.allocate('threshold', 'float32', 1);
    const coverage = builder.allocate('coverage', 'uint32', DEMAND_COUNT);
    const selected = builder.allocate('selected', 'uint32', CANDIDATE_COUNT);
    const assignments = builder.allocate('assignments', 'uint32', DEMAND_COUNT);
    const objective = builder.allocate('objective', 'float32', 1);
    const status = builder.allocate('status', 'uint32', 4);
    builder.graph.add(
      new GPUNetworkFacilityOptimization({
        operation: 'backup-coverage',
        facilityCount: CANDIDATE_COUNT,
        selectCount: Math.round(values.selectCount),
        costs: matrix.view,
        demand: builder.upload('demand', 'float32', scene.population).view,
        capacities: builder.upload(
          'capacities',
          'float32',
          new Float32Array(CANDIDATE_COUNT).fill(1000)
        ).view,
        threshold: threshold.view,
        output: {
          selectedFacilities: selected.view,
          assignments: assignments.view,
          objective: objective.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        assignments,
        selected,
        matrix,
        selectedIds: addFlaggedIds(builder, {
          id: 'selected',
          flags: selected,
          count: CANDIDATE_COUNT
        }),
        coverage,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        candidates
      },
      update: current => {
        candidateNodes.write(scene.facilityNodes);
        candidates.write(scene.facilityPositions);
        search.costLimit.write(Float32Array.of(4));
        threshold.write(Float32Array.of(current.threshold));
      }
    };
  },
  rebuildOn: ['selectCount'],
  // Drag: the candidate site nearest the pointer moves to the nearest intersection; the chosen sites,
  // the threshold rings and the demand's backup coverage follow.
  interaction: {
    kind: 'drag',
    hint: 'drag a candidate site',
    rest: ({inputs, toUnit}) =>
      toUnit([inputs.drag.defaultPositions[0], inputs.drag.defaultPositions[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const move = moveFacility(inputs, inputs.drag, pointer.x, pointer.y);
      const selected = peek(outputs.selected);
      const costs = peek(outputs.matrix);
      const glyphs: RunGlyph[] = [
        {
          shape: 'crosshair',
          at: [inputs.network.nodes[move.node * 2], inputs.network.nodes[move.node * 2 + 1]],
          size: 14
        }
      ];
      let backup = 0;
      let chosen = 0;
      if (selected && costs) {
        const classes = new Uint32Array(DEMAND_COUNT);
        for (let site = 0; site < CANDIDATE_COUNT; site++) {
          if (selected[site] !== 1) continue;
          chosen++;
          for (let demand = 0; demand < DEMAND_COUNT; demand++) {
            if (costs[demand * CANDIDATE_COUNT + site] <= values.threshold) classes[demand]++;
          }
        }
        for (let demand = 0; demand < DEMAND_COUNT; demand++) {
          classes[demand] = Math.min(2, classes[demand]);
          if (classes[demand] === 2) backup++;
        }
        outputs.coverage.write(classes);
      }
      return {
        rerun: move.changed,
        glyphs,
        readout: chosen
          ? `site ${move.facility + 1} · threshold ${values.threshold.toFixed(2)} · backup ${Math.round((100 * backup) / DEMAND_COUNT)}%`
          : `site ${move.facility + 1} · no selection covers every demand`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    // Demand by how many chosen sites lie within the threshold: none, one, or backup (two or more).
    {
      layer: 'dots',
      positions: outputs.demands,
      values: outputs.coverage,
      ramp: 'categorical',
      categories: [
        [...theme.categories[2], 1],
        [...theme.muted, 1],
        [...theme.categories[4], 1]
      ],
      radius: 8
    },
    ...getFacilityMarkers(outputs.candidates, theme, {ids: outputs.selectedIds, radius: 11})
  ]
});
