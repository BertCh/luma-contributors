// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkLouvain finds modularity communities among Chicago tracts. Larger observed
// home-to-work flows strengthen a link, and resolution controls the size of communities.

import {GPUNetworkLouvain} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {COMMUTE_FLOWS, TRACT_LOCATIONS, getCommutePositions} from '../../../data/commute-flows';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {fitPositions} from './shared/graphs';
import {parameters} from './network-louvain.parameters';

const RESOLUTION = findParameter(parameters, 'resolution');
const MINIMUM_JOBS = findParameter(parameters, 'minimumJobs');

const NODE_COUNT = 32;

export default defineOperationRun({
  parameters,
  caption: 'Commuter communities among 32 Chicago tracts, weighted by jobs flowing between them.',
  createInputs: ({world}) => {
    // Keep the busiest 32 tracts within the bounded dense solver. Combine reciprocal job flows.
    const totals = new Float64Array(TRACT_LOCATIONS.length);
    for (const [home, work, jobs] of COMMUTE_FLOWS) {
      totals[home] += jobs;
      totals[work] += jobs;
    }
    const tracts = Array.from(totals.keys())
      .sort((first, second) => totals[second] - totals[first])
      .slice(0, NODE_COUNT);
    const pairs = new Float32Array(NODE_COUNT * NODE_COUNT);
    for (const [home, work, jobs] of COMMUTE_FLOWS) {
      const first = tracts.indexOf(home);
      const second = tracts.indexOf(work);
      if (first < 0 || second < 0) continue;
      pairs[first * NODE_COUNT + second] += jobs;
      pairs[second * NODE_COUNT + first] += jobs;
    }
    const allPositions = getCommutePositions(world);
    return {
      pairs,
      positions: fitPositions(
        Float32Array.from(
          tracts.flatMap(tract => [allPositions[tract * 2], allPositions[tract * 2 + 1]])
        ),
        world
      ),
      weights: new Float32Array(NODE_COUNT * NODE_COUNT),
      lines: new Float32Array(NODE_COUNT * NODE_COUNT * 4),
      lineCount: {value: 0}
    };
  },
  build(builder, inputs) {
    const weights = builder.allocate('weights', 'float32', NODE_COUNT * NODE_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const communities = builder.allocate('communities', 'uint32', NODE_COUNT);
    const objectiveTrace = builder.allocate('objective-trace', 'float32', 129);
    const summary = builder.allocate('summary', 'uint32', 4);
    builder.graph.add(
      new GPUNetworkLouvain({
        nodeCount: NODE_COUNT,
        seed: 4332,
        weights: weights.view,
        parameters: parameterBuffer.view,
        output: {
          communities: communities.view,
          objectiveTrace: objectiveTrace.view,
          summary: summary.view
        }
      })
    );
    const ties = builder.upload('ties', 'float32x4', inputs.lines);
    return {
      outputs: {
        communities,
        ties,
        positions: builder.upload('positions', 'float32x2', inputs.positions)
      },
      update: values => {
        // Drop links below the selected job count.
        let lineCount = 0;
        for (let first = 0; first < NODE_COUNT; first++) {
          for (let second = 0; second < NODE_COUNT; second++) {
            const jobs = inputs.pairs[first * NODE_COUNT + second];
            const kept = jobs >= values.minimumJobs ? jobs : 0;
            inputs.weights[first * NODE_COUNT + second] = kept;
            if (kept > 0 && first < second) {
              inputs.lines.set(
                [
                  inputs.positions[first * 2],
                  inputs.positions[first * 2 + 1],
                  inputs.positions[second * 2],
                  inputs.positions[second * 2 + 1]
                ],
                lineCount++ * 4
              );
            }
          }
        }
        inputs.lines.fill(-100, lineCount * 4);
        inputs.lineCount.value = lineCount;
        weights.write(inputs.weights);
        ties.write(inputs.lines);
        parameterBuffer.write(Float32Array.of(values.resolution, 0.000001));
      }
    };
  },
  // x is the Louvain resolution, y the weakest link (in workers) the graph keeps.
  interaction: {
    kind: 'pad',
    hint: 'tune the communities',
    rest: [
      getFractionOf(RESOLUTION, RESOLUTION.default),
      1 - getFractionOf(MINIMUM_JOBS, MINIMUM_JOBS.default)
    ],
    resolve(pointer, {outputs, peek}) {
      const resolution = getParameterAt(RESOLUTION, pointer.unitX);
      const minimumJobs = getParameterAt(MINIMUM_JOBS, 1 - pointer.unitY);
      const labels = peek(outputs.communities);
      const communityCount = labels ? new Set(labels).size : 0;
      return {
        values: {resolution, minimumJobs},
        readout: `resolution ${resolution.toFixed(1)} · minimumJobs ${minimumJobs} · ${communityCount} communities`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.ties, radius: 1.2},
    {layer: 'dots', positions: outputs.positions, values: outputs.communities, radius: 9}
  ]
});
