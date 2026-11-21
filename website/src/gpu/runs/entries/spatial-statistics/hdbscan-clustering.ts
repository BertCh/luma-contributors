// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHDBSCAN (HDBSCAN): builds the density hierarchy and keeps the stable
// clusters, so clusters of different density are found without an epsilon. Noise points are
// drawn in the muted color; each cluster gets its own color. Hovering a point rings its core
// distance (the distance to its minimumSamples-th neighbor) and lights the members of its cluster.

import {GPUHDBSCAN} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './hdbscan-clustering.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Stable density clusters; noise is muted.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 700, 5, 0.12);
    return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const labels = builder.allocate('labels', 'uint32', points.count);
    const coreDistances = builder.allocate('core-distances', 'float32', points.count);
    const members = builder.allocate('members', 'uint32', points.count);
    builder.graph.add(
      new GPUHDBSCAN({
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        minimumClusterSize: values.minimumClusterSize,
        minimumSamples: values.minimumSamples,
        output: {labels: labels.view, coreDistances: coreDistances.view}
      })
    );
    return {
      outputs: {
        labels,
        coreDistances,
        members,
        positions: builder.upload('points', 'float32x2', points.positions)
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.6, 0.45],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const labels = peek(outputs.labels);
      const coreDistances = peek(outputs.coreDistances);
      const mask = new Uint32Array(inputs.count);
      if (row < 0 || !labels || !coreDistances) {
        outputs.members.write(mask);
        return {};
      }
      const label = labels[row];
      const isNoise = label === 0xffffffff;
      let memberCount = 0;
      for (let other = 0; other < inputs.count; other++) {
        if (!isNoise && labels[other] === label) {
          mask[other] = 1;
          memberCount++;
        }
      }
      outputs.members.write(mask);
      const at = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      return {
        glyphs: [
          {shape: 'ring', at, radius: coreDistances[row], dashed: true},
          {shape: 'crosshair', at, size: 12}
        ],
        readout: `${isNoise ? 'noise' : `cluster ${label} · ${memberCount} points`} · core distance ${coreDistances[
          row
        ].toFixed(3)} (k ${values.minimumSamples})`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 4.5},
    // The hovered cluster's members, ringed: the faded mask hides every other point.
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.members,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      alpha: 0.6,
      radius: 7.5
    }
  ]
});
