// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegmentRingAssembly chains loose directed segments into closed rings. The input is the
// shuffled, slightly jittered edge list of a 3 x 2 grid of cells (muted). With
// `cancelOpposingSegments`, the two opposite copies of every shared edge cancel and one outer ring
// remains; without it each cell closes on its own. Endpoints join when they are within
// `vertexTolerance`, so jitter beyond the tolerance leaves rings open.

import {GPUSegmentRingAssembly} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments} from './shared/lines';
import {createCoverage, createRingSegments} from './shared/polygons';
import {parameters} from './segment-ring-assembly.parameters';

function getHash(index: number): number {
  const value = Math.sin(index * 12.9898 + 4.1) * 43758.5453;
  return value - Math.floor(value);
}

export default defineOperationRun({
  parameters,
  caption: 'Loose segments assembled into rings; one color per assembled ring.',
  createInputs: ({random, world}) => {
    const coverage = createCoverage(random, world, {
      rows: 2,
      columns: 3,
      edgeVertexCount: 2,
      wiggle: 0.05
    });
    const segments = createRingSegments(coverage);
    const segmentCount = segments.length / 4;
    const order = Array.from({length: segmentCount}, (_, index) => index).sort(
      (first, second) => getHash(first) - getHash(second)
    );
    const getSegments = (jitter: number) =>
      Float32Array.from({length: segments.length}, (_, slot) => {
        const row = Math.floor(slot / 4);
        const source = order[row];
        return (
          segments[source * 4 + (slot % 4)] +
          (getHash(row * 4 + (slot % 4) + 99) - 0.5) * 2 * jitter
        );
      });
    return {segmentCount, getSegments};
  },
  build(builder, inputs, values) {
    const initial = inputs.getSegments(values.jitter);
    const endpoints = builder.upload('endpoints', 'float32x4', initial);
    const ringCapacity = inputs.segmentCount;
    const ringOffsets = builder.allocate('ring-offsets', 'uint32', ringCapacity + 1);
    const positions = builder.allocate(
      'ring-positions',
      'float32x2',
      inputs.segmentCount + ringCapacity
    );
    const count = builder.allocate('ring-count', 'uint32', 1);
    const overflow = builder.allocate('ring-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUSegmentRingAssembly({
        endpoints: endpoints.view,
        vertexTolerance: values.vertexTolerance,
        cancelOpposingSegments: values.cancelOpposingSegments === 1,
        normalizeWinding: true,
        output: {
          ringOffsets: ringOffsets.view,
          positions: positions.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const drawing = addPackedSegments(builder, 'rings', positions, ringOffsets, {lineCount: count});
    return {
      outputs: {endpoints, ...drawing},
      update: current => endpoints.write(inputs.getSegments(current.jitter))
    };
  },
  rebuildOn: ['cancelOpposingSegments', 'vertexTolerance'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.endpoints, radius: 4},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.2
    }
  ]
});
