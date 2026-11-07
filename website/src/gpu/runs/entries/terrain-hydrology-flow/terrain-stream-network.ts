// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainStreamNetwork on the D8 directions and stream mask of GPUTerrainFlow: reach polylines
// between headwaters, confluences and outlets, drawn from the vertex positions it wrote.

import {GPUTerrainStreamNetwork} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-stream-network.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const REACH_CAPACITY = 1024;
const VERTEX_CAPACITY = 16384;

export default defineOperationRun({
  parameters,
  caption: 'Reach polylines extracted from the stream mask; they split at every confluence.',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    const sourceCells = builder.allocate('source-cells', 'uint32', REACH_CAPACITY);
    const targetCells = builder.allocate('target-cells', 'uint32', REACH_CAPACITY);
    const vertexOffsets = builder.allocate('vertex-offsets', 'uint32', REACH_CAPACITY + 1);
    const positions = builder.allocate('positions', 'float32x2', VERTEX_CAPACITY);
    const lengths = builder.allocate('lengths', 'float32', REACH_CAPACITY);
    const reachCount = builder.allocate('reach-count', 'uint32', 1);
    const vertexCount = builder.allocate('vertex-count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const unresolvedCount = builder.allocate('unresolved-count', 'uint32', 1);
    builder.graph.add(
      new GPUTerrainStreamNetwork({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        streams: flow.streams.view,
        output: {
          sourceCells: sourceCells.view,
          targetCells: targetCells.view,
          vertexOffsets: vertexOffsets.view,
          positions: positions.view,
          lengths: lengths.view,
          reachCount: reachCount.view,
          vertexCount: vertexCount.view,
          status: {overflow: overflow.view},
          unresolvedCount: unresolvedCount.view
        }
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, positions},
      update: values => {
        flow.writeStreamThreshold(values.streamThreshold);
        // Rows past the vertex count are never written; keep them out of the picture.
        positions.write(new Float32Array(VERTEX_CAPACITY * 2).fill(Number.NaN));
      }
    };
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {layer: 'dots', positions: outputs.positions, radius: 3.5, color: [0.35, 0.8, 1, 1]}
  ]
});
