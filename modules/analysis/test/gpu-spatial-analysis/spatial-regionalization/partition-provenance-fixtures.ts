// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {compactRegionLabels} from '../../../src/gpu-spatial-analysis/spatial-regionalization/compact-region-labels';
import type {WeightsRig} from '../spatial-weights/spatial-weights-harness';

/** The same declarations are replayed on a null device before entering the GPU queue. */
export function createSparsePartitionComparison(rig: WeightsRig) {
  const source = Uint32Array.of(0, 0, 2, 2);
  const compact = compactRegionLabels(source);
  const outputs = [2, 4, compact.count].map((capacity, route) => {
    const summary = rig.output('float32', 9);
    const sizes = rig.output('uint32', capacity);
    rig.graph.add(
      new GPURegionPartitionEvaluation({
        id: `sparse-partition-${route}`,
        values: rig.input(Float32Array.of(0, 0, 10, 10), 'float32', 4),
        labels: rig.input(route === 2 ? compact.labels : source, 'uint32', 4),
        labelCapacity: capacity,
        output: {summary: summary.view, regionSizes: sizes.view}
      })
    );
    return {summary, sizes, capacity};
  });
  return outputs;
}

/** Large fixture used to check the 257th row tile independently of the first 256. */
export function createStagedPartitionInput() {
  const rowCount = 65537;
  const values = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    const first = row === rowCount - 1 ? 4096 : row % 2 ? 2 : 0;
    values[2 * row] = first;
    values[2 * row + 1] = 2 * first + 3;
  }
  return {
    rowCount,
    columnCount: 2,
    values,
    labels: Uint32Array.from({length: rowCount}, (_, row) => (row % 2 ? 65536 : 0))
  };
}
