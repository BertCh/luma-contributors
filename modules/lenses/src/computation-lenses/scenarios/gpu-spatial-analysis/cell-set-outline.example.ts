// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCellSetOutline,
  type GPUCellSourceInput
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Sorted, distinct H3 resolution-8 cell table produced by the lens scenario, with its row count. */
export type CellSetOutlineExampleInput = GPUCellSourceInput & {
  /** Boundary edges the outline may emit, and rings the assembly may write. */
  segmentCapacity: number;
  ringCapacity: number;
};

/** Traces the outer boundary of the cover cells and chains it into closed rings. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {cells, count, source, segmentCapacity, ringCapacity}: CellSetOutlineExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const make = <Format extends 'uint32' | 'uint32x2' | 'float32x2' | 'float32x4'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  // #region example
  const output = {
    rows: make('segmentRows', 'uint32', segmentCapacity),
    cells: make('segmentCells', 'uint32x2', segmentCapacity),
    edgeIndices: make('edgeIndices', 'uint32', segmentCapacity),
    endpoints: make('endpoints', 'float32x4', segmentCapacity),
    status: {
      count: make('segmentCount', 'uint32', 1),
      overflow: make('outlineOverflow', 'uint32', 1)
    }
  };
  const rings = {
    ringOffsets: make('ringOffsets', 'uint32', ringCapacity + 1),
    positions: make('ringPositions', 'float32x2', segmentCapacity + ringCapacity),
    status: {count: make('ringCount', 'uint32', 1), overflow: make('ringOverflow', 'uint32', 1)}
  };
  graph.add(
    new GPUCellSetOutline({
      family: 'h3',
      // Retain bare cells/count calls and accept a counted source on its own.
      ...(source ? {source, cells, count} : {cells: cells!, count}),
      output,
      rings: {normalizeWinding: true, output: rings}
    })
  );
  const {ringOffsets, positions} = rings; // closed rings, longitude/latitude degrees
  // #endregion example
  return {...output, rings, ringOffsets, positions};
}
