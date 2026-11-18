// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUCellPyramid} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Overture places (longitude/latitude degrees) and the table capacity of each level. */
export type CellPyramidExampleInput = {
  places: GraphDataView<'float32x2'>;
  capacities: readonly number[];
};

/** Builds place counts at Quadbin zooms 18, 17 and 16 in one pass over the places. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {places, capacities}: CellPyramidExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1, usage);
  const levels = capacities.map((capacity, level) => ({
    cells: createTransientView(graph, `level${level}Cells`, 'uint32x2', capacity, usage),
    counts: createTransientView(graph, `level${level}Counts`, 'uint32', capacity, usage),
    status: {
      count: scalar(`level${level}Count`),
      overflow: scalar(`level${level}Overflow`),
      candidateOverflow: scalar(`level${level}CandidateOverflow`)
    }
  }));
  const levelCounts = createTransientView(graph, 'levelCounts', 'uint32', 3, usage);
  graph.add(
    new GPUCellPyramid({
      family: 'quadbin',
      positions: places,
      resolutions: [18, 17, 16], // finest first
      output: {levels, levelCounts}
    })
  );
  // #endregion example
  const [finest, , coarsest] = levels;
  return {levels, levelCounts, finest, coarsest};
}
