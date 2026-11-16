// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

export function createRasterContributorFixture(width = 5, height = 1) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const grid = {
    width,
    height,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length = width * height) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const band = {
    id: 'band',
    format: 'uint32' as const,
    storage: {kind: 'buffer' as const, values: view('uint32')},
    validity: view('uint32'),
    scale: 2,
    noDataValue: 99
  };
  return {graph, grid, view, band};
}
