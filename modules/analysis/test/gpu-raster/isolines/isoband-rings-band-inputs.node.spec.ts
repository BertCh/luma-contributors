// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/gpu-isoband-rings';

import {createRasterContributorFixture} from '../raster-grid/raster-contributor-test-fixture';

it('accepts a published band at the isoband-ring public surface', () => {
  const {graph, grid, view, band} = createRasterContributorFixture(3, 3);
  const contributor = new GPUIsobandRings({
    grid,
    values: band,
    breaks: view('float32', 1),
    parameters: view('float32', 12),
    edgeCapacity: 32,
    output: {
      ringOffsets: view('uint32', 9),
      positions: createTransientView(graph, 'positions', 'float32x2', 32),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    }
  });
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});
