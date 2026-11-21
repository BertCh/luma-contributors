// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCanonicalGeometryEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createEquality(outputLength: number, tolerance = 0): GPUCanonicalGeometryEquality {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'canonical-contract'});
  const tree = (name: string) => ({
    positions: createTransientView(graph, `${name}-positions`, 'float32x2', 1),
    nodes: createTransientView(graph, `${name}-nodes`, 'uint32x4', 1),
    children: createTransientView(graph, `${name}-children`, 'uint32', 1)
  });
  return new GPUCanonicalGeometryEquality({
    id: 'custom-equality',
    left: tree('left'),
    right: tree('right'),
    pairs: createTransientView(graph, 'pairs', 'uint32x2', 2),
    output: createTransientView(graph, 'output', 'uint32', outputLength),
    tolerance
  });
}

it('GEO-122 names the custom ID, output length and required pair row count', () => {
  for (const outputLength of [1, 3]) {
    expect(() => createEquality(outputLength)).toThrow(
      'custom-equality output.length must equal pairs.length (2)'
    );
  }
  expect(() => createEquality(2)).not.toThrow();
});

it('GEO-122 validates tolerance independently and retains shader representability bounds', () => {
  expect(() => createEquality(2, -1)).toThrow(/custom-equality tolerance.*0/);
  for (const tolerance of [NaN, Infinity, 3.5e38]) {
    expect(() => createEquality(2, tolerance)).toThrow(/custom-equality tolerance/);
  }
  expect(() => createEquality(2, 3e38)).not.toThrow();
});
