// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterMedialAxis} from '../../../src/raster-analysis/gpu-raster-skeleton';
import {computeMedialAxisReference, computeFilledRectangleReference} from './medial-axis-oracle';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';

/** CPU statement of the proved primitive-step criterion, compared with dense containment. */
function classifyPrimitiveSteps(
  width: number,
  height: number,
  reference: ReturnType<typeof computeMedialAxisReference>
) {
  return reference.squaredRadius.map((radius, cell) => {
    if (radius === 0) return 0;
    const column = cell % width;
    const row = Math.floor(cell / width);
    const horizontal = column - reference.nearestBackground[cell][0];
    const vertical = row - reference.nearestBackground[cell][1];
    let divisor = Math.abs(horizontal);
    let remainder = Math.abs(vertical);
    while (remainder !== 0) [divisor, remainder] = [remainder, divisor % remainder];
    const nextColumn = column + horizontal / divisor;
    const nextRow = row + vertical / divisor;
    if (nextColumn < 0 || nextColumn >= width || nextRow < 0 || nextRow >= height) return 1;
    const expectedSquare =
      (horizontal + horizontal / divisor) ** 2 + (vertical + vertical / divisor) ** 2;
    return Number(reference.squaredRadius[nextRow * width + nextColumn] !== expectedSquare);
  });
}

it('primitive-step classification equals exact all-pairs disk containment for all 65536 binary 4x4 masks', {
  timeout: 30000
}, () => {
  for (let bits = 0; bits < 65536; bits++) {
    const values = Array.from({length: 16}, (_, cell) => (bits >>> cell) & 1);
    const reference = computeMedialAxisReference(4, 4, values);
    const actual = classifyPrimitiveSteps(4, 4, reference);
    if (actual.some((value, cell) => value !== reference.medialAxis[cell])) {
      throw new Error(`Primitive-step mismatch for mask ${bits}`);
    }
  }
});

it('filled-rectangle oracle preserves corners and even-width plateaus rather than thinning', () => {
  for (const [width, height] of [
    [1, 7],
    [7, 1],
    [2, 2],
    [4, 8],
    [9, 7]
  ]) {
    const reference = computeMedialAxisReference(width, height, Array(width * height).fill(1));
    const analytic = computeFilledRectangleReference(width, height);
    expect(analytic.medialAxis).toEqual(reference.medialAxis);
    expect(analytic.radius).toEqual(reference.radius);
  }
});

function createContributor(width: number, height: number, extraRows = 0) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const count = width * height;
  const contributor = new GPURasterMedialAxis({
    grid: {width, height, affine: [2, 3, 400, 5, -7, 600], pixelInterpretation: 'area'},
    input: createTransientView(graph, 'input', 'uint32', count + extraRows),
    output: {
      medialAxis: createTransientView(graph, 'axis', 'uint32', count + extraRows),
      radius: createTransientView(graph, 'radius', 'float32', count + extraRows)
    }
  });
  return {graph, contributor};
}

it('medial axis composes exact distance nodes beyond the old limit, retaining oversized output capacities', () => {
  const {graph, contributor} = createContributor(257, 193, 17);
  try {
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(sources.size).toBeGreaterThan(2);
    for (const [name, source] of sources) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), name).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
  } finally {
    graph.device.destroy();
  }
});

it('medial axis rejects unsafe squared-distance dimensions and device-limited scratch', () => {
  expect(() => createContributor(32769, 1)).toThrow();
  const {graph, contributor} = createContributor(8192, 8192);
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/storage|binding/);
  } finally {
    graph.device.destroy();
  }
});
