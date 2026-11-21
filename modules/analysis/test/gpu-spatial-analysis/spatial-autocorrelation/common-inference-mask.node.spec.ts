// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocalMoran} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('CMP-07 composes scalar observation completeness and row membership for both inference consumers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const valuesComplete = createTransientView(graph, 'values-complete', 'uint32', 1);
    const mask = createTransientView(graph, 'membership', 'uint32', 4);
    const candidateOverflow = createTransientView(graph, 'membership-loss', 'uint32', 1);
    const status = {candidateOverflow};
    const result = GPULocalMoran.createCommonMaskNodes(graph, {
      rowCount: 4,
      valuesComplete,
      mask,
      membershipStatus: status
    });
    status.candidateOverflow = valuesComplete;
    expect(result.membershipStatus.candidateOverflow).toBe(candidateOverflow);
    expect(result.mask.length).toBe(4);
    expect(getParsedKernelSources(result.nodes).size).toBe(2);
    const source = getGPUCommandNodeKernelSources(result.nodes)
      .map(kernel => kernel.source)
      .join('\n');
    expect(source).toContain('valuesComplete[valuesCompleteOffset] != 0u');
    expect(source).toContain('candidateOverflow[candidateOverflowOffset] == 0u');
    expect(source).toContain('membership[membershipOffset + index]');
  } finally {
    device.destroy();
  }
});

it('B2 preserves and tests a false membership certificate independently of values completeness', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const complete = createTransientView(graph, 'membership-complete', 'uint32', 1);
    const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
    const valuesComplete = createTransientView(graph, 'values-complete', 'uint32', 1);
    const mask = createTransientView(graph, 'membership', 'uint32', 4);
    const status = {complete, overflow};
    const result = GPULocalMoran.createCommonMaskNodes(graph, {
      rowCount: 4,
      valuesComplete,
      mask,
      membershipStatus: status
    });
    expect(result.membershipStatus['complete']).toBe(complete);
    const source = getGPUCommandNodeKernelSources(result.nodes)[0].source;
    expect(source).toContain('membershipComplete[membershipCompleteOffset] != 0u');
    const predicate = source
      .match(/complete\[completeOffset\] = u32\(([^;]+)\);/)![1]
      .replaceAll(/(\d+)u\b/g, '$1');
    const evaluate = new Function(
      'valuesComplete',
      'overflow',
      'membershipComplete',
      `const valuesCompleteOffset = 0, overflowOffset = 0, membershipCompleteOffset = 0; return ${predicate};`
    );
    expect(evaluate([1], [0], [0])).toBe(false);
    expect(evaluate([1], [0], [1])).toBe(true);
    expect(getParsedKernelSources(result.nodes).size).toBe(2);
  } finally {
    device.destroy();
  }
});
