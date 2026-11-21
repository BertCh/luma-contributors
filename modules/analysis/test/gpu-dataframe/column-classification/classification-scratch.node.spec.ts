// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {
  NATURAL_BREAKS_MAXIMUM_EXACT_ROWS,
  NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS,
  NATURAL_BREAKS_MAXIMUM_MONOTONE_ROWS
} from '../../../src/dataframe-analysis/column-classification/natural-breaks-limits';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

function createFixture(rows: number, mode: 'exact' | 'monotone' = 'monotone') {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    id: 'classification-scratch',
    values: createTransientView(graph, 'values', 'float32', rows),
    parameters: createTransientView(graph, 'parameters', 'float32', 32),
    maximumClassCount: 2,
    methods: ['natural-breaks'] as const,
    naturalBreaksMode: mode,
    output: {
      breaks: createTransientView(graph, 'breaks', 'float32', 3),
      classCount: createTransientView(graph, 'classes', 'uint32', 1)
    }
  };
  return {device, graph, props};
}

it('retains explicit exact and approximate monotone route envelopes as named limits', () => {
  expect(NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS).toBe(1024);
  expect(NATURAL_BREAKS_MAXIMUM_EXACT_ROWS).toBe(8192);
  expect(NATURAL_BREAKS_MAXIMUM_MONOTONE_ROWS).toBe(262144);
  for (const [mode, limit] of [
    ['exact', NATURAL_BREAKS_MAXIMUM_EXACT_ROWS],
    ['monotone', NATURAL_BREAKS_MAXIMUM_MONOTONE_ROWS]
  ] as const) {
    const {device, graph, props} = createFixture(limit, mode);
    try {
      expect(() => new GPUClassBreaks(props)).not.toThrow();
      expect(
        () =>
          new GPUClassBreaks({
            ...props,
            values: createTransientView(graph, 'too-many-values', 'float32', limit + 1)
          })
      ).toThrow();
    } finally {
      device.destroy();
    }
  }
});

it('rejects row scratch above maxBufferSize before declaring any resources', () => {
  const {device, graph, props} = createFixture(64);
  try {
    // Inputs fit 512 bytes. Sorted moment/prefix scratch is checked first and needs
    // 64 * 5 * 4 = 1280 bytes; the later task lists also exceed this limit (2048 bytes).
    Object.defineProperty(device, 'limits', {value: {...device.limits, maxBufferSize: 512}});
    const contributor = new GPUClassBreaks(props);
    const declareScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /sorted moments\/prefix.*maxBufferSize/
    );
    expect(declareScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('preflights packed integer conversion scratch across otherwise valid chunks', () => {
  const {device, graph, props} = createFixture(100);
  try {
    Object.defineProperty(device, 'limits', {value: {...device.limits, maxBufferSize: 512}});
    const chunks = [100, 100, 100].map((length, index) =>
      createTransientView(graph, `integer-${index}`, 'uint32', length)
    );
    const contributor = new GPUClassBreaks({
      ...props,
      methods: ['quantile'],
      naturalBreaksMode: undefined,
      values: createVectorView('integers', 'uint32', chunks)
    });
    const declareScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /packed value conversion\/moment input.*maxBufferSize/
    );
    expect(declareScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
