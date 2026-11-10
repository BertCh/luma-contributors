// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {
  GPUTimeWordsRelative,
  type GPUTimeWordsRelativeProps
} from '../../../src/dataframe-analysis/time-words-relative/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  timeZone: 'UTC',
  axis: {kind: 'regular', step: 1000}
};
let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUTimeWordsRelativeProps> = {}
): GPUTimeWordsRelativeProps {
  return {
    timestamps: createTransientView(graph, `times-${serial++}`, 'uint32x2', 5),
    temporalContext: CONTEXT,
    origin: 1_700_000_000_000n,
    output: createTransientView(graph, `relative-${serial++}`, 'float32', 5),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUTimeWordsRelativeProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUTimeWordsRelative(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

describe('GPUTimeWordsRelative', () => {
  it('exposes the input context as float32-relative counted from the origin', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const node = new GPUTimeWordsRelative(createProps(graph));
    expect(node.outputTemporalContext).toEqual({
      ...CONTEXT,
      encoding: 'float32-relative',
      epoch: 1_700_000_000_000n
    });
    // The input is not modified.
    expect(CONTEXT.encoding).toBe('int64-words');
    expect(CONTEXT.epoch).toBe('unix');
    expect(node.getCommandNodes(graph).map(command => command.id)).toEqual([
      'time-words-relative-convert'
    ]);
    device.destroy();
  });

  it('takes the origin as an instant, whatever epoch the words count from', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const node = new GPUTimeWordsRelative(
      createProps(graph, {temporalContext: {...CONTEXT, epoch: 1_000n}, origin: 5_000n})
    );
    expect(node.outputTemporalContext.epoch).toBe(5_000n);
    device.destroy();
  });

  it('resolves the input as Int64 words only', () => {
    expectThrows(
      () => ({temporalContext: {...CONTEXT, encoding: 'float32-relative', epoch: 5n}}),
      /int64-words/
    );
    expectThrows(() => ({temporalContext: {...CONTEXT, encoding: 'double-single'}}), /int64-words/);
    expectThrows(() => ({temporalContext: {...CONTEXT, unit: 'minute' as never}}), /temporal unit/);
  });

  it('validates the origin', () => {
    expectThrows(() => ({origin: 2n ** 63n}), /origin must be a signed Int64/);
    expectThrows(() => ({origin: 12 as never}), /origin must be a signed Int64/);
    expectThrows(
      () => ({temporalContext: {...CONTEXT, epoch: -(2n ** 62n)}, origin: 2n ** 62n + 1n}),
      /more than 2\^63 ticks from the input epoch/
    );
  });

  it('validates the views', () => {
    expectThrows(
      graph => ({timestamps: createTransientView(graph, 'empty', 'uint32x2', 0)}),
      /at least one row/
    );
    expectThrows(
      graph => ({output: createTransientView(graph, 'short', 'float32', 4)}),
      /output must have 5 rows/
    );
    expectThrows(
      graph => ({output: createTransientView(graph, 'wrong', 'uint32', 5) as never}),
      /output/
    );
  });

  it('refuses timestamps larger than one storage binding at declaration', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const limit = device.limits.maxStorageBufferBindingSize / 8;
    const add = (rows: number) =>
      graph.add(
        new GPUTimeWordsRelative({
          ...createProps(graph),
          id: `time-words-relative-${rows}`,
          timestamps: createTransientView(graph, `big-times-${rows}`, 'uint32x2', rows),
          output: createTransientView(graph, `big-relative-${rows}`, 'float32', rows)
        })
      );
    expect(() => add(limit)).not.toThrow();
    expect(() => add(limit + 1)).toThrow(/timestamps needs 134217736 bytes in one binding/);
    device.destroy();
  });
});
