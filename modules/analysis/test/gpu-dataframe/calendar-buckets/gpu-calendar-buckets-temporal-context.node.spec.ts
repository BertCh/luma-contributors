// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {
  GPUCalendarBuckets,
  type GPUCalendarBucketsProps
} from '../../../src/dataframe-analysis/calendar-buckets/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const WORD_CONTEXT: GPUTemporalContext = {
  unit: 'nanosecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

function createCalendar(overrides: Partial<GPUCalendarBucketsProps>): GPUCalendarBuckets {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    return new GPUCalendarBuckets({
      timestamps: createTransientView(graph, 'timestamps', 'uint32x2', 4),
      parameters: createTransientView(graph, 'parameters', 'sint32', 2),
      output: {hour: createTransientView(graph, 'hour', 'uint32', 4)},
      ...overrides
    });
  } finally {
    device.destroy();
  }
}

describe('GPUCalendarBuckets temporalContext', () => {
  for (const unit of ['second', 'millisecond', 'microsecond', 'nanosecond'] as const) {
    it(`takes the ${unit} unit from the context`, () => {
      expect(createCalendar({temporalContext: {...WORD_CONTEXT, unit}}).unit).toBe(unit);
    });
  }

  it('keeps the plain unit prop and its millisecond default', () => {
    expect(createCalendar({}).unit).toBe('millisecond');
    expect(createCalendar({unit: 'second'}).unit).toBe('second');
  });

  it('accepts an explicit unit that agrees with the context', () => {
    expect(createCalendar({unit: 'nanosecond', temporalContext: WORD_CONTEXT}).unit).toBe(
      'nanosecond'
    );
  });

  it('rejects an explicit unit that contradicts the context', () => {
    expect(() => createCalendar({unit: 'millisecond', temporalContext: WORD_CONTEXT})).toThrow(
      /contradicts/
    );
  });

  it('rejects the encodings it cannot read', () => {
    expect(() =>
      createCalendar({
        temporalContext: {...WORD_CONTEXT, encoding: 'float32-relative', epoch: 1000n}
      })
    ).toThrow(/int64-words/);
    expect(() =>
      createCalendar({temporalContext: {...WORD_CONTEXT, encoding: 'double-single'}})
    ).toThrow(/int64-words/);
  });

  it('rejects words that are not counted from the Unix epoch', () => {
    expect(() => createCalendar({temporalContext: {...WORD_CONTEXT, epoch: 1000n}})).toThrow(
      /Unix epoch/
    );
    expect(createCalendar({temporalContext: {...WORD_CONTEXT, epoch: 0n}}).unit).toBe('nanosecond');
  });

  it('validates the context it is given', () => {
    expect(() =>
      createCalendar({temporalContext: {...WORD_CONTEXT, axis: {kind: 'regular', step: 0}}})
    ).toThrow(/step/);
  });
});
