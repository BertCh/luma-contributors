// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUCalendarBucketsParameterValues,
  GPUCalendarBuckets
} from '../../../src/dataframe-analysis/calendar-buckets/index';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const CASES = [
  ['nanosecond', 1_700_000_000_000_000_000n],
  ['microsecond', 1_700_000_000_000_000n],
  ['second', 1_700_000_000n]
] as const;

for (const [unit, timestamp] of CASES) {
  it(`GPUCalendarBuckets decodes ${unit} words with the unit from temporalContext`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const temporalContext: GPUTemporalContext = {
      unit,
      epoch: 'unix',
      encoding: 'int64-words',
      axis: {kind: 'irregular'}
    };
    const graph = new GPUCommandGraph(device, {id: 'calendar-context-graph'});
    const parameters = new GPUParameterBuffer(device, {
      id: 'calendar-context-parameters',
      format: 'sint32',
      length: 2
    });
    const timestampBuffer = createInputBuffer(
      device,
      getInt64TimeWords(BigInt64Array.of(timestamp))
    );
    const yearBuffer = createOutputBuffer(device, 1);
    const monthBuffer = createOutputBuffer(device, 1);
    const hourBuffer = createOutputBuffer(device, 1);
    graph.add(
      new GPUCalendarBuckets({
        id: 'calendar-context',
        temporalContext,
        timestamps: importGraphBuffer(graph, 'times', timestampBuffer, 'uint32x2', 1),
        parameters: parameters.importToGraph(graph),
        output: {
          year: importGraphBuffer(graph, 'year', yearBuffer, 'sint32', 1),
          month: importGraphBuffer(graph, 'month', monthBuffer, 'uint32', 1),
          hour: importGraphBuffer(graph, 'hour', hourBuffer, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    parameters.write(getGPUCalendarBucketsParameterValues(0, 0));
    submitGraph(device, compiled, undefined);
    // 1,700,000,000 seconds is 2023-11-14T22:13:20Z.
    expect(await readUint32(yearBuffer, 1)).toEqual([2023]);
    expect(await readUint32(monthBuffer, 1)).toEqual([11]);
    expect(await readUint32(hourBuffer, 1)).toEqual([22]);
    compiled.destroy();
    parameters.destroy();
    for (const buffer of [timestampBuffer, yearBuffer, monthBuffer, hourBuffer]) {
      buffer.destroy();
    }
  });
}
