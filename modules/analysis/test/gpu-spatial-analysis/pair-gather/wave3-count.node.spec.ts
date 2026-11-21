// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUPairGather,
  type GPUPairGatherOutput
} from '../../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('OUT-11 accepts a pair-table output and rejects a missing status.count at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const integers = (length = 1) => createTransientView(graph, `view-${serial++}`, 'uint32', length);
  const pairs = {
    leftRows: integers(2),
    rightRows: integers(2),
    status: {count: integers(), overflow: integers()}
  };
  const count = integers();
  const output = {
    leftRows: integers(3),
    rightRows: integers(3),
    status: {count, overflow: integers()}
  };
  const construct = (destination: GPUPairGatherOutput) =>
    new GPUPairGather({pairs, output: destination});
  try {
    const gather = construct(output);
    expect(() => gather.getCommandNodes(graph)).not.toThrow();
    const missingCount = {...output, status: {overflow: output.status.overflow}};
    const legacyOutput = {...missingCount, rowCount: count};
    const missingDestinations = [
      missingCount,
      legacyOutput,
      {...output, status: undefined},
      undefined
    ];
    for (const destination of missingDestinations) {
      expect(() => construct(destination as GPUPairGatherOutput)).toThrow(
        /^pair-gather output\.status\.count is required$/
      );
    }
    const mixedOutput = {...output, rowCount: count};
    expect(() => construct(mixedOutput)).toThrow(
      /^pair-gather output\.rowCount was removed; use output\.status\.count$/
    );
  } finally {
    device.destroy();
  }
});

it('names the output status fields when their aligned binding ranges overlap', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const integers = (id: string, length = 1) => createTransientView(graph, id, 'uint32', length);
  try {
    const storage = integers('status', 2);
    expect(
      () =>
        new GPUPairGather({
          pairs: {
            leftRows: integers('input-left'),
            rightRows: integers('input-right'),
            status: {count: integers('input-count'), overflow: integers('input-overflow')}
          },
          output: {
            leftRows: integers('output-left'),
            rightRows: integers('output-right'),
            status: {
              count: graph.createDataView(storage.buffer, {format: 'uint32', length: 1}),
              overflow: graph.createDataView(storage.buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: 4
              })
            }
          }
        })
    ).toThrow(
      'pair-gather output.status.count chunk 0 and output.status.overflow chunk 0 aligned binding byte ranges must not overlap'
    );
  } finally {
    device.destroy();
  }
});

it('B2 captures output.status and output.status.count getters exactly once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const integers = () => createTransientView(graph, `getter-${serial++}`, 'uint32', 1);
  const count = integers();
  let statusReads = 0;
  let countReads = 0;
  const status = {
    overflow: integers(),
    get count() {
      countReads++;
      return countReads === 1 ? count : integers();
    }
  };
  const output = {
    leftRows: integers(),
    rightRows: integers(),
    get status() {
      if (++statusReads > 1) throw new Error('read twice: output.status');
      return status;
    }
  };
  try {
    const gather = new GPUPairGather({
      pairs: {
        leftRows: integers(),
        rightRows: integers(),
        status: {count: integers(), overflow: integers()}
      },
      output
    });
    expect(statusReads).toBe(1);
    expect(countReads).toBe(1);
    expect(gather.outputCount).toBe(count);
    expect(gather.props.output.status.count).toBe(count);
  } finally {
    device.destroy();
  }
});
