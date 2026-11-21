// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createPublishNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGeometryFixture} from '../gpu-spatial-analysis/outline-geometry/geometry-fixture';
import {expectCompactPublicationReuse} from './compact-publisher-status-fixture';
import {requireTestDevice} from './require-test-device';

it('compact publisher separates candidate loss, final truncation and cleared status on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      total: {format: 'uint32', values: new Uint32Array([1])},
      candidateFlag: {format: 'uint32', values: new Uint32Array([1])}
    },
    outputs: {
      ids: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidateOverflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) => ({
      getCommandNodes: graph => [
        createPublishNode(graph, {
          id: 'publish',
          operation: 'CompactPublisherStatusTest',
          requiredCount: inputs['total'] as never,
          candidateOverflowSources: [inputs['candidateFlag'] as never],
          output: {
            ids: outputs['ids'] as never,
            status: {
              count: outputs['count'] as never,
              requiredCount: outputs['requiredCount'] as never,
              overflow: outputs['overflow'] as never,
              candidateOverflow: outputs['candidateOverflow'] as never
            }
          }
        })
      ]
    })
  });
  try {
    const incomplete = await fixture.run();
    expect(incomplete['count']).toEqual([1]);
    expect(incomplete['requiredCount']).toEqual([1]);
    expect(incomplete['overflow']).toEqual([0]);
    expect(incomplete['candidateOverflow']).toEqual([1]);
    fixture.writeInput('candidateFlag', new Uint32Array([0]));
    const complete = await fixture.run();
    expect(complete['candidateOverflow']).toEqual([0]);
    fixture.writeInput('total', new Uint32Array([3]));
    const truncated = await fixture.run();
    expect(truncated['count']).toEqual([2]);
    expect(truncated['requiredCount']).toEqual([3]);
    expect(truncated['overflow']).toEqual([1]);
    expect(truncated['candidateOverflow']).toEqual([0]);
    await expectCompactPublicationReuse(fixture);
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});
