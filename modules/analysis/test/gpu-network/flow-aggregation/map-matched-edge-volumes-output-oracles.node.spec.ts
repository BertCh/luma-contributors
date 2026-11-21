// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUMapMatchedEdgeVolumes} from '../../../src/gpu-network/flow-aggregation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('edge-volume transition refusals accept omitted and undefined outputs including inherited properties', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const name of ['zoneOutWeights', 'zoneInWeights'])
      for (const mode of [
        'omitted',
        'undefined',
        'inherited-undefined',
        'view',
        'inherited-view'
      ]) {
        const context = createOracleContext(device);
        const {input, output} = context;
        const transitions = {
          ids: output('ids', 'uint32', 2),
          status: {count: output('count', 'uint32'), overflow: output('overflow', 'uint32')}
        };
        const fields = {
          [name]: mode.endsWith('view') ? output('weights', 'float32', 8) : undefined
        };
        if (mode.startsWith('inherited-')) Object.setPrototypeOf(transitions, fields);
        else if (mode !== 'omitted') Object.assign(transitions, fields);
        try {
          const createContributor = () =>
            new GPUMapMatchedEdgeVolumes({
              traversedEdges: {
                edges: input('edges', 'uint32', [2, 5, 7]),
                counts: input('counts', 'uint32', [3]),
                completeness: 'known-complete'
              },
              trackCount: 1,
              traversedEdgeCapacity: 3,
              edgeCount: 8,
              transitionPairCapacity: 8,
              output: {
                edgeVolumes: output('volumes', 'uint32', 8),
                status: {candidateOverflow: output('loss', 'uint32')},
                transitions
              }
            });
          if (mode.endsWith('view')) expect(createContributor).toThrow(new RegExp(name));
          else
            expect(
              () => createContributor().getCommandNodes(context.graph),
              `${name}: ${mode}`
            ).not.toThrow();
        } finally {
          context.destroy();
        }
      }
  } finally {
    device.destroy();
  }
});

it('edge-volume transition zone weight ports currently refuse because no weight input exists', () => {
  const device = createNullWebGPUDevice();
  for (const name of ['zoneOutWeights', 'zoneInWeights']) {
    const context = createOracleContext(device);
    const {input, output} = context;
    const createContributor = () =>
      new GPUMapMatchedEdgeVolumes({
        traversedEdges: {
          edges: input('edges', 'uint32', [2, 5, 7]),
          counts: input('counts', 'uint32', [3]),
          completeness: 'known-complete'
        },
        trackCount: 1,
        traversedEdgeCapacity: 3,
        edgeCount: 8,
        transitionPairCapacity: 8,
        output: {
          edgeVolumes: output('volumes', 'uint32', 8),
          status: {candidateOverflow: output('loss', 'uint32')},
          transitions: {
            ids: output('ids', 'uint32', 2),
            status: {count: output('count', 'uint32'), overflow: output('overflow', 'uint32')},
            [name]: output('weights', 'float32', 8)
          }
        }
      });
    try {
      expect(createContributor).toThrow(new RegExp('map-matched-edge-volumes.*' + name + '.*use'));
    } finally {
      context.destroy();
    }
  }
  device.destroy();
});
