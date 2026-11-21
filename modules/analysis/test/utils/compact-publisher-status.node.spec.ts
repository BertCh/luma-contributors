// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {createPublishNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUPairGather} from '../../src/gpu-spatial-analysis/pair-gather/index';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';

test('compact publisher keeps candidate loss separate from final capacity loss', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const props = {
    id: 'publish',
    operation: 'PublisherStatusTest',
    requiredCount: scalar('required'),
    candidateOverflowSources: [scalar('candidate-in')],
    overflowSources: [scalar('final-in')],
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 2),
      status: {
        count: scalar('count'),
        overflow: scalar('overflow'),
        candidateOverflow: scalar('candidate')
      }
    }
  };
  const [kernel] = getGPUCommandNodeKernelSources([createPublishNode(graph, props)]);
  expect(kernel!.source).toContain(
    'candidateOverflowOut[candidateOverflowOutOffset] = select(0u, 1u, candidateOverflowIn0[candidateOverflowIn0Offset] != 0u)'
  );
  expect(kernel!.source).toContain(
    'overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY || overflowIn0[overflowIn0Offset] != 0u)'
  );
  device.destroy();
});

test('compact publisher refuses to discard a supplied candidate status', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  try {
    expect(() =>
      createPublishNode(graph, {
        id: 'publish-refusal',
        operation: 'PublisherStatusTest',
        requiredCount: scalar('required'),
        candidateOverflowSources: [scalar('candidate')],
        output: {
          ids: scalar('ids'),
          status: {count: scalar('count'), overflow: scalar('overflow')}
        }
      })
    ).toThrow(/publish-refusal.*output.status.candidateOverflow is required/);
  } finally {
    device.destroy();
  }
});

test('pair gather refuses to discard a supplied candidate status', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  expect(
    () =>
      new GPUPairGather({
        pairs: {
          leftRows: scalar('left'),
          rightRows: scalar('right'),
          status: {
            count: scalar('count'),
            overflow: scalar('overflow'),
            candidateOverflow: scalar('candidate')
          }
        },
        output: {
          leftRows: scalar('left-out'),
          rightRows: scalar('right-out'),
          status: {count: scalar('row-count'), overflow: scalar('output-overflow')}
        }
      })
  ).toThrow(/pair-gather.*output.status.candidateOverflow is required/);
  device.destroy();
});

test('pair gather propagates the supplied pair candidate flag into a separate status', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const candidateOverflow = scalar('pair-candidate');
  const outputCandidateOverflow = scalar('output-candidate');
  const nodes = new GPUPairGather({
    pairs: {
      leftRows: scalar('left'),
      rightRows: scalar('right'),
      status: {count: scalar('count'), overflow: scalar('overflow'), candidateOverflow}
    },
    output: {
      leftRows: createTransientView(graph, 'left-out', 'uint32', 2),
      rightRows: createTransientView(graph, 'right-out', 'uint32', 2),
      status: {
        count: scalar('row-count'),
        overflow: scalar('output-overflow'),
        candidateOverflow: outputCandidateOverflow
      }
    }
  }).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(
    kernels.some(kernel =>
      kernel.source.includes(
        'candidateOverflowOut[candidateOverflowOutOffset] = select(0u, 1u, candidateIn0[candidateIn0Offset] != 0u)'
      )
    )
  ).toBe(true);
  expect(
    nodes
      .flatMap(node => node.resources ?? [])
      .some(resource => 'buffer' in resource && resource.buffer === candidateOverflow)
  ).toBe(true);
  device.destroy();
});
