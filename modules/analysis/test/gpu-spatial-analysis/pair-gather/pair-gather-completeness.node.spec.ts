// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

for (const perQuery of [true, false]) {
  it(`GEO-113 captures nearest status and declares loss-aware unmatched kernels (${perQuery})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const candidate = scalar('candidate');
    const truncated = createTransientView(graph, 'truncated', 'uint32', 2);
    const neighbors = {
      rows: createTransientView(graph, 'neighbors', 'uint32', 2),
      counts: createTransientView(graph, 'counts', 'uint32', 2),
      capacity: 1,
      status: {overflow: scalar('producer-overflow'), candidateOverflow: candidate},
      ...(perQuery ? {candidateTruncated: truncated} : {})
    };
    const output = {
      leftRows: createTransientView(graph, 'left', 'uint32', 2),
      rightRows: createTransientView(graph, 'right', 'uint32', 2),
      status: {
        count: scalar('count'),
        overflow: scalar('overflow'),
        candidateOverflow: scalar('gather-candidate')
      }
    };
    const contributor = new GPUPairGather({neighbors, output, how: 'left'});
    neighbors.status.candidateOverflow = scalar('replacement');
    try {
      expect(contributor.props.neighbors!.status!.candidateOverflow).toBe(candidate);
      const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect(kernels.get('pair-gather-empty-mask')).toContain(
        perQuery ? 'truncated[' : 'neighborCounts['
      );
      if (!perQuery)
        expect(kernels.get('pair-gather-unmatched-loss-0')).toContain(
          'maskOut[maskOutOffset + index] = 0u'
        );
      for (const {source} of kernels.configurations)
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    } finally {
      device.destroy();
    }
  });
}

it('GEO-113 validates one per-query truncation word per neighbor count', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  try {
    expect(
      () =>
        new GPUPairGather({
          neighbors: {
            rows: scalar('rows'),
            counts: scalar('counts'),
            capacity: 1,
            candidateTruncated: createTransientView(graph, 'truncated', 'uint32', 2)
          },
          output: {
            leftRows: scalar('left'),
            rightRows: scalar('right'),
            status: {
              count: scalar('count'),
              overflow: scalar('overflow'),
              candidateOverflow: scalar('candidate')
            }
          }
        })
    ).toThrow(/neighbors.candidateTruncated needs one row per query/);
  } finally {
    device.destroy();
  }
});

for (const configuration of ['pairs', 'neighbors', 'truncation-only'] as const) {
  it(`B1 constructs ${configuration} without aggregate loss flags`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    try {
      const inputs =
        configuration === 'pairs'
          ? {
              pairs: {
                leftRows: scalar('pair-left'),
                rightRows: scalar('pair-right'),
                status: {count: scalar('pair-count'), overflow: scalar('pair-overflow')}
              }
            }
          : {
              neighbors: {
                rows: scalar('rows'),
                counts: scalar('counts'),
                capacity: 1,
                ...(configuration === 'truncation-only'
                  ? {candidateTruncated: scalar('truncated')}
                  : {})
              }
            };
      // Legacy pair/neighbor constructors also remain valid with no candidate destination.
      for (const withCandidate of configuration === 'truncation-only' ? [true] : [false, true]) {
        expect(
          () =>
            new GPUPairGather({
              ...inputs,
              output: {
                leftRows: scalar(`left-${withCandidate}`),
                rightRows: scalar(`right-${withCandidate}`),
                status: {
                  count: scalar(`count-${withCandidate}`),
                  overflow: scalar(`overflow-${withCandidate}`),
                  ...(withCandidate
                    ? {candidateOverflow: scalar(`candidate-${withCandidate}`)}
                    : {})
                }
              }
            })
        ).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });
}

it('B2 final capacity flags preserve established unmatched rows in the production mask kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const integers = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const contributor = new GPUPairGather({
    how: 'left',
    neighbors: {
      rows: integers('neighbors', 3),
      counts: integers('counts', 3),
      capacity: 1,
      status: {
        overflow: integers('producer-overflow'),
        candidateOverflow: integers('producer-candidate')
      }
    },
    overflowSources: [integers('extra-final-capacity')],
    output: {
      leftRows: integers('left', 3),
      rightRows: integers('right', 3),
      status: {
        count: integers('count'),
        overflow: integers('overflow'),
        candidateOverflow: integers('candidate')
      }
    }
  });
  try {
    const kernels = getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(
          node =>
            node.id === 'pair-gather-empty-mask' ||
            node.id.startsWith('pair-gather-unmatched-loss-')
        )
    );
    const buffers: Record<string, Uint32Array> = {
      neighborCounts: Uint32Array.of(1, 0, 0),
      maskOut: new Uint32Array(3),
      loss0: Uint32Array.of(0),
      loss1: Uint32Array.of(0),
      loss2: Uint32Array.of(1)
    };
    for (const kernel of kernels) {
      const bindings = Object.fromEntries(
        new WgslReflect(kernel.source).storage.map(declaration => [
          declaration.binding,
          buffers[declaration.name]
        ])
      );
      new WgslExec(
        new WgslParser().parse(kernel.source.replace(/^\s*_ = &\w+;$/gm, ''))
      ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: kernel.constants});
    }
    // Counts alone establish queries 1 and 2 as unmatched. An unrelated final row-capacity
    // flag cannot invalidate that evidence; all producer search-loss flags are zero.
    expect(Array.from(buffers['maskOut'])).toEqual([0, 1, 1]);
  } finally {
    device.destroy();
  }
});

it('S2 reduces truncation flags without a query-sized compaction output', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const integers = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const contributor = new GPUPairGather({
    neighbors: {
      rows: integers('neighbors', 1025),
      counts: integers('counts', 1025),
      capacity: 1,
      candidateTruncated: integers('truncated', 1025),
      status: {overflow: integers('producer-overflow')}
    },
    output: {
      leftRows: integers('left', 1025),
      rightRows: integers('right', 1025),
      status: {
        count: integers('count'),
        overflow: integers('overflow'),
        candidateOverflow: integers('candidate')
      }
    }
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    contributor.getCommandNodes(graph);
    const truncationAllocations = allocate.mock.calls
      .map(([descriptor]) => descriptor)
      .filter(descriptor => descriptor.id?.startsWith('pair-gather-truncated'));
    expect(truncationAllocations.length).toBeGreaterThan(0);
    for (const descriptor of truncationAllocations)
      expect(descriptor.byteLength).toBeLessThan(1025 * 4);
  } finally {
    allocate.mockRestore();
    device.destroy();
  }
});
