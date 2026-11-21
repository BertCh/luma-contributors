// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPURasterEmbeddingZoneMean} from '@luma-contributors/analysis/raster-analysis';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

// RT-126: the producer's completeness controls (requiredCount, overflow, candidateOverflow) and
// zone validity must reach the reducer, or a truncated tuple prefix publishes a partial mean.
it('RT-126 zone mean binds the fractional coverage completeness status and zone validity', () => {
  const context = createKernelAuditContext('embedding-zone-mean-completeness');
  try {
    const contributor = new GPURasterEmbeddingZoneMean({
      pixelCount: 2,
      dimensions: 2,
      zoneCount: 1,
      vectors: context.view('float32', 4, 'vectors'),
      tuples: {
        zoneIds: context.view('uint32', 1, 'zone-ids'),
        cellIds: context.view('uint32', 1, 'cell-ids'),
        fractions: context.view('float32', 1, 'fractions'),
        status: {
          count: context.view('uint32', 1, 'count'),
          requiredCount: context.view('uint32', 1, 'required-count'),
          overflow: context.view('uint32', 1, 'overflow'),
          candidateOverflow: context.view('uint32', 1, 'candidate-overflow')
        },
        zoneValidity: context.view('uint32', 1, 'zone-validity')
      },
      output: {
        means: context.view('float32', 2, 'means'),
        weights: context.view('float32', 1, 'weights'),
        meanLengths: context.view('float32', 1, 'mean-lengths')
      }
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph));
    for (const kernel of kernels) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
    const sources = kernels.map(kernel => kernel.source).join('\n');
    for (const name of ['requiredCount', 'overflow', 'candidateOverflow', 'zoneValidity']) {
      expect(sources, name).toMatch(new RegExp(`\\b${name}\\[${name}Offset`));
    }
    // The means kernel suppresses both the mean and the weight of an unsupported zone.
    expect(sources).toMatch(/select\(missing\(\), weight, supported\)/);
  } finally {
    context.device.destroy();
  }
});

it('RT-126 zone mean rejects a short zoneValidity', () => {
  const context = createKernelAuditContext('embedding-zone-mean-validity-length');
  try {
    expect(
      () =>
        new GPURasterEmbeddingZoneMean({
          pixelCount: 2,
          dimensions: 2,
          zoneCount: 2,
          vectors: context.view('float32', 4),
          tuples: {
            zoneIds: context.view('uint32', 1),
            cellIds: context.view('uint32', 1),
            fractions: context.view('float32', 1),
            status: {count: context.view('uint32', 1)},
            zoneValidity: context.view('uint32', 1)
          },
          output: {means: context.view('float32', 4)}
        })
    ).toThrow(
      /^raster-embedding-zone-mean tuples\.zoneValidity must hold at least zoneCount \(2\)/
    );
  } finally {
    context.device.destroy();
  }
});

it('review N1 rejects disjoint zoneValidity and writable means in one physical buffer', () => {
  const context = createKernelAuditContext('embedding-zone-mean-validity-alias');
  const backing = context.view('float32', 66, 'pool');
  const zoneValidity = context.graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const means = context.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 2,
    byteOffset: 256
  });
  try {
    expect(
      () =>
        new GPURasterEmbeddingZoneMean({
          pixelCount: 1,
          dimensions: 2,
          zoneCount: 1,
          vectors: context.view('float32', 2),
          tuples: {
            zoneIds: context.view('uint32', 1),
            cellIds: context.view('uint32', 1),
            fractions: context.view('float32', 1),
            status: {count: context.view('uint32', 1)},
            zoneValidity
          },
          output: {means}
        })
    ).toThrow(/share buffers|same physical buffer/);
  } finally {
    context.device.destroy();
  }
});

it('S4 candidate-only reducer binds its sole loss control independently of other status', () => {
  const context = createKernelAuditContext('embedding-zone-mean-candidate-only');
  try {
    const contributor = new GPURasterEmbeddingZoneMean({
      pixelCount: 2,
      dimensions: 2,
      zoneCount: 1,
      vectors: context.view('float32', 4),
      tuples: {
        zoneIds: context.view('uint32', 2),
        cellIds: context.view('uint32', 2),
        fractions: context.view('float32', 2),
        status: {count: context.view('uint32', 1), candidateOverflow: context.view('uint32', 1)}
      },
      output: {means: context.view('float32', 2)}
    });
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph)).find(
      kernel => kernel.source.includes('complete[completeOffset] =')
    );
    expect(kernel).toBeDefined();
    expect(kernel!.source).toContain('candidateOverflow[candidateOverflowOffset] == 0u');
    expect(kernel!.source).not.toContain('requiredCount[');
    expect(kernel!.source).not.toContain('overflow[');
    expect(kernel!.source).not.toContain('zoneValidity[');
  } finally {
    context.device.destroy();
  }
});
