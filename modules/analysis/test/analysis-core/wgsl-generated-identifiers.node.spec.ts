// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getBinarySearchWGSL} from '../../src/analysis-core/kernels/binary-search-wgsl';
import {createGraphChunkGroupKernelNode} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {validateWGSLIdentifier} from '../../src/analysis-core/kernels/wgsl-identifiers';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../src/testing/gpu-contributor-test-utils';
import {WGSL_KEYWORDS, WGSL_RESERVED_WORDS} from './kernel-audit/kernel-audit-wgsl';

it('rejects every canonical keyword and reserved word in generated identifiers', () => {
  for (const name of [
    ...WGSL_KEYWORDS,
    ...WGSL_RESERVED_WORDS,
    '_',
    '__private',
    'bad-name',
    '1values'
  ]) {
    expect(() => validateWGSLIdentifier('identifiers', 'binding', name), name).toThrow(
      /WGSL identifier/
    );
    expect(() => getBinarySearchWGSL(name), name).toThrow(/WGSL identifier/);
  }
  // sample is a context-dependent name, not a reserved word.
  expect(() => getBinarySearchWGSL('sample')).not.toThrow();
  expect(() => getBinarySearchWGSL('values', 'u64', 'Words2')).not.toThrow();
});

it('rejects generated kernel bindings before declaring any resources', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createTransientView(graph, 'values', 'uint32', 1);
  try {
    for (const name of ['target', 'active', 'filter', 'module', 'meta']) {
      const props = {
        id: 'identifiers',
        operation: 'Identifiers',
        invocationCount: 1,
        bindings: [{name, view, type: 'u32' as const, access: 'read_write' as const}],
        body: ''
      };
      expect(() => createWGSLKernelNode(graph, props)).toThrow(/identifiers binding name/);
      expect(() => createGraphChunkGroupKernelNode(graph, props)).toThrow(
        /identifiers binding name/
      );
    }
  } finally {
    device.destroy();
  }
});

it('keeps chunk pipeline sources constant as uneven dispatch counts grow', async () => {
  const {getGPUCommandNodePipelineKeys} = await import(
    '../../src/testing/gpu-contributor-test-utils'
  );
  const device = createNullWebGPUDevice();
  try {
    const keys = [[1], [1, 257, 1025]].map(lengths => {
      const graph = new GPUCommandGraph(device);
      const nodes = lengths.map((length, chunkIndex) => {
        const values = createTransientView(graph, `values-${chunkIndex}`, 'uint32', length);
        return createGraphChunkGroupKernelNode(graph, {
          id: `chunk-${chunkIndex}`,
          operation: 'GeneratedIdentifiers',
          invocationCount: length,
          rowBase: chunkIndex * 2048,
          totalLength: 4096,
          bindings: [{name: 'values', view: values, type: 'u32', access: 'read_write'}],
          body: 'values[valuesOffset + index] = index;'
        });
      });
      expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        lengths.map(length => Math.ceil(length / 256))
      );
      return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    });
    expect([...keys[0]]).toEqual([...keys[1]]);
    expect(keys[1].size).toBe(1);
  } finally {
    device.destroy();
  }
});

it('accepts the Unicode XID identifier grammar in the public kernel builder', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = createTransientView(graph, 'values', 'uint32', 1);
  try {
    for (const name of ['réflexion', '朝焼け', 'Δέλτα', '𐰓𐰏𐰇', 'שָׁלוֹם', '_réflexion', 'e\u0301']) {
      expect(() => validateWGSLIdentifier('unicode', 'binding', name)).not.toThrow();
      expect(() => getBinarySearchWGSL(name)).not.toThrow();
      const node = createWGSLKernelNode(graph, {
        id: 'unicode',
        operation: 'Unicode',
        invocationCount: 1,
        bindings: [{name, view, type: 'u32', access: 'read_write'}],
        body: ''
      });
      expect(getGPUCommandNodeKernelSources([node])).toHaveLength(1);
    }
    for (const name of ['\u0301letter', '😀', 'name space', '__réflexion']) {
      expect(() => validateWGSLIdentifier('unicode', 'binding', name)).toThrow();
    }
  } finally {
    device.destroy();
  }
});
