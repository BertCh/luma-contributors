// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView, addGPUCommandNodes} from '@luma.gl/graph/gpu-core';
import {GPUComputationRecorder} from '../../src/computation-record/computation-recorder';
import {NullDevice} from '@luma.gl/core/test-utils';
import {describe, expect, test} from 'vitest';
import {GPUTransferMask} from '../../src/computation-record/gpu-transfer-mask';
import {hashGPUComputationText} from '../../src/computation-record/hash-computation-text';

describe('GPUTransferMask', () => {
  for (const recording of [undefined, {}, {sourceText: true}] as const) {
    for (const layout of ['index', 'csr'] as const) {
      for (const mode of ['any', 'count'] as const) {
        test(`${layout}-${mode} emits clear/scatter nodes with recording ${JSON.stringify(recording)}`, () => {
          const device = createDevice('transfer-device');
          const graph = new GPUCommandGraph(device);
          const recorder = recording && new GPUComputationRecorder(recording);
          try {
            const source = createTransientView(graph, 'source', 'uint32', 3);
            const target = createTransientView(graph, 'target', 'uint32', 2);
            const targets = createTransientView(graph, 'targets', 'uint32', 3);
            const offsets = createTransientView(graph, 'offsets', 'uint32', 4);
            const relation =
              layout === 'index' ? {layout, targets} : {layout, offsets, neighbors: targets};
            const nodes = new GPUTransferMask({
              id: 'transfer',
              source,
              target,
              relation,
              mode,
              recorder
            }).getCommandNodes(graph);
            expect(nodes.map(node => node.id)).toEqual(['transfer-clear', 'transfer-scatter']);
            expect(nodes.map(node => node.type)).toEqual(['compute', 'compute']);
            expect(nodes[1].dependsOn).toEqual(['transfer-clear']);
            expect(nodes[0].resources).toEqual([{buffer: target, usage: 'storage-read-write'}]);
            expect(nodes[1].resources).toEqual([
              {buffer: source, usage: 'storage-read'},
              ...(layout === 'csr' ? [{buffer: offsets, usage: 'storage-read'}] : []),
              {buffer: targets, usage: 'storage-read'},
              {buffer: target, usage: 'storage-read-write'}
            ]);
            for (const node of nodes) {
              expect(node.workload).toMatchObject({
                operation: 'GPUTransferMask',
                variant: `${layout}-${mode}`,
                commandCount: 1
              });
              const annotation = recorder?.getNodeRecord(node.id);
              if (recording) {
                expect(annotation).toMatchObject({
                  op: 'GPUTransferMask',
                  attrs: {layout, mode}
                });
                if ('sourceText' in recording) {
                  expect(annotation?.source?.hash).toBe(
                    hashGPUComputationText(annotation!.source!.text!)
                  );
                } else {
                  expect(annotation?.source).not.toHaveProperty('text');
                }
              } else {
                // Unrecorded production nodes carry nothing beyond the published node fields.
                expect(node).not.toHaveProperty('annotation');
              }
            }
            addGPUCommandNodes(graph, nodes);
          } finally {
            device.destroy();
          }
        });
      }
    }
  }

  test('validates packed formats, relation lengths, graph ownership and separate output storage', () => {
    const device = createDevice('validation-device');
    const graph = new GPUCommandGraph(device);
    try {
      const source = createTransientView(graph, 'source', 'uint32', 3);
      const target = createTransientView(graph, 'target', 'uint32', 2);
      const targets = createTransientView(graph, 'targets', 'uint32', 3);
      const invalid = createTransientView(graph, 'invalid', 'float32', 3);
      const props = {
        id: 'validate-transfer',
        source,
        target,
        relation: {layout: 'index' as const, targets}
      };
      expect(() => new GPUTransferMask({...props, source: invalid})).toThrow('validate-transfer');
      expect(() => new GPUTransferMask({...props, target: invalid})).toThrow('validate-transfer');
      expect(
        () => new GPUTransferMask({...props, relation: {layout: 'index', targets: invalid}})
      ).toThrow('validate-transfer');
      expect(
        () => new GPUTransferMask({...props, relation: {layout: 'index', targets: target}})
      ).toThrow('validate-transfer');
      expect(
        () =>
          new GPUTransferMask({
            ...props,
            relation: {layout: 'csr', offsets: source, neighbors: targets}
          })
      ).toThrow('offsets length');
      expect(() => new GPUTransferMask({...props, target: source})).toThrow('separate storage');
      const padded = graph.createDataView(
        graph.createTransientBuffer({id: 'padded', byteLength: 24, usage: Buffer.STORAGE}),
        {format: 'uint32', length: 3, byteStride: 8}
      );
      expect(() => new GPUTransferMask({...props, source: padded})).toThrow('validate-transfer');
      expect(() => new GPUTransferMask(props).getCommandNodes(new GPUCommandGraph(device))).toThrow(
        'validate-transfer'
      );
    } finally {
      device.destroy();
    }
  });

  test('bounds dispatch in three dimensions and preserves nonaligned storage-view offsets', () => {
    const device = createDevice('bounded-device');
    Object.defineProperty(device.limits, 'maxComputeWorkgroupsPerDimension', {value: 2});
    const graph = new GPUCommandGraph(device);
    const recorder = new GPUComputationRecorder({sourceText: true});
    try {
      const source = createTransientView(graph, 'source', 'uint32', 2048);
      const targets = createTransientView(graph, 'targets', 'uint32', 2048);
      const target = graph.createDataView(
        graph.createTransientBuffer({id: 'target', byteLength: 280, usage: Buffer.STORAGE}),
        {format: 'uint32', length: 2, byteOffset: 260}
      );
      const nodes = new GPUTransferMask({
        id: 'bounded',
        source,
        relation: {layout: 'index', targets},
        target,
        recorder
      }).getCommandNodes(graph);
      const annotation = recorder.getNodeRecord(nodes[1].id);
      expect(annotation?.dispatch?.workgroups).toEqual([2, 2, 2]);
      expect(annotation?.source?.text).toContain('const targetMaskOffset: u32 = 1u;');
      expect(annotation?.source?.text).toContain('targetIndex < 2u');
    } finally {
      device.destroy();
    }
  });
});

function createDevice(id: string): NullDevice {
  const device = new NullDevice({id});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device.limits, 'maxComputeWorkgroupsPerDimension', {
    value: 65535,
    configurable: true
  });
  return device;
}
