// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUValueCube} from '../../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('UDF-14 rejects legacy mode props from untyped inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const keys = createTransientView(graph, 'keys', 'uint32', 2);
  const props = {leftKeys: keys, rightKeys: keys, output: {}};
  try {
    for (const legacyProp of ['kind', 'how']) {
      const message = new RegExp(`${legacyProp} was removed.*compactMatchedLeftRows`);
      expect(() => new GPUKeyLookup({...props, [legacyProp]: 'left'})).toThrow(message);
      const virtualProps = new Proxy(props, {
        get: (target, property) =>
          property === legacyProp ? 'inner' : Reflect.get(target, property)
      });
      expect(legacyProp in virtualProps).toBe(false);
      expect(() => new GPUKeyLookup(virtualProps)).toThrow(message);
    }
  } finally {
    device.destroy();
  }
});

it('CMP-03 validates unsigned unmatched fills', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const keys = createTransientView(graph, 'keys', 'uint32', 2);
  const output = createTransientView(graph, 'output', 'uint32', 2);
  try {
    expect(
      () =>
        new GPUKeyLookup({
          leftKeys: keys,
          rightKeys: keys,
          gather: [{column: keys, output, unmatchedValue: -1}],
          output: {}
        })
    ).toThrow(/unmatchedValue/);
  } finally {
    device.destroy();
  }
});

it('W2-03 cube means publish exact words without a rounded sum dependency', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `cube-${serial++}`, format, length);
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPUValueCube({
        cellCount: 1,
        periodCount: 2,
        cells: view('uint32', 7),
        periods: view('uint32', 7),
        values: view('float32', 7),
        output: {means: view('float32', 2)}
      }).getCommandNodes(graph)
    );
    const source = kernels.find(kernel => kernel.id === 'value-cube-means')!.source;
    expect(source).toContain('getFiniteFixedPointMean(words, unitExponent, count)');
    expect(source).not.toContain('sumsIn');
  } finally {
    device.destroy();
  }
});

it('FLT-15 integer count and distinct publication preserve integer bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `count-${serial++}`, format, length);
  try {
    for (const operation of ['count', 'count-distinct'] as const) {
      const kernels = getGPUCommandNodeKernelSources(
        new GPUKeyLookup({
          id: `count-${operation}`,
          leftKeys: view('uint32', 2),
          rightKeys: view('uint32', 3),
          aggregates: [{operation, column: view('float32', 3), output: view('uint32', 2)}],
          output: {}
        }).getCommandNodes(graph)
      );
      const finish = kernels.find(kernel => kernel.id.endsWith('-finish'))!;
      expect(finish.source).not.toContain('f32(count');
      expect(finish.source).toContain('array<u32>');
    }
  } finally {
    device.destroy();
  }
});

it('wave three independently audits generated shared and independent-scale kernels', async () => {
  const {GPUGroupStatistics} = await import(
    '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics'
  );
  const {auditWGSLModule} = await import('../../analysis-core/kernel-audit/kernel-audit-wgsl');
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `audit-${serial++}`, format, length);
  try {
    for (const scaleMode of ['shared', 'per-group'] as const) {
      const producers = [
        new GPUKeyLookup({
          id: `join-${scaleMode}`,
          scaleMode,
          leftKeys: view('uint32', 2),
          rightKeys: view('uint32', 3),
          aggregates: [{operation: 'mean', column: view('float32', 3), output: view('float32', 2)}],
          output: {status: {complete: view('uint32', 1)}}
        }),
        new GPUGroupStatistics({
          id: `groups-${scaleMode}`,
          scaleMode,
          scratchTileSize: scaleMode === 'per-group' ? 2 : undefined,
          keys: view('uint32', 3),
          columns: [
            {
              values: view('float32', 3),
              statistics: ['sum', 'mean'],
              output: {sumValues: view('float32', 2), means: view('float32', 2)}
            }
          ],
          output: {
            keys: view('uint32', 2),
            counts: view('uint32', 2),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }),
        new GPUValueCube({
          id: `cube-${scaleMode}`,
          scaleMode: scaleMode === 'shared' ? 'shared' : 'per-bin',
          cellCount: 1,
          periodCount: 2,
          cells: view('uint32', 3),
          periods: view('uint32', 3),
          values: view('float32', 3),
          output: {means: view('float32', 2)}
        })
      ];
      for (const producer of producers) {
        const kernels = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
        if (producer instanceof GPUKeyLookup) {
          expect(kernels.filter(kernel => kernel.id.endsWith('-sum-scale'))).toHaveLength(1);
        }
        for (const kernel of kernels) {
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
          if (kernel.id.startsWith('join-') && kernel.id.endsWith('-sum-scale')) {
            expect(kernel.source).not.toContain(' + unique');
            if (scaleMode === 'shared') {
              expect(kernel.source).toContain('SUM_SCALE_INDEX], workgroupMaximum');
              expect(kernel.source).not.toContain('atomicAdd(&counts');
            } else {
              expect(kernel.source).toContain('SUM_SCALE_INDEX + group]');
              expect(kernel.source).toContain('atomicAdd(&counts[countsOffset + group], 1u)');
              expect(kernel.source).toContain('(bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u');
            }
          }
        }
      }
    }
  } finally {
    device.destroy();
  }
});

it('DEP snapshots nested loss status and rejects the flat loss port', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `loss-${serial++}`, format, length);
  try {
    const sumOverflow = view('uint32', 1);
    const aggregate = {
      operation: 'sum' as const,
      column: view('float32', 2),
      output: view('float32', 1),
      sums: view('uint32x2', 1),
      status: {sumOverflow}
    };
    const props = {
      leftKeys: view('uint32', 1),
      rightKeys: view('uint32', 2),
      aggregates: [aggregate],
      output: {}
    };
    const producer = new GPUKeyLookup(props);
    aggregate.status.sumOverflow = view('uint32', 1);
    expect(producer.props.aggregates![0].status?.sumOverflow).toBe(sumOverflow);
    const legacyAggregate = {...aggregate, status: undefined, sumOverflow};
    expect(() => new GPUKeyLookup({...props, aggregates: [legacyAggregate]})).toThrow(
      /sumOverflow was removed.*status\.sumOverflow/
    );
    const virtualAggregate = new Proxy(aggregate, {
      get: (target, property) =>
        property === 'sumOverflow' ? sumOverflow : Reflect.get(target, property)
    });
    expect('sumOverflow' in virtualAggregate).toBe(false);
    expect(() => new GPUKeyLookup({...props, aggregates: [virtualAggregate]})).toThrow(
      /sumOverflow was removed.*status\.sumOverflow/
    );
    expect(
      () => new GPUKeyLookup({...props, aggregates: [{...aggregate, sums: undefined}]})
    ).toThrow(/requires sums/);
  } finally {
    device.destroy();
  }
});
