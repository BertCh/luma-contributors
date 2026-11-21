// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {GPUValueCube} from '../../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {computeColumnStatistics} from '../group-statistics/group-statistics-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('B1 B2 carries both upstream loss words into certification and compact publication', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = (length = 1) => createTransientView(graph, `review-${serial++}`, 'uint32', length);
  try {
    const keys = view(2);
    const overflow = view();
    const candidateOverflow = view();
    const nodes = new GPUKeyLookup({
      compactMatchedLeftRows: true,
      leftKeys: view(2),
      rightKeys: keys,
      rightTable: {keys, status: {overflow, candidateOverflow}},
      output: {
        status: {complete: view(), candidateOverflow: view()},
        rows: {ids: view(), status: {count: view(), overflow: view(), candidateOverflow: view()}}
      }
    }).getCommandNodes(graph);
    expect(
      () =>
        new GPUKeyLookup({
          leftKeys: view(2),
          rightKeys: keys,
          rightTable: {keys, status: {overflow, candidateOverflow: view(2)}},
          output: {status: {complete: view()}}
        })
    ).toThrow(/candidateOverflow must hold one row/);
    for (const suffix of ['certify', 'publish']) {
      const node = nodes.find(candidate => candidate.id.endsWith(`-${suffix}`))!;
      expect(
        node.resources!.some(
          resource => 'buffer' in resource && resource.buffer === candidateOverflow
        ),
        suffix
      ).toBe(true);
    }
  } finally {
    device.destroy();
  }
});

it('B4 captures inherited and non-enumerable declared props and nested ports', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `snapshot-${serial++}`, format, length);
  try {
    const mask = view('uint32', 2);
    const group = new GPUGroupStatistics(
      Object.assign(
        Object.create({
          get mask() {
            return mask;
          }
        }),
        {
          keys: view('uint32', 2),
          columns: [],
          output: {
            keys: view('uint32', 2),
            counts: view('uint32', 2),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }
      )
    );
    expect(group.props.mask).toBe(mask);
    const means = view('float32', 1);
    const cube = new GPUValueCube(
      Object.assign(
        Object.create({
          get mask() {
            return mask;
          }
        }),
        {
          cellCount: 1,
          periodCount: 1,
          cells: view('uint32', 2),
          periods: view('uint32', 2),
          values: view('float32', 2),
          output: Object.create({
            get means() {
              return means;
            }
          })
        }
      )
    );
    expect(cube.props.mask).toBe(mask);
    expect(cube.props.output.means).toBe(means);
    expect(
      getGPUCommandNodeKernelSources(cube.getCommandNodes(graph)).find(kernel =>
        kernel.id.endsWith('-accumulate')
      )!.source
    ).toContain('mask[maskOffset');
    const aggregate = Object.create({
      get operation() {
        return 'count';
      },
      get output() {
        return view('float32', 2);
      }
    });
    expect(
      () =>
        new GPUKeyLookup({
          leftKeys: view('uint32', 2),
          rightKeys: view('uint32', 2),
          aggregates: [aggregate],
          output: {}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('earlier oracle fix directly rounds a sum above a float32 midpoint', () => {
  const result = computeColumnStatistics([1, 2 ** -24, 2 ** -60], {
    variance: 'population',
    fractions: [],
    sumScale: 1
  });
  // Sum is strictly above midpoint 1+2^-24. Nearest binary32 is 1+2^-23.
  expect(result.sumValue).toBe(1.0000001192092896);
});

it('B3 independent reductions derive headroom from resident local counts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `local-${serial++}`, format, length);
  try {
    const producers = [
      new GPUKeyLookup({
        scaleMode: 'per-group',
        leftKeys: view('uint32', 1),
        rightKeys: view('uint32', 4),
        aggregates: [{operation: 'sum', column: view('float32', 4), output: view('float32', 1)}],
        output: {}
      }),
      new GPUGroupStatistics({
        scaleMode: 'per-group',
        keys: view('uint32', 4),
        columns: [
          {
            values: view('float32', 4),
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
        scaleMode: 'per-bin',
        cellCount: 1,
        periodCount: 2,
        cells: view('uint32', 4),
        periods: view('uint32', 4),
        values: view('float32', 4),
        output: {sums: view('float32', 2), means: view('float32', 2)}
      })
    ];
    for (const producer of producers) {
      const sources = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
      for (const kernel of sources)
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      expect(
        sources.some(
          kernel =>
            kernel.source.includes('firstLeadingBit(max(') && kernel.source.includes('atomicLoad(&')
        ),
        producer.id
      ).toBe(true);
      const finish = sources.find(
        kernel =>
          kernel.id.endsWith('-finish') ||
          kernel.id.endsWith('-finish-sums') ||
          kernel.id.endsWith('-means')
      )!;
      expect(finish.source, producer.id).toContain('firstLeadingBit(max(');
    }
  } finally {
    device.destroy();
  }
});

it('B4 captures a non-enumerable cube mask once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `nonenum-${serial++}`, format, length);
  let reads = 0;
  try {
    const mask = view('uint32', 2);
    const props = {
      cellCount: 1,
      periodCount: 1,
      cells: view('uint32', 2),
      periods: view('uint32', 2),
      values: view('float32', 2),
      output: {means: view('float32', 1)}
    };
    Object.defineProperty(props, 'mask', {
      get() {
        reads++;
        return mask;
      },
      enumerable: false
    });
    const cube = new GPUValueCube(props);
    expect(cube.props.mask).toBe(mask);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});
