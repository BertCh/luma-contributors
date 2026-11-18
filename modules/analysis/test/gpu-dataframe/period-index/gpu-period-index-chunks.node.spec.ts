// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {readFileSync} from 'node:fs';
import {expect, it, vi} from 'vitest';
import {
  GPUPeriodIndex,
  type GPUPeriodIndexProps
} from '../../../src/dataframe-analysis/period-index/gpu-period-index';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../calendar-buckets/chunk-shader-validation.test-utils';

function createFixture(lengths: number[], kind: 'calendar' | 'fixed' = 'calendar') {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const column = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length + 128);
        return graph.createDataView(backing.buffer, {format, byteOffset: (index % 64) * 4, length});
      })
    );
  const props: GPUPeriodIndexProps<GraphVectorView> = {
    id: 'periods',
    temporalContext: {
      unit: 'millisecond',
      epoch: 'unix',
      encoding: 'int64-words',
      axis: {kind: 'irregular'}
    },
    periods:
      kind === 'calendar'
        ? {kind, step: {calendarUnit: 'hour'}, origin: {year: 2024}, count: 12}
        : {kind, origin: 0n, width: 1000, count: 12},
    columns:
      kind === 'calendar'
        ? {
            year: column('year', 'sint32'),
            month: column('month', 'uint32'),
            dayOfMonth: column('day', 'uint32'),
            hour: column('hour', 'uint32')
          }
        : undefined,
    timestamps: kind === 'fixed' ? column('timestamps', 'uint32x2') : undefined,
    mask: column('mask', 'uint32'),
    output: {
      periods: column('periods', 'uint32'),
      status: {invalidCount: createTransientView(graph, 'invalid-count', 'uint32', 1)}
    }
  };
  return {device, graph, props, column};
}

it('documents aligned chunk limits instead of refusing total rows', () => {
  const {device, graph, props} = createFixture([1, 2], 'fixed');
  try {
    // Three timestamp rows need 24 bytes in total, but each chunk fits a 20-byte binding.
    // The second chunk starts four bytes into its backing buffer and needs 16 bytes of data.
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 20});
    const contributor = new GPUPeriodIndex(props);
    expect(contributor.rowCount).toBe(3);
    expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
      'periods-clear-invalid-count',
      'periods-periods-chunk-0',
      'periods-periods-chunk-1'
    ]);
    const guide = readFileSync(
      new URL('../../../../../docs/api/gpu-dataframe-analysis.md', import.meta.url),
      'utf8'
    )
      .split('## `GPUPeriodIndex`')[1]
      .split('## `GPUCubeLayoutTranspose`')[0];
    expect(guide).not.toContain('Inputs must be single packed views');
    expect(guide).toContain('matching ordered chunk lengths');
    expect(guide).toContain('per chunk');
  } finally {
    device.destroy();
  }
});

it('validates row topology and every writable field and chunk', () => {
  const {device, graph, props, column} = createFixture([1, 3, 2]);
  try {
    expect(
      () => new GPUPeriodIndex({...props, mask: column('bad-mask', 'uint32').data[0]})
    ).toThrow(/mask must have 6 rows/);
    const reordered = createVectorView(
      'bad-columns',
      'uint32',
      [...column('reorder', 'uint32').data].reverse()
    );
    expect(
      () => new GPUPeriodIndex({...props, columns: {...props.columns!, month: reordered}})
    ).toThrow(/calendarMonth chunk 0.*topology/);
    const output = column('output', 'uint32');
    expect(
      () =>
        new GPUPeriodIndex({
          ...props,
          output: {periods: createVectorView('repeat', 'uint32', [output.data[1], output.data[1]])}
        })
    ).toThrow(/must not overlap/);
    const invalidCount = graph.createDataView(output.data[1].buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    expect(
      () => new GPUPeriodIndex({...props, output: {periods: output, status: {invalidCount}}})
    ).toThrow(/must not overlap/);
  } finally {
    device.destroy();
  }
});

it('rejects calendar operands that cannot be emitted as signed WGSL literals', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    for (const interval of [NaN, Infinity, -Infinity, 1e100, 0x80000000]) {
      expect(
        () =>
          new GPUPeriodIndex({
            ...props,
            periods: {
              kind: 'calendar',
              step: {calendarUnit: 'year', interval},
              origin: {year: 1970},
              count: 1
            }
          })
      ).toThrow(/periods.step.*(positive integer|MAXIMUM_CALENDAR_STEPS_PER_PERIOD)/);
    }
    expect(
      () =>
        new GPUPeriodIndex({
          ...props,
          periods: {
            kind: 'calendar',
            step: {calendarUnit: 'week', interval: 306_783_379},
            origin: {year: 1970},
            count: 1
          }
        })
    ).toThrow(/MAXIMUM_CALENDAR_STEPS_PER_PERIOD.*2147483647/);
    for (const year of [-1_000_000, 1_000_000]) {
      expect(
        () =>
          new GPUPeriodIndex({
            ...props,
            periods: {kind: 'calendar', step: {calendarUnit: 'hour'}, origin: {year}, count: 1}
          })
      ).toThrow(/periods.origin.*CALENDAR_STEP_INDEX/);
    }
    for (const [calendarUnit, interval] of [
      ['year', 0x7fffffff],
      ['week', 306_783_378]
    ] as const) {
      const contributor = new GPUPeriodIndex({
        ...props,
        periods: {kind: 'calendar', step: {calendarUnit, interval}, origin: {year: 1970}, count: 1}
      });
      const nodes = contributor.getCommandNodes(graph);
      getParsedKernelSources(nodes);
      expectValidShaderIdentifiers(
        getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)
      );
    }
    for (const hourIndex of [-0x80000000, 0x7fffffff]) {
      const instant = new Date(hourIndex * 3_600_000);
      const contributor = new GPUPeriodIndex({
        ...props,
        periods: {
          kind: 'calendar',
          step: {calendarUnit: 'hour'},
          count: 1,
          origin: {
            year: instant.getUTCFullYear(),
            month: instant.getUTCMonth() + 1,
            day: instant.getUTCDate(),
            hour: instant.getUTCHours()
          }
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      getParsedKernelSources(nodes);
      expectValidShaderIdentifiers(
        getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)
      );
    }
  } finally {
    device.destroy();
  }
});

it('prechecks later input/output chunks, status bindings and graph ownership before graph changes', () => {
  for (const failure of ['input', 'output', 'status']) {
    const {device, graph, props} = createFixture([1, 3, 2], 'fixed');
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value: failure === 'input' ? 24 : 256
    });
    if (failure === 'output') {
      const backing = createTransientView(graph, 'large-output', 'uint32', 128);
      const view = graph.createDataView(backing.buffer, {
        format: 'uint32',
        byteOffset: 252,
        length: 3
      });
      const output = props.output.periods as ReturnType<typeof createVectorView<'uint32'>>;
      props.output.periods = createVectorView('offset-output', 'uint32', [
        output.data[0],
        view,
        output.data[2]
      ]);
    }
    if (failure === 'status')
      props.output.status = {
        invalidCount: createTransientView(graph, 'large-status', 'uint32', 65)
      };
    const contributor = new GPUPeriodIndex(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs .*bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
      /views must belong/
    );
    device.destroy();
  }
});

for (const kind of ['calendar', 'fixed'] as const) {
  it(`keeps ${kind} shader/pipeline counts constant and accepts 30 million rows`, () => {
    const keys = new Set<string>();
    const sources = new Set<string>();
    for (const lengths of [
      [513],
      [513, 1, 257],
      Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
      [0, 10_000_000, 9_000_000, 11_000_000]
    ]) {
      const {device, graph, props} = createFixture(lengths, kind);
      const contributor = new GPUPeriodIndex(props);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBe(
        lengths.filter(length => length > 0).length + 1
      );
      expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(2);
      if (lengths[0] === 513)
        for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
          keys.add(key);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
      device.destroy();
    }
    expect(keys.size).toBe(2);
    expect(sources.size).toBe(2);
    expectValidShaderIdentifiers(sources);
  });
}

it('does not bind explicitly ignored period inputs, even when they exceed the device limit', () => {
  for (const kind of ['calendar', 'fixed'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const ignored = createTransientView(graph, 'ignored', 'uint32', 33_554_433);
    const contributor = new GPUPeriodIndex({
      id: 'ignored-inputs',
      temporalContext: {
        unit: 'millisecond',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      },
      periods:
        kind === 'calendar'
          ? {kind, step: {calendarUnit: 'year'}, origin: {year: 2024}, count: 2}
          : {kind, origin: 0n, width: 1n, count: 2},
      columns: {
        year: createTransientView(graph, 'year', 'sint32', kind === 'calendar' ? 1 : 33_554_433),
        month: ignored,
        dayOfMonth: ignored,
        hour: ignored
      },
      timestamps: createTransientView(
        graph,
        'timestamps',
        'uint32x2',
        kind === 'fixed' ? 1 : 16_777_217
      ),
      output: {periods: createTransientView(graph, 'periods', 'uint32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(1);
    expect(
      nodes[0].resources?.some(
        resource =>
          'buffer' in resource &&
          ('buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer) ===
            ignored.buffer
      )
    ).toBe(false);
    expect(getParsedKernelSources(nodes).size).toBe(1);
    device.destroy();
  }
});
