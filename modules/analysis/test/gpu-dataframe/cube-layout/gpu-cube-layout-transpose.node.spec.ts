// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUCubeLayoutTranspose,
  type GPUCubeLayoutTransposeProps
} from '../../../src/dataframe-analysis/cube-layout/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCubeLayoutTransposeProps> = {}
): GPUCubeLayoutTransposeProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    values: view('uint32', 12),
    from: 'cell-major',
    cellCount: 4,
    periodCount: 3,
    output: view('uint32', 12),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUCubeLayoutTransposeProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUCubeLayoutTranspose(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const viewOf = <Format extends 'uint32' | 'float32' | 'sint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `v-${serial++}`, format, length);

describe('GPUCubeLayoutTranspose validation', () => {
  it('accepts both value formats and both directions', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    expect(new GPUCubeLayoutTranspose(createProps(graph)).elementCount).toBe(12);
    const floats = new GPUCubeLayoutTranspose(
      createProps(graph, {
        from: 'period-major',
        values: viewOf(graph, 'float32', 12),
        output: viewOf(graph, 'float32', 12)
      })
    );
    expect(floats.getCommandNodes(graph).map(node => node.id)).toEqual([
      'cube-layout-transpose-transpose'
    ]);
    device.destroy();
  });

  it('checks the shape against the cube', () => {
    expectThrows(() => ({cellCount: 0}), /cellCount must be an integer in \[1, 4294967295\]/);
    expectThrows(() => ({periodCount: 1.5}), /periodCount must be an integer/);
    expectThrows(
      () => ({cellCount: 2 ** 20, periodCount: 2 ** 20}),
      /cellCount \* periodCount must be at most 4294967295/
    );
    expectThrows(() => ({cellCount: 5}), /values must have 15 rows/);
    expectThrows(graph => ({output: viewOf(graph, 'uint32', 11)}), /output must have 12 rows/);
    expectThrows(() => ({from: 'row-major' as never}), /from must be cell-major or period-major/);
  });

  it('checks the formats and buffers', () => {
    expectThrows(graph => ({values: viewOf(graph, 'sint32', 12) as never}), /values/);
    expectThrows(graph => ({output: viewOf(graph, 'float32', 12)}), /output/);
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    expect(() => new GPUCubeLayoutTranspose({...props, output: props.values})).toThrow(
      /outputs must not share buffers with inputs/
    );
    device.destroy();
  });

  it('refuses a cube larger than one storage binding at declaration', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const limit = device.limits.maxStorageBufferBindingSize / 4;
    const add = (cellCount: number, periodCount: number) => {
      const length = cellCount * periodCount;
      graph.add(
        new GPUCubeLayoutTranspose({
          id: `transpose-${length}`,
          values: viewOf(graph, 'uint32', length),
          output: viewOf(graph, 'uint32', length),
          from: 'cell-major',
          cellCount,
          periodCount
        })
      );
    };
    expect(() => add(limit, 1)).not.toThrow();
    expect(() => add(limit + 1, 1)).toThrow(/cube needs 134217732 bytes in one binding/);
    device.destroy();
  });

  for (const format of ['uint32', 'float32'] as const) {
    for (const from of ['cell-major', 'period-major'] as const) {
      for (const [cellCount, periodCount] of [
        [268_435_456, 1],
        [1, 268_435_456]
      ]) {
        it(`declares a large degenerate ${format} cube from ${from}: ${cellCount}x${periodCount}`, () => {
          const device = createNullWebGPUDevice();
          Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2 ** 30});
          Object.defineProperty(device.limits, 'maxBufferSize', {value: 2 ** 30});
          const graph = new GPUCommandGraph(device);
          const length = cellCount * periodCount;
          const transpose = new GPUCubeLayoutTranspose({
            values: viewOf(graph, format, length),
            output: viewOf(graph, format, length),
            from,
            cellCount,
            periodCount
          });
          expect(transpose.getCommandNodes(graph).map(node => node.id)).toEqual([
            'cube-layout-transpose-transpose'
          ]);
          device.destroy();
        });
      }
    }

    for (const offsetView of ['values', 'output'] as const) {
      it(`rejects an aligned ${format} ${offsetView} binding above the limit`, () => {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device);
        const length = device.limits.maxStorageBufferBindingSize / 4;
        const storage = viewOf(graph, format, length + 1);
        const offset = graph.createDataView(storage.buffer, {format, length, byteOffset: 4});
        const transpose = new GPUCubeLayoutTranspose({
          values: offsetView === 'values' ? offset : viewOf(graph, format, length),
          output: offsetView === 'output' ? offset : viewOf(graph, format, length),
          from: 'cell-major',
          cellCount: length,
          periodCount: 1
        });
        const bindingName = offsetView === 'values' ? 'cube' : 'transposed';
        expect(() => transpose.getCommandNodes(graph)).toThrow(
          `GPUCubeLayoutTranspose: kernel cube-layout-transpose-transpose binding ${bindingName} ` +
            'needs 134217732 bytes, above the device limit of 134217728'
        );
        device.destroy();
      });
    }
  }
});
