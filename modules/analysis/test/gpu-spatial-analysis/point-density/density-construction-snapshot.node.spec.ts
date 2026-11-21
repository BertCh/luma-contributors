// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUAdaptiveBandwidthKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {GPUSpaceTimeRelativeRisk} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-relative-risk';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

for (const kind of ['fixed', 'adaptive', 'space-time', 'risk'] as const) {
  it(`W2-02 ${kind} captures construction metadata and routing before declaration`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 2);
    const times = createTransientView(graph, 'times', 'float32', 2);
    const parameters = createTransientView(graph, 'parameters', 'float32', 16);
    const values = createTransientView(graph, 'values', 'float32', 1);
    const invalidCount = createTransientView(graph, 'invalid', 'uint32', 1);
    const props = {
      id: 'captured',
      positions,
      times,
      parameters,
      casePositions: positions,
      caseTimes: times,
      controlPositions: positions,
      controlTimes: times,
      grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
      indexGridSize: [1, 1] as [number, number],
      indexBounds: [0, 0, 1, 1] as [number, number, number, number],
      spatialContext: {coordinateSpace: 'planar', metric: 'native'} as const,
      temporalContext: {
        encoding: 'float32-relative',
        unit: 'second',
        epoch: 0n,
        axis: {kind: 'irregular'}
      } as const,
      sliceCount: 1,
      output: {values, status: {invalidCount, caseInvalidCount: invalidCount}}
    };
    // Risk requires disjoint diagnostic outputs; the control status is omitted.
    const contributor =
      kind === 'fixed'
        ? new GPUKernelDensity(props)
        : kind === 'adaptive'
          ? new GPUAdaptiveBandwidthKernelDensity(props)
          : kind === 'space-time'
            ? new GPUSpaceTimeKernelDensity(props)
            : new GPUSpaceTimeRelativeRisk(props);
    props.id = 'mutated';
    props.grid.width = 7;
    (props.grid.affine as unknown as number[])[0] = 99;
    props.indexBounds[0] = -99;
    props.indexGridSize[0] = 7;
    props.sliceCount = 7;
    (props.temporalContext as {epoch: bigint}).epoch = 99n;
    if ('temporalContext' in contributor.props) {
      expect(contributor.props.temporalContext?.epoch).toBe(0n);
    }
    props.output.values = createTransientView(graph, 'replacement', 'float32', 1);
    props.output.status.invalidCount = createTransientView(
      graph,
      'replacement-status',
      'uint32',
      1
    );
    props.output.status.caseInvalidCount = props.output.status.invalidCount;
    expect(contributor.props.grid.width).toBe(1);
    expect(contributor.props.grid.affine[0]).toBe(1);
    expect(contributor.props.indexBounds).toEqual([0, 0, 1, 1]);
    expect(contributor.props.indexGridSize).toEqual([1, 1]);
    expect(contributor.props.output.values).toBe(values);
    expect(contributor.props.parameters).toBe(parameters);
    expect(contributor.props.spatialContext).not.toBe(props.spatialContext);
    if (kind !== 'adaptive') {
      const status = (contributor.props.output as {status?: Record<string, unknown>}).status;
      expect(status).toBeDefined();
      expect(status).not.toBe(props.output.status);
      expect(status?.[kind === 'risk' ? 'caseInvalidCount' : 'invalidCount']).toBe(invalidCount);
    }
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeGreaterThan(0);
    if (kind !== 'adaptive') {
      expect(
        nodes.some(node =>
          node.resources?.some(resource => 'buffer' in resource && resource.buffer === invalidCount)
        )
      ).toBe(true);
    }
    expect(
      getGPUCommandNodeKernelSources(nodes).every(kernel => !kernel.source.includes('mutated'))
    ).toBe(true);
    device.destroy();
  });
}

it('W2-02 fixed KDE captures inherited getters once, including nested routing', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 1);
  const reads = {id: 0, grid: 0, width: 0, output: 0, values: 0};
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  Object.defineProperty(grid, 'width', {
    get: () => {
      reads.width++;
      return 1;
    }
  });
  const output = Object.create({
    get values() {
      reads.values++;
      return values;
    }
  }) as {values: typeof values};
  const props = Object.create({
    get id() {
      reads.id++;
      return 'density-getter';
    },
    get grid() {
      reads.grid++;
      return grid;
    },
    get output() {
      reads.output++;
      return output;
    }
  });
  Object.assign(props, {
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    indexBounds: [0, 0, 1, 1],
    indexGridSize: [1, 1]
  });
  const contributor = new GPUKernelDensity(props);
  expect(contributor.id).toBe('density-getter');
  expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  expect(reads).toEqual({id: 1, grid: 1, width: 1, output: 1, values: 1});
  expect(
    () =>
      new GPUKernelDensity({
        ...contributor.props,
        id: 'density-requirement',
        parameters: createTransientView(graph, 'short-parameters', 'float32', 8)
      })
  ).toThrow(/density-requirement parameters must hold 9 float32 values/);
  device.destroy();
});

for (const kind of ['fixed', 'adaptive', 'space-time', 'risk'] as const) {
  it(`R1 B6 ${kind} captures only declared metadata on cyclic structural wrappers`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
    const reads = {unrelated: 0};
    Object.assign(grid, {layer: {grid}});
    Object.defineProperty(grid, 'unrelated', {
      get() {
        reads.unrelated++;
        throw new Error('unrelated getter');
      }
    });
    const spatialContext = {coordinateSpace: 'planar', metric: 'native'} as const;
    Object.assign(spatialContext, {owner: {grid}});
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    const times = createTransientView(graph, 'times', 'float32', 1);
    const props = {
      id: 'structural-wrapper',
      grid,
      spatialContext,
      positions,
      times,
      casePositions: positions,
      controlPositions: positions,
      caseTimes: times,
      controlTimes: times,
      parameters: createTransientView(graph, 'parameters', 'float32', 14),
      sliceCount: 1,
      indexBounds: [0, 0, 1, 1] as const,
      indexGridSize: [1, 1] as const,
      output: {values: createTransientView(graph, 'values', 'float32', 1)}
    };
    const contributor =
      kind === 'fixed'
        ? new GPUKernelDensity(props)
        : kind === 'adaptive'
          ? new GPUAdaptiveBandwidthKernelDensity(props)
          : kind === 'space-time'
            ? new GPUSpaceTimeKernelDensity(props)
            : new GPUSpaceTimeRelativeRisk(props);
    expect(reads.unrelated).toBe(0);
    expect('layer' in contributor.props.grid).toBe(false);
    expect('owner' in contributor.props.spatialContext!).toBe(false);
    expect(contributor.props.grid.width).toBe(1);
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
    device.destroy();
  });
}

it('R1 S2 malformed study windows reach the ID-prefixed structural validator', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    id: 'window-validation',
    grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    indexBounds: [0, 0, 1, 1] as const,
    indexGridSize: [1, 1] as const,
    output: {values: createTransientView(graph, 'values', 'float32', 1)}
  };
  for (const studyWindow of [
    {},
    {
      kind: 'polygon',
      id: 'bad-window',
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      getViews: 42
    }
  ]) {
    expect(() => new GPUKernelDensity({...props, studyWindow: studyWindow as never})).toThrow(
      /window-validation studyWindow must be 'rectangle' or a GPUStudyWindowPreparation/
    );
  }
  device.destroy();
});

it('R1 S3 relative risk rejects a short output when both intermediate densities are supplied', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 1);
  const times = createTransientView(graph, 'times', 'float32', 1);
  expect(
    () =>
      new GPUSpaceTimeRelativeRisk({
        id: 'risk-capacity',
        casePositions: positions,
        controlPositions: positions,
        caseTimes: times,
        controlTimes: times,
        parameters: createTransientView(graph, 'parameters', 'float32', 14),
        grid: makeExtentGrid(3, 1, [0, 0, 3, 1]),
        sliceCount: 1,
        indexBounds: [0, 0, 3, 1],
        indexGridSize: [1, 1],
        output: {
          values: createTransientView(graph, 'risk', 'float32', 1),
          caseDensity: createTransientView(graph, 'case-density', 'float32', 3),
          controlDensity: createTransientView(graph, 'control-density', 'float32', 3)
        }
      })
  ).toThrow(/risk-capacity output.values.*3/);
  device.destroy();
});
