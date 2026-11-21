// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createWrapperLossFixture,
  createLateCrossingFixture,
  createLateSharedFixture,
  LOSS_WRAPPERS,
  type WrapperView
} from './wrapper-loss-fixture';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';
import {useCPUExecutorTestHooks} from './cpu-executor-test-hooks';

useCPUExecutorTestHooks();
function createMemoryHarness() {
  const {device, graph} = createHarness();
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: WrapperView = (format, length, values) => {
    const result = createTransientView(graph, `late-${serial++}`, format, length);
    const data = values ?? Array(length * (format === 'uint32' ? 1 : 2)).fill(99);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  return {device, graph, memory, view};
}
it.each([4, 16])('a late crossing remains observable with %s leaves', leafCapacity => {
  const {device, graph, memory, view} = createMemoryHarness();
  const fixture = createLateCrossingFixture(leafCapacity, view);
  try {
    executeCommandNodesOnCPU(
      [...fixture.predicates.getCommandNodes(graph), ...fixture.split.getCommandNodes(graph)],
      memory
    );
    expect(Array.from(memory.readUint32(fixture.intersectionCount))).toEqual([
      leafCapacity === 4 ? 0 : 1
    ]);
    expect(Array.from(memory.readUint32(fixture.predicateOverflow))).toEqual([
      leafCapacity === 4 ? 1 : 0
    ]);
    expect(Array.from(memory.readUint32(fixture.simple))).toEqual([0]);
    expect(Array.from(memory.readUint32(fixture.pieceCount))).toEqual([leafCapacity === 4 ? 1 : 3]);
    expect(Array.from(memory.readUint32(fixture.pieceOverflow))).toEqual([0]);
    expect(Array.from(memory.readUint32(fixture.candidateOverflow))).toEqual([
      leafCapacity === 4 ? 1 : 0
    ]);
  } finally {
    device.destroy();
  }
});
it.each([4, 8])('a late shared segment remains observable with %s leaves', leafCapacity => {
  const {device, graph, memory, view} = createMemoryHarness();
  const fixture = createLateSharedFixture(leafCapacity, view);
  try {
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.count))).toEqual([leafCapacity === 4 ? 0 : 1]);
    expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([leafCapacity === 4 ? 1 : 0]);
    expect(Array.from(memory.readUint32(fixture.candidateOverflow))).toEqual([
      leafCapacity === 4 ? 1 : 0
    ]);
  } finally {
    device.destroy();
  }
});

function createHarness() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view: WrapperView = (format, length) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  return {device, graph, view};
}

it.each(LOSS_WRAPPERS)('%s refuses hidden loss from restricted leaves', kind => {
  const {device, view} = createHarness();
  try {
    const fixture = createWrapperLossFixture(kind, view);
    const contributor = fixture.contributor;
    expect(() => {
      if (contributor instanceof GPULineSplit) {
        const props = contributor.props;
        return new GPULineSplit({
          ...props,
          output: {
            pieces: {
              ...props.output.pieces,
              status: {count: view('uint32', 1), overflow: view('uint32', 1)}
            }
          }
        });
      }
      if (contributor instanceof GPULineClipByPolygon) {
        const props = contributor.props;
        return new GPULineClipByPolygon({
          ...props,
          output: {
            pieces: {
              ...props.output.pieces,
              // A JavaScript caller can omit `overflow`; the constructor must reject that.
              status: {count: view('uint32', 1)} as typeof props.output.pieces.status
            }
          }
        });
      }
      if (contributor instanceof GPUSharedPaths)
        return new GPUSharedPaths({
          ...contributor.props,
          output: {...contributor.props.output, status: {count: view('uint32', 1)}}
        });
      return new GPUGeometryPredicates({
        ...contributor.props,
        output: {status: {intersectionCount: view('uint32', 1)}}
      });
    }).toThrow(/leafCapacity.*(candidateOverflow|overflow)/);
  } finally {
    device.destroy();
  }
});

it('large status-only point geometry does not need a segment table', () => {
  const {device, graph, view} = createHarness();
  try {
    const contributor = new GPUGeometryPredicates({
      geometry: {kind: 'points', positions: view('float32x2', 4194305)},
      intersectionCapacity: 1,
      output: {status: {overflow: view('uint32', 1)}}
    });
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

for (const name of ['complete', 'unresolvedCount'] as const) {
  it.each([
    'split',
    'clip'
  ] as const)(`%s refuses unsupported ${name} status before snapshot loss`, kind => {
    const {device, view} = createHarness();
    try {
      const contributor = createWrapperLossFixture(kind, view).contributor;
      if (!(contributor instanceof GPULineSplit || contributor instanceof GPULineClipByPolygon))
        throw new Error('invalid fixture');
      const status = {...contributor.props.output.pieces.status, [name]: view('uint32', 1)};
      expect(() =>
        contributor instanceof GPULineSplit
          ? new GPULineSplit({
              ...contributor.props,
              output: {pieces: {...contributor.props.output.pieces, status}}
            })
          : new GPULineClipByPolygon({
              ...contributor.props,
              output: {pieces: {...contributor.props.output.pieces, status}}
            })
      ).toThrow(new RegExp(`output.pieces.status.${name}`));
    } finally {
      device.destroy();
    }
  });
}
