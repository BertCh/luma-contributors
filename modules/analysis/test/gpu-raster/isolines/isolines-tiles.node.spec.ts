// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {computeIsolinesOnCPU} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {
  validateIsolinesTile,
  validateIsolinesMaximumSegmentCount
} from '../../../src/raster-analysis/isolines/isolines-tile';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  expectSameSegments,
  expectSkimageSegments,
  getSegmentRows,
  getTileScenes,
  getWholeScene,
  getLargeOffsetScene,
  tileFixtures
} from './isolines-tiles-test-utils';

for (const fixture of tileFixtures) {
  it(`CPU tiled union equals whole-domain and independent skimage: ${fixture.name}`, () => {
    const wholeScene = getWholeScene(fixture);
    const whole = computeIsolinesOnCPU(wholeScene, {...wholeScene, segmentCapacity: 256});
    const tileRows = getTileScenes(fixture).flatMap(scene => {
      const result = computeIsolinesOnCPU(scene, {
        ...scene,
        segmentCapacity: 256,
        vertexCapacity: 512
      });
      expect(result.overflow[0]).toBe(0);
      expect(result.polylineOverflow[0]).toBe(0);
      // Every directed edge in a fragment's polyline comes from an owned segment.
      expect(result.vertexCount[0]).toBe(result.count[0] + result.polylineCount[0]);
      return getSegmentRows(result);
    });
    expectSameSegments(tileRows, getSegmentRows(whole));
    expectSkimageSegments(tileRows, fixture);
    expect(new Set(tileRows.map(row => row.slice(0, 3).join(':'))).size).toBe(tileRows.length);
  });
}

it('counts only owned cells before applying segment and complete-polyline capacities', () => {
  const scene = getTileScenes(tileFixtures[0])[3];
  const full = computeIsolinesOnCPU(scene, {...scene, segmentCapacity: 256});
  const limited = computeIsolinesOnCPU(scene, {...scene, segmentCapacity: 1, vertexCapacity: 2});
  expect(full.count[0]).toBeGreaterThan(1);
  expect(limited.requiredCount[0]).toBe(full.count[0]);
  expect(limited.count[0]).toBe(1);
  expect(limited.overflow[0]).toBe(1);
  expect(limited.polylineOverflow[0]).toBe(1);
  expect(limited.polylineCount[0]).toBe(0);
  const emptyTile = {...scene.tile!, ownedCells: [1, 1, 1, 2] as const};
  const empty = computeIsolinesOnCPU(scene, {
    ...scene,
    tile: emptyTile,
    segmentCapacity: 1,
    vertexCapacity: 2
  });
  for (const port of [
    'count',
    'requiredCount',
    'overflow',
    'polylineCount',
    'vertexCount',
    'polylineOverflow'
  ] as const)
    expect(empty[port][0]).toBe(0);
});

it('retains zero-length equality bridges and domain edge identities at four-tile corners', () => {
  const fixture = {...tileFixtures[0], levels: [9]};
  const wholeScene = getWholeScene(fixture);
  const whole = computeIsolinesOnCPU(wholeScene, {...wholeScene, segmentCapacity: 256});
  const tileRows = getTileScenes(fixture).flatMap(scene =>
    getSegmentRows(computeIsolinesOnCPU(scene, {...scene, segmentCapacity: 256}))
  );
  // At the split's sample (column 3, row 3), the lower-left cell owns this equality bridge.
  expect(tileRows).toContainEqual([0, 23, 68, 7, 10.5, 7, 10.5]);
  expectSameSegments(tileRows, getSegmentRows(whole));
});

it('validates half-open ownership, halo placement and domain identifier limits by name', () => {
  const scene = getTileScenes(tileFixtures[0])[3];
  const tile = scene.tile!;
  for (const [changes, prop] of [
    [{columnOffset: -1}, 'columnOffset'],
    [{rowOffset: 0.5}, 'rowOffset'],
    [{domainWidth: 65536, domainHeight: 65536}, 'domain samples'],
    [{columnOffset: tile.domainWidth}, 'columnOffset'],
    [{ownedCells: [2, 0, 1, 2]}, 'maxColumn'],
    [{ownedCells: [0, 0, scene.grid.width, 2]}, 'maxColumn'],
    [{ownedCells: [0, -1, 1, 2]}, 'minRow']
  ] as const) {
    const invalid = {...tile, ...changes};
    expect(() => validateIsolinesTile('seam-test', scene.grid, invalid)).toThrow(
      new RegExp(`seam-test.*${prop}`)
    );
    expect(() =>
      computeIsolinesOnCPU(scene, {...scene, tile: invalid, segmentCapacity: 256})
    ).toThrow(new RegExp(prop));
  }
});

it('schedules domain edge publication after local stitching and checks scratch limits before nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const scene = getTileScenes(tileFixtures[0])[3];
    const contributor = new GPUIsolines({
      ...scene,
      values: createTransientView(graph, 'values', 'float32', scene.values.length),
      mask: createTransientView(graph, 'mask', 'uint32', scene.mask.length),
      levels: createTransientView(graph, 'levels', 'float32', scene.levels.length),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      output: {
        segments: createTransientView(graph, 'segments', 'float32x4', 32),
        segmentLevels: createTransientView(graph, 'segmentLevels', 'uint32', 32),
        segmentEdges: createTransientView(graph, 'segmentEdges', 'uint32x2', 32),
        polylines: {
          vertices: createTransientView(graph, 'vertices', 'float32x2', 64),
          polylineOffsets: createTransientView(graph, 'polylineOffsets', 'uint32', 33),
          polylineLevels: createTransientView(graph, 'polylineLevels', 'uint32', 32),
          polylineClosed: createTransientView(graph, 'polylineClosed', 'uint32', 32),
          polylineCount: createTransientView(graph, 'polylineCount', 'uint32', 1),
          vertexCount: createTransientView(graph, 'vertexCount', 'uint32', 1)
        },
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          requiredCount: createTransientView(graph, 'requiredCount', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          polylineOverflow: createTransientView(graph, 'polylineOverflow', 'uint32', 1)
        }
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.at(-1)?.id).toBe('isolines-domain-edges');
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.some(source => source.source.includes('fn ownsCell'))).toBe(true);
    expect(sources.some(source => source.source.includes('var domainEdge'))).toBe(true);
    // This tile is 6x5 at domain offset (2, 2) in an 8x7 domain. Exercise both edge families,
    // extreme local edges, and a dirty fourth segment outside the published three-row count.
    const memory = createCPUGraphMemory();
    memory.write(contributor.props.output.status.count, Uint32Array.of(3));
    memory.write(
      contributor.props.output.segmentEdges!,
      Uint32Array.of(0, 24, 25, 54, 4, 30, 99, 99)
    );
    executeCommandNodesOnCPU([nodes.at(-1)!], memory);
    expect(Array.from(memory.readUint32(contributor.props.output.segmentEdges!, 8))).toEqual([
      16, 48, 67, 104, 20, 72, 99, 99
    ]);
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 16}
    });
    // Caller bindings are checked before scratch: this tile's 30 float32 samples need 120 bytes.
    const buffers = vi.spyOn(graph, 'createTransientBuffer');
    const views = vi.spyOn(graph, 'createDataView');
    const passes = vi.spyOn(graph, 'addComputePass');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/isolines.*needs 120 bytes.*limit 16/);
    expect(buffers).not.toHaveBeenCalled();
    expect(views).not.toHaveBeenCalled();
    expect(passes).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('bounds the worst-case required count in uint32 using owned cells rather than halo cells', () => {
  const grid = {width: 4096, height: 4096};
  expect(() =>
    validateIsolinesMaximumSegmentCount('count-budget', grid, undefined, 128)
  ).not.toThrow();
  expect(() => validateIsolinesMaximumSegmentCount('count-budget', grid, undefined, 256)).toThrow(
    /count-budget.*owned cells.*levels.length.*4294967295/
  );
  expect(() =>
    validateIsolinesMaximumSegmentCount(
      'count-budget',
      grid,
      {
        domainWidth: 4096,
        domainHeight: 4096,
        columnOffset: 0,
        rowOffset: 0,
        ownedCells: [0, 0, 1, 1]
      },
      256
    )
  ).not.toThrow();
});

for (const axis of ['column', 'row'] as const) {
  for (const originSample of [16777218, 16777217]) {
    it(`rebases large ${axis} indices before float32 conversion with origin ${originSample}`, () => {
      const scene = getLargeOffsetScene(axis, originSample);
      const expectedCoordinate = 16777218.5 - originSample;
      const expected =
        axis === 'column'
          ? [expectedCoordinate, 0.5, expectedCoordinate, 0.5]
          : [0.5, expectedCoordinate, 0.5, expectedCoordinate];
      const cpuResult = computeIsolinesOnCPU(scene, {...scene, segmentCapacity: 1});
      expect(Array.from(cpuResult.segments)).toEqual(expected);
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const values = createTransientView(graph, 'values', 'float32', 4);
        const contributor = new GPUIsolines({
          ...scene,
          values,
          mask: createTransientView(graph, 'mask', 'uint32', 4),
          levels: createTransientView(graph, 'levels', 'float32', 1),
          parameters: createTransientView(graph, 'parameters', 'float32', 8),
          output: {
            segments: createTransientView(graph, 'segments', 'float32x4', 1),
            segmentLevels: createTransientView(graph, 'segmentLevels', 'uint32', 1),
            status: {
              count: createTransientView(graph, 'count', 'uint32', 1),
              overflow: createTransientView(graph, 'overflow', 'uint32', 1)
            }
          }
        });
        const memory = createCPUGraphMemory();
        memory.write(values, scene.values);
        memory.write(contributor.props.mask!, scene.mask);
        memory.write(contributor.props.levels, scene.levels);
        memory.write(contributor.props.parameters, scene.parameters);
        // One cell's exclusive scan offset is zero. Execute the real scatter shader on the CPU;
        // no scan emulation or large-domain raster allocation is needed.
        const nodes = contributor
          .getCommandNodes(graph)
          .filter(node => node.id === 'isolines-scatter');
        expect(nodes.length).toBe(1);
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readFloat32(contributor.props.output.segments))).toEqual(expected);
      } finally {
        device.destroy();
      }
    });
  }
}
