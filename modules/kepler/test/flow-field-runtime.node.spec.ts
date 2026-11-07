// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';

const simulationCalls = vi.hoisted(() => ({
  setField: [] as unknown[],
  setSpawnRegion: [] as unknown[]
}));

vi.mock('@luma.gl/graph/experimental/simulation', () => ({
  isFlowParticleSimulationSupported: () => true,
  FlowParticleSimulation: class {
    byteLength = 0;
    step() {
      return {};
    }
    setField(field: unknown) {
      simulationCalls.setField.push(field);
    }
    setSpawnRegion(region: unknown) {
      simulationCalls.setSpawnRegion.push(region);
    }
    destroy() {}
  }
}));
vi.mock('@deck.gl-community/gpu-layers', () => ({
  FlowParticleLayer: class {
    props: unknown;
    constructor(props: unknown) {
      this.props = props;
    }
  }
}));

import {FlowFieldRuntime} from '../../../examples/deck/kepler-compute/src/flow-field-runtime';
import {planFlowField} from '../../../examples/deck/kepler-compute/src/flow-field-plan';

function makeDevice() {
  const textures: {destroy: () => void}[] = [];
  return {
    textures,
    createTexture: () => {
      const texture = {destroy: vi.fn()};
      textures.push(texture);
      return texture;
    }
  };
}

function makeGpuGrid(measure: () => Promise<[number, number, number, number] | null>) {
  return {
    hasAltitude: true,
    buildCount: 0,
    byteLength: 0,
    readbackCount: 0,
    extentReadbackCount: 0,
    emptyField: false,
    activeBounds: [0, 0, 10, 10] as [number, number, number, number],
    options: {
      extent: {bounds: [0, 0, 10, 10] as [number, number, number, number]},
      velocityScaleNumerator: 5
    },
    build: vi.fn(function (this: {buildCount: number}) {
      this.buildCount++;
    }),
    measureSelectionBounds: vi.fn(measure),
    destroy: vi.fn()
  };
}

function makeRuntime(grid: ReturnType<typeof makeGpuGrid>, revision: {value: number}) {
  const device = makeDevice();
  const plan = planFlowField({}, {bounds: [0, 0, 10, 10]});
  const selection = {
    mask: {} as never,
    getRevision: () => revision.value,
    requestUpdate: vi.fn()
  };
  const runtime = new FlowFieldRuntime(
    device as never,
    'layer',
    {columns: 4, rows: 4, bounds: [0, 0, 10, 10]},
    plan,
    1,
    {grid: grid as never, selection}
  );
  return {runtime, device, selection, plan};
}

describe('FlowFieldRuntime filtered bounds', () => {
  test('a filter revision measures the selected rows, moves the bounds and then builds', async () => {
    simulationCalls.setField.length = 0;
    const grid = makeGpuGrid(async () => [2, 3, 4, 5]);
    const revision = {value: 1};
    const {runtime, selection} = makeRuntime(grid, revision);
    expect(grid.build).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(1));
    expect(runtime.grid.bounds).toEqual([2, 3, 4, 5]);
    expect(grid.activeBounds).toEqual([2, 3, 4, 5]);
    expect(grid.emptyField).toBe(false);
    // The visual-policy scale follows the new extent.
    expect(grid.options.velocityScaleNumerator).not.toBe(5);
    expect(simulationCalls.setField).toHaveLength(1);
    expect(runtime.stats.selectionRevision).toBe(1);
    expect(selection.requestUpdate).toHaveBeenCalled();
    // Same revision: nothing more to do.
    runtime.rebuildGpuGrid();
    // The reduction ran twice: the second, one frame later, confirmed the first (the mask for a new
    // revision can be encoded after the first reduction was submitted).
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(2);
    // A new revision measures again, one at a time.
    revision.value = 2;
    runtime.rebuildGpuGrid();
    runtime.rebuildGpuGrid();
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(3);
  });

  test('too few selected rows leave an empty field on the previous bounds', async () => {
    const grid = makeGpuGrid(async () => null);
    const {runtime} = makeRuntime(grid, {value: 1});
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(1));
    expect(grid.emptyField).toBe(true);
    expect(runtime.grid.bounds).toEqual([0, 0, 10, 10]);
  });

  test('a failed reduction falls back to the full extent and records the reason', async () => {
    const grid = makeGpuGrid(async () => {
      throw new Error('reduction failed');
    });
    const {runtime} = makeRuntime(grid, {value: 1});
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(1));
    expect(runtime.stats.extentFailure).toBe('reduction failed');
    expect(grid.emptyField).toBe(false);
  });

  test('an unfiltered grid builds synchronously on the full extent', () => {
    const grid = makeGpuGrid(async () => [1, 1, 2, 2]);
    const device = makeDevice();
    const runtime = new FlowFieldRuntime(
      device as never,
      'layer',
      {columns: 4, rows: 4, bounds: [0, 0, 10, 10]},
      planFlowField({}, {bounds: [0, 0, 10, 10]}),
      1,
      {grid: grid as never}
    );
    expect(grid.build).toHaveBeenCalledTimes(1);
    expect(grid.measureSelectionBounds).not.toHaveBeenCalled();
    expect(runtime.stats.hasAltitude).toBe(true);
  });
});

describe('FlowFieldRuntime viewport particles and styling', () => {
  test('advance concentrates spawning on the viewport and releases it for a full view', () => {
    simulationCalls.setSpawnRegion.length = 0;
    const grid = makeGpuGrid(async () => null);
    const device = makeDevice();
    const runtime = new FlowFieldRuntime(
      device as never,
      'layer',
      {columns: 4, rows: 4, bounds: [0, 0, 10, 10]},
      planFlowField({}, {bounds: [0, 0, 10, 10]}),
      1,
      {grid: grid as never}
    );
    runtime.advance(1, [4, 4, 6, 6]);
    expect(runtime.stats.spawnRegion).not.toBeNull();
    runtime.advance(2, [-50, -50, 50, 50]);
    expect(runtime.stats.spawnRegion).toBeNull();
    expect(simulationCalls.setSpawnRegion).toHaveLength(2);
  });

  test('createLayer passes the palette and the altitude only when requested', () => {
    const grid = makeGpuGrid(async () => null);
    const device = makeDevice();
    const runtime = new FlowFieldRuntime(
      device as never,
      'layer',
      {columns: 4, rows: 4, bounds: [0, 0, 10, 10]},
      planFlowField({}, {bounds: [0, 0, 10, 10]}),
      1,
      {grid: grid as never}
    );
    const plain = runtime.createLayer({id: 'a', color: [1, 2, 3], visible: true});
    expect((plain.props as {speedColoring?: unknown}).speedColoring).toBeUndefined();
    const styled = runtime.createLayer({
      id: 'b',
      color: [1, 2, 3],
      visible: true,
      speedPalette: [[1, 0, 0]],
      elevationMultiplier: 2
    });
    const props = styled.props as {
      speedColoring: {palette: number[][]};
      altitudeField: {scale: number};
    };
    expect(props.speedColoring.palette).toEqual([[1, 0, 0]]);
    expect(props.altitudeField.scale).toBe(2);
    const draped = runtime.createLayer({
      id: 'c',
      color: [1, 2, 3],
      visible: true,
      elevationMultiplier: 0
    });
    expect((draped.props as {altitudeField?: unknown}).altitudeField).toBeUndefined();
  });
});
