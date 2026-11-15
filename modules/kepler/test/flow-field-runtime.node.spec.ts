// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, beforeEach, describe, expect, test, vi} from 'vitest';

// Node uses isolate:false: discard previously evaluated consumers before applying this file's mocks.
vi.hoisted(() => vi.resetModules());

const simulationCalls = vi.hoisted(() => ({
  drawCount: 0,
  setField: [] as unknown[],
  crops: [] as {source: unknown; region: unknown}[],
  destroyedCrops: 0,
  constructionCount: 0,
  constructionFailures: [] as number[]
}));

beforeEach(() => {
  simulationCalls.constructionCount = 0;
  simulationCalls.constructionFailures = [];
});

vi.mock('../../../examples/deck/kepler-compute/src/flow-field-viewport-texture', () => ({
  FlowFieldViewportTexture: class {
    texture = {cropped: true};
    byteLength = 288;
    constructor() {
      simulationCalls.constructionCount++;
      if (simulationCalls.constructionFailures.includes(simulationCalls.constructionCount))
        throw new Error('Injected crop construction failure');
    }
    update(source: unknown, region: unknown) {
      simulationCalls.crops.push({source, region});
      return true;
    }
    destroy() {
      simulationCalls.destroyedCrops++;
    }
  }
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
    destroy() {}
  }
}));
vi.mock('../../../examples/deck/kepler-compute/gpu-layers/index', () => ({
  FlowParticleLayer: class {
    props: unknown;
    constructor(props: unknown) {
      this.props = props;
    }
    draw() {
      simulationCalls.drawCount++;
    }
    getNeedsRedraw() {
      return false;
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

  test('analysis awaits the current revision even when an older grid has already built', async () => {
    const grid = makeGpuGrid(async () => [0, 0, 10, 10]);
    const revision = {value: 1};
    const {runtime} = makeRuntime(grid, revision);
    expect(runtime.stats.gridPending).toBe(true);
    expect(runtime.stats.gridReady).toBe(false);
    await runtime.waitForGridReady();
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.gridReady).toBe(true);
    expect(grid.buildCount).toBe(1);
    let release!: (bounds: [number, number, number, number]) => void;
    const measurement = new Promise<[number, number, number, number]>(resolve => {
      release = resolve;
    });
    grid.measureSelectionBounds.mockImplementation(() => measurement);
    revision.value = 2;
    expect(runtime.stats.gridPending).toBe(true);
    expect(runtime.stats.gridReady).toBe(false);
    let completed = false;
    const firstWait = runtime.waitForGridReady().then(() => {
      completed = true;
    });
    const secondWait = runtime.waitForGridReady();
    await Promise.resolve();
    expect(grid.buildCount).toBe(1);
    expect(runtime.stats.gridReady).toBe(false);
    expect(completed).toBe(false);
    release([2, 3, 4, 5]);
    await Promise.all([firstWait, secondWait]);
    expect(grid.buildCount).toBe(2);
    expect(runtime.stats.selectionRevision).toBe(2);
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.gridReady).toBe(true);
    expect(runtime.grid.bounds).toEqual([2, 3, 4, 5]);
    // One stable-bounds measurement for each revision, shared by both analysis callers.
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(4);
    runtime.destroy();
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.gridReady).toBe(false);
  });

  test('a superseded measurement never publishes against the newer selection mask', async () => {
    let release!: (bounds: [number, number, number, number]) => void;
    const measurement = new Promise<[number, number, number, number]>(resolve => {
      release = resolve;
    });
    const grid = makeGpuGrid(async () => [2, 3, 4, 5]);
    grid.measureSelectionBounds.mockImplementationOnce(() => measurement);
    const revision = {value: 1};
    const {runtime} = makeRuntime(grid, revision);
    const ready = runtime.waitForGridReady();
    revision.value = 2;
    runtime.rebuildGpuGrid();
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(1);
    release([0, 0, 10, 10]);
    await ready;
    expect(grid.build).toHaveBeenCalledTimes(1);
    expect(runtime.stats.selectionRevision).toBe(2);
    expect(runtime.grid.bounds).toEqual([2, 3, 4, 5]);
    expect(runtime.stats.gridReady).toBe(true);
    runtime.destroy();
  });

  test('analysis waiting has a deadline and stops when its runtime is destroyed', async () => {
    vi.useFakeTimers();
    try {
      const grid = makeGpuGrid(() => new Promise(() => {}));
      const {runtime} = makeRuntime(grid, {value: 1});
      const timeout = expect(runtime.waitForGridReady(5)).rejects.toThrow('analysis timeout');
      await vi.advanceTimersByTimeAsync(5);
      await timeout;
      const destroyed = expect(runtime.waitForGridReady()).rejects.toThrow('destroyed');
      runtime.destroy();
      await destroyed;
      expect(runtime.stats.gridReady).toBe(false);
      expect(grid.build).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  test.each([
    'resolve',
    'reject'
  ] as const)('analysis ignores a stale %s after its mask is replaced at the same revision', async completion => {
    let resolveMeasurement!: (bounds: [number, number, number, number]) => void;
    let rejectMeasurement!: (error: Error) => void;
    const measurement = new Promise<[number, number, number, number]>((resolve, reject) => {
      resolveMeasurement = resolve;
      rejectMeasurement = reject;
    });
    const grid = makeGpuGrid(async () => [2, 3, 4, 5]);
    grid.measureSelectionBounds.mockImplementationOnce(() => measurement);
    const {runtime, selection} = makeRuntime(grid, {value: 1});
    const previousMask = selection.mask;
    const ready = runtime.waitForGridReady();
    selection.mask = {} as never;
    runtime.rebuildGpuGrid();
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(1);
    expect(runtime.stats.gridPending).toBe(true);
    expect(runtime.stats.gridReady).toBe(false);
    if (completion === 'resolve') resolveMeasurement([0, 0, 10, 10]);
    else rejectMeasurement(new Error('superseded reduction failed'));
    await ready;
    expect(grid.build).toHaveBeenCalledExactlyOnceWith(
      runtime.texture,
      selection.mask,
      runtime.altitudeTexture
    );
    expect(grid.measureSelectionBounds).toHaveBeenCalledWith(previousMask);
    expect(grid.measureSelectionBounds).toHaveBeenLastCalledWith(selection.mask);
    expect(runtime.stats.bounds).toEqual([2, 3, 4, 5]);
    expect(runtime.stats.selectionRevision).toBe(1);
    expect(runtime.stats.extentFailure).toBeUndefined();
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.gridReady).toBe(true);
    runtime.destroy();
  });

  test.each([
    'resolve',
    'reject'
  ] as const)('destroy ignores a late reduction %s without publishing or requesting updates', async completion => {
    vi.useFakeTimers();
    try {
      let resolveMeasurement!: (bounds: [number, number, number, number]) => void;
      let rejectMeasurement!: (error: Error) => void;
      const measurement = new Promise<[number, number, number, number]>((resolve, reject) => {
        resolveMeasurement = resolve;
        rejectMeasurement = reject;
      });
      const grid = makeGpuGrid(() => measurement);
      const {runtime, device, selection} = makeRuntime(grid, {value: 1});
      const ready = expect(runtime.waitForGridReady()).rejects.toThrow('destroyed');
      runtime.destroy();
      await ready;
      if (completion === 'resolve') resolveMeasurement([2, 3, 4, 5]);
      else rejectMeasurement(new Error('destroyed reduction failed'));
      await vi.advanceTimersByTimeAsync(32);
      runtime.destroy();
      expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(1);
      expect(grid.build).not.toHaveBeenCalled();
      expect(selection.requestUpdate).not.toHaveBeenCalled();
      expect(runtime.stats.bounds).toEqual([0, 0, 10, 10]);
      expect(runtime.stats.selectionRevision).toBeNull();
      expect(runtime.stats.extentFailure).toBeUndefined();
      expect(runtime.stats.gridPending).toBe(false);
      expect(runtime.stats.gridReady).toBe(false);
      expect(runtime.isGridReady).toBe(false);
      expect(runtime.stats.viewportBuildCount).toBe(0);
      expect(grid.destroy).toHaveBeenCalledOnce();
      for (const texture of device.textures) expect(texture.destroy).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
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

describe('FlowFieldRuntime particle styling', () => {
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
    expect(styled.props.speedColoring?.palette).toEqual([[1, 0, 0]]);
    expect(styled.props.altitudeField?.scale).toBe(2);
    const draped = runtime.createLayer({
      id: 'c',
      color: [1, 2, 3],
      visible: true,
      elevationMultiplier: 0
    });
    expect((draped.props as {altitudeField?: unknown}).altitudeField).toBeUndefined();
  });
});

describe('FlowFieldRuntime viewport on the public API', () => {
  test.each([
    'cpu',
    'gpu'
  ] as const)('%s grid crops velocity and altitude and reuses the crop on stable frames', mode => {
    simulationCalls.crops.length = 0;
    simulationCalls.setField.length = 0;
    simulationCalls.destroyedCrops = 0;
    const device = makeDevice();
    const grid = makeGpuGrid(async () => null);
    const reservation = {release: vi.fn()};
    const memoryRegistry = {reserveMemory: vi.fn(() => reservation)};
    const runtime = new FlowFieldRuntime(
      device as never,
      'viewport',
      {
        columns: 4,
        rows: 4,
        bounds: [0, 0, 10, 10],
        texels: new Float32Array(64),
        altitude: new Float32Array(16)
      },
      planFlowField({}, {bounds: [0, 0, 10, 10], maximumSpeed: 1}),
      1,
      mode === 'gpu' ? {grid: grid as never} : undefined,
      memoryRegistry as never
    );
    runtime.advance(0);
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.gridReady).toBe(true);
    const initialCalls = simulationCalls.setField.length;
    runtime.advance(10, [4, 4, 6, 6]);
    runtime.advance(159, [4, 4, 6, 6]);
    expect(simulationCalls.crops).toHaveLength(0);
    runtime.advance(160, [4, 4, 6, 6]);
    expect(runtime.stats.simulationBounds).toEqual([3.8, 3.8, 6.2, 6.2]);
    expect(runtime.stats.bounds).toEqual([0, 0, 10, 10]);
    expect(runtime.stats.particleCount).toBe(runtime.plan.particleCount);
    expect(simulationCalls.crops).toHaveLength(2);
    expect(simulationCalls.crops[0].source).toBe(runtime.texture);
    expect(simulationCalls.crops[1].source).toBe(runtime.altitudeTexture);
    expect(simulationCalls.setField).toHaveLength(initialCalls + 1);
    const layer = runtime.createLayer({
      id: 'crop',
      color: [],
      visible: true,
      speedPalette: [[1, 0, 0]]
    });
    const props = layer.props as unknown as {
      bounds: number[];
      speedColoring: {velocityTexture: unknown};
      altitudeField: {texture: unknown};
    };
    expect(props.bounds).toEqual(runtime.stats.simulationBounds);
    expect(props.speedColoring.velocityTexture).toEqual({cropped: true});
    expect(props.altitudeField.texture).toEqual({cropped: true});
    runtime.advance(200, [4.01, 4.01, 6.01, 6.01]);
    expect(simulationCalls.crops).toHaveLength(2);
    // Zoom out restores the original textures, while the already-created layer sees new bounds.
    runtime.advance(300);
    runtime.advance(450);
    expect(props.bounds).toEqual([0, 0, 10, 10]);
    expect(props.speedColoring.velocityTexture).toBe(runtime.texture);
    expect(props.altitudeField.texture).toBe(runtime.altitudeTexture);
    runtime.destroy();
    expect(simulationCalls.destroyedCrops).toBe(2);
    expect(memoryRegistry.reserveMemory).toHaveBeenCalledExactlyOnceWith(576, 'gpu', 'scratch');
    expect(reservation.release).toHaveBeenCalledOnce();
  });

  test('filter-driven grid changes recrop the latest bounds without a new viewport delay', async () => {
    simulationCalls.crops.length = 0;
    const grid = makeGpuGrid(async () => [2, 2, 8, 8]);
    const revision = {value: 1};
    const {runtime} = makeRuntime(grid, revision);
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(1));
    runtime.advance(0, [4, 4, 6, 6]);
    expect(runtime.stats.bounds).toEqual([2, 2, 8, 8]);
    expect(runtime.stats.simulationBounds).toEqual([3.8, 3.8, 6.2, 6.2]);
    const previousBuilds = runtime.stats.viewportBuildCount;
    revision.value = 2;
    runtime.advance(20, [4, 4, 6, 6]);
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(2));
    runtime.advance(40, [4, 4, 6, 6]);
    expect(runtime.stats.viewportBuildCount).toBe(previousBuilds + 1);
    expect(runtime.stats.gridReadbackCount).toBe(0);
    runtime.destroy();
  });
});

function makeAdmissionRuntime(reserveMemory: () => {release(): void}) {
  const runtime = new FlowFieldRuntime(
    makeDevice() as never,
    'admission',
    {
      columns: 4,
      rows: 4,
      bounds: [0, 0, 10, 10],
      texels: new Float32Array(64),
      altitude: new Float32Array(16)
    },
    planFlowField({}, {bounds: [0, 0, 10, 10], maximumSpeed: 1}),
    1,
    undefined,
    {reserveMemory} as never
  );
  runtime.advance(0);
  return runtime;
}

describe('viewport admission cannot interrupt animation', () => {
  test('budget refusal preserves the usable field, reports once, and recovers after a bounded retry', () => {
    const release = vi.fn();
    const reserveMemory = vi
      .fn()
      .mockImplementationOnce(() => {
        throw new Error('GPU budget refused');
      })
      .mockReturnValue({release});
    const runtime = makeAdmissionRuntime(reserveMemory);
    const notify = vi.fn();
    runtime.onViewportAvailabilityChange = notify;
    const bytes = runtime.stats.byteLength;
    const calls = simulationCalls.setField.length;
    runtime.advance(10, [4, 4, 6, 6]);
    expect(() => runtime.advance(160, [4, 4, 6, 6])).not.toThrow();
    expect(runtime.stats.viewportFailure).toContain('GPU budget refused');
    expect(runtime.stats.simulationBounds).toEqual([0, 0, 10, 10]);
    expect(runtime.stats.spawnRegion).toBeNull();
    expect(runtime.stats.byteLength).toBe(bytes);
    expect(simulationCalls.setField).toHaveLength(calls);
    const steps = runtime.stats.stepCount;
    for (const timestamp of [200, 400, 800]) runtime.advance(timestamp, [4, 4, 6, 6]);
    expect(runtime.stats.stepCount).toBeGreaterThan(steps);
    expect(reserveMemory).toHaveBeenCalledTimes(1);
    expect(notify).toHaveBeenCalledTimes(1);
    for (const timestamp of [1160, 1320, 1480]) runtime.advance(timestamp, [4, 4, 6, 6]);
    expect(reserveMemory).toHaveBeenCalledTimes(2);
    expect(runtime.stats.viewportFailure).toBeUndefined();
    expect(runtime.stats.simulationBounds).toEqual([3.8, 3.8, 6.2, 6.2]);
    expect(notify).toHaveBeenCalledTimes(2);
    runtime.destroy();
    expect(release).toHaveBeenCalledOnce();
  });

  test.each([
    1, 2
  ])('failed crop %i rolls back both allocations and reservations on repeated attempts', failureIndex => {
    simulationCalls.destroyedCrops = 0;
    simulationCalls.constructionFailures = failureIndex === 1 ? [1, 2] : [2, 4];
    const reservations: {release: ReturnType<typeof vi.fn>}[] = [];
    const reserveMemory = vi.fn(() => {
      const reservation = {release: vi.fn()};
      reservations.push(reservation);
      return reservation;
    });
    const runtime = makeAdmissionRuntime(reserveMemory);
    const bytes = runtime.stats.byteLength;
    runtime.advance(10, [4, 4, 6, 6]);
    expect(() => runtime.advance(160, [4, 4, 6, 6])).not.toThrow();
    runtime.advance(200, [4, 4, 6, 6]);
    expect(() => runtime.advance(1320, [4, 4, 6, 6])).not.toThrow();
    expect(reservations).toHaveLength(2);
    expect(reservations.every(reservation => reservation.release.mock.calls.length === 1)).toBe(
      true
    );
    expect(simulationCalls.destroyedCrops).toBe(failureIndex === 1 ? 0 : 2);
    expect(runtime.stats.byteLength).toBe(bytes);
    expect(runtime.stats.simulationBounds).toEqual([0, 0, 10, 10]);
    expect(runtime.stats.viewportFailure).toContain('Injected crop construction failure');
    runtime.destroy();
    expect(reservations.every(reservation => reservation.release.mock.calls.length === 1)).toBe(
      true
    );
  });
});

describe('flow field late initial extent and selection completion', () => {
  test('waits for the first GPU field, then reuses one crop indefinitely at a fixed camera', async () => {
    let finishReduction!: (bounds: [number, number, number, number]) => void;
    const pending = new Promise<[number, number, number, number]>(resolve => {
      finishReduction = resolve;
    });
    const grid = makeGpuGrid(() => pending);
    const {runtime} = makeRuntime(grid, {value: 1});
    const viewport = [4, 4, 6, 6] as const;
    for (const timestamp of [0, 16, 32, 64, 500, 1000]) runtime.advance(timestamp, viewport);
    expect(runtime.stats.gridPending).toBe(true);
    expect(runtime.stats.viewportBuildCount).toBe(0);
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(1);
    finishReduction([0, 0, 10, 10]);
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(1));
    runtime.advance(1100, viewport);
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.viewportBuildCount).toBe(1);
    const bounds = runtime.stats.simulationBounds;
    // A long quiet camera interval must not restart the settle timer or recrop the same source.
    for (let timestamp = 1116; timestamp <= 4000; timestamp += 16)
      runtime.advance(timestamp, viewport);
    expect(runtime.stats.viewportBuildCount).toBe(1);
    expect(runtime.stats.simulationBounds).toEqual(bounds);
    expect(grid.build).toHaveBeenCalledTimes(1);
    runtime.destroy();
  });

  test.each([
    'revision',
    'mask'
  ] as const)('discards a late extent after the selection %s changes', async change => {
    let finishReduction!: (bounds: [number, number, number, number]) => void;
    const pending = new Promise<[number, number, number, number]>(resolve => {
      finishReduction = resolve;
    });
    const grid = makeGpuGrid(() => pending);
    const revision = {value: 1};
    const {runtime, selection} = makeRuntime(grid, revision);
    runtime.advance(0, [4, 4, 6, 6]);
    if (change === 'revision') revision.value = 2;
    else selection.mask = {} as never;
    runtime.advance(16, [4, 4, 6, 6]);
    // No overlapping reduction or stale grid publication, even if the camera never moved.
    expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(1);
    grid.measureSelectionBounds.mockResolvedValue([2, 2, 8, 8]);
    finishReduction([1, 1, 3, 3]);
    await vi.waitFor(() => expect(grid.measureSelectionBounds).toHaveBeenCalledTimes(3));
    expect(grid.build).not.toHaveBeenCalled();
    expect(runtime.stats.gridPending).toBe(true);
    expect(runtime.stats.bounds).toEqual([0, 0, 10, 10]);
    runtime.advance(600, [4, 4, 6, 6]);
    await vi.waitFor(() => expect(grid.build).toHaveBeenCalledTimes(1));
    runtime.advance(650, [4, 4, 6, 6]);
    expect(runtime.stats.gridPending).toBe(false);
    expect(runtime.stats.bounds).toEqual([2, 2, 8, 8]);
    expect(runtime.stats.viewportBuildCount).toBe(1);
    for (const timestamp of [800, 1000, 1500, 2000]) runtime.advance(timestamp, [4, 4, 6, 6]);
    expect(runtime.stats.viewportBuildCount).toBe(1);
    expect(grid.build).toHaveBeenCalledTimes(1);
    runtime.destroy();
  });
});

test('split viewports restore the full shared field and draw both disjoint views', () => {
  const device = makeDevice();
  const runtime = new FlowFieldRuntime(
    device as never,
    'split-field',
    {columns: 4, rows: 4, bounds: [0, 0, 10, 10], texels: new Float32Array(64)},
    planFlowField({}, {bounds: [0, 0, 10, 10]}),
    1
  );
  const left = {getBounds: () => [0, 0, 2, 2]};
  const right = {getBounds: () => [8, 8, 10, 10]};
  const viewports = [right];
  const layer = runtime.createLayer({id: 'split-field', visible: true, color: undefined});
  Object.assign(layer, {context: {viewport: right, deck: {getViewports: () => viewports}}});
  const timestamp = vi.spyOn(performance, 'now').mockReturnValue(0);
  try {
    layer.getNeedsRedraw();
    expect(runtime.stats.spawnRegion).not.toBeNull();
    viewports.push(left);
    timestamp.mockReturnValue(200);
    layer.getNeedsRedraw();
    expect(runtime.stats.simulationBounds).toEqual([0, 0, 10, 10]);
    expect(runtime.stats.spawnRegion).toBeNull();
    simulationCalls.drawCount = 0;
    for (const viewport of [left, right]) {
      Object.assign(layer, {context: {viewport}});
      layer.draw({} as never);
    }
    expect(simulationCalls.drawCount).toBe(2);
    Object.assign(layer, {context: {viewport: {getBounds: () => [20, 20, 30, 30]}}});
    layer.draw({} as never);
    expect(simulationCalls.drawCount).toBe(2);
  } finally {
    timestamp.mockRestore();
    runtime.destroy();
  }
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('../../../examples/deck/kepler-compute/src/flow-field-viewport-texture');
  vi.doUnmock('@luma.gl/graph/experimental/simulation');
  vi.doUnmock('../../../examples/deck/kepler-compute/gpu-layers/index');
  vi.resetModules();
});
