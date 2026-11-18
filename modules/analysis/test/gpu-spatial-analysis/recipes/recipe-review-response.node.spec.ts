// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {addExcessRiskRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {addZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/zonal-distribution-recipe';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  addRateClusterMapRecipe,
  type GPURateClusterMapRecipeProps
} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {addRasterZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {addTrackQualityRecipe} from '../../../src/gpu-network/recipes/track-quality-recipe';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2' | 'uint32x2' | 'sint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `input-${serial++}`, format, length);
  return {device, graph, view, bufferCount: () => serial};
}

it('R1 captures Proxy population values once for both declarations', () => {
  const {device, graph, view} = createContext();
  const events = view('float32', 4);
  const populations = view('float32', 4);
  const short = view('float32', 1);
  let reads = 0;
  const props = new Proxy(
    {id: 'retry', events, populations},
    {
      get(target, key, receiver) {
        if (key === 'populations') return ++reads <= 2 ? populations : short;
        return Reflect.get(target, key, receiver);
      }
    }
  );
  try {
    expect(() => addExcessRiskRecipe(graph, props)).not.toThrow();
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('R1 rejects a captured short population without graph mutations and permits same-ID retry', () => {
  const {device, graph, view, bufferCount} = createContext();
  const events = view('float32', 4);
  const populations = view('float32', 4);
  const short = view('float32', 1);
  let reads = 0;
  const props = new Proxy(
    {id: 'retry', events, populations},
    {
      get(target, key, receiver) {
        if (key === 'populations') {
          reads++;
          return short;
        }
        return Reflect.get(target, key, receiver);
      }
    }
  );
  try {
    expect(() => addExcessRiskRecipe(graph, props)).toThrow(/populations length/);
    expect(reads).toBe(1);
    const compiled = graph.compile();
    expect(compiled.stats.logicalBufferCount).toBe(bufferCount());
    expect(compiled.stats.nodeOrder).toEqual([]);
    compiled.destroy();
    // Independent mutable instance checks retry, because public inspection freezes a graph.
    const retry = createContext();
    try {
      const retryEvents = retry.view('float32', 4);
      const retryPopulations = retry.view('float32', 4);
      const retryShort = retry.view('float32', 1);
      expect(() =>
        addExcessRiskRecipe(retry.graph, {
          id: 'retry',
          events: retryEvents,
          populations: retryShort
        })
      ).toThrow();
      expect(() =>
        addExcessRiskRecipe(retry.graph, {
          id: 'retry',
          events: retryEvents,
          populations: retryPopulations
        })
      ).not.toThrow();
    } finally {
      retry.device.destroy();
    }
  } finally {
    device.destroy();
  }
});

it('R1 does not inspect an unrelated non-enumerable raster preview getter', () => {
  const {device, graph, view} = createContext();
  const input = {
    id: 'band',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('float32', 4)}
  };
  let reads = 0;
  Object.defineProperty(input, 'preview', {
    get() {
      reads++;
      throw new Error('unrelated preview');
    }
  });
  try {
    expect(() =>
      addSurfacePreparationRecipe(graph, {
        id: 'retry',
        grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        input,
        maximumDistance: 1
      })
    ).not.toThrow();
    expect(reads).toBe(0);
  } finally {
    device.destroy();
  }
});

function createPreparedContext() {
  const context = createContext();
  const {view} = context;
  const polygons = {
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2)
  };
  const prepared = new GPUSpatialJoinPrepared({
    id: 'prepared',
    geometry: polygons,
    edgeSlabs: true
  });
  const props = {
    id: 'retry',
    values: view('float32', 2),
    source: {
      kind: 'points' as const,
      join: {
        points: {kind: 'points' as const, positions: view('float32x2', 2)},
        polygons,
        prepared,
        candidateCapacity: 8
      }
    }
  };
  return {...context, prepared, props};
}

it('R2 reuses an already-declared prepared join and preserves the committed handle identity', () => {
  const {device, graph, prepared, props} = createPreparedContext();
  try {
    graph.add(prepared);
    const storage = prepared.storage;
    expect(prepared.isDeclaredIn(graph)).toBe(true);
    const result = addZonalDistributionRecipe(graph, props);
    const join = result.contributors.find(
      contributor => contributor instanceof GPUPointInPolygonJoin
    ) as GPUPointInPolygonJoin;
    expect(join.props.prepared).toBe(prepared);
    expect(prepared.storage).toBe(storage);
    expect(prepared.isDeclaredIn(graph)).toBe(true);
    expect(prepared.encodedBuildCount).toBe(0);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('R2 rejects an undeclared handle without registering recipe resources, then retries after declaration', () => {
  const {device, graph, prepared, props} = createPreparedContext();
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const imported = vi.spyOn(graph, 'importBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => addZonalDistributionRecipe(graph, props)).toThrow(/prepared.*(added|declared)/);
    expect(allocate).not.toHaveBeenCalled();
    expect(imported).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    graph.add(prepared);
    expect(() => addZonalDistributionRecipe(graph, props)).not.toThrow();
  } finally {
    vi.restoreAllMocks();
    prepared.destroy();
    device.destroy();
  }
});

function createRateContext() {
  const context = createContext();
  const {view, device} = context;
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 16
  });
  const props: GPURateClusterMapRecipeProps = {
    id: 'retry',
    events: view('float32', 4),
    populations: view('float32', 4),
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 16),
      featureOffsets: view('uint32', 5),
      polygonOffsets: view('uint32', 5),
      ringOffsets: view('uint32', 5)
    },
    parameters,
    neighborCapacity: 16,
    pairCapacity: 16,
    palette: view('uint32', 5)
  };
  return {...context, parameters, props};
}

it.each([
  'colors',
  'quadrants',
  'palette'
] as const)('R3 rejects short %s logical views despite spare offset backing capacity, and retries', port => {
  for (const inspect of [true, false]) {
    const {device, graph, view, parameters, props, bufferCount} = createRateContext();
    const backing = view('uint32', 8);
    const short = graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const invalid =
      port === 'palette' ? {...props, palette: short} : {...props, outputs: {[port]: short}};
    try {
      expect(() => addRateClusterMapRecipe(graph, invalid)).toThrow(new RegExp(port));
      if (inspect) {
        const compiled = graph.compile();
        expect(compiled.stats.logicalBufferCount).toBe(bufferCount());
        expect(compiled.stats.nodeOrder).toEqual([]);
        compiled.destroy();
      } else {
        const corrected = graph.createDataView(backing.buffer, {
          format: 'uint32',
          length: port === 'palette' ? 5 : 4,
          byteOffset: 4
        });
        const correctedProps =
          port === 'palette'
            ? {...props, palette: corrected}
            : {...props, outputs: {[port]: corrected}};
        const result = addRateClusterMapRecipe(graph, correctedProps);
        expect(result.colors.length).toBe(4);
        expect(result.quadrants.length).toBe(4);
        if (port !== 'palette') expect(result[port]).toBe(corrected);
      }
    } finally {
      parameters.destroy();
      device.destroy();
    }
  }
});

it('R4 rejects independent smoothed-rate and z-score output aliases before mutation and permits retry', () => {
  for (const inspect of [true, false]) {
    const {device, graph, view, parameters, props, bufferCount} = createRateContext();
    const shared = view('float32', 4);
    try {
      expect(() =>
        addRateClusterMapRecipe(graph, {
          ...props,
          outputs: {smoothedRates: shared, zScores: shared}
        })
      ).toThrow(/share|alias|overlap/);
      if (inspect) {
        const compiled = graph.compile();
        expect(compiled.stats.logicalBufferCount).toBe(bufferCount());
        expect(compiled.stats.nodeOrder).toEqual([]);
        compiled.destroy();
      } else {
        const result = addRateClusterMapRecipe(graph, {...props, outputs: {smoothedRates: shared}});
        expect(result.smoothedRates).toBe(shared);
        expect(result.zScores.buffer).not.toBe(shared.buffer);
      }
    } finally {
      parameters.destroy();
      device.destroy();
    }
  }
});

it('R4 checks a nested raster median against the original raw input including offset aliases', () => {
  for (const inspect of [true, false]) {
    const {device, graph, view, bufferCount} = createContext();
    const values = view('float32', 4);
    const medians = graph.createDataView(values.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 4
    });
    const props = {
      id: 'retry',
      grid: {
        width: 2,
        height: 2,
        affine: [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number],
        pixelInterpretation: 'area' as const
      },
      input: {id: 'raw', format: 'float32' as const, storage: {kind: 'buffer' as const, values}},
      zones: view('uint32', 4),
      zoneCount: 2
    };
    try {
      expect(() => addRasterZonalDistributionRecipe(graph, {...props, outputs: {medians}})).toThrow(
        /share|alias|overlap/
      );
      if (inspect) {
        const compiled = graph.compile();
        expect(compiled.stats.logicalBufferCount).toBe(bufferCount());
        expect(compiled.stats.nodeOrder).toEqual([]);
        compiled.destroy();
      } else {
        const result = addRasterZonalDistributionRecipe(graph, props);
        expect(result.outputs.medians.buffer).not.toBe(values.buffer);
      }
    } finally {
      device.destroy();
    }
  }
});

it('R5 accepts a frozen parameter wrapper and uploads exactly once after successful scheduling', () => {
  const {device, graph, view} = createContext();
  const parameters = Object.freeze(
    new GPUParameterBuffer(device, {id: 'parameters', format: 'float32', length: 8})
  );
  parameters.write(new Float32Array(8));
  const scheduled = vi.spyOn(graph, 'add');
  const originalWrite = parameters.buffer.write.bind(parameters.buffer);
  const upload = vi.spyOn(parameters.buffer, 'write').mockImplementation((...args) => {
    expect(scheduled).toHaveBeenCalled();
    return originalWrite(...args);
  });
  try {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {
        id: 'retry',
        timestamps: view('uint32x2', 4),
        calendarParameters: view('sint32', 2),
        slices: {step: 'month', origin: {year: 2023}, count: 6},
        cells: {
          kind: 'lattice',
          positions: view('float32x2', 4),
          width: 2,
          height: 2,
          bounds: [0, 0, 2, 2]
        },
        settings: {temporalWindow: 1},
        parameters
      })
    ).not.toThrow();
    expect(upload).toHaveBeenCalledTimes(1);
  } finally {
    vi.restoreAllMocks();
    parameters.destroy();
    device.destroy();
  }
});

it('R6 rejects unequal similarity pairs before registering track-quality nodes/resources and retries', () => {
  for (const inspect of [true, false]) {
    const {device, graph, view, bufferCount} = createContext();
    const pairB = view('uint32', 2);
    const short = view('uint32', 1);
    const props = {
      id: 'retry',
      trajectory: {
        positions: view('float32x2', 4),
        timestamps: view('float32', 4),
        trackOffsets: view('uint32', 2)
      },
      thresholds: view('float32', 2),
      similarity: {pairA: view('uint32', 2), pairB}
    };
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const add = vi.spyOn(graph, 'add');
    try {
      expect(() =>
        addTrackQualityRecipe(graph, {...props, similarity: {...props.similarity, pairB: short}})
      ).toThrow(/pairA and pairB/);
      expect(allocate).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
      if (inspect) {
        const compiled = graph.compile();
        expect(compiled.stats.logicalBufferCount).toBe(bufferCount());
        expect(compiled.stats.nodeOrder).toEqual([]);
        compiled.destroy();
      } else {
        expect(() => addTrackQualityRecipe(graph, props)).not.toThrow();
      }
    } finally {
      vi.restoreAllMocks();
      device.destroy();
    }
  }
});

it('R6 keeps metric producers before the raw quality/count stages in the deferred ledger', () => {
  const {device, graph, view} = createContext();
  const compute = vi.spyOn(graph, 'addComputePass');
  try {
    const result = addTrackQualityRecipe(graph, {
      id: 'retry',
      trajectory: {
        positions: view('float32x2', 4),
        timestamps: view('float32', 4),
        trackOffsets: view('uint32', 2)
      },
      thresholds: view('float32', 2)
    });
    const nodes = compute.mock.calls.map(([node]) => node);
    const writer = nodes.findIndex(node =>
      node.resources?.some(resource => {
        if (!('buffer' in resource)) return false;
        const buffer =
          resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
        return (
          buffer === result.outputs.metrics.stepSpeeds.buffer &&
          (resource.usage === 'storage-write' || resource.usage === 'storage-read-write')
        );
      })
    );
    const quality = nodes.findIndex(node => node.id === 'retry-quality');
    const counts = nodes.findIndex(node => node.id === 'retry-flag-counts');
    expect(writer).toBeGreaterThanOrEqual(0);
    expect(quality).toBeGreaterThan(writer);
    expect(counts).toBeGreaterThan(quality);
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});
