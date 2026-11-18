// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPU_COLOR_SCALE_PARAMETER_LENGTH} from '../../../src/dataframe-analysis/column-classification/index';
import {createFillNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addSpaceTimeHotSpotsRecipe,
  type GPUSpaceTimeHotSpotsRecipeProps
} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {
  addFleetDwellRecipe,
  type GPUFleetDwellRecipeProps
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';

it.each([
  'overflow',
  'candidateOverflow'
] as const)('G2 rejects OLS writes aliasing weights %s before registration and permits retry', flagName => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const storage = createTransientView(graph, 'loss-and-code', 'uint32', 2).buffer;
  const flag = graph.createDataView(storage, {format: 'uint32', length: 1});
  const separate = graph.createDataView(storage, {format: 'uint32', length: 1, byteOffset: 4});
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'reg-parameters',
    format: 'float32',
    length: 8
  });
  const props = {
    id: 'reg',
    predictorCount: 1,
    parameters,
    predictors: createTransientView(graph, 'predictors', 'float32', 6),
    response: createTransientView(graph, 'response', 'float32', 6),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 7),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 6),
      weights: createTransientView(graph, 'weights', 'float32', 6),
      status: {[flagName]: flag}
    }
  };
  const original = getRegisteredBufferIds(graph);
  try {
    expect(() =>
      addSpatialRegressionRecipe(graph, {...props, outputs: {ols: {status: {code: flag}}}})
    ).toThrow(new RegExp(flagName));
    expect(getRegisteredBufferIds(graph)).toEqual(original);
    expect(getScheduledNodes(graph)).toEqual([]);
    expect(() =>
      addSpatialRegressionRecipe(graph, {...props, outputs: {ols: {status: {code: separate}}}})
    ).not.toThrow();
  } finally {
    parameters.destroy();
    graph.device.destroy();
  }
});

it('G6 rejects a stop mask overlapping public durations before registration', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createFleetProps(graph);
  const durations = createTransientView(graph, 'durations', 'float32', 2);
  const stopMask = graph.createDataView(durations.buffer, {format: 'uint32', length: 2});
  const original = getRegisteredBufferIds(graph);
  try {
    expect(() =>
      addFleetDwellRecipe(graph, {...props, outputs: {stops: {durations}}, scratch: {stopMask}})
    ).toThrow(/logical/);
    expect(getRegisteredBufferIds(graph)).toEqual(original);
    expect(getScheduledNodes(graph)).toEqual([]);
    expect(() => addFleetDwellRecipe(graph, props)).not.toThrow();
  } finally {
    graph.device.destroy();
  }
});

it('G9 retries an out-of-range hourly origin with the same ID', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSpaceTimeProps(graph);
  const original = getRegisteredBufferIds(graph);
  try {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {
        ...props,
        slices: {...props.slices, step: 'hour', origin: {year: 300000}}
      })
    ).toThrow(/step index/);
    expect(getRegisteredBufferIds(graph)).toEqual(original);
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {...props, slices: {...props.slices, step: 'hour'}})
    ).not.toThrow();
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

it.each([
  'inherited-skipped-count',
  'invalid-count'
] as const)('G4 protects weights evidence from %s and permits retry', mode => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSpaceTimeProps(graph);
  const flag = createTransientView(graph, 'loss', 'uint32', 1);
  const cells = {
    kind: 'ids' as const,
    cellCount: 4,
    cellIds: createTransientView(graph, 'cell-ids', 'uint32', 2),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
      weights: createTransientView(graph, 'weights', 'float32', 4),
      status: mode === 'invalid-count' ? {invalidCount: flag} : {candidateOverflow: flag}
    }
  };
  const status =
    mode === 'invalid-count'
      ? {skippedEventCount: flag}
      : Object.create({
          get skippedEventCount() {
            return flag;
          }
        });
  const original = getRegisteredBufferIds(graph);
  try {
    expect(() => addSpaceTimeHotSpotsRecipe(graph, {...props, cells, outputs: {status}})).toThrow();
    expect(getRegisteredBufferIds(graph)).toEqual(original);
    expect(getScheduledNodes(graph)).toEqual([]);
    expect(() => addSpaceTimeHotSpotsRecipe(graph, {...props, cells})).not.toThrow();
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

// Read-only inspection verifies actual scheduling, not calls to graph.add before it rejects a node.
it('G4 captures an inherited changing status getter once', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSpaceTimeProps(graph);
  const skipped = createTransientView(graph, 'skipped', 'uint32', 1);
  let reads = 0;
  const status = Object.create({
    get skippedEventCount() {
      reads++;
      return reads === 1 ? skipped : (props.cells as {positions: unknown}).positions;
    }
  });
  try {
    const result = addSpaceTimeHotSpotsRecipe(graph, {...props, outputs: {status}});
    expect(reads).toBe(1);
    expect(
      result.status.stages.find(stage => stage.stage === 'slice-range')?.status.invalidCount
    ).toBe(skipped);
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

function getRegisteredBufferIds(graph: GPUCommandGraph): string[] {
  return [...(graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()];
}

function getScheduledNodes(graph: GPUCommandGraph): readonly GPUCommandNode[] {
  return [...(graph as unknown as {nodes: GPUCommandNode[]}).nodes];
}

function createSpaceTimeProps(graph: GPUCommandGraph): GPUSpaceTimeHotSpotsRecipeProps {
  return {
    timestamps: createTransientView(graph, 'timestamps', 'uint32x2', 2),
    calendarParameters: createTransientView(graph, 'calendar-parameters', 'sint32', 2),
    parameters: new GPUParameterBuffer(graph.device, {
      id: 'settings',
      format: 'float32',
      length: 8
    }),
    slices: {step: 'month', origin: {year: 2023}, count: 6},
    cells: {
      kind: 'lattice',
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      width: 2,
      height: 2,
      bounds: [0, 0, 2, 2]
    },
    settings: {temporalWindow: 1}
  };
}

it.each([
  'foreign-positions',
  'overlapping-skipped-count'
] as const)('R1 rejects %s without changing the scheduled node list', invalid => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createSpaceTimeProps(graph);
  const marker = createTransientView(graph, 'marker', 'uint32', 1);
  graph.add(
    createFillNode(graph, {
      id: 'original',
      operation: 'Test',
      view: marker,
      type: 'u32',
      value: '0u'
    })
  );
  const original = getScheduledNodes(graph);
  const positions = props.cells.kind === 'lattice' ? props.cells.positions : undefined;
  const broken =
    invalid === 'foreign-positions'
      ? {
          ...props,
          cells: {
            ...props.cells,
            positions: createTransientView(new GPUCommandGraph(device), 'foreign', 'float32x2', 2)
          }
        }
      : {
          ...props,
          outputs: {
            status: {
              skippedEventCount: graph.createDataView(positions!.buffer, {
                format: 'uint32',
                length: 1
              })
            }
          }
        };
  try {
    expect(() => addSpaceTimeHotSpotsRecipe(graph, broken)).toThrow();
    expect(getScheduledNodes(graph).map(node => node.id)).toEqual(original.map(node => node.id));
    expect(() => addSpaceTimeHotSpotsRecipe(graph, props)).not.toThrow();
  } finally {
    props.parameters.destroy();
    device.destroy();
  }
});

function createFleetProps(graph: GPUCommandGraph): GPUFleetDwellRecipeProps {
  return {
    trajectories: {
      kind: 'trajectories',
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      timestamps: createTransientView(graph, 'timestamps', 'float32', 2),
      trackOffsets: createTransientView(graph, 'tracks', 'uint32', 2)
    },
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    stopCapacity: 2,
    zones: {
      kind: 'polygons',
      positions: createTransientView(graph, 'zones', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      candidateCapacity: 2
    }
  };
}

it('R2 preserves adjacent centroids and mask in one caller buffer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createFleetProps(graph);
  const storage = createTransientView(graph, 'shared-output', 'uint32', 6).buffer;
  const centroids = graph.createDataView(storage, {format: 'float32x2', length: 2, byteOffset: 0});
  const stopMask = graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 16});
  try {
    expect(() =>
      addFleetDwellRecipe(graph, {...props, outputs: {stops: {centroids}}, scratch: {stopMask}})
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('F7 rejects logically overlapping public centroids and stop mask before scheduling', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createFleetProps(graph);
  const storage = createTransientView(graph, 'shared-output', 'uint32', 6).buffer;
  const centroids = graph.createDataView(storage, {format: 'float32x2', length: 2});
  const stopMask = graph.createDataView(storage, {format: 'uint32', length: 2});
  try {
    expect(() =>
      addFleetDwellRecipe(graph, {...props, outputs: {stops: {centroids}}, scratch: {stopMask}})
    ).toThrow(/logical/);
    expect(getScheduledNodes(graph)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it.each([
  'month',
  'week'
] as const)('F11 retries an oversized %s interval with the same ID', step => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSpaceTimeProps(graph);
  try {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {
        ...props,
        slices: {...props.slices, step, interval: step === 'week' ? 306783379 : 2147483648}
      })
    ).toThrow(/steps per period/);
    expect(getScheduledNodes(graph)).toEqual([]);
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {...props, slices: {...props.slices, step, interval: 1}})
    ).not.toThrow();
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

it('F3 preserves supplied weights loss and rejects aliasing its evidence before mutation', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSpaceTimeProps(graph);
  const candidateOverflow = createTransientView(graph, 'candidate-loss', 'uint32', 1);
  const overflow = createTransientView(graph, 'final-loss', 'uint32', 1);
  const cells = {
    kind: 'ids' as const,
    cellCount: 4,
    cellIds: createTransientView(graph, 'cell-ids', 'uint32', 2),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
      weights: createTransientView(graph, 'weights', 'float32', 4),
      status: {overflow, candidateOverflow}
    }
  };
  try {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {
        ...props,
        cells,
        outputs: {status: {skippedEventCount: candidateOverflow}}
      })
    ).toThrow();
    expect(getScheduledNodes(graph)).toEqual([]);
    const result = addSpaceTimeHotSpotsRecipe(graph, {...props, cells});
    expect(result.status.stages).toContainEqual({
      stage: 'weights',
      status: {overflow, candidateOverflow}
    });
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

it('F12 accepts spare offset group status words', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createSpaceTimeProps(graph);
  const storage = createTransientView(graph, 'group-status', 'uint32', 4).buffer;
  const groupOverflow = graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 4});
  try {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {...props, outputs: {status: {groupOverflow}}})
    ).not.toThrow();
  } finally {
    props.parameters.destroy();
    graph.device.destroy();
  }
});

it.each([
  'parameters',
  'palette',
  'output-length'
] as const)('R3 retries space-time with corrected %s on the same graph and ID', invalid => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createSpaceTimeProps(graph);
  const short = new GPUParameterBuffer(device, {id: 'short', format: 'float32', length: 7});
  const broken = {
    ...props,
    ...(invalid === 'parameters' ? {parameters: short} : {}),
    ...(invalid === 'output-length'
      ? {outputs: {trendZ: createTransientView(graph, 'short-trend', 'float32', 1)}}
      : {}),
    ...(invalid === 'palette'
      ? {
          color: {
            maximumPaletteCount: 0,
            palette: createTransientView(graph, 'palette', 'uint32', 18),
            parameters: createTransientView(graph, 'colors-parameters', 'float32', 8)
          }
        }
      : {})
  };
  try {
    expect(() => addSpaceTimeHotSpotsRecipe(graph, broken)).toThrow();
    expect(() => addSpaceTimeHotSpotsRecipe(graph, props)).not.toThrow();
  } finally {
    short.destroy();
    props.parameters.destroy();
    device.destroy();
  }
});

it('R3 retries stops after correcting the declared timestamp encoding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createFleetProps(graph);
  try {
    expect(() =>
      addFleetDwellRecipe(graph, {
        ...props,
        trajectories: {
          ...props.trajectories,
          temporalContext: {
            unit: 'second',
            epoch: 0n,
            encoding: 'int64-words',
            axis: {kind: 'irregular'}
          }
        }
      })
    ).toThrow(/times/);
    expect(() => addFleetDwellRecipe(graph, props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('R4 a rejected second recipe does not write a working recipe parameter buffer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createSpaceTimeProps(graph);
  try {
    addSpaceTimeHotSpotsRecipe(graph, props);
    const original = getScheduledNodes(graph);
    const write = vi.spyOn(props.parameters, 'write');
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {
        ...props,
        id: 'rejected',
        settings: {temporalWindow: 2},
        color: {
          maximumPaletteCount: 0,
          palette: createTransientView(graph, 'palette', 'uint32', 18),
          parameters: createTransientView(graph, 'color-parameters', 'float32', 8)
        }
      })
    ).toThrow();
    expect(getScheduledNodes(graph)).toEqual(original);
    expect(write).not.toHaveBeenCalled();
  } finally {
    props.parameters.destroy();
    device.destroy();
  }
});

it('R4 writes settings only after scheduling succeeds', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createSpaceTimeProps(graph);
  const write = vi.spyOn(props.parameters, 'write');
  vi.spyOn(graph, 'add').mockImplementationOnce(() => {
    throw new Error('scheduling rejected');
  });
  try {
    expect(() => addSpaceTimeHotSpotsRecipe(graph, props)).toThrow('scheduling rejected');
    expect(write).not.toHaveBeenCalled();
    expect(getScheduledNodes(graph)).toEqual([]);
  } finally {
    props.parameters.destroy();
    device.destroy();
  }
});

it('R3 permits shared read-only color domain and parameter bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createSpaceTimeProps(graph);
  const colorParameters = createTransientView(
    graph,
    'color-parameters',
    'float32',
    GPU_COLOR_SCALE_PARAMETER_LENGTH
  );
  const colorDomain = graph.createDataView(colorParameters.buffer, {format: 'float32', length: 2});
  try {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(graph, {
        ...props,
        color: {
          palette: createTransientView(graph, 'palette', 'uint32', 18),
          parameters: colorParameters
        },
        scratch: {colorDomain}
      })
    ).not.toThrow();
  } finally {
    props.parameters.destroy();
    device.destroy();
  }
});

it.each([
  'short-positions',
  'short-cell-ids',
  'strided-positions'
] as const)('R5 rejects %s before resource registration and allows correction', invalid => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createSpaceTimeProps(graph);
  const positions =
    invalid === 'strided-positions'
      ? graph.createDataView(createTransientView(graph, 'padded', 'uint32', 8).buffer, {
          format: 'float32x2',
          length: 2,
          byteStride: 16
        })
      : createTransientView(graph, 'short-positions', 'float32x2', 1);
  const cells =
    invalid === 'short-cell-ids'
      ? {
          kind: 'ids' as const,
          cellIds: createTransientView(graph, 'cell-ids', 'uint32', 1),
          cellCount: 4,
          weights: {
            offsets: createTransientView(graph, 'offsets', 'uint32', 5),
            neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
            weights: createTransientView(graph, 'weights', 'float32', 4)
          }
        }
      : {
          ...props.cells,
          kind: 'lattice' as const,
          positions,
          width: 2,
          height: 2,
          bounds: [0, 0, 2, 2] as const
        };
  try {
    expect(() => addSpaceTimeHotSpotsRecipe(graph, {...props, cells})).toThrow();
    expect(getScheduledNodes(graph)).toEqual([]);
    expect(() => addSpaceTimeHotSpotsRecipe(graph, props)).not.toThrow();
  } finally {
    props.parameters.destroy();
    device.destroy();
  }
});
