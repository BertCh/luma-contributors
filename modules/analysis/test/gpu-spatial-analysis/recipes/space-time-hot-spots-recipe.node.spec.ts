// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUPeriodIndex} from '../../../src/dataframe-analysis/period-index/index';
import {GPUCalendarBuckets} from '../../../src/dataframe-analysis/calendar-buckets/index';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGPUEmergingHotSpotsParameterValues} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';

it('H1 initial classification settings match an identical update in every parameter word', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'settings',
    format: 'float32',
    length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
  });
  const write = vi.spyOn(parameters, 'write');
  const settings = {
    temporalWindow: 0,
    persistentFraction: 0.75,
    minimumTrendSliceCount: 3,
    sporadicRequiresFinalSlice: false
  };
  try {
    const recipe = addSpaceTimeHotSpotsRecipe(graph, {
      timestamps: createTransientView(graph, 'timestamps', 'uint32x2', 1),
      calendarParameters: createTransientView(graph, 'calendar', 'sint32', 2),
      slices: {step: 'month', origin: {year: 2023}, count: 6},
      cells: {
        kind: 'lattice',
        positions: createTransientView(graph, 'positions', 'float32x2', 1),
        width: 5,
        height: 5,
        bounds: [0, 0, 5, 5]
      },
      settings,
      parameters
    });
    const initial = Array.from(write.mock.calls[0][0]);
    expect([initial[4], initial[5], initial[7]]).toEqual([0.75, 3, 1]);
    recipe.setSettings(settings);
    expect(Array.from(write.mock.calls[1][0])).toEqual(initial);
    expect(initial).toEqual(Array.from(getGPUEmergingHotSpotsParameterValues(settings)));
  } finally {
    parameters.destroy();
    device.destroy();
  }
});

it('addSpaceTimeHotSpotsRecipe rejects cyclic fields and invalid slices, and forwards the options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'sint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  let serial = 0;
  const props = (overrides: Record<string, unknown>) => {
    const key = `${Object.keys(overrides).join('-')}-${serial++}`;
    return {
      id: `rejects-${key}`,
      timestamps: view('uint32x2', 4, `ts-${key}`),
      calendarParameters: view('sint32', 2, `cp-${key}`),
      slices: {step: 'month', origin: {year: 2023, month: 1}, count: 6},
      cells: {
        kind: 'lattice',
        positions: view('float32x2', 4, `pos-${key}`),
        width: 2,
        height: 2,
        bounds: [0, 0, 2, 2]
      },
      settings: {temporalWindow: 1},
      parameters: new GPUParameterBuffer(device, {
        id: `par-${key}`,
        format: 'float32',
        length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      }),
      ...overrides
    } as never;
  };
  const timeZone = {
    transitions: view('uint32x2', 1, 'zone-transitions'),
    offsets: view('sint32', 2, 'zone-offsets')
  };
  const utcOffsets = view('sint32', 4, 'row-offsets');
  expect(() => addSpaceTimeHotSpotsRecipe(graph, props({timeZone, utcOffsets}))).toThrow(
    'addSpaceTimeHotSpotsRecipe pass utcOffsets or timeZone, not both'
  );
  for (const options of [{timeZone}, {utcOffsets}]) {
    const calendarRecipe = addSpaceTimeHotSpotsRecipe(graph, props(options));
    const calendar = calendarRecipe.contributors.find(
      contributor => contributor instanceof GPUCalendarBuckets
    );
    expect(calendar).toBeInstanceOf(GPUCalendarBuckets);
    if (calendar instanceof GPUCalendarBuckets) {
      expect(calendar.props.timeZone).toEqual(options.timeZone);
      if (options.timeZone) expect(calendar.props.timeZone).not.toBe(options.timeZone);
      expect(calendar.props.timeZone?.transitions).toBe(options.timeZone?.transitions);
      expect(calendar.props.timeZone?.offsets).toBe(options.timeZone?.offsets);
      expect(calendar.props.utcOffsets).toBe(options.utcOffsets);
    }
  }
  // Cyclic calendar fields are no longer accepted.
  expect(() =>
    addSpaceTimeHotSpotsRecipe(graph, props({slices: {field: 'month', firstValue: 1, count: 6}}))
  ).toThrow(/cyclic calendar fields/);
  expect(() =>
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({slices: {step: 'weekday', origin: {year: 2023}, count: 6}})
    )
  ).toThrow(/slices.step/);
  expect(() =>
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({slices: {step: 'day', interval: 0, origin: {year: 2023}, count: 6}})
    )
  ).toThrow(/slices.interval/);
  expect(() =>
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({slices: {step: 'day', origin: {year: 2023, month: 13}, count: 6}})
    )
  ).toThrow(/slices.origin.month/);
  // multipleTesting reaches the result's plain inference object.
  const recipe = addSpaceTimeHotSpotsRecipe(
    graph,
    props({multipleTesting: 'benjamini-yekutieli', finalSlice: 'last-valid'})
  );
  expect(recipe.contributors.some(contributor => contributor instanceof GPUPeriodIndex)).toBe(true);
  expect(recipe.inference.multipleTesting).toBe('benjamini-yekutieli');
  expect(recipe.inference.alternative).toBe('two-sided');
  expect(recipe.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  const spatialContext = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'area-frame', kind: 'local-lambert-equal-area'}
  } as const;
  const lattice = {
    kind: 'lattice',
    positions: view('float32x2', 4, 'context-positions'),
    width: 2,
    height: 2,
    bounds: [0, 0, 2, 2],
    spatialContext
  };
  expect(addSpaceTimeHotSpotsRecipe(graph, props({cells: lattice})).spatialContext).toEqual(
    spatialContext
  );
  for (const metric of ['none', 'great-circle']) {
    expect(() =>
      addSpaceTimeHotSpotsRecipe(
        graph,
        props({
          cells: {
            ...lattice,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric}
          }
        })
      )
    ).toThrow(/project positions to a planar frame first/);
  }
  const geographic = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const weights = {
    offsets: view('uint32', 5, 'context-offsets'),
    neighbors: view('uint32', 4, 'context-neighbors'),
    weights: view('float32', 4, 'context-weights'),
    spatialContext: geographic
  };
  const cells = {kind: 'ids', cellIds: view('uint32', 4, 'context-ids'), cellCount: 4, weights};
  expect(addSpaceTimeHotSpotsRecipe(graph, props({cells})).spatialContext).toEqual(geographic);
  expect(
    addSpaceTimeHotSpotsRecipe(
      graph,
      props({
        cells: {
          ...cells,
          weights: {...weights, spatialContext: undefined}
        }
      })
    ).spatialContext
  ).toBeUndefined();
  device.destroy();
});

it.each([
  'parameters',
  'slice-count',
  'output-length',
  'settings',
  'palette'
] as const)('rejects invalid %s without scheduling a partial recipe', invalid => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH - (invalid === 'parameters' ? 1 : 0)
  });
  const props = {
    timestamps: createTransientView(graph, 'timestamps', 'uint32x2', 2),
    calendarParameters: createTransientView(graph, 'calendar-parameters', 'sint32', 2),
    slices: {
      step: 'month' as const,
      origin: {year: 2023},
      count: invalid === 'slice-count' ? 257 : 6
    },
    cells: {
      kind: 'lattice' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      width: 2,
      height: 2,
      bounds: [0, 0, 2, 2] as const
    },
    settings: {temporalWindow: invalid === 'settings' ? -1 : 1},
    parameters,
    ...(invalid === 'output-length'
      ? {outputs: {trendZ: createTransientView(graph, 'short-trend', 'float32', 1)}}
      : {}),
    ...(invalid === 'palette'
      ? {
          color: {
            palette: createTransientView(graph, 'palette', 'uint32', 1),
            parameters: createTransientView(graph, 'color-parameters', 'float32', 4),
            maximumPaletteCount: 0
          }
        }
      : {})
  };
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => addSpaceTimeHotSpotsRecipe(graph, props)).toThrow(
      invalid === 'parameters' ? /parameters needs 8 values/ : undefined
    );
    expect(add).not.toHaveBeenCalled();
  } finally {
    parameters.destroy();
    device.destroy();
  }
});
