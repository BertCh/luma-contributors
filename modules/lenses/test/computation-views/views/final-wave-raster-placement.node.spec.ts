// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../../analysis/src/testing/gpu-contributor-test-utils';
import {LensCommandGraph} from '../../../src/computation-lenses/lens-view-snapshot';
import * as harness from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-harness';
import {loadDixieDem} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-data';
import {run} from '../../../src/computation-lenses/scenarios/gpu-raster/cost-distance.lens';
import {run as runJoin} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-join.lens';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';
import {
  buildD8StencilModel,
  renderD8Stencil,
  D8_STENCIL_VIEW
} from '../../../src/gpu-terrain/computation-pack/views/d8-stencil-view';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {getViewNodeText} from '../../../src/computation-views/view-node';

test('UNI-10 real loader and scenario preserve CRS through grid-only capture decoration', async () => {
  const data = await loadDixieDem();
  const device = createNullWebGPUDevice();
  const owned: {destroy(): void}[] = [];
  const interception = vi.spyOn(harness, 'runRasterLens').mockImplementation(async props => {
    expect(props.crs).toBeUndefined(); // The production scenario forwards only data.grid.
    const graph = new LensCommandGraph(device);
    const registered: Parameters<typeof harness.arrangeCaptures>[1][number][] = [];
    const register = (
      name: string,
      view: Parameters<harness.RasterLensSession['capture']>[1],
      spec: harness.RasterLensCaptureSpec | false
    ) => {
      if (spec) registered.push({name, view, spec});
      return view;
    };
    await props.build({
      device,
      graph,
      input(name, format, values, spec = {role: 'input', geometry: 'raster'}) {
        const view = createTransientView(graph, name, format, values.length);
        register(name, view, spec);
        return view;
      },
      output(
        name,
        format,
        spec = {role: 'output', geometry: 'raster'},
        length = props.grid.width * props.grid.height
      ) {
        const view = createTransientView(graph, name, format, length);
        register(name, view, spec);
        return view;
      },
      capture: (name, view, spec = {}) => {
        register(name, view, spec);
      },
      track: resource => {
        if (resource) owned.push(resource);
      }
    });
    const captures = harness.arrangeCaptures(
      props,
      registered,
      registered.map(({name}) => makeY2Capture(name, [0]))
    );
    const raster = captures.snapshots.find(
      capture =>
        capture.name === (props.scenarioId === 'gpu-raster.cost-distance' ? 'costs' : 'joinZones')
    )!;
    expect(raster.geometry).toMatchObject({
      crs: 'EPSG:32610',
      affine: data.grid.affine,
      pixelInterpretation: 'area',
      shape: [64, 64]
    });
    return {...makeY2Record([]), captures: {...makeY2Record([]).captures!, ...captures}};
  });
  try {
    await run(device);
    await runJoin(device);
    expect(interception).toHaveBeenCalledTimes(2);
  } finally {
    interception.mockRestore();
    for (const resource of owned) resource.destroy();
    device.destroy();
  }
});

test('UNI-10 explicitly authored full placements preserve affine, point samples and frame', () => {
  const grid = {
    width: 2,
    height: 3,
    affine: [4, 2, 10, 1, -6, 20] as const,
    pixelInterpretation: 'point' as const,
    coordinateReferenceSystem: {authority: 'EPSG:32610'}
  };
  const props = {grid, coordinateModel: 'projected' as const, unit: 'm', frameId: 'survey'};
  const geometry = harness.getRasterLensGeometry(props);
  expect(geometry.kind).toBe('raster');
  if (geometry.kind !== 'raster') throw new Error('Expected raster geometry');
  const decorated = harness.getRasterLensGeometry(props, geometry);
  expect(decorated).toEqual({
    kind: 'raster',
    shape: [2, 3],
    affine: [4, 2, 10, 1, -6, 20],
    pixelInterpretation: 'point',
    coordinateModel: 'projected',
    unit: 'm',
    crs: 'EPSG:32610',
    frameId: 'survey'
  });
  const device = createNullWebGPUDevice();
  try {
    const graph = new LensCommandGraph(device);
    const view = createTransientView(graph, 'authored', 'float32', 2);
    const props = {
      device,
      ...{grid, coordinateModel: 'projected' as const, unit: 'm', frameId: 'survey'},
      scenarioId: 'gpu-core.cards',
      domain: 'gpu-core',
      label: 'authored',
      dataset: {id: 'fixture', version: '1'},
      build: async () => {}
    };
    const authored = {
      kind: 'raster' as const,
      shape: [1, 2] as const,
      affine: [3, 3, 7, 4, -4, 9] as const,
      pixelInterpretation: 'area' as const,
      unit: 'm'
    };
    const arranged = harness.arrangeCaptures(
      props,
      [{name: 'authored', view, spec: {geometry: authored}}],
      [makeY2Capture('authored', [1, 2])]
    );
    expect(arranged.snapshots[0].geometry).toEqual({
      ...authored,
      crs: 'EPSG:32610',
      frameId: 'survey'
    });
  } finally {
    device.destroy();
  }
});

test('S1 sheared D8 step vectors add before taking their length', () => {
  const record = makeY2Record([
    makeY2Capture('elevation', [20, 20, 20, 20, 20, 14, 20, 8, 2], {
      role: 'input',
      meaning: 'elevation',
      geometry: {kind: 'raster', shape: [3, 3], affine: [3, 3, 0, 4, -4, 0], unit: 'm'}
    })
  ]);
  const model = buildD8StencilModel(createViewContext(record, {bindings: {entity: 4}}));
  // (3,4) + (3,-4) = (6,0): length 6; drop 18 / 6 = gradient 3.
  expect(model.neighbours[1].distance).toBe(6);
  expect(model.neighbours[1].gradient).toBe(3);
});

test.each([
  [4, 0, 10, 0, -6, 20],
  [0, -6, 10, 4, 0, 20]
])('W5-23 affine neighbor distances use both step vectors: %j', (...affine) => {
  const record = makeY2Record([
    makeY2Capture('elevation', [20, 20, 20, 20, 20, 14, 20, 8, 2], {
      role: 'input',
      meaning: 'elevation',
      geometry: {
        kind: 'raster',
        shape: [3, 3],
        affine: affine as [number, number, number, number, number, number],
        unit: 'm'
      }
    })
  ]);
  const context = createViewContext(record, {bindings: {entity: 4}});
  const model = buildD8StencilModel(context);
  // Hand computation: east drop 6 / 4 = 1.5; south drop 12 / 6 = 2;
  // diagonal length sqrt(4^2 + 6^2) = sqrt(52), in either rotated frame.
  expect(model.neighbours[0].distance).toBe(4);
  expect(model.neighbours[0].gradient).toBe(1.5);
  expect(model.neighbours[2].distance).toBe(6);
  expect(model.neighbours[2].gradient).toBe(2);
  expect(model.neighbours[1].distance).toBeCloseTo(Math.sqrt(52), 12);
  expect(Number.isFinite(model.neighbours[1].gradient)).toBe(true);
  expect(model.neighbours[1].gradient).toBeCloseTo(18 / Math.sqrt(52), 12);
  expect(getViewNodeText(renderD8Stencil(model, context))).toContain('6 m');
  expect(D8_STENCIL_VIEW.describe(model, context).facts.join(' ')).not.toContain(
    '1.414 cell sizes'
  );
});

test('W5-23 legacy square stencil keeps cardinal and diagonal distances', () => {
  const record = makeY2Record([
    makeY2Capture('elevation', new Array(9).fill(0), {
      role: 'input',
      meaning: 'elevation',
      geometry: {kind: 'raster', shape: [3, 3], cellSize: 5, unit: 'm'}
    })
  ]);
  const model = buildD8StencilModel(createViewContext(record, {bindings: {entity: 4}}));
  expect(model.neighbours[0].distance).toBe(5);
  expect(model.neighbours[1].distance).toBe(5 * Math.SQRT2);
});
