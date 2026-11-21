// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getRasterGridGeoreference} from '../../../src/analysis-core/contracts/raster-grid';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPU_TERRAIN_WEISS_LANDFORMS,
  GPUTerrainTopographicPosition,
  GPUTerrainWeissLandforms,
  getGPUTerrainWeissLandformsParameterValues,
  type GPUTerrainTopographicPositionProps,
  type GPUTerrainWeissLandformsProps
} from '../../../src/gpu-terrain/topographic-position/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';
import {computeWeissLandforms} from './weiss-landforms-oracle';

const GRID = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3], origin: [0, 5]});

it.each([
  'standardization',
  'grid',
  'getters'
] as const)('final2 G2 Weiss captures inherited fields and reads %s once', inherited => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const reads: number[] = [];
    const getterFields = <Fields extends object>(fields: Fields): Fields =>
      Object.create(
        null,
        Object.fromEntries(
          Object.entries(fields).map(([key, value]) => {
            const index = reads.push(0) - 1;
            return [
              key,
              {
                get: () => {
                  reads[index]++;
                  return reads[index] === 1 ? value : undefined;
                }
              }
            ];
          })
        )
      );
    let props: GPUTerrainWeissLandformsProps = {
      grid: makeGPURasterMetadata({width: 9, height: 9}),
      smallScale: {radius: 1},
      largeScale: {radius: 2},
      standardization: 'local',
      elevation: createBand(graph, 'inherited-elevation', 81),
      parameters: createTransientView(graph, 'inherited-settings', 'float32', 8),
      output: {landforms: createTransientView(graph, 'inherited-landforms', 'uint32', 81)}
    };
    if (inherited === 'standardization') {
      const {standardization, ...ownProps} = props;
      props = Object.assign(
        Object.create({
          get standardization() {
            return standardization;
          }
        }),
        ownProps
      );
    } else if (inherited === 'grid') {
      props.grid = Object.create(props.grid);
    } else {
      props = getterFields({
        ...props,
        grid: getterFields(props.grid),
        smallScale: getterFields(props.smallScale!),
        largeScale: getterFields(props.largeScale!),
        output: getterFields(props.output)
      });
    }
    const contributor = new GPUTerrainWeissLandforms(props);
    expect(contributor.getHaloStage()).toEqual({requiredHalo: 2});
    const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.get('terrain-weiss-landforms-position-evaluate')).toContain(
      'vec2<i32>(1i, 0i), vec2<i32>(2i, 0i)'
    );
    expect([...kernels.keys()].some(id => id.includes('-squares'))).toBe(false);
    expect(reads.every(count => count === 1)).toBe(true);
  } finally {
    device.destroy();
  }
});

it('final1 F4 Weiss fixture distinguishes the captured radius from a larger caller radius', () => {
  const elevation = new Float32Array(81);
  elevation[4 * 9 + 7] = 100;
  const options = {cellSize: 1, standardization: 'local' as const, standardThreshold: 0.05};
  const captured = computeWeissLandforms(elevation, undefined, 9, 9, {
    ...options,
    scales: [{radius: 1}, {radius: 2}]
  });
  const changed = computeWeissLandforms(elevation, undefined, 9, 9, {
    ...options,
    scales: [{radius: 1}, {radius: 4}]
  });
  expect(captured.landforms[4 * 9 + 4]).toBe(5);
  expect(changed.landforms[4 * 9 + 4]).toBe(4);
});

it.each([
  'scales',
  'grid'
] as const)('final1 F4 Weiss retains its validated footprint after caller mutation of %s', mutation => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const grid = {
      width: 9,
      height: 9,
      affine: [1, 0, 0, 0, -1, 9] as [number, number, number, number, number, number],
      pixelInterpretation: 'area' as const
    };
    const smallScale = {radius: 1, innerRadius: 0};
    const largeScale = {radius: 2, innerRadius: 0};
    const contributor = new GPUTerrainWeissLandforms({
      grid,
      smallScale,
      largeScale,
      standardization: 'local',
      elevation: createBand(graph, 'mutation-elevation', 81),
      parameters: createTransientView(graph, 'mutation-settings', 'float32', 8),
      output: {landforms: createTransientView(graph, 'mutation-landforms', 'uint32', 81)}
    });
    if (mutation === 'scales') {
      largeScale.radius = 4;
      largeScale.innerRadius = 1;
    } else {
      grid.width = 1;
      grid.height = 1;
      grid.affine[2] = 100;
    }
    expect(contributor.getHaloStage()).toEqual({requiredHalo: 2});
    expect(contributor.props.largeScale).toEqual({radius: 2, innerRadius: 0});
    expect(contributor.props.grid.width).toBe(9);
    expect(contributor.props.grid.height).toBe(9);
    expect(contributor.props.grid.affine).toEqual([1, 0, 0, 0, -1, 9]);
    const evaluate = getParsedKernelSources(contributor.getCommandNodes(graph)).get(
      'terrain-weiss-landforms-position-evaluate'
    )!;
    expect(evaluate).toContain('vec2<i32>(1i, 0i), vec2<i32>(2i, 0i)');
    expect(evaluate).toContain('const WIDTH: u32 = 9u;');
  } finally {
    device.destroy();
  }
});

it('GPUTerrainTopographicPosition builds one exact summed-area table and validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainTopographicPositionProps> = {}) => {
    instance++;
    return new GPUTerrainTopographicPosition({
      grid: GRID,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      scales: [{radius: 1}, {radius: 4, innerRadius: 2}],
      output: {deviationFromMean: createTransientView(graph, `dev-${instance}`, 'float32', 60)},
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.requiredHalo).toBe(4);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('terrain-topographic-position-elevation');
  expect(ids[1]).toBe('terrain-topographic-position-summed-area-quantize');
  expect(ids.at(-1)).toBe('terrain-topographic-position-evaluate');
  expect(ids.filter(id => id.includes('-transpose-'))).toHaveLength(5);
  expect(ids.every(id => id.startsWith('terrain-topographic-position-'))).toBe(true);

  // @ts-expect-error Empty output is deliberately exercised through the JavaScript guard.
  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() => create({scales: []})).toThrow(/scales must contain/);
  expect(() => create({scales: [{radius: 0}]})).toThrow(/integer radius/);
  expect(() => create({scales: [{radius: 2, innerRadius: 2}]})).toThrow(/innerRadius < radius/);
  expect(() => create({quantum: 0.003})).toThrow(/power of two/);
  expect(() =>
    create({output: {deviationFromMean: createTransientView(graph, 'short-dev', 'float32', 30)}})
  ).toThrow(/must contain 60 values/);
  const shared = createTransientView(graph, 'shared', 'float32', 60);
  expect(() =>
    create({output: {deviationFromMean: shared, topographicPositionIndex: shared}})
  ).toThrow(/^terrain-topographic-position .*logical byte ranges must not overlap/);

  // The table needs 3 * pixelCount words per binding: at most 11184810 cells.
  expect(() =>
    new GPUTerrainTopographicPosition({
      id: 'large',
      grid: makeGPURasterMetadata({width: 8192, height: 2048}),
      elevation: createBand(graph, 'large-elevation', 8192 * 2048),
      scales: [{radius: 1}],
      output: {
        maximumDeviation: createTransientView(graph, 'large-dev', 'float32', 8192 * 2048)
      }
    }).getCommandNodes(graph)
  ).toThrow('large grid cells must be at most 11184810');
});

it('GPUTerrainWeissLandforms composes position, slope, statistics, and classification', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainWeissLandformsProps> = {}) => {
    instance++;
    return new GPUTerrainWeissLandforms({
      id: `weiss-${instance}`,
      grid: GRID,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {landforms: createTransientView(graph, `landforms-${instance}`, 'uint32', 30)},
      ...overrides
    });
  };
  const global = create();
  expect(global.localRadius).toBe(15);
  const globalIds = global.getCommandNodes(graph).map(node => node.id);
  expect(globalIds).toContain('weiss-1-slope');
  expect(globalIds).toContain('weiss-1-squares');
  expect(globalIds.at(-1)).toBe('weiss-1-classify');
  const local = create({
    standardization: 'local',
    smallScale: {radius: 2},
    largeScale: {radius: 6}
  });
  expect(local.getHaloStage().requiredHalo).toBe(6);
  const localIds = local.getCommandNodes(graph).map(node => node.id);
  expect(localIds.some(id => id === 'weiss-2-squares' || id.startsWith('weiss-2-sum-'))).toBe(
    false
  );
  expect(globalIds.some(id => id.startsWith('weiss-1-sum-0'))).toBe(true);

  // Rows of 3 units advance south from y = 5 over 5 rows: edges 5 and -10.
  expect(Array.from(getGPUTerrainWeissLandformsParameterValues({}, GRID))).toEqual([
    2, 3, 1, 5, -10, 1, 5, 0
  ]);
  const georeference = getRasterGridGeoreference(GRID);
  expect(Array.from(getGPUTerrainWeissLandformsParameterValues({}, GRID).subarray(0, 5))).toEqual([
    ...georeference.cellSize,
    1,
    georeference.northEdge,
    georeference.southEdge
  ]);
  expect(() => getGPUTerrainWeissLandformsParameterValues({}, GRID, new Float32Array(7))).toThrow(
    /getGPUTerrainWeissLandformsParameterValues/
  );
  expect(GPU_TERRAIN_WEISS_LANDFORMS.mountainTop).toBe(10);
  // @ts-expect-error Empty output is deliberately exercised through the JavaScript guard.
  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() => create({standardization: 'median' as never})).toThrow(/global or local/);
  expect(() => create({smallScale: {radius: 1.5}})).toThrow(/integer radius/);
  expect(() =>
    create({parameters: createTransientView(graph, 'short-settings', 'float32', 7)})
  ).toThrow(/at least 8/);
  expect(() =>
    create({
      output: {standardizedPosition: createTransientView(graph, 'short-std', 'float32', 30)}
    })
  ).toThrow(/two values per pixel/);
});
