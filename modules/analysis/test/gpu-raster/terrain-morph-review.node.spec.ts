// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {computeRasterFillHolesOnCPU} from '../../src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation';
import {computeRasterReconstructionOnCPU} from '../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';
import {createTerrainFlowAccumulationNodes} from '../../src/gpu-terrain/hydrology/terrain-flow-passes';
import {createKernelAuditContext} from '../analysis-core/kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../utils/gpu-contributor-test-utils';
import {updateMissingObservationsStructureChart} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/raster/missing-observations-chart';
import type {ChartData} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/chart-types';

it('F11 selected-cell chart suppresses unfinished reconstruction and shows the converged value', () => {
  const marker = new Float32Array(16);
  marker[0] = 1;
  const mask = new Float32Array(16).fill(1);
  const grid = {
    width: 16,
    height: 1,
    affine: [1, 0, 0, 0, -1, 1] as const,
    pixelInterpretation: 'area' as const
  };
  for (const maximumIterations of [8, 32]) {
    const reconstruction = computeRasterReconstructionOnCPU(marker, mask, {
      grid,
      maximumIterations
    });
    expect(reconstruction.validity[9]).toBe(1);
    expect(reconstruction.values[9]).toBe(maximumIterations === 8 ? 0 : 1);
    const charts = new Map<string, ChartData | null>();
    for (const selectedCell of [0, 9]) {
      updateMissingObservationsStructureChart(
        {
          setChart: (id, chart) => {
            charts.set(id, chart);
          }
        },
        selectedCell,
        mask,
        {
          morphology: {
            dilate: {values: mask},
            erode: {values: mask},
            opening: {values: mask},
            closing: {values: mask}
          },
          reconstruction: {values: reconstruction.values, converged: reconstruction.converged[0]}
        },
        1
      );
      const chart = charts.get('structureChart');
      expect(chart?.kind).toBe('bars');
      if (!chart || chart.kind !== 'bars' || !chart.labels)
        throw new Error('structure chart was not emitted');
      expect(chart.labels.includes('reconstruction')).toBe(maximumIterations === 32);
      if (maximumIterations === 32)
        expect(chart.values[chart.labels.indexOf('reconstruction')]).toBe(1);
      expect(chart.labels).toContain('measured July');
    }
  }
});

vi.mock('../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));

type SceneArray = Float32Array | Uint32Array;
type SceneBuffer = {
  id: string;
  length: number;
  view: GraphDataView;
  buffer: {readAsync(): Promise<Uint8Array>};
  write(values: SceneArray): void;
};
type SceneState = {
  inputs: {
    columns: number;
    rows: number;
    cellSize: number;
    values: Float32Array;
    draws: Float32Array;
    stamp: [number, number];
  };
  outputs: Record<string, SceneBuffer>;
  values: Record<string, number>;
  world: {width: number; height: number};
  peek(buffer: SceneBuffer): SceneArray | null;
};
type SceneSpecification = {
  build(
    builder: {
      graph: unknown;
      allocate(id: string, format: 'float32' | 'uint32', length: number): SceneBuffer;
      upload(id: string, format: 'float32' | 'uint32', values: SceneArray): SceneBuffer;
    },
    inputs: SceneState['inputs'],
    values: SceneState['values']
  ): {outputs: SceneState['outputs']};
  afterRun(state: SceneState): Promise<{converged: boolean}>;
  interaction: {
    resolve(
      pointer: {x: number; y: number; unitX: number; unitY: number},
      state: SceneState
    ): {readout: string};
  };
  draw(state: SceneState & {extra: {converged: boolean}}): {values: SceneBuffer}[];
};

for (const sceneName of ['fill-holes', 'morphological-reconstruction']) {
  it(`R3 ${sceneName} propagates convergence to its readout and overlay at budgets four and five`, async () => {
    const modulePath = `../../../../website/src/gpu/runs/entries/raster-bands-indices/${sceneName}`;
    const scene = (await import(modulePath)).default as SceneSpecification;
    for (const maximumIterations of [4, 5]) {
      const context = createKernelAuditContext(`${sceneName}-${maximumIterations}`);
      try {
        const arrays = new Map<SceneBuffer, SceneArray>();
        const create = (
          id: string,
          format: 'float32' | 'uint32',
          data: SceneArray
        ): SceneBuffer => {
          const buffer: SceneBuffer = {
            id,
            length: data.length,
            view: createTransientView(context.graph, id, format, data.length),
            buffer: {
              readAsync: async () => new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            },
            write: values => data.set(values)
          };
          arrays.set(buffer, data);
          return buffer;
        };
        const builder = {
          graph: context.graph,
          upload: create,
          allocate: (id: string, format: 'float32' | 'uint32', length: number) =>
            create(
              id,
              format,
              format === 'float32' ? new Float32Array(length) : new Uint32Array(length)
            )
        };
        const inputs: SceneState['inputs'] = {
          columns: 11,
          rows: 11,
          cellSize: 1,
          values: new Float32Array(121).fill(sceneName === 'fill-holes' ? 0 : 1),
          draws: new Float32Array(121).fill(1),
          stamp: [5, 5]
        };
        const values = {maximumIterations, method: 0, height: 0.1, threshold: 0.5, pits: 0};
        const built = scene.build(builder, inputs, values);
        expect(built.outputs['converged']).toBeDefined();
        const grid = {
          width: 11,
          height: 11,
          affine: [1, 0, 0, 0, -1, 11] as const,
          pixelInterpretation: 'area' as const
        };
        const marker = new Float32Array(121);
        marker[60] = 1;
        const result =
          sceneName === 'fill-holes'
            ? computeRasterFillHolesOnCPU(new Uint32Array(121), {
                grid,
                connectivity: 8,
                maximumIterations
              })
            : computeRasterReconstructionOnCPU(marker, new Float32Array(121).fill(1), {
                grid,
                connectivity: 8,
                maximumIterations
              });
        arrays.get(built.outputs['result'])!.set(result.values);
        arrays.get(built.outputs['converged'])!.set(result.converged);
        if (built.outputs['removed'])
          arrays
            .get(built.outputs['removed'])!
            .set(Float32Array.from(result.values, value => 1 - value));
        const state: SceneState = {
          inputs,
          outputs: built.outputs,
          values,
          world: {width: 11, height: 11},
          peek: buffer => arrays.get(buffer) ?? null
        };
        const extra = await scene.afterRun(state);
        expect(extra.converged).toBe(maximumIterations === 5);
        const readout = scene.interaction.resolve(
          {x: 5, y: 5, unitX: 0.5, unitY: 0.5},
          state
        ).readout;
        expect(readout.includes('unfinished propagation')).toBe(maximumIterations === 4);
        const layers = scene.draw({...state, extra});
        const derived = built.outputs[sceneName === 'fill-holes' ? 'result' : 'removed'];
        expect(layers.some(layer => layer.values === derived)).toBe(maximumIterations === 5);
        if (maximumIterations === 5)
          expect(readout).toContain(
            sceneName === 'fill-holes' ? 'filled 0 cells' : '0% of cells cut'
          );
      } finally {
        context.device.destroy();
      }
    }
  });
}

it('R8 the public pyramid catalog explains cancelling support and all three input modes', async () => {
  const modulePath = '../../../../website/src/catalog/entries/raster-embeddings-flow';
  const {entries} = (await import(modulePath)) as {
    entries: {
      id: string;
      explanation: string;
      params: {name: string}[];
      outputs: {name: string; description: string}[];
    }[];
  };
  const entry = entries.find(candidate => candidate.id === 'raster-embedding-pyramid-level')!;
  expect(entry.params.some(parameter => parameter.name === 'aggregation')).toBe(true);
  expect(entry.outputs.some(output => output.name === 'vectorSums')).toBe(true);
  expect(entry.outputs.find(output => output.name === 'validity')!.description).toContain(
    'nonzero'
  );
  expect(entry.explanation).toContain('[east, west]');
  expect(entry.explanation).toContain('count 2');
  expect(entry.explanation).toContain('validity 0');
});

it('R1 fills accumulation words at the original offset with the exact unresolved sentinel', () => {
  const context = createKernelAuditContext('flow-word-fill-review');
  try {
    const allocation = context.buffer(512);
    const accumulation = context.graph.createDataView(allocation, {
      format: 'float32',
      byteOffset: 256,
      length: 2
    });
    const nodes = createTerrainFlowAccumulationNodes(context.graph, {
      id: 'flow',
      width: 2,
      height: 1,
      cellSizeMode: 'uniform',
      maxIterations: 1,
      elevation: context.view('float32', 2),
      receivers: context.view('uint32', 2),
      accumulation,
      settings: context.view('float32', 16),
      runoff: context.view('float32', 2),
      area: false
    });
    const fill = nodes.find(node => node.id === 'flow-accumulate-init')!;
    const target = fill.resources!.find(resource => resource.usage === 'storage-read-write')!;
    expect('buffer' in target).toBe(true);
    if (!('buffer' in target)) throw new Error('missing fill target');
    const view = target.buffer as GraphDataView;
    expect(view.format).toBe('uint32');
    expect(view.buffer).toBe(accumulation.buffer);
    expect(view.byteOffset).toBe(256);
    expect(view.length).toBe(2);
    const [kernel] = getGPUCommandNodeKernelSources([fill]);
    expect(kernel.source).toContain('= 0xffffffffu;');
  } finally {
    context.device.destroy();
  }
});
