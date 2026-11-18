// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {computeSinkhornDivergenceOnCPU} from '../../../src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation';
import type {GPUSinkhornDivergence} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

// Exercise the real scene callbacks without loading the canvas renderer in Node.
vi.mock('../../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));

// Load the scene at runtime: its renderer uses the website's tsconfig, rather than the module
// test project's compiler options. These structural types describe only the callbacks we use.
const SCENE_MODULE =
  '../../../../../website/src/gpu/runs/entries/categories-cells-tests/sinkhorn-divergence';
type SceneArray = Float32Array | Uint32Array | Int32Array;
type SceneFormat = 'float32' | 'uint32' | 'float32x2';
type SceneBuffer = {
  id: string;
  format: SceneFormat;
  length: number;
  buffer: {readAsync(): Promise<Uint8Array>};
  view: GraphDataView;
  write(values: SceneArray): void;
};
type SceneBuilder = {
  graph: GPUCommandGraph;
  upload(id: string, format: SceneFormat, values: SceneArray): SceneBuffer;
  allocate(id: string, format: SceneFormat, length: number): SceneBuffer;
};

type SceneInputs = {
  world: {width: number; height: number};
  source: Float32Array;
  shift: [number, number];
  distance: number;
};
type SceneState = {
  inputs: SceneInputs;
  outputs: Record<string, SceneBuffer>;
  values: {epsilon: number; maximumIterations: number};
};
type SceneSpecification = {
  createInputs(context: {
    random(): number;
    world: SceneInputs['world'];
    seed: number;
    aspect: number;
  }): SceneInputs;
  build(
    builder: SceneBuilder,
    inputs: SceneInputs,
    values: SceneState['values']
  ): {
    outputs: SceneState['outputs'];
    update?(values: SceneState['values']): void;
  };
  afterRun(state: SceneState & Pick<SceneBuilder, 'upload' | 'allocate'>): Promise<unknown>;
  interaction: {
    resolve(
      pointer: {x: number; y: number; unitX: number; unitY: number},
      state: SceneState & {
        world: SceneInputs['world'];
        pixelSize: number;
        toUnit(point: readonly [number, number]): readonly [number, number];
        peek(buffer: SceneBuffer): SceneArray | null;
        build: number;
      }
    ): {readout?: string};
  };
};

it('labels the Honshu two-iteration estimate and clears the limit indication at sixty iterations', async () => {
  const specification = (await import(SCENE_MODULE)).default as SceneSpecification;
  // The website's seed-1 random stream, independent of the solver.
  let randomState = 1;
  const random = () => {
    randomState = (randomState + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(randomState ^ (randomState >>> 15), 1 | randomState);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  const world = {width: 320, height: 180};
  const inputs = specification.createInputs({
    random,
    world,
    seed: 1,
    aspect: world.width / world.height
  });
  const device = createNullWebGPUDevice();
  try {
    for (const maximumIterations of [2, 60]) {
      const graph = new GPUCommandGraph(device);
      const add = vi.spyOn(graph, 'add');
      const arrays = new Map<unknown, Float32Array | Uint32Array | Int32Array>();
      const namedArrays = new Map<string, Float32Array | Uint32Array | Int32Array>();
      const createBuffer = (
        id: string,
        format: SceneFormat,
        values: Float32Array | Uint32Array | Int32Array
      ): SceneBuffer => {
        const buffer = {
          readAsync: async () => new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
        };
        arrays.set(buffer, values);
        namedArrays.set(id, values);
        const components = Number(format.match(/x(\d)$/)?.[1] ?? 1);
        return {
          id,
          format,
          length: values.length / components,
          buffer,
          view: createTransientView(graph, id, format, values.length / components),
          write: updated => values.set(updated)
        };
      };
      const builder: SceneBuilder = {
        graph,
        upload: (id, format, values) => createBuffer(id, format, values),
        allocate: (id, format, length) =>
          createBuffer(
            id,
            format,
            format === 'uint32' ? new Uint32Array(length) : new Float32Array(length)
          )
      };
      const values = {epsilon: 12, maximumIterations};
      const built = specification.build(builder, inputs, values);
      built.update?.(values);
      const contributor = add.mock.calls[0][0] as GPUSinkhornDivergence;
      // This assertion catches the original scene even before checking its readout.
      expect(contributor.props.output.status?.iterationLimitReached).toBeDefined();
      expect(contributor.props.output.status?.iterationLimitReached).toBe(
        built.outputs['iterationLimitReached'].view
      );
      const result = computeSinkhornDivergenceOnCPU(
        namedArrays.get('source') as Float32Array,
        namedArrays.get('target') as Float32Array,
        namedArrays.get('parameters') as Float32Array,
        {grid: contributor.props.grid, maximumIterations}
      );
      expect(result.iterationLimitReached[0]).toBe(maximumIterations === 2 ? 1 : 0);
      expect(result.converged[0]).toBe(maximumIterations === 2 ? 0 : 1);
      expect(Number.isFinite(result.divergence[0])).toBe(true);
      expect(result.divergence[0]).toBeCloseTo(maximumIterations === 2 ? 15.14211 : 18.12424, 4);
      for (const name of ['divergence', 'components', 'iterationLimitReached'] as const) {
        namedArrays.get(name)?.set(result[name]);
      }
      await specification.afterRun({
        inputs,
        outputs: built.outputs,
        values,
        upload: builder.upload,
        allocate: builder.allocate
      });
      const readout = specification.interaction!.resolve(
        {x: 0, y: 0, unitX: 0.5 + 4 / 24, unitY: 0.5 + 2 / 14},
        {
          inputs,
          outputs: built.outputs,
          values,
          world,
          pixelSize: 1,
          toUnit: point => point,
          peek: buffer => arrays.get(buffer.buffer) ?? null,
          build: maximumIterations
        }
      ).readout;
      expect(readout).toContain('typical distance');
      expect(readout?.includes('iteration limit reached (estimate)')).toBe(maximumIterations === 2);
      add.mockRestore();
    }
  } finally {
    device.destroy();
  }
});
