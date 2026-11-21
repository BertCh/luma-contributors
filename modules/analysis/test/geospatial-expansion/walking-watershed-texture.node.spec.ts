// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createWalkingWatershedTextureScene} from './walking-watershed-texture-scenes';
import {computeRasterWalkingCostOnCPU} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {computeRasterWatershedSegmentationOnCPU} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';
import {Texture} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it, vi} from 'vitest';
import {
  GPURasterWalkingCost,
  getGPURasterWalkingCostParameterValues,
  type GPURasterWalkingCostModel
} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const grid = makeGPURasterMetadata({width: 3, height: 1});
function createTextureBand(graph: GPUCommandGraph, usage = Texture.SAMPLE): GPURasterBand {
  const texture = graph.createTransientTexture({
    id: 'rgba',
    format: 'rgba32float',
    width: 3,
    height: 1,
    usage
  });
  return {
    id: 'blue',
    format: 'float32',
    scale: 2,
    offset: 10,
    noDataValue: -9,
    storage: {kind: 'texture', view: graph.createTextureView(texture), channel: 2}
  };
}

for (const route of [
  'walking-elevation',
  'walking-friction',
  'fifo',
  'minimum-spanning-forest'
] as const) {
  function createContributor(graph: GPUCommandGraph, input: GPURasterBand) {
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    if (route === 'walking-elevation' || route === 'walking-friction') {
      const buffer: GPURasterBand = {
        id: 'buffer',
        format: 'float32',
        storage: {kind: 'buffer', values: view('values', 'float32', 3)}
      };
      return new GPURasterWalkingCost({
        grid,
        elevation: route === 'walking-elevation' ? input : buffer,
        friction: route === 'walking-friction' ? input : buffer,
        sources: view('sources', 'uint32', 3),
        parameters: view('parameters', 'float32', 16),
        maximumIterations: 2,
        output: {
          costs: view('costs', 'float32', 3),
          validity: view('validity', 'uint32', 3),
          status: {converged: view('converged', 'uint32', 1)}
        }
      });
    }
    return new GPURasterWatershedSegmentation({
      grid,
      input,
      method: route,
      markers: view('markers', 'uint32', 3),
      output: {
        labels: view('labels', 'uint32', 3),
        status: {converged: view('converged', 'uint32', 1)}
      }
    });
  }

  if (route === 'walking-elevation' || route === 'walking-friction') {
    it(`earlier-fix note: ${route} rejects unknown model kinds before snapshotting`, () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      try {
        const contributor = createContributor(graph, createTextureBand(graph));
        if (!(contributor instanceof GPURasterWalkingCost))
          throw new Error('expected walking contributor');
        const model = {
          kind: 'misspelling',
          coefficients: [1, 2, 0, 0],
          downhillSlopeThreshold: -1
        } as unknown as GPURasterWalkingCostModel;
        expect(() => new GPURasterWalkingCost({...contributor.props, model})).toThrow(
          /model.kind.*tobler.*grass.*misspelling/
        );
      } finally {
        graph.device.destroy();
      }
    });
  }

  it(`${route} accepts sampled texture bands`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    try {
      expect(() =>
        createContributor(graph, createTextureBand(graph)).getCommandNodes(graph)
      ).not.toThrow();
    } finally {
      graph.device.destroy();
    }
  });

  it(`${route} rejects texture usage without SAMPLE at construction`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    try {
      expect(() => createContributor(graph, createTextureBand(graph, Texture.COPY_DST))).toThrow(
        /Texture.SAMPLE/
      );
    } finally {
      graph.device.destroy();
    }
  });

  it(`${route} rejects foreign textures before allocating scratch`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const foreign = new GPUCommandGraph(createNullWebGPUDevice());
    try {
      const contributor = createContributor(graph, createTextureBand(foreign));
      const allocations = vi.spyOn(graph, 'createTransientBuffer');
      const additions = vi.spyOn(graph, 'add');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/target graph/);
      expect(allocations).not.toHaveBeenCalled();
      expect(additions).not.toHaveBeenCalled();
    } finally {
      foreign.device.destroy();
      graph.device.destroy();
    }
  });
}

it('earlier-fix note: parameter packing rejects unknown model kinds without changing the target', () => {
  const model = {
    kind: 'misspelling',
    coefficients: [1, 2, 0, 0],
    downhillSlopeThreshold: -1
  } as unknown as GPURasterWalkingCostModel;
  const target = new Float32Array(16).fill(123);
  expect(() => getGPURasterWalkingCostParameterValues({model}, grid, target)).toThrow(
    /model.kind.*tobler.*grass.*misspelling/
  );
  expect(Array.from(target)).toEqual(new Array(16).fill(123));
});

// These independent literals also check that replacing all usable elevations with zero
// cannot pass the GPU scene: reading the selected texture channel must matter.
it('B5 walking texture scene makes elevation change the directed transition cost', () => {
  const scene = createWalkingWatershedTextureScene('walking');
  const parameters = getGPURasterWalkingCostParameterValues(
    {model: scene.model, frictionWeight: 1},
    scene.grid
  );
  const run = (frame: (typeof scene.frames)[number], zero: boolean) =>
    computeRasterWalkingCostOnCPU(
      {
        elevation: {
          id: 'elevation',
          format: 'float32',
          values: Float32Array.from(frame.elevation, value => (zero && value !== -9 ? -5 : value)),
          validity: Uint32Array.from(frame.elevationValidity),
          noDataValue: -9,
          scale: 2,
          offset: 10
        },
        friction: {
          id: 'friction',
          format: 'float32',
          values: Float32Array.from(frame.friction),
          validity: Uint32Array.from(frame.frictionValidity),
          noDataValue: -9,
          scale: 0.5,
          offset: 10
        },
        sources: Uint32Array.from(scene.seeds),
        parameters
      },
      {grid: scene.grid, model: scene.model, maximumIterations: 8}
    );
  // Rise 2 metres costs another 2 * 2 = 4; distance + friction are 11.75 / 13.75.
  for (const [frame, cell] of [
    [0, 1],
    [1, 4]
  ]) {
    expect(Number.isFinite(run(scene.frames[frame], false).costs[cell])).toBe(true);
    expect(Number.isFinite(run(scene.frames[frame], true).costs[cell])).toBe(true);
  }
  expect(run(scene.frames[0], false).costs[1]).toBe(15.75);
  expect(run(scene.frames[1], false).costs[4]).toBe(17.75);
  expect(run(scene.frames[0], true).costs[1]).toBe(11.75);
  expect(run(scene.frames[1], true).costs[4]).toBe(13.75);
});
for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  it(`B5 ${method} texture scene changes plateau ownership when elevations reverse`, () => {
    const scene = createWalkingWatershedTextureScene(method);
    const labels = scene.frames.map(
      frame =>
        computeRasterWatershedSegmentationOnCPU(
          {
            id: 'elevation',
            format: 'float32',
            values: Float32Array.from(frame.elevation),
            validity: Uint32Array.from(frame.elevationValidity),
            noDataValue: -9,
            scale: 2,
            offset: 10
          },
          Uint32Array.from(scene.seeds),
          {grid: scene.grid, method}
        ).labels
    );
    // The lower endpoint reaches cell 1 before the higher endpoint in either rooted method.
    expect(Array.from(labels[0].slice(0, 3))).toEqual([5, 5, 9]);
    expect(Array.from(labels[1].slice(0, 3))).toEqual([5, 9, 9]);
    scene.frames.forEach((frame, index) =>
      expect(Array.from(labels[index])).toEqual(frame.expected)
    );
    const frame = scene.frames[1];
    const zero = computeRasterWatershedSegmentationOnCPU(
      {
        id: 'zero',
        format: 'float32',
        values: Float32Array.from(frame.elevation, value => (value === -9 ? -9 : -5)),
        validity: Uint32Array.from(frame.elevationValidity),
        noDataValue: -9,
        scale: 2,
        offset: 10
      },
      Uint32Array.from(scene.seeds),
      {grid: scene.grid, method}
    );
    expect(Array.from(zero.labels.slice(0, 3))).toEqual([5, 5, 9]);
  });
}
