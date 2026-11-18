// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPUReduction,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {Texture} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {
  GPUEmpiricalBayesRates,
  GPUSpatialEmpiricalBayesRates
} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {createNullWebGPUDevice, createInputBuffer} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addPreparedRecipe} from '../../../src/gpu-spatial-analysis/recipes/prepared-recipe';
import {RecipeBuilder} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {getGPULonLatPolygonPreparationParameterValues} from '../../../src/gpu-spatial-analysis/lon-lat-polygons/index';
import {FIJI_TRANSMERIDIAN, createLonLatPolygonFixtureViews} from '../lon-lat-polygons/fixtures';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const events = createTransientView(graph, 'events', 'uint32', 2);
  const populations = createTransientView(graph, 'populations', 'uint32', 2);
  const shortPopulations = createTransientView(graph, 'short-populations', 'uint32', 1);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 3),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
    weights: createTransientView(graph, 'weights', 'float32', 2)
  };
  return {device, graph, props: {id: 'retry', events, populations, weights}, shortPopulations};
}

const recipes = [
  {name: 'addExcessRiskRecipe', add: addExcessRiskRecipe},
  {name: 'addSpatialRateRecipe', add: addSpatialRateRecipe}
] as const;

it.each(
  recipes
)('$name rejects an invalid constructor before changing the caller graph', recipe => {
  const {device, graph, props, shortPopulations} = createContext();
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const importBuffer = vi.spyOn(graph, 'importBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => recipe.add(graph, {...props, populations: shortPopulations})).toThrow(
      /populations length/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(importBuffer).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => recipe.add(graph, props)).not.toThrow();
    expect(add).toHaveBeenCalled();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it.each(recipes)('$name rejects a later expansion before changing the caller graph', recipe => {
  const {device, graph, props} = createContext();
  // Excess risk has a final invalid-count reduction after the rate contributor and glue stage.
  const reduction = GPUReduction.prototype.getCommandNodes;
  const expansion = vi
    .spyOn(GPUReduction.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUReduction, target) {
      if (this.id === 'retry-invalid-rows-invalid-sum') {
        throw new Error('invalid later expansion');
      }
      return reduction.call(this, target);
    });
  const spatialExpansion = GPUSpatialEmpiricalBayesRates.prototype.getCommandNodes;
  const spatial = vi
    .spyOn(GPUSpatialEmpiricalBayesRates.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUSpatialEmpiricalBayesRates, target) {
      spatialExpansion.call(this, target);
      throw new Error('invalid later expansion');
    });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const importBuffer = vi.spyOn(graph, 'importBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => recipe.add(graph, props)).toThrow('invalid later expansion');
    expect(allocate).not.toHaveBeenCalled();
    expect(importBuffer).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expansion.mockRestore();
    spatial.mockRestore();
    expect(() => recipe.add(graph, props)).not.toThrow();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it.each([
  {name: 'excess risk', Contributor: GPUEmpiricalBayesRates},
  {name: 'spatial rate', Contributor: GPUSpatialEmpiricalBayesRates}
])('$name contributors construct without registering graph resources', ({Contributor}) => {
  const {device, graph, props} = createContext();
  const output = createTransientView(graph, 'output', 'float32', 2);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    if (Contributor === GPUEmpiricalBayesRates) {
      new GPUEmpiricalBayesRates({...props, output: {excessRisk: output}});
    } else {
      new GPUSpatialEmpiricalBayesRates({...props, output: {spatialRates: output}});
    }
    expect(allocate).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it.each(
  recipes
)('$name preserves the public compiled node and resource counts on failure', recipe => {
  const {device, graph, props, shortPopulations} = createContext();
  try {
    graph.addCopyPass({
      id: 'existing',
      resources: [],
      compile: () => ({encode: () => {}})
    });
    expect(() => recipe.add(graph, {...props, populations: shortPopulations})).toThrow();
    // Compilation freezes the graph, so this complements the independent same-ID retry tests.
    const compiled = graph.compile();
    try {
      expect(compiled.stats.nodeOrder).toEqual(['existing']);
      expect(compiled.stats.logicalBufferCount).toBe(6);
      expect(compiled.stats.logicalTransientBufferCount).toBe(6);
    } finally {
      compiled.destroy();
    }
  } finally {
    device.destroy();
  }
});

it.each(recipes)('$name retains caller output identity without allocating GPU storage', recipe => {
  const {device, graph, props} = createContext();
  const output = createTransientView(graph, 'caller-output', 'float32', 2);
  const allocate = vi.spyOn(device, 'createBuffer');
  try {
    const result = recipe.add(graph, {...props, output});
    expect(Object.values(result.outputs)).toContain(output);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it.each(recipes)('$name retains aliases between offset views during preparation', recipe => {
  const {device, graph, props} = createContext();
  const shared = createTransientView(graph, 'shared', 'uint32', 4);
  const events = graph.createDataView(shared.buffer, {format: 'uint32', length: 2, byteOffset: 8});
  const output = graph.createDataView(shared.buffer, {format: 'float32', length: 2});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => recipe.add(graph, {...props, events, output})).toThrow(/alias|overlap|share/);
    expect(allocate).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => recipe.add(graph, props)).not.toThrow();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it.each(
  recipes
)('$name retains physical aliases across imported handles during preparation', recipe => {
  const {device, graph, props} = createContext();
  const buffer = createInputBuffer(device, new Float32Array([2, 6]));
  const events = importGraphBuffer(graph, 'imported-events', buffer, 'float32', 2);
  const output = importGraphBuffer(graph, 'imported-output', buffer, 'float32', 2);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => recipe.add(graph, {...props, events, output})).toThrow(/alias|overlap|share/);
    expect(allocate).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => recipe.add(graph, props)).not.toThrow();
  } finally {
    vi.restoreAllMocks();
    buffer.destroy();
    device.destroy();
  }
});

it.each(
  recipes
)('$name rejects foreign inputs before preparation can hide their ownership', recipe => {
  const {device, graph, props} = createContext();
  const foreign = new GPUCommandGraph(device);
  const populations = createTransientView(foreign, 'foreign-populations', 'uint32', 2);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => recipe.add(graph, {...props, populations})).toThrow(/target graph/);
    expect(allocate).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => recipe.add(graph, props)).not.toThrow();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('preparation snapshots a getter once across both declarations', () => {
  const {device, graph, props, shortPopulations} = createContext();
  const getPopulations = vi.fn(() =>
    getPopulations.mock.calls.length === 1 ? props.populations : shortPopulations
  );
  try {
    expect(() =>
      addExcessRiskRecipe(graph, {
        ...props,
        get populations() {
          return getPopulations();
        }
      })
    ).not.toThrow();
    expect(getPopulations).toHaveBeenCalledTimes(1);
  } finally {
    device.destroy();
  }
});

it('preparation preserves texture subresources and vector chunk layouts without changing caller identities', () => {
  const {device, graph, props} = createContext();
  const texture = graph.createTransientTexture({
    id: 'texture',
    format: 'r32float',
    width: 4,
    height: 4,
    mipLevels: 2,
    usage: Texture.SAMPLE
  });
  const selected = graph.createTextureView(texture, {baseMipLevel: 1, mipLevelCount: 1});
  const vector = new GraphVectorView({
    id: 'chunks',
    name: 'chunks',
    format: 'uint32',
    length: 4,
    valueLength: 4,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: [props.events, props.populations]
  });
  let declarations = 0;
  try {
    const result = addPreparedRecipe(graph, {selected, vector}, (target, captured) => {
      declarations++;
      expect(captured.selected.texture.graph).toBe(target);
      expect(captured.selected.baseMipLevel).toBe(1);
      expect(captured.selected.width).toBe(2);
      expect(captured.vector.data.map(chunk => chunk.length)).toEqual([2, 2]);
      for (const chunk of captured.vector.data) expect(chunk.buffer.graph).toBe(target);
      return captured;
    });
    expect(declarations).toBe(2);
    expect(result.selected).toBe(selected);
    expect(result.vector).toBe(vector);
  } finally {
    device.destroy();
  }
});

it('a later expansion destroys owned preparation contributors and keeps the caller graph empty', () => {
  const {device, graph} = createContext();
  const destroyed = vi.fn();
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() =>
      addPreparedRecipe(graph, {}, target => {
        const builder = new RecipeBuilder(target);
        builder.add({
          getCommandNodes: current => {
            createTransientView(current, 'owned-scratch', 'uint32', 1);
            return [];
          },
          destroy: destroyed
        });
        builder.add({
          getCommandNodes: () => {
            throw new Error('later expansion');
          }
        });
        return builder.contributors;
      })
    ).toThrow('later expansion');
    expect(destroyed).toHaveBeenCalledTimes(1);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

for (const backend of ['zonal', 'group'] as const) {
  for (const edgeModel of ['linear', 'great-circle'] as const) {
    it(`choropleth ${backend}/${edgeModel} preserves opaque Fiji buffer metadata without reflecting on native state`, () => {
      const {device, graph} = createContext();
      const polygons = createLonLatPolygonFixtureViews(device, graph, FIJI_TRANSMERIDIAN);
      const values = getGPULonLatPolygonPreparationParameterValues(edgeModel);
      const parameterBuffer = createInputBuffer(device, values);
      // Browser buffers carry native state. Model a brand-checked native accessor with the
      // real Map size accessor: entering this unrelated resource metadata must throw.
      const readNativeState = vi.fn(() => Reflect.get(Map.prototype, 'size'));
      Object.defineProperty(polygons.buffers[0], 'unrelatedNativeState', {get: readNativeState});
      try {
        const result = addPointsInPolygonsChoroplethRecipe(graph, {
          backend,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel},
          prepare: true,
          preparedPositionCapacity: 512,
          preparationParameters: importGraphBuffer(
            graph,
            'preparation',
            parameterBuffer,
            'float32',
            values.length
          ),
          points: {kind: 'points', positions: createTransientView(graph, 'points', 'float32x2', 6)},
          polygons: {...polygons, kind: 'polygons', candidateCapacity: 64}
        });
        expect(result.featureCount).toBe(1);
        expect(result.status.stages[0].stage).toBe('polygon-preparation');
        expect(readNativeState).not.toHaveBeenCalled();
      } finally {
        polygons.destroy();
        parameterBuffer.destroy();
        device.destroy();
      }
    });
  }
}

it('opaque resources keep their identity while class-based recipe getters are still captured once', () => {
  const {device, graph, props} = createContext();
  const buffer = createInputBuffer(device, new Uint32Array([2, 6]));
  const readNativeState = vi.fn(() => Reflect.get(Map.prototype, 'size'));
  Object.defineProperty(buffer, 'unrelatedNativeState', {get: readNativeState});
  let reads = 0;
  class InputConfiguration {
    readonly populations = props.populations;
    readonly resources = {buffer, device};
    get events() {
      reads++;
      return props.events;
    }
  }
  try {
    const configuration = new InputConfiguration();
    const captured = addPreparedRecipe(graph, configuration, (target, input) => {
      expect(input.events.buffer.graph).toBe(target);
      expect(input.resources.buffer).toBe(buffer);
      expect(input.resources.device).toBe(device);
      return input;
    });
    expect(captured.events).toBe(props.events);
    expect(reads).toBe(1);
    expect(readNativeState).not.toHaveBeenCalled();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});
