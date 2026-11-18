// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it, vi} from 'vitest';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
import {GPURasterFillNoData} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPURasterTileReprojection} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {captureGraphCommandNodes} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  addSurfacePreparationRecipe,
  type GPUSurfacePreparationOutput
} from '../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {snapshotRasterBand} from '../../src/raster-analysis/raster-band-utils';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

const GRID = {
  width: 3,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
};
const OPERATORS = ['rank', 'fill', 'reprojection', 'tiles'] as const;

it.each([
  'buffer',
  'texture'
] as const)('captures inherited and non-enumerable %s band fields (raster-terrain:F3)', kind => {
  const {graph, input} = createFixture();
  const values = createTransientView(graph, 'raw-values', 'float32', 3);
  const view = input.storage.kind === 'texture' ? input.storage.view : undefined;
  const expectedStorage = kind === 'buffer' ? {kind, values} : {kind, view, channel: 2};
  const storage = Object.create(expectedStorage);
  const band = Object.create({
    get id() {
      return 'inherited-band';
    },
    get format() {
      return 'float32';
    },
    get storage() {
      return storage;
    },
    get scale() {
      return 2;
    },
    get offset() {
      return 10;
    },
    get noDataValue() {
      return -9;
    },
    get validity() {
      return input.validity;
    }
  }) as GPURasterBand;
  try {
    const captured = snapshotRasterBand(band);
    expect(captured).toEqual({
      id: 'inherited-band',
      format: 'float32',
      storage: expectedStorage,
      scale: 2,
      offset: 10,
      noDataValue: -9,
      validity: input.validity
    });
    expect(captured.storage).not.toBe(storage);
  } finally {
    graph.device.destroy();
  }
});

it.each([
  ...OPERATORS,
  'surface-recipe'
] as const)('%s preserves inherited calibration and storage channel (raster-terrain:F3)', operator => {
  const {graph, input, createContributor} = createFixture();
  const band = Object.assign(
    Object.create({
      get scale() {
        return 2;
      },
      get offset() {
        return 10;
      }
    }),
    {
      id: input.id,
      format: input.format,
      validity: input.validity,
      noDataValue: input.noDataValue,
      storage: Object.assign(
        Object.create({
          get channel() {
            return 2;
          }
        }),
        {kind: 'texture', view: input.storage.kind === 'texture' ? input.storage.view : undefined}
      )
    }
  ) as GPURasterBand;
  const contributor = operator === 'surface-recipe' ? undefined : createContributor(operator, band);
  try {
    const nodes = contributor
      ? contributor.getCommandNodes(graph)
      : captureGraphCommandNodes(graph, () => {
          addSurfacePreparationRecipe(graph, {
            grid: GRID,
            input: band,
            maximumDistance: 1,
            filter: {radius: 0}
          });
        });
    const conversion = [...getParsedKernelSources(nodes).values()].find(source =>
      source.includes('textureLoad(')
    );
    expect(conversion).toContain('f32(rawSample) * 2.0 + 10.0');
    expect(conversion).toContain('textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0).z');
  } finally {
    if (contributor && 'destroy' in contributor) contributor.destroy();
    graph.device.destroy();
  }
});

it('F5/G7 rejects aliased surface outputs before allocation and allows same-ID retry', () => {
  const {graph, input} = createFixture();
  const buffer = graph.device.createBuffer({byteLength: 12, usage: Buffer.STORAGE});
  const views = ['original', 'unresolved'].map(id =>
    graph.createDataView(graph.importBuffer({id, byteLength: 12, usage: Buffer.STORAGE}, buffer), {
      format: 'uint32',
      length: id === 'original' ? 3 : 1
    })
  );
  const correctedCount = createTransientView(graph, 'corrected-count', 'uint32', 1);
  const add = vi.spyOn(graph, 'add');
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() =>
      addSurfacePreparationRecipe(graph, {
        id: 'sur',
        grid: GRID,
        input,
        maximumDistance: 1,
        outputs: {originalValidity: views[0], unresolvedCount: views[1]}
      })
    ).toThrow(/overlap|share a buffer/);
    expect(add).not.toHaveBeenCalled();
    const allocationCount = allocations.mock.calls.length;
    let retryError: string | undefined;
    try {
      const result = addSurfacePreparationRecipe(graph, {
        id: 'sur',
        grid: GRID,
        input,
        maximumDistance: 1,
        outputs: {originalValidity: views[0], unresolvedCount: correctedCount}
      });
      expect(result.outputs.originalValidity).toBe(views[0]);
      expect(result.outputs.unresolvedCount).toBe(correctedCount);
    } catch (error) {
      retryError = String(error);
    }
    expect({allocationCount, retryError}).toEqual({allocationCount: 0, retryError: undefined});
  } finally {
    buffer.destroy();
    graph.device.destroy();
  }
});

for (const name of [
  'values',
  'validity',
  'originalValidity',
  'filled',
  'filtered',
  'unresolvedCount'
] as const) {
  it.each([
    'rows',
    'format',
    'packing'
  ] as const)(`H2 rejects invalid ${name} %s before allocation and allows same-ID retry`, defect => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const grid = {...GRID, width: 2};
    const format = name === 'values' ? 'float32' : 'uint32';
    const expectedLength = name === 'unresolvedCount' ? 1 : 2;
    const input: GPURasterBand = {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'source', 'float32', 2)}
    };
    const corrected = createTransientView(graph, 'corrected', format, expectedLength);
    const invalid =
      defect === 'packing'
        ? graph.createDataView(
            graph.createTransientBuffer({
              id: 'invalid',
              byteLength: expectedLength * 8,
              usage: Buffer.STORAGE
            }),
            {format, length: expectedLength, byteStride: 8}
          )
        : createTransientView(
            graph,
            'invalid',
            defect === 'format' ? (format === 'float32' ? 'uint32' : 'float32') : format,
            defect === 'rows' ? (expectedLength === 1 ? 2 : 1) : expectedLength
          );
    const allocations = vi.spyOn(graph, 'createTransientBuffer');
    const add = vi.spyOn(graph, 'add');
    try {
      const props = {id: 'sur', grid, input, maximumDistance: 1};
      expect(() =>
        addSurfacePreparationRecipe(graph, {
          ...props,
          outputs: {[name]: invalid} as Partial<GPUSurfacePreparationOutput>
        })
      ).toThrow(
        defect === 'rows'
          ? /provided output must have \d+ rows/
          : defect === 'packing'
            ? /packed/
            : new RegExp(`${format} GPU data`)
      );
      const allocationCount = allocations.mock.calls.length;
      expect(add).not.toHaveBeenCalled();
      let retryError: string | undefined;
      try {
        const result = addSurfacePreparationRecipe(graph, {
          ...props,
          outputs: {[name]: corrected} as Partial<GPUSurfacePreparationOutput>
        });
        expect(result.outputs[name]).toBe(corrected);
      } catch (error) {
        retryError = String(error);
      }
      expect({allocationCount, retryError}).toEqual({allocationCount: 0, retryError: undefined});
    } finally {
      graph.device.destroy();
    }
  });
}

it('G7 captures inherited output getters once before validating and wiring the boundary', () => {
  const {graph, input} = createFixture();
  const originalValidity = createTransientView(graph, 'provided-original', 'uint32', 3);
  const count = createTransientView(graph, 'provided-count', 'uint32', 1);
  let currentCount = count;
  let originalReads = 0;
  let countReads = 0;
  const provided = Object.create({
    get originalValidity() {
      originalReads++;
      return originalValidity;
    },
    get unresolvedCount() {
      countReads++;
      return countReads === 1 ? currentCount : originalValidity;
    }
  });
  try {
    const nodes = captureGraphCommandNodes(graph, () => {
      const result = addSurfacePreparationRecipe(graph, {
        grid: GRID,
        input,
        maximumDistance: 1,
        outputs: provided
      });
      expect(result.outputs.originalValidity).toBe(originalValidity);
      expect(result.outputs.unresolvedCount).toBe(count);
    });
    currentCount = originalValidity;
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
    expect({originalReads, countReads}).toEqual({originalReads: 1, countReads: 1});
    expect(
      nodes.some(node =>
        node.resources?.some(resource => 'buffer' in resource && resource.buffer === count)
      )
    ).toBe(true);
  } finally {
    graph.device.destroy();
  }
});

it('surface preparation composes texture-backed fill and rank stages (RT-09)', () => {
  const {graph, input} = createFixture();
  try {
    const nodes = captureGraphCommandNodes(graph, () => {
      const result = addSurfacePreparationRecipe(graph, {
        grid: GRID,
        input,
        maximumDistance: 1,
        filter: {radius: 0}
      });
      expect(result.band.storage.kind).toBe('buffer');
      expect(result.band.validity).toBe(result.outputs.validity);
    });
    const sources = getParsedKernelSources(nodes);
    expect([...sources.values()].some(source => source.includes('textureLoad('))).toBe(true);
    expect(sources.has('surface-preparation-recipe-filter')).toBe(true);
  } finally {
    graph.device.destroy();
  }
});

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const texture = graph.createTransientTexture({
    id: 'source',
    format: 'rgba32float',
    width: 3,
    height: 1,
    usage: Texture.SAMPLE
  });
  const input: GPURasterBand = {
    id: 'source',
    format: 'float32',
    storage: {kind: 'texture', view: graph.createTextureView(texture), channel: 2},
    noDataValue: -9,
    scale: 2,
    offset: 10,
    validity: createTransientView(graph, 'mask', 'uint32', 3)
  };
  const output = {
    values: createTransientView(graph, 'values', 'float32', 3),
    validity: createTransientView(graph, 'validity', 'uint32', 3)
  };
  const filled = createTransientView(graph, 'filled', 'uint32', 3);
  function createContributor(operator: (typeof OPERATORS)[number], band: GPURasterBand = input) {
    switch (operator) {
      case 'rank':
        return new GPURasterRankFilter({id: operator, grid: GRID, input: band, radius: 0, output});
      case 'fill':
        return new GPURasterFillNoData({
          id: operator,
          grid: GRID,
          input: band,
          maximumDistance: 1,
          output: {...output, filled, status: {}}
        });
      case 'reprojection':
        return new GPURasterReprojection({
          id: operator,
          sourceGrid: GRID,
          destinationGrid: GRID,
          input: band,
          output
        });
      case 'tiles':
        return new GPURasterTileReprojection({
          id: operator,
          sourceGrid: GRID,
          destinationGrid: GRID,
          sources: [{pixelBounds: [0, 0, 3, 1], input: band}],
          output
        });
    }
  }
  return {graph, input, createContributor};
}

it.each(OPERATORS)('%s rejects an absent texture channel at construction (RT-09)', operator => {
  const {graph, input, createContributor} = createFixture();
  try {
    const invalid = {...input, storage: {...input.storage, channel: 4}} as unknown as GPURasterBand;
    expect(() => createContributor(operator, invalid)).toThrow(/channel/);
  } finally {
    graph.device.destroy();
  }
});

for (const timing of ['before wiring', 'after wiring'] as const) {
  it.each(OPERATORS)('%s captures band metadata and bindings ' + timing + ' (R1)', operator => {
    const {graph, input, createContributor} = createFixture();
    const originalMask = input.validity;
    const originalTexture = input.storage.kind === 'texture' ? input.storage.view : undefined;
    const contributor = createContributor(operator);
    const mutateInput = () => {
      input.scale = 3;
      input.offset = 99;
      input.noDataValue = 1;
      input.validity = createTransientView(graph, 'replacement-mask', 'uint32', 3);
      if (input.storage.kind === 'texture') {
        input.storage.channel = 0;
        input.storage.view = graph.createTextureView(
          graph.createTransientTexture({
            id: 'replacement-texture',
            format: 'rgba32float',
            width: 3,
            height: 1,
            usage: Texture.SAMPLE
          })
        );
      }
    };
    try {
      if (timing === 'before wiring') mutateInput();
      const nodes = contributor.getCommandNodes(graph);
      if (timing === 'after wiring') mutateInput();
      const conversion = [...getParsedKernelSources(nodes).values()].find(source =>
        source.includes('textureLoad(')
      );
      expect(conversion).toContain('textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0).z');
      expect(conversion).toContain('f32(rawSample) * 2.0 + 10.0');
      expect(conversion).toContain('rawSample != -9.0');
      expect(conversion).toContain('sourceValidity[SOURCE_VALIDITY_OFFSET + pixelIndex] != 0u');
      expect(
        nodes.some(node =>
          node.resources?.some(resource => 'buffer' in resource && resource.buffer === originalMask)
        )
      ).toBe(true);
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource => 'texture' in resource && resource.texture === originalTexture
          )
        )
      ).toBe(true);
    } finally {
      if ('destroy' in contributor) contributor.destroy();
      graph.device.destroy();
    }
  });
}

it.each([
  'foreign',
  'extent',
  'channel'
] as const)('surface recipe rejects %s input before allocating and allows same-ID retry (R2)', defect => {
  const {graph, input} = createFixture();
  const foreign = createFixture();
  let invalid: GPURasterBand;
  if (defect === 'foreign') invalid = {...foreign.input, validity: undefined};
  else if (defect === 'channel')
    invalid = {...input, storage: {...input.storage, channel: 4}} as unknown as GPURasterBand;
  else
    invalid = {
      ...input,
      storage: {
        kind: 'texture',
        view: graph.createTextureView(
          graph.createTransientTexture({
            id: 'bad-extent',
            format: 'rgba32float',
            width: 2,
            height: 1,
            usage: Texture.SAMPLE
          })
        )
      }
    };
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    const props = {id: 'surface', grid: GRID, maximumDistance: 1};
    expect(() => addSurfacePreparationRecipe(graph, {...props, input: invalid})).toThrow();
    const allocationCount = allocations.mock.calls.length;
    let retryError: string | undefined;
    try {
      addSurfacePreparationRecipe(graph, {...props, input});
    } catch (error) {
      retryError = String(error);
    }
    expect({allocationCount, retryError}).toEqual({allocationCount: 0, retryError: undefined});
  } finally {
    foreign.graph.device.destroy();
    graph.device.destroy();
  }
});

it.each(
  OPERATORS
)('%s rejects nonsampleable texture at construction and allows same-ID retry (R3)', operator => {
  const {graph, input, createContributor} = createFixture();
  const invalid: GPURasterBand = {
    ...input,
    storage: {
      kind: 'texture',
      view: graph.createTextureView(
        graph.createTransientTexture({
          id: 'copy-only',
          format: 'rgba32float',
          width: 3,
          height: 1,
          usage: Texture.COPY_DST
        })
      )
    }
  };
  const valid: GPURasterBand = {
    ...input,
    storage: {
      kind: 'texture',
      view: graph.createTextureView(
        graph.createTransientTexture({
          id: 'sampleable',
          format: 'rgba32float',
          width: 3,
          height: 1,
          usage: Texture.COPY_DST | Texture.SAMPLE
        })
      ),
      channel: 2
    }
  };
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => createContributor(operator, invalid)).toThrow(
      new RegExp(`^${operator} input.*Texture.SAMPLE`)
    );
    expect(allocations).not.toHaveBeenCalled();
    const contributor = createContributor(operator, valid);
    try {
      expect(() => graph.add(contributor)).not.toThrow();
    } finally {
      if ('destroy' in contributor) contributor.destroy();
    }
  } finally {
    graph.device.destroy();
  }
});

for (const [property, value] of [
  ['scale', 1e39],
  ['offset', -1e39]
] as const) {
  it.each(
    OPERATORS
  )(`%s rejects float32-overflowing input.${property} at construction (R4)`, operator => {
    const {graph, input, createContributor} = createFixture();
    try {
      expect(() => createContributor(operator, {...input, [property]: value})).toThrow(
        new RegExp(`^${operator} input\\.${property} must be a finite number`)
      );
    } finally {
      graph.device.destroy();
    }
  });
}

it.each(OPERATORS)('%s accepts finite float32 calibration boundaries (R4)', operator => {
  const {graph, input, createContributor} = createFixture();
  const maximumFloat32 = Math.fround(3.4028234663852886e38);
  try {
    const contributor = createContributor(operator, {
      ...input,
      scale: maximumFloat32,
      offset: -maximumFloat32
    });
    try {
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    } finally {
      if ('destroy' in contributor) contributor.destroy();
    }
  } finally {
    graph.device.destroy();
  }
});

it.each(OPERATORS)('%s rejects the wrong texture extent at construction (RT-09)', operator => {
  const {graph, input, createContributor} = createFixture();
  const texture = graph.createTransientTexture({
    id: 'wrong-extent',
    format: 'rgba32float',
    width: 2,
    height: 1,
    usage: Texture.SAMPLE
  });
  try {
    const invalid: GPURasterBand = {
      ...input,
      storage: {kind: 'texture', view: graph.createTextureView(texture)}
    };
    expect(() => createContributor(operator, invalid)).toThrow(/texture extent/);
  } finally {
    graph.device.destroy();
  }
});

it.each(
  OPERATORS
)('%s rejects foreign textures before allocating or adding nodes (RT-09)', operator => {
  const {graph, createContributor} = createFixture();
  const foreign = createFixture();
  // No mask here: a buffer-only ownership check would miss the foreign texture.
  const contributor = createContributor(operator, {...foreign.input, validity: undefined});
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  const additions = vi.spyOn(graph, 'add');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/target graph/);
    expect(allocations).not.toHaveBeenCalled();
    expect(additions).not.toHaveBeenCalled();
  } finally {
    if ('destroy' in contributor) contributor.destroy();
    foreign.graph.device.destroy();
    graph.device.destroy();
  }
});

it.each(OPERATORS)('%s emits selected-channel calibrated texture conversion (RT-09)', operator => {
  const {graph, createContributor} = createFixture();
  const contributor = createContributor(operator);
  try {
    const nodes = contributor.getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(1);
    const conversion = [...sources.values()].find(source => source.includes('textureLoad('));
    expect(conversion).toContain('textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0).z');
    expect(conversion).toContain('rawSample != -9.0');
    expect(conversion).toContain('f32(rawSample) * 2.0 + 10.0');
    expect(conversion).toContain('sourceValidity[SOURCE_VALIDITY_OFFSET + pixelIndex] != 0u');
    if (contributor instanceof GPURasterTileReprojection) {
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    }
  } finally {
    if ('destroy' in contributor) contributor.destroy();
    graph.device.destroy();
  }
});
