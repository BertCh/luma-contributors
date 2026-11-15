// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it, vi} from 'vitest';
import {GPURasterArithmetic} from '../../../src/raster-analysis/raster-algebra/gpu-raster-arithmetic';
import {GPURasterSampling} from '../../../src/raster-analysis/raster-sampling/gpu-raster-sampling';
import {GPURasterProfile} from '../../../src/raster-analysis/raster-sampling/gpu-raster-profile';
import {GPURasterConditional} from '../../../src/raster-analysis/raster-algebra/gpu-raster-conditional';
import {GPURasterReclassify} from '../../../src/raster-analysis/raster-algebra/gpu-raster-reclassify';
import {GPURasterStretch} from '../../../src/raster-analysis/raster-stretch/gpu-raster-stretch';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createFixture(width = 3, height = 3) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width, height});
  const cellCount = width * height;
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'sint32' | 'float32x2'>(
    format: Format,
    length = cellCount
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const band: GPURasterBand = {
    id: 'source',
    format: 'uint32',
    storage: {kind: 'buffer', values: view('uint32', cellCount + 3)},
    validity: view('uint32'),
    noDataValue: 99,
    scale: 2,
    offset: 10
  };
  const parameters = view('float32', 16);
  const createReclassify = (input: GPURasterBand = band, output = view('uint32')) =>
    new GPURasterReclassify({
      id: 'band-reclassify',
      grid,
      values: input,
      parameters,
      breaks: view('float32', 2),
      output: {classes: output}
    });
  const createConditional = (input: GPURasterBand = band) =>
    new GPURasterConditional({
      id: 'band-conditional',
      grid,
      cellCount,
      conditionValues: input,
      a: band,
      b: band,
      parameters,
      output: {values: view('float32')}
    });
  const createStretch = (input: GPURasterBand = band) =>
    new GPURasterStretch({
      id: 'band-stretch',
      grid,
      values: input,
      parameters,
      binCount: 16,
      lutSize: 2,
      output: {stretched: view('float32')}
    });
  const createArithmetic = (input: GPURasterBand = band) =>
    new GPURasterArithmetic({
      id: 'band-arithmetic',
      grid,
      cellCount,
      a: input,
      b: band,
      parameters,
      output: {values: view('float32')}
    });
  const createSampling = (input: GPURasterBand = band) =>
    new GPURasterSampling({
      id: 'band-sampling',
      grid,
      values: input,
      positions: view('float32x2'),
      parameters,
      output: {values: view('float32')}
    });
  const createProfile = (input: GPURasterBand = band) =>
    new GPURasterProfile({
      id: 'band-profile',
      grid,
      values: input,
      positions: view('float32x2'),
      lineOffsets: view('uint32', 2),
      parameters,
      output: {
        sampleValues: view('float32'),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    });
  return {
    createArithmetic,
    createSampling,
    createProfile,
    device,
    graph,
    grid,
    view,
    band,
    parameters,
    createReclassify,
    createConditional,
    createStretch
  };
}

it.each([
  'buffer',
  'texture'
] as const)('accepts published %s bands through public raster primitives', storageKind => {
  const fixture = createFixture();
  try {
    const {
      graph,
      grid,
      band,
      createReclassify,
      createConditional,
      createStretch,
      createArithmetic,
      createSampling,
      createProfile
    } = fixture;
    const textureBand: GPURasterBand = {
      ...band,
      storage: {
        kind: 'texture',
        view: graph.createTextureView(
          graph.createTransientTexture({
            id: 'raster-texture',
            format: 'r32uint',
            ...grid,
            usage: Texture.SAMPLE
          })
        )
      }
    };
    const input = storageKind === 'texture' ? textureBand : band;
    for (const create of [
      createReclassify,
      createConditional,
      createStretch,
      createArithmetic,
      createSampling,
      createProfile
    ]) {
      const contributor = create(input);
      const nodes = contributor.getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      // Every band is canonicalized before downstream kernels, not treated as a bare view.
      expect(sources.some(({source}) => source.includes('sourceValidity'))).toBe(true);
      expect(sources.every(({source}) => (source.match(/var<storage/g) ?? []).length <= 8)).toBe(
        true
      );
    }
    expect(createReclassify().rowCount).toBe(9); // Ignore the backing view's padding.
  } finally {
    fixture.device.destroy();
  }
});

it('rejects missing grids, short validity, wrong formats and aliases at construction', () => {
  const fixture = createFixture();
  try {
    const {
      band,
      view,
      parameters,
      createReclassify,
      createConditional,
      createStretch,
      createArithmetic,
      createSampling,
      createProfile
    } = fixture;
    expect(
      () =>
        new GPURasterReclassify({
          values: band,
          parameters,
          breaks: view('float32', 2),
          output: {classes: view('uint32')}
        })
    ).toThrow(/^raster-reclassify values requires grid/);
    expect(
      () =>
        new GPURasterConditional({
          cellCount: 9,
          mask: view('uint32'),
          a: band,
          parameters,
          output: {values: view('float32')}
        })
    ).toThrow(/^raster-conditional a requires grid/);
    for (const create of [
      createReclassify,
      createConditional,
      createStretch,
      createArithmetic,
      createSampling,
      createProfile
    ]) {
      expect(() => create({...band, validity: view('uint32', 8)})).toThrow(/validity.*at least 9/);
      expect(() =>
        create({
          ...band,
          storage: {kind: 'buffer', values: view('float32')}
        } as unknown as GPURasterBand)
      ).toThrow(/uint32/);
    }
    expect(() => createReclassify(band, band.validity!)).toThrow(/^band-reclassify outputs/);
    expect(() =>
      createReclassify(
        band,
        band.storage.kind === 'buffer'
          ? (band.storage.values as GraphDataView<'uint32'>)
          : view('uint32')
      )
    ).toThrow(/^band-reclassify outputs/);
  } finally {
    fixture.device.destroy();
  }
});

it('rejects a band texture from another graph before contributing nodes', () => {
  const fixture = createFixture();
  try {
    const foreignGraph = new GPUCommandGraph(fixture.device);
    const band: GPURasterBand = {
      ...fixture.band,
      validity: undefined,
      storage: {
        kind: 'texture',
        view: foreignGraph.createTextureView(
          foreignGraph.createTransientTexture({
            id: 'foreign',
            format: 'r32uint',
            width: 3,
            height: 3,
            usage: Texture.SAMPLE
          })
        )
      }
    };
    for (const create of [
      fixture.createReclassify,
      fixture.createConditional,
      fixture.createStretch,
      fixture.createArithmetic,
      fixture.createSampling,
      fixture.createProfile
    ]) {
      expect(() => create(band).getCommandNodes(fixture.graph)).toThrow(/target graph/);
    }
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  'a',
  'b'
] as const)('rejects a foreign arithmetic %s texture without allocating, allowing a corrected retry', operand => {
  const fixture = createFixture(1, 1);
  try {
    const foreignGraph = new GPUCommandGraph(fixture.device);
    const foreignBand: GPURasterBand = {
      ...fixture.band,
      validity: undefined,
      storage: {
        kind: 'texture',
        view: foreignGraph.createTextureView(
          foreignGraph.createTransientTexture({
            id: 'foreign-arithmetic-texture',
            format: 'r32uint',
            width: 1,
            height: 1,
            usage: Texture.SAMPLE
          })
        )
      }
    };
    const props = {
      id: 'arithmetic-retry',
      grid: fixture.grid,
      cellCount: 1,
      a: fixture.band,
      b: fixture.band,
      parameters: fixture.parameters,
      output: {values: fixture.view('float32')}
    };
    const invalid = new GPURasterArithmetic({...props, [operand]: foreignBand});
    const allocations = vi.spyOn(fixture.graph, 'createTransientBuffer');
    expect(() => invalid.getCommandNodes(fixture.graph)).toThrow(/target graph/);
    const failedAllocations = allocations.mock.calls.length;
    // Reuse exactly the same graph, contributor ID and outputs after correcting the operand.
    const corrected = new GPURasterArithmetic(props);
    expect(() => {
      const nodes = corrected.getCommandNodes(fixture.graph);
      for (const node of nodes) fixture.graph.add(node);
    }).not.toThrow();
    expect(failedAllocations).toBe(0);
  } finally {
    vi.restoreAllMocks();
    fixture.device.destroy();
  }
});

it.each([
  {property: 'noDataValue', value: 0.5, range: '[0, 4294967295]'},
  {property: 'noDataValue', value: -1, range: '[0, 4294967295]'},
  {property: 'noDataValue', value: 0x100000000, range: '[0, 4294967295]'},
  {property: 'scale', value: 1e100, range: 'finite number'},
  {property: 'offset', value: -1e100, range: 'finite number'}
])('rejects invalid band $property=$value during construction', ({property, value, range}) => {
  const fixture = createFixture(1, 1);
  try {
    const invalidBand = {...fixture.band, [property]: value};
    for (const [id, inputName, create] of [
      ['band-reclassify', 'values', fixture.createReclassify],
      ['band-conditional', 'conditionValues', fixture.createConditional],
      ['band-stretch', 'values', fixture.createStretch],
      ['band-arithmetic', 'a', fixture.createArithmetic],
      ['band-sampling', 'values', fixture.createSampling],
      ['band-profile', 'values', fixture.createProfile]
    ] as const) {
      let failure: unknown;
      try {
        create(invalidBand);
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(Error);
      const message = (failure as Error).message;
      expect(message.startsWith(`${id} ${inputName}.${property} `)).toBe(true);
      expect(message).toContain(range);
    }
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  {
    format: 'r32float' as const,
    channel: 0,
    property: 'storage.view.format',
    range: 'r32uint or rgba32uint'
  },
  {format: 'r32uint' as const, channel: 1, property: 'storage.channel', range: '[0, 0]'},
  {format: 'rgba32uint' as const, channel: 4, property: 'storage.channel', range: '[0, 3]'}
])('rejects invalid band texture $property during construction', ({
  format,
  channel,
  property,
  range
}) => {
  const fixture = createFixture();
  try {
    const invalidBand = {
      ...fixture.band,
      storage: {
        kind: 'texture',
        view: fixture.graph.createTextureView(
          fixture.graph.createTransientTexture({
            id: 'invalid-texture',
            format,
            width: 3,
            height: 3,
            usage: Texture.SAMPLE
          })
        ),
        channel
      }
    } as unknown as GPURasterBand;
    for (const [id, inputName, create] of [
      ['band-reclassify', 'values', fixture.createReclassify],
      ['band-conditional', 'conditionValues', fixture.createConditional],
      ['band-stretch', 'values', fixture.createStretch],
      ['band-arithmetic', 'a', fixture.createArithmetic],
      ['band-sampling', 'values', fixture.createSampling],
      ['band-profile', 'values', fixture.createProfile]
    ] as const) {
      let failure: unknown;
      try {
        create(invalidBand);
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(Error);
      const message = (failure as Error).message;
      expect(message.startsWith(`${id} ${inputName}.${property} `)).toBe(true);
      expect(message).toContain(range);
    }
  } finally {
    fixture.device.destroy();
  }
});

it('accepts native sentinel limits, float NaN, and finite float32 calibration limits', () => {
  const fixture = createFixture(1, 1);
  try {
    for (const [format, noDataValue] of [
      ['uint32', 0],
      ['uint32', 0xffffffff],
      ['sint32', -0x80000000],
      ['sint32', 0x7fffffff],
      ['float32', NaN],
      ['float32', 3.4028235e38],
      ['float32', -3.4028235e38]
    ] as const) {
      const band = {
        ...fixture.band,
        format,
        noDataValue,
        scale: 3.4028235e38,
        offset: -3.4028235e38,
        storage: {kind: 'buffer', values: fixture.view(format)}
      } as GPURasterBand;
      for (const create of [
        fixture.createReclassify,
        fixture.createConditional,
        fixture.createStretch,
        fixture.createArithmetic,
        fixture.createSampling,
        fixture.createProfile
      ]) {
        expect(() => create(band)).not.toThrow();
      }
    }
  } finally {
    fixture.device.destroy();
  }
});

it('rejects invalid signed and floating sentinels and conditional operand metadata at construction', () => {
  const fixture = createFixture(1, 1);
  try {
    for (const [format, noDataValue] of [
      ['sint32', -0x80000001],
      ['sint32', 0x80000000],
      ['sint32', 0.5],
      ['uint32', NaN],
      ['float32', Infinity],
      ['float32', 1e100]
    ] as const) {
      const band = {
        ...fixture.band,
        format,
        noDataValue,
        storage: {kind: 'buffer', values: fixture.view(format)}
      } as GPURasterBand;
      for (const create of [
        fixture.createReclassify,
        fixture.createConditional,
        fixture.createStretch,
        fixture.createArithmetic,
        fixture.createSampling,
        fixture.createProfile
      ]) {
        expect(() => create(band)).toThrow(
          /^(band-reclassify values|band-conditional conditionValues|band-stretch values|band-arithmetic a|band-sampling values|band-profile values)\.noDataValue .*\[/
        );
      }
    }
    for (const property of ['a', 'b'] as const) {
      expect(
        () =>
          new GPURasterConditional({
            id: 'conditional-operands',
            grid: fixture.grid,
            cellCount: 1,
            mask: fixture.view('uint32'),
            [property]: {...fixture.band, offset: 1e100},
            parameters: fixture.parameters,
            output: {values: fixture.view('float32')}
          })
      ).toThrow(new RegExp(`^conditional-operands ${property}\\.offset .*\\[`));
    }
  } finally {
    fixture.device.destroy();
  }
});

it('rejects multi-mip, array and multisampled raster textures at construction', () => {
  const fixture = createFixture();
  try {
    for (const {texture, error} of [
      {
        texture: fixture.graph.createTransientTexture({
          id: 'multiple-mips',
          format: 'r32uint',
          width: 3,
          height: 3,
          mipLevels: 2,
          usage: Texture.SAMPLE
        }),
        error: /storage\.view\.mipLevelCount .*\[1, 1\]/
      },
      {
        texture: fixture.graph.createTransientTexture({
          id: 'array-texture',
          format: 'r32uint',
          width: 3,
          height: 3,
          dimension: '2d-array',
          depth: 2,
          usage: Texture.SAMPLE
        }),
        error: /storage\.view\.dimension must be 2d/
      },
      {
        texture: fixture.graph.createTransientTexture({
          id: 'multiple-samples',
          format: 'r32uint',
          width: 3,
          height: 3,
          samples: 4,
          usage: Texture.SAMPLE
        }),
        error: /storage\.view\.texture\.samples .*\[1, 1\]/
      }
    ]) {
      const band = {
        ...fixture.band,
        storage: {kind: 'texture', view: fixture.graph.createTextureView(texture)}
      } as GPURasterBand;
      for (const create of [
        fixture.createReclassify,
        fixture.createConditional,
        fixture.createStretch,
        fixture.createArithmetic,
        fixture.createSampling,
        fixture.createProfile
      ]) {
        expect(() => create(band)).toThrow(error);
      }
    }
  } finally {
    fixture.device.destroy();
  }
});

it('rejects band validity owned by another graph at construction', () => {
  const fixture = createFixture();
  try {
    const foreignGraph = new GPUCommandGraph(fixture.device);
    const band = {
      ...fixture.band,
      validity: createTransientView(foreignGraph, 'foreign-validity', 'uint32', 9)
    };
    for (const create of [
      fixture.createReclassify,
      fixture.createConditional,
      fixture.createStretch,
      fixture.createArithmetic,
      fixture.createSampling,
      fixture.createProfile
    ]) {
      expect(() => create(band)).toThrow(/validity must belong to the same graph/);
    }
  } finally {
    fixture.device.destroy();
  }
});

it.each(
  (['arithmetic', 'arithmetic-second', 'sampling', 'profile'] as const).flatMap(contributor =>
    (['buffer', 'texture'] as const).flatMap(storageKind =>
      [64, 80].map(limit => ({contributor, storageKind, limit}))
    )
  )
)('preflights $contributor $storageKind conversion at a $limit-byte limit before allocation and retry', ({
  contributor,
  storageKind,
  limit
}) => {
  const fixture = createFixture(1, 1);
  const {device, graph, view, parameters} = fixture;
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: limit}
  });
  const positions = view('float32x2', 1);
  const lineOffsets = view('uint32', 2);
  const outputValues = view('float32', 1);
  const status = {count: view('uint32', 1), overflow: view('uint32', 1)};
  const legacyOperand = view('float32', 1);
  const build = (width: number) => {
    const grid = makeGPURasterMetadata({width, height: width});
    const metadata = {id: `source-${width}`, format: 'float32' as const};
    const band: GPURasterBand =
      storageKind === 'buffer'
        ? {...metadata, storage: {kind: 'buffer', values: view('float32', width * width)}}
        : {
            ...metadata,
            storage: {
              kind: 'texture',
              view: graph.createTextureView(
                graph.createTransientTexture({
                  id: `texture-${width}`,
                  format: 'r32float',
                  width,
                  height: width,
                  usage: Texture.SAMPLE
                })
              )
            }
          };
    const shared = {id: 'conversion-retry', grid, parameters};
    if (contributor === 'arithmetic' || contributor === 'arithmetic-second') {
      return new GPURasterArithmetic({
        ...shared,
        cellCount: 1,
        a: contributor === 'arithmetic' ? band : legacyOperand,
        b: band,
        output: {values: outputValues}
      });
    }
    if (contributor === 'sampling') {
      return new GPURasterSampling({
        ...shared,
        values: band,
        positions,
        output: {values: outputValues}
      });
    }
    return new GPURasterProfile({
      ...shared,
      values: band,
      positions,
      lineOffsets,
      output: {sampleValues: outputValues, status}
    });
  };
  // Prepare both inputs before spying: only contributor scratch allocations count.
  const invalid = build(5);
  const corrected = build(4);
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    let failure: unknown;
    try {
      invalid.getCommandNodes(graph);
    } catch (error) {
      failure = error;
    }
    const failedAllocations = allocations.mock.calls.length;
    expect(() => {
      for (const node of corrected.getCommandNodes(graph)) graph.add(node);
    }).not.toThrow();
    expect(failedAllocations).toBe(0);
    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toMatch(
      new RegExp(`^conversion-retry.*100 bytes.*limit ${limit}.*reduce grid`)
    );
  } finally {
    allocations.mockRestore();
    device.destroy();
  }
});
