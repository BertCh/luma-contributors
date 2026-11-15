// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Texture} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPURasterSpectralFilter,
  getGPURasterSpectralFilterParameterValues,
  MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION,
  MINIMUM_RASTER_SPECTRAL_SIGMA,
  type GPURasterSpectralFilterProps
} from '../../../src/raster-analysis/spectral-filter/index';
import {GPURasterRegistration} from '../../../src/raster-analysis/remote-sensing/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createFixture(width = 16, height = 8) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'sint32' | 'float32x2'>(
    format: Format,
    length = width * height
  ) => createTransientView(graph, `spectral-${serial++}`, format, length);
  const props: GPURasterSpectralFilterProps = {
    id: 'spectral-test',
    grid: makeGPURasterMetadata({width, height}),
    input: {
      id: 'band',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32')},
      validity: view('uint32')
    },
    filter: 'low-pass',
    parameters: view('float32', 8),
    output: {
      values: view('float32'),
      validity: view('uint32'),
      status: {invalidCount: view('uint32', 1)}
    }
  };
  return {device, graph, view, props};
}

it('packs frequency settings and enforces named float32 limits', () => {
  expect(
    Array.from(
      getGPURasterSpectralFilterParameterValues({
        cutoff: 0.125,
        lowerCutoff: 0.25,
        upperCutoff: 0.5,
        sigma: 0.25,
        notchCenter: [-0.5, 0.5],
        notchRadius: 0
      })
    )
  ).toEqual([0.125, 0.25, 0.5, 0.25, -0.5, 0.5, 0, 0]);
  expect(getGPURasterSpectralFilterParameterValues({sigma: MINIMUM_RASTER_SPECTRAL_SIGMA})[3]).toBe(
    MINIMUM_RASTER_SPECTRAL_SIGMA
  );
  for (const settings of [
    {cutoff: -1},
    {cutoff: Infinity},
    {upperCutoff: 1},
    {lowerCutoff: 0.5, upperCutoff: 0.25},
    {sigma: 0},
    {sigma: 1e-40},
    {notchRadius: NaN},
    {notchCenter: [0.6, 0] as const}
  ]) {
    expect(() => getGPURasterSpectralFilterParameterValues(settings)).toThrow(
      /^getGPURasterSpectralFilterParameterValues /
    );
  }
  expect(() => getGPURasterSpectralFilterParameterValues({}, new Float32Array(7))).toThrow(
    /target must hold 8/
  );
});

it('rejects unsupported dimensions, short views, aliases and foreign graphs', () => {
  const {device, graph, view, props} = createFixture();
  try {
    for (const width of [1, 3, MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION * 2]) {
      expect(() => new GPURasterSpectralFilter({...props, grid: {...props.grid, width}})).toThrow(
        /^spectral-test grid.width/
      );
    }
    expect(() => new GPURasterSpectralFilter({...props, parameters: view('float32', 7)})).toThrow(
      /parameters.*8/
    );
    expect(
      () =>
        new GPURasterSpectralFilter({
          ...props,
          input: {...props.input, validity: view('uint32', 1)}
        })
    ).toThrow(/input.validity.*128/);
    expect(
      () =>
        new GPURasterSpectralFilter({
          ...props,
          output: {...props.output, values: view('float32', 127)}
        })
    ).toThrow(/output.values.*128/);
    expect(
      () =>
        new GPURasterSpectralFilter({...props, output: {...props.output, values: props.parameters}})
    ).toThrow();
    expect(
      () =>
        new GPURasterSpectralFilter({
          ...props,
          output: {...props.output, status: {invalidCount: view('uint32', 0)}}
        })
    ).toThrow(/invalidCount.*one uint32/);
    expect(
      () =>
        new GPURasterSpectralFilter({
          ...props,
          filter: 'invalid' as GPURasterSpectralFilterProps['filter']
        })
    ).toThrow(/filter must be/);
    const foreign = new GPUCommandGraph(device);
    expect(() => new GPURasterSpectralFilter(props).getCommandNodes(foreign)).toThrow(
      /belong to the target graph/
    );
    expect(graph).toBeDefined();
  } finally {
    device.destroy();
  }
});

it('parses all filter kernels and bounds actual nodes at the minimum and maximum radix-two sizes', () => {
  for (const [width, height] of [
    [2, 2],
    [16, 8],
    [2048, 2048]
  ]) {
    const {device, graph, props} = createFixture(width, height);
    try {
      for (const filter of ['low-pass', 'high-pass', 'band-pass', 'gaussian', 'notch'] as const) {
        for (const status of [props.output.status, undefined]) {
          const contributor = new GPURasterSpectralFilter({
            ...props,
            id: `${props.id}-${filter}-${status ? 'status' : 'no-status'}`,
            filter,
            output: {...props.output, status}
          });
          const nodes = contributor.getCommandNodes(graph);
          expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
          const sources = getParsedKernelSources(nodes);
          expect(sources.size).toBeGreaterThan(0);
          for (const source of sources.values()) {
            expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
            expect(auditWGSLModule(source)).toEqual([]);
          }
        }
      }
    } finally {
      device.destroy();
    }
  }
});

it('checks device scratch limits before creating any nodes', () => {
  const {device, graph, props} = createFixture(2048, 2048);
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 1024});
    expect(() => new GPURasterSpectralFilter(props).getCommandNodes(graph)).toThrow(
      /spectral-test complex scratch.*limit 1024/
    );
  } finally {
    device.destroy();
  }
});

it('validates optional peak-ratio outputs for both registration modes', () => {
  const {device, graph, view, props} = createFixture();
  try {
    for (const mode of ['periodic', 'masked-linear'] as const) {
      const registration = {
        id: 'ratio-test',
        mode,
        grid: props.grid,
        reference: view('float32'),
        moving: view('float32'),
        output: {
          shift: view('float32x2', 1),
          peak: view('float32', 1),
          status: view('uint32', 1),
          secondPeakRatio: view('float32', 1)
        }
      };
      const sources = getParsedKernelSources(
        new GPURasterRegistration(registration).getCommandNodes(graph)
      );
      expect([...sources.values()].some(source => source.includes('secondPeakRatio'))).toBe(true);
      for (const source of sources.values()) {
        expect(auditWGSLModule(source)).toEqual([]);
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
      }
      expect(
        () =>
          new GPURasterRegistration({
            ...registration,
            output: {...registration.output, secondPeakRatio: view('float32', 0)}
          })
      ).toThrow(/ratio-test/);
      expect(
        () =>
          new GPURasterRegistration({
            ...registration,
            output: {...registration.output, secondPeakRatio: registration.reference}
          })
      ).toThrow(/ratio-test/);
    }
  } finally {
    device.destroy();
  }
});

it('supports calibrated texture bands with padded validity within the same command-node budget', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const texture = graph.createTransientTexture({
      id: 'texture',
      format: 'r32float',
      width: 16,
      height: 8,
      usage: Texture.SAMPLE
    });
    const validity = view('uint32', 132);
    const contributor = new GPURasterSpectralFilter({
      ...props,
      input: {
        id: 'texture-band',
        format: 'float32',
        validity: graph.createDataView(validity.buffer, {
          format: 'uint32',
          byteOffset: 8,
          length: 130
        }),
        storage: {
          kind: 'texture',
          view: graph.createTextureView(texture, {mipLevelCount: 1}),
          channel: 0
        },
        scale: 0.5,
        offset: 2,
        noDataValue: -999
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const source of getParsedKernelSources(nodes).values()) {
      expect(auditWGSLModule(source)).toEqual([]);
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    }
  } finally {
    device.destroy();
  }
});

it('canonicalizes only the raster prefix of padded buffer bands and validity views', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const values = view('float32', 131);
    const validity = view('uint32', 132);
    const input = {
      id: 'padded-band',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: graph.createDataView(values.buffer, {format: 'float32', byteOffset: 4, length: 130})
      },
      validity: graph.createDataView(validity.buffer, {
        format: 'uint32',
        byteOffset: 8,
        length: 130
      }),
      scale: 1.5,
      offset: -2
    };
    const contributor = new GPURasterSpectralFilter({...props, input});
    const nodes = contributor.getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(0);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    // Adapting the band must not shrink or replace the caller's views.
    expect(input.storage.values.length).toBe(130);
    expect(input.storage.values.byteOffset).toBe(4);
    expect(input.validity.length).toBe(130);
    expect(input.validity.byteOffset).toBe(8);
  } finally {
    device.destroy();
  }
});

it('rejects unrepresentable float32 band metadata at construction with named limits', () => {
  const {device, props} = createFixture(2, 2);
  try {
    for (const name of ['scale', 'offset', 'noDataValue'] as const) {
      for (const value of [
        1e100,
        -1e100,
        Infinity,
        -Infinity,
        ...(name === 'noDataValue' ? [] : [NaN])
      ]) {
        expect(
          () =>
            new GPURasterSpectralFilter({
              ...props,
              input: {...props.input, [name]: value}
            }),
          `${name}=${value}`
        ).toThrow(new RegExp(`^spectral-test input\\.${name} must be a finite number in \\[`));
      }
    }
  } finally {
    device.destroy();
  }
});

it('rejects fractional and out-of-range integer nodata at construction with named limits', () => {
  const {device, props, view} = createFixture(2, 2);
  try {
    for (const format of ['uint32', 'sint32'] as const) {
      const invalidValues =
        format === 'uint32' ? [0.5, -1, 4294967296, NaN] : [0.5, -2147483649, 2147483648, NaN];
      for (const noDataValue of invalidValues) {
        const input = {
          id: 'metadata-band',
          format,
          storage: {kind: 'buffer', values: view(format, 4)},
          noDataValue
        } as GPURasterSpectralFilterProps['input'];
        expect(
          () => new GPURasterSpectralFilter({...props, input}),
          `${format} nodata=${noDataValue}`
        ).toThrow(/^spectral-test input\.noDataValue must be an integer in \[/);
      }
    }
  } finally {
    device.destroy();
  }
});

it('accepts float32 metadata limits, NaN float nodata and exact integer nodata limits', () => {
  const {device, graph, props, view} = createFixture(2, 2);
  const maximumFloat32 = 3.4028234663852886e38;
  const inputs: GPURasterSpectralFilterProps['input'][] = [
    {
      id: 'float-limits',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32', 4)},
      scale: maximumFloat32,
      offset: -maximumFloat32,
      noDataValue: NaN
    },
    {
      id: 'float-nodata',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32', 4)},
      scale: 0,
      offset: 0,
      noDataValue: maximumFloat32
    },
    {
      id: 'uint-nodata',
      format: 'uint32',
      storage: {kind: 'buffer', values: view('uint32', 4)},
      noDataValue: 4294967295
    },
    {
      id: 'sint-minimum',
      format: 'sint32',
      storage: {kind: 'buffer', values: view('sint32', 4)},
      noDataValue: -2147483648
    },
    {
      id: 'sint-maximum',
      format: 'sint32',
      storage: {kind: 'buffer', values: view('sint32', 4)},
      noDataValue: 2147483647
    }
  ];
  try {
    for (const input of inputs) {
      const contributor = new GPURasterSpectralFilter({...props, id: input.id, input});
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect(sources.size).toBeGreaterThan(0);
      for (const source of sources.values()) expect(auditWGSLModule(source)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('rejects absent scalar texture channels during construction', () => {
  const {device, graph, props} = createFixture(2, 2);
  try {
    const texture = graph.createTransientTexture({
      id: 'invalid-channel',
      format: 'r32float',
      width: 2,
      height: 2,
      usage: Texture.SAMPLE
    });
    const input: GPURasterSpectralFilterProps['input'] = {
      id: 'source',
      format: 'float32',
      storage: {kind: 'texture', view: graph.createTextureView(texture), channel: 1}
    };
    expect(() => new GPURasterSpectralFilter({...props, input})).toThrow(
      /^spectral-test input\.storage\.channel must be an integer in \[0, 0\]/
    );
  } finally {
    device.destroy();
  }
});

it('rejects source and validity formats that disagree with the raster contract at construction', () => {
  const {device, props, view} = createFixture(2, 2);
  try {
    const input = {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: view('uint32', 4)}
    } as unknown as GPURasterSpectralFilterProps['input'];
    expect(() => new GPURasterSpectralFilter({...props, input})).toThrow(
      /^spectral-test input\.storage\.values .*float32/
    );
    const invalidValidity = {
      ...props.input,
      validity: view('float32', 4)
    } as unknown as GPURasterSpectralFilterProps['input'];
    expect(() => new GPURasterSpectralFilter({...props, input: invalidValidity})).toThrow(
      /^spectral-test input\.validity .*uint32/
    );
  } finally {
    device.destroy();
  }
});

it('rejects unsupported texture layouts and scalar representations at construction', () => {
  const {device, graph, props} = createFixture(2, 2);
  try {
    const textures = [
      {
        texture: graph.createTransientTexture({
          id: 'mips',
          format: 'r32float',
          width: 2,
          height: 2,
          mipLevels: 2,
          usage: Texture.SAMPLE
        }),
        error: /^spectral-test input\.storage\.view\.mipLevelCount .*\[1, 1\]/
      },
      {
        texture: graph.createTransientTexture({
          id: 'array',
          format: 'r32float',
          width: 2,
          height: 2,
          dimension: '2d-array',
          depth: 2,
          usage: Texture.SAMPLE
        }),
        error: /^spectral-test input\.storage\.view\.dimension .*2d/
      },
      {
        texture: graph.createTransientTexture({
          id: 'samples',
          format: 'r32float',
          width: 2,
          height: 2,
          samples: 4,
          usage: Texture.SAMPLE
        }),
        error: /^spectral-test input\.storage\.view\.texture\.samples .*\[1, 1\]/
      },
      {
        texture: graph.createTransientTexture({
          id: 'uint-texture',
          format: 'r32uint',
          width: 2,
          height: 2,
          usage: Texture.SAMPLE
        }),
        error: /^spectral-test input\.storage\.view\.format .*r32float.*rgba32float/
      }
    ];
    for (const {texture, error} of textures) {
      const input = {
        id: 'source',
        format: 'float32',
        storage: {kind: 'texture', view: graph.createTextureView(texture)}
      } as unknown as GPURasterSpectralFilterProps['input'];
      expect(() => new GPURasterSpectralFilter({...props, input}), texture.id).toThrow(error);
    }
  } finally {
    device.destroy();
  }
});

it('accepts supported scalar and RGBA channels and a single selected mip and array layer', () => {
  const {device, graph, props} = createFixture(2, 2);
  const textureView = <Format extends 'r32float' | 'rgba32float' | 'rgba32uint' | 'rgba32sint'>(
    id: string,
    format: Format
  ) =>
    graph.createTextureView(
      graph.createTransientTexture({id, format, width: 2, height: 2, usage: Texture.SAMPLE})
    );
  const arrayTexture = graph.createTransientTexture({
    id: 'selected-layer',
    format: 'r32float',
    width: 4,
    height: 4,
    depth: 2,
    dimension: '2d-array',
    mipLevels: 2,
    usage: Texture.SAMPLE
  });
  const inputs: GPURasterSpectralFilterProps['input'][] = [
    {
      id: 'scalar-default',
      format: 'float32',
      storage: {kind: 'texture', view: textureView('scalar-default', 'r32float')}
    },
    {
      id: 'rgba-float',
      format: 'float32',
      storage: {kind: 'texture', view: textureView('rgba-float', 'rgba32float'), channel: 3}
    },
    {
      id: 'rgba-uint',
      format: 'uint32',
      storage: {kind: 'texture', view: textureView('rgba-uint', 'rgba32uint'), channel: 3}
    },
    {
      id: 'rgba-sint',
      format: 'sint32',
      storage: {kind: 'texture', view: textureView('rgba-sint', 'rgba32sint'), channel: 3}
    },
    {
      id: 'selected-range',
      format: 'float32',
      storage: {
        kind: 'texture',
        view: graph.createTextureView(arrayTexture, {
          dimension: '2d',
          baseMipLevel: 1,
          mipLevelCount: 1,
          baseArrayLayer: 1,
          arrayLayerCount: 1
        }),
        channel: 0
      }
    }
  ];
  try {
    for (const input of inputs) {
      const contributor = new GPURasterSpectralFilter({...props, id: input.id, input});
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect(sources.size).toBeGreaterThan(0);
      for (const source of sources.values()) expect(auditWGSLModule(source)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
