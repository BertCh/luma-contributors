// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {
  GPURasterContrastEqualization,
  getGPURasterContrastEqualizationParameterValues,
  type GPURasterContrastEqualizationProps
} from '../../../src/raster-analysis/raster-contrast-equalization/index';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/index';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {RASTER_CONTRAST_EQUALIZATION_CASES} from '../../analysis-core/kernel-audit/cases-raster-contrast-equalization';
import {registerKernelAudit} from '../../analysis-core/kernel-audit/register-kernel-audit';
import reference from './reference.json';

registerKernelAudit('CART-03 raster contrast', RASTER_CONTRAST_EQUALIZATION_CASES);

function withProps(
  callback: (
    props: GPURasterContrastEqualizationProps,
    context: ReturnType<typeof createKernelAuditContext>
  ) => void
): void {
  const context = createKernelAuditContext('contrast-validation');
  const {view} = context;
  const grid = makeGPURasterMetadata({width: 7, height: 5});
  const props: GPURasterContrastEqualizationProps = {
    grid,
    input: makeGPURasterBand(view('float32', 35), grid, {validity: view('uint32', 35)}),
    parameters: view('float32', 8),
    output: {
      values: view('float32', 35),
      validity: view('uint32', 35),
      status: {approximation: view('uint32', 1)}
    }
  };
  try {
    callback(props, context);
  } finally {
    context.device.destroy();
  }
}

it('packs and validates float32 domains, target size and normalized clip limit', () => {
  expect(Array.from(getGPURasterContrastEqualizationParameterValues())).toEqual([
    0,
    1,
    0,
    1,
    Math.fround(0.01),
    0,
    0,
    0
  ]);
  const target = new Float32Array(10).fill(9);
  expect(
    getGPURasterContrastEqualizationParameterValues(
      {inputDomain: [-10, 20], referenceDomain: [2, 5], clipLimit: 0},
      target
    )
  ).toBe(target);
  expect(Array.from(target)).toEqual([-10, 20, 2, 5, 0, 0, 0, 0, 9, 9]);
  for (const domain of [
    [1, 1],
    [2, 1],
    [0, NaN],
    [0, Infinity],
    [-3e38, 3e38],
    [1, 1 + 1e-12]
  ] as const) {
    expect(() => getGPURasterContrastEqualizationParameterValues({inputDomain: domain})).toThrow(
      /inputDomain.*finite.*float32/
    );
    expect(() =>
      getGPURasterContrastEqualizationParameterValues({referenceDomain: domain})
    ).toThrow(/referenceDomain.*finite.*float32/);
  }
  for (const clipLimit of [-0.1, 1.1, NaN, Infinity]) {
    expect(() => getGPURasterContrastEqualizationParameterValues({clipLimit})).toThrow(/clipLimit/);
  }
  expect(() => getGPURasterContrastEqualizationParameterValues({}, new Float32Array(7))).toThrow(
    /target.*8/
  );
});

it('validates all named construction limits and the method/reference relationship', () => {
  withProps((props, {view}) => {
    expect(() => new GPURasterContrastEqualization(props)).not.toThrow();
    for (const binCount of [1, 257, 3.5]) {
      expect(() => new GPURasterContrastEqualization({...props, binCount})).toThrow(
        /raster-contrast-equalization binCount.*256/
      );
    }
    for (const tileWidth of [0, 257, 2.5]) {
      expect(() => new GPURasterContrastEqualization({...props, tileWidth})).toThrow(
        /tileWidth.*256/
      );
      expect(() => new GPURasterContrastEqualization({...props, tileHeight: tileWidth})).toThrow(
        /tileHeight.*256/
      );
    }
    expect(() => new GPURasterContrastEqualization({...props, mode: 'other' as 'clahe'})).toThrow(
      /mode/
    );
    expect(() => new GPURasterContrastEqualization({...props, mode: 'match-histograms'})).toThrow(
      /reference/
    );
    expect(
      () =>
        new GPURasterContrastEqualization({
          ...props,
          reference: {grid: props.grid, input: props.input}
        })
    ).toThrow(/reference/);
    expect(
      () => new GPURasterContrastEqualization({...props, parameters: view('float32', 7)})
    ).toThrow(/parameters.*8/);
    expect(
      () =>
        new GPURasterContrastEqualization({
          ...props,
          output: {...props.output, values: view('float32', 34)}
        })
    ).toThrow(/output.values.*35/);
    expect(
      () =>
        new GPURasterContrastEqualization({
          ...props,
          output: {...props.output, validity: view('float32', 35) as never}
        })
    ).toThrow(/output.validity/);
    expect(
      () =>
        new GPURasterContrastEqualization({
          ...props,
          output: {
            ...props.output,
            values:
              props.input.storage.kind === 'buffer'
                ? (props.input.storage.values as never)
                : props.output.values
          }
        })
    ).toThrow(/share buffers/);
  });
});

it('rejects oversized images, virtual reflection padding and histogram tables at construction', () => {
  withProps((props, {view}) => {
    const makeProps = (width: number, height: number): GPURasterContrastEqualizationProps => {
      const grid = makeGPURasterMetadata({width, height});
      return {
        ...props,
        grid,
        input: makeGPURasterBand(view('float32', width * height), grid),
        output: {
          ...props.output,
          values: view('float32', width * height),
          validity: view('uint32', width * height)
        }
      };
    };
    expect(() => new GPURasterContrastEqualization(makeProps(4097, 4096))).toThrow(
      /grid cell count.*16777216/
    );
    expect(
      () => new GPURasterContrastEqualization({...makeProps(4095, 4096), tileWidth: 255})
    ).toThrow(/reflected tile cell count.*16777216/);
    expect(
      () =>
        new GPURasterContrastEqualization({...makeProps(4096, 4096), tileWidth: 8, tileHeight: 8})
    ).toThrow(/histogramLength.*16777216/);
  });
});

it('checks device binding limits and graph ownership before declaring any nodes', () => {
  withProps((props, context) => {
    const other = createKernelAuditContext('contrast-other');
    try {
      expect(() =>
        new GPURasterContrastEqualization({
          ...props,
          parameters: other.view('float32', 8)
        }).getCommandNodes(context.graph)
      ).toThrow(/target graph/);
      const originalLimit = context.device.limits.maxStorageBufferBindingSize;
      Object.defineProperty(context.device, 'limits', {
        value: {...context.device.limits, maxStorageBufferBindingSize: 32},
        configurable: true
      });
      expect(() => new GPURasterContrastEqualization(props).getCommandNodes(context.graph)).toThrow(
        /needs.*bytes.*limit 32/
      );
      Object.defineProperty(context.device, 'limits', {
        value: {...context.device.limits, maxStorageBufferBindingSize: originalLimit},
        configurable: true
      });
    } finally {
      other.device.destroy();
    }
  });
});

it('keeps oracle coverage for clips, padding, singleton/constant inputs, ties and empty distributions', () => {
  expect(reference.oracle.skimage).toBe('0.24.0');
  expect(reference.cases.length).toBe(18);
  for (const scene of reference.cases) {
    expect(scene.values.length).toBe(scene.width * scene.height);
    expect(scene.expected.length).toBe(scene.values.length);
    expect(scene.expected.every(value => value === null || Number.isFinite(value))).toBe(true);
  }
  const clipped = reference.cases.find(scene => scene.name === 'odd-reflected')!;
  const unclipped = reference.cases.find(scene => scene.name === 'odd-reflected-unclipped')!;
  expect(clipped.expected).not.toEqual(unclipped.expected);
});

it('rejects incompatible source and reference bands at construction without declaring resources', () => {
  withProps((props, context) => {
    const {graph, view} = context;
    const grid = makeGPURasterMetadata({width: 2, height: 2});
    const makeBand = (format: 'float32' | 'uint32' | 'sint32' = 'float32'): GPURasterBand =>
      ({
        id: 'band',
        format,
        storage: {kind: 'buffer', values: view(format, 4)},
        validity: view('uint32', 4)
      }) as GPURasterBand;
    const input = makeBand();
    const referenceInput = makeBand();
    const matchingProps: GPURasterContrastEqualizationProps = {
      ...props,
      grid,
      input,
      mode: 'match-histograms',
      reference: {grid, input: referenceInput}
    };
    const malformedBands: {name: string; band: GPURasterBand; error: RegExp}[] = [
      ...[3, 5].map(length => ({
        name: `values length ${length}`,
        band: {
          ...makeBand(),
          storage: {kind: 'buffer', values: view('float32', length)}
        } as GPURasterBand,
        error: /values.*4/
      })),
      ...[3, 5].map(length => ({
        name: `validity length ${length}`,
        band: {...makeBand(), validity: view('uint32', length)},
        error: /validity.*4/
      })),
      {
        name: 'declared format mismatch',
        band: {...makeBand(), format: 'uint32'} as GPURasterBand,
        error: /format|uint32/
      },
      {
        name: 'unsupported format',
        band: {...makeBand(), format: 'float64'} as never,
        error: /format/
      },
      ...[Infinity, -Infinity, 1e39].map(noDataValue => ({
        name: 'invalid float nodata',
        band: {...makeBand(), noDataValue},
        error: /nodata|noDataValue/
      })),
      ...[-1, 2 ** 32, 0.5, NaN].map(noDataValue => ({
        name: 'invalid uint32 nodata',
        band: {...makeBand('uint32'), noDataValue},
        error: /nodata|noDataValue/
      })),
      ...[-(2 ** 31) - 1, 2 ** 31, 0.5, NaN].map(noDataValue => ({
        name: 'invalid sint32 nodata',
        band: {...makeBand('sint32'), noDataValue},
        error: /nodata|noDataValue/
      })),
      {name: 'missing band identifier', band: {...makeBand(), id: ''}, error: /identifier/},
      {
        name: 'texture format mismatch',
        band: {
          id: 'texture',
          format: 'float32',
          storage: {
            kind: 'texture',
            view: graph.createTextureView(context.texture('r32uint', 2, 2))
          }
        } as never,
        error: /format/
      },
      {
        name: 'texture absent channel',
        band: {
          id: 'texture',
          format: 'float32',
          storage: {
            kind: 'texture',
            view: graph.createTextureView(context.texture('r32float', 2, 2)),
            channel: 1
          }
        },
        error: /channel/
      }
    ];
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    const addNode = vi.spyOn(graph, 'addComputePass');
    try {
      for (const target of ['input', 'reference.input'] as const) {
        for (const {name, band, error} of malformedBands) {
          const invalidProps =
            target === 'input'
              ? {...matchingProps, input: band}
              : {...matchingProps, reference: {grid, input: band}};
          for (let attempt = 0; attempt < 2; attempt++) {
            expect(
              () => new GPURasterContrastEqualization(invalidProps),
              `${target}: ${name}`
            ).toThrow(error);
          }
        }
      }
      expect(createBuffer).not.toHaveBeenCalled();
      expect(addNode).not.toHaveBeenCalled();
      expect(() =>
        new GPURasterContrastEqualization(matchingProps).getCommandNodes(graph)
      ).not.toThrow();
      // Float NaN nodata and exact integer boundary sentinels remain valid on either band.
      for (const band of [
        {...makeBand(), noDataValue: NaN},
        {...makeBand('uint32'), noDataValue: 0xffffffff},
        {...makeBand('sint32'), noDataValue: -(2 ** 31)}
      ]) {
        expect(
          () =>
            new GPURasterContrastEqualization({
              ...matchingProps,
              input: band,
              reference: {grid, input: band}
            })
        ).not.toThrow();
      }
    } finally {
      createBuffer.mockRestore();
      addNode.mockRestore();
    }
  });
});

it('preflights the alignment prefix at the storage binding limit without mutating the graph', () => {
  withProps((props, context) => {
    const {graph} = context;
    const limit = context.device.limits.maxStorageBufferBindingSize;
    expect(limit).toBe(134217728);
    const buffer = context.buffer(limit + 4, 'unaligned-output');
    const values = graph.createDataView(buffer, {
      format: 'float32',
      length: limit / 4,
      byteOffset: 4
    });
    const contributor = new GPURasterContrastEqualization({
      ...props,
      output: {...props.output, values}
    });
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    const addNode = vi.spyOn(graph, 'addComputePass');
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        expect(() => contributor.getCommandNodes(graph)).toThrow(/134217732.*134217728/);
        expect(createBuffer).not.toHaveBeenCalled();
        expect(addNode).not.toHaveBeenCalled();
      }
      expect(() => new GPURasterContrastEqualization(props).getCommandNodes(graph)).not.toThrow();
    } finally {
      createBuffer.mockRestore();
      addNode.mockRestore();
    }
  });
});

it('accepts an exactly fitting aligned storage binding at a nonzero byte offset', () => {
  withProps((props, {graph, device, buffer}) => {
    const limit = device.limits.maxStorageBufferBindingSize;
    const values = graph.createDataView(buffer(limit + 256, 'aligned-output'), {
      format: 'float32',
      length: limit / 4,
      byteOffset: 256
    });
    expect(() =>
      new GPURasterContrastEqualization({
        ...props,
        output: {...props.output, values}
      }).getCommandNodes(graph)
    ).not.toThrow();
  });
});

it('preserves positive clip semantics when packing underflowing or subnormal fractions', () => {
  for (const clipLimit of [1e-50, Number.MIN_VALUE, 2 ** -149, 2 ** -127]) {
    const packed = getGPURasterContrastEqualizationParameterValues({clipLimit});
    // Minimum normal float32: neither host conversion nor GPU denormal flushing selects zero.
    expect(packed[4]).toBe(2 ** -126);
    expect(Math.max(1, Math.floor(packed[4] * 65536))).toBe(1);
  }
  expect(getGPURasterContrastEqualizationParameterValues({clipLimit: 0})[4]).toBe(0);
});

it('rejects float32 domain widths or endpoints that can flush to zero', () => {
  const minimumNormal = 2 ** -126;
  const minimumSubnormal = 2 ** -149;
  const domains: readonly (readonly [number, number])[] = [
    [minimumNormal, minimumNormal + minimumSubnormal],
    [-minimumNormal - minimumSubnormal, -minimumNormal],
    [0, minimumSubnormal],
    [-minimumSubnormal, 0],
    [-minimumNormal / 2, minimumNormal / 2],
    [minimumSubnormal, 1],
    [-1, -minimumSubnormal]
  ];
  for (const domain of domains) {
    for (const name of ['inputDomain', 'referenceDomain'] as const) {
      const target = new Float32Array(8).fill(9);
      expect(() =>
        getGPURasterContrastEqualizationParameterValues({[name]: domain}, target)
      ).toThrow(new RegExp(`${name}.*normal float32`));
      expect(Array.from(target)).toEqual(new Array(8).fill(9));
    }
  }
  // Zero is an allowed endpoint; the smallest normal width is supported on either side.
  expect(
    Array.from(
      getGPURasterContrastEqualizationParameterValues({inputDomain: [0, minimumNormal]})
    ).slice(0, 2)
  ).toEqual([0, minimumNormal]);
  expect(
    Array.from(
      getGPURasterContrastEqualizationParameterValues({referenceDomain: [-minimumNormal, 0]})
    ).slice(2, 4)
  ).toEqual([-minimumNormal, 0]);
});
