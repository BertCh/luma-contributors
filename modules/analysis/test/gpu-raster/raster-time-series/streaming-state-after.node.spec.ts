// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPURasterStreamingAccumulatorParameterValues,
  getGPURasterFixedPointValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {getGPURasterClimateRunAccumulatorParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator';
import {getGPURasterStandardizedIndexCalibrationParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {getGPURasterStandardizedIndexParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';

const packers = [
  {
    name: 'streaming',
    length: 5,
    activeWord: 1,
    pack: (activeSliceCount: number, target: Float32Array) =>
      getGPURasterStreamingAccumulatorParameterValues(
        {reset: false, maximumValueMagnitude: 4, activeSliceCount},
        2,
        target
      )
  },
  {
    name: 'climate',
    length: 4,
    activeWord: 1,
    pack: (activeSliceCount: number, target: Float32Array) =>
      getGPURasterClimateRunAccumulatorParameterValues(
        {reset: false, threshold: 0, maximumIncrement: 4, activeSliceCount},
        2,
        target
      )
  },
  {
    name: 'calibration',
    length: 7,
    activeWord: 1,
    pack: (activeSliceCount: number, target: Float32Array) =>
      getGPURasterStandardizedIndexCalibrationParameterValues(
        {reset: false, maximumValueMagnitude: 4, activeSliceCount},
        2,
        target
      )
  },
  {
    name: 'evaluation',
    length: 2,
    activeWord: 0,
    pack: (activeSliceCount: number, target: Float32Array) =>
      getGPURasterStandardizedIndexParameterValues({activeSliceCount}, 2, target)
  }
];
for (const {name, length, activeWord, pack} of packers) {
  describe(name, () => {
    it('rejects empty/short targets before writing', () => {
      for (const size of [0, length - 1]) {
        const target = new Float32Array(size).fill(77);
        expect(() => pack(1, target)).toThrow();
        expect([...target]).toEqual(Array(size).fill(77));
      }
    });
    it('accepts exact/padded targets and active zero', () => {
      for (const size of [length, length + 2]) {
        const target = new Float32Array(size).fill(77);
        expect(pack(0, target)).toBe(target);
        expect(target[activeWord]).toBe(0);
        expect([...target.slice(length)]).toEqual(Array(size - length).fill(77));
      }
    });
    it('rejects negative, fractional and excessive counts without changing bytes', () => {
      for (const count of [-1, 0.5, 3, NaN]) {
        const target = new Float32Array(length + 2).fill(77);
        expect(() => pack(count, target)).toThrow();
        expect([...target]).toEqual(Array(length + 2).fill(77));
      }
    });
  });
}
it('rejects a late bad stream bound without changing bytes', () => {
  const target = new Float32Array(5).fill(77);
  expect(() =>
    getGPURasterStreamingAccumulatorParameterValues(
      {reset: true, maximumValueMagnitude: 1, maximumStreamSliceCount: -1},
      2,
      target
    )
  ).toThrow();
  expect([...target]).toEqual([77, 77, 77, 77, 77]);
});
it('rejects nonpositive headroom for nonempty streams', () => {
  expect(() =>
    getGPURasterStreamingAccumulatorParameterValues(
      {reset: false, maximumValueMagnitude: 1, maximumStreamSliceCount: 0},
      1
    )
  ).toThrow();
});
it('decodes only bounded bytes and whole triples', () => {
  // 96-bit integer 1 at exponent 0 is exactly one; surrounding words are not part of the read.
  const words = Uint32Array.of(99, 1, 0, 0, 88);
  const bytes = new Uint8Array(words.buffer, 4, 12);
  expect([...getGPURasterFixedPointValues(bytes, 0)]).toEqual([1]);
  expect(() => getGPURasterFixedPointValues(new Uint8Array(11), 0)).toThrow();
  expect(() => getGPURasterFixedPointValues([1, 0, 0, 2], 0)).toThrow();
});

it('validates positive finite float32 bounds and preserves the previous packet on late errors', () => {
  for (const maximum of [-1, 0, NaN, Infinity, 1e39, 1e-46]) {
    for (const [length, pack] of [
      [
        5,
        (target: Float32Array) =>
          getGPURasterStreamingAccumulatorParameterValues(
            {reset: true, maximumValueMagnitude: maximum},
            2,
            target
          )
      ],
      [
        4,
        (target: Float32Array) =>
          getGPURasterClimateRunAccumulatorParameterValues(
            {reset: true, threshold: 0, maximumIncrement: maximum},
            2,
            target
          )
      ],
      [
        7,
        (target: Float32Array) =>
          getGPURasterStandardizedIndexCalibrationParameterValues(
            {reset: true, maximumValueMagnitude: maximum},
            2,
            target
          )
      ]
    ] as const) {
      const target = new Float32Array(length + 2).fill(77);
      expect(() => pack(target)).toThrow();
      expect([...target]).toEqual(Array(length + 2).fill(77));
    }
  }
  for (const maximumStreamSliceCount of [-1, 0, 0.5, 2 ** 32]) {
    for (const [length, pack] of [
      [
        4,
        (target: Float32Array) =>
          getGPURasterClimateRunAccumulatorParameterValues(
            {reset: true, threshold: 0, maximumIncrement: 1, maximumStreamSliceCount},
            2,
            target
          )
      ],
      [
        7,
        (target: Float32Array) =>
          getGPURasterStandardizedIndexCalibrationParameterValues(
            {reset: true, maximumValueMagnitude: 1, maximumStreamSliceCount},
            2,
            target
          )
      ]
    ] as const) {
      const target = new Float32Array(length).fill(77);
      expect(() => pack(target)).toThrow();
      expect([...target]).toEqual(Array(length).fill(77));
    }
  }
});

import {
  createTransientView,
  GPUCommandGraph,
  GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  decodeGPURasterStreamingAccumulatorStatus
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {
  GPURasterClimateRunAccumulator,
  createGPURasterClimateRunAccumulatorEncodingPlan
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator';
import {
  GPURasterStandardizedIndexCalibration,
  createGPURasterStandardizedIndexCalibrationEncodingPlan,
  resolveGPURasterStandardizedIndexZeroAtom
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {GPURasterStandardizedIndex} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {GPURasterRollingMean} from '../../../src/raster-analysis/raster-time-series/gpu-raster-rolling-mean';

it('captures immutable units for every retained-state model and refuses incompatible frame edits', () => {
  const settings = {maximumValueMagnitude: 1, maximumStreamSliceCount: 1024};
  const plan = createGPURasterStreamingAccumulatorEncodingPlan(settings, 1);
  settings.maximumValueMagnitude = 2;
  expect(Object.isFrozen(plan)).toBe(true);
  expect(plan.maximumValueMagnitude).toBe(1);
  const first = getGPURasterStreamingAccumulatorParameterValues(plan, {reset: true});
  const second = getGPURasterStreamingAccumulatorParameterValues(plan, {reset: false});
  expect([...first.slice(2)]).toEqual([...second.slice(2)]);
  // Hand encoding: two samples of 1 are the integer 2 * 2^(-unitExponent), not 3.
  const integer = 2n << BigInt(-plan.sumUnitExponent);
  const words = [
    Number(integer & 0xffffffffn),
    Number((integer >> 32n) & 0xffffffffn),
    Number(integer >> 64n)
  ];
  expect([...getGPURasterFixedPointValues(words, plan.sumUnitExponent)]).toEqual([2]);
  const target = new Float32Array(5).fill(77);
  expect(() =>
    getGPURasterStreamingAccumulatorParameterValues(
      plan,
      {...{reset: false, maximumValueMagnitude: 2}},
      target
    )
  ).toThrow(/reset/);
  expect([...target]).toEqual([77, 77, 77, 77, 77]);
  const climate = createGPURasterClimateRunAccumulatorEncodingPlan(
    {threshold: -1, maximumIncrement: 2},
    2
  );
  expect(Object.isFrozen(climate)).toBe(true);
  expect(
    getGPURasterClimateRunAccumulatorParameterValues(climate, {
      reset: false,
      activeSliceCount: 1
    })[2]
  ).toBe(-1);
  expect(() =>
    getGPURasterClimateRunAccumulatorParameterValues(climate, {...{reset: false, threshold: 0}})
  ).toThrow(/reset/);
  const calibration = createGPURasterStandardizedIndexCalibrationEncodingPlan(
    {location: -1, maximumValueMagnitude: 2},
    2
  );
  expect(Object.isFrozen(calibration)).toBe(true);
  expect(
    getGPURasterStandardizedIndexCalibrationParameterValues(calibration, {reset: false})[2]
  ).toBe(-1);
  expect(() =>
    getGPURasterStandardizedIndexCalibrationParameterValues(calibration, {
      ...{reset: false, location: 0}
    })
  ).toThrow(/reset/);
});

it('rejects invalid physical chunk widths, including for rolling-mean packets', () => {
  for (const chunkSliceCount of [0, -1, 0.5, 257]) {
    expect(() =>
      getGPURasterStreamingAccumulatorParameterValues(
        {reset: true, maximumValueMagnitude: 1},
        chunkSliceCount
      )
    ).toThrow();
    expect(() =>
      getGPURasterClimateRunAccumulatorParameterValues(
        {reset: true, threshold: 0, maximumIncrement: 1},
        chunkSliceCount
      )
    ).toThrow();
    expect(() =>
      getGPURasterStandardizedIndexCalibrationParameterValues(
        {reset: true, maximumValueMagnitude: 1},
        chunkSliceCount
      )
    ).toThrow();
  }
  for (const chunkSliceCount of [0, -1, 0.5, 2 ** 24 + 1]) {
    expect(() => getGPURasterStandardizedIndexParameterValues({}, chunkSliceCount)).toThrow();
  }
});

it('resolves zeroAtom and rejects the retired prop', () => {
  expect(resolveGPURasterStandardizedIndexZeroAtom('atom', {})).toBe(false);
  expect(resolveGPURasterStandardizedIndexZeroAtom('atom', {zeroAtom: true})).toBe(true);
  const retiredAtom = {zeroAtom: true, zeroInflated: true};
  expect(() => resolveGPURasterStandardizedIndexZeroAtom('atom', retiredAtom)).toThrow(
    /atom.*zeroInflated.*removed.*use zeroAtom/
  );
});

it('decodes bounded sticky overflow status and rejects missing certificates', () => {
  expect(decodeGPURasterStreamingAccumulatorStatus(Uint32Array.of(0))).toEqual({
    complete: true,
    overflow: false
  });
  const words = Uint32Array.of(99, 1, 88);
  expect(decodeGPURasterStreamingAccumulatorStatus(new Uint8Array(words.buffer, 4, 4))).toEqual({
    complete: false,
    overflow: true
  });
  expect(() => decodeGPURasterStreamingAccumulatorStatus([])).toThrow();
  expect(() => decodeGPURasterStreamingAccumulatorStatus([2])).toThrow();
});

it('rejects substituted formats on every port of the five streaming constructors before reading lengths', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = (format: 'float32' | 'uint32' | 'unorm8x4', length = 16) => {
      return createTransientView(graph, `port-${serial++}`, format, length);
    };
    const floats = () => view('float32') as GraphDataView<'float32'>;
    const integers = () => view('uint32') as GraphDataView<'uint32'>;
    const cases = [
      {
        make: (props: unknown) =>
          new GPURasterStreamingAccumulator(
            props as ConstructorParameters<typeof GPURasterStreamingAccumulator>[0]
          ),
        props: {
          id: 'stream',
          cellCount: 1,
          chunkSliceCount: 2,
          values: floats(),
          mask: integers(),
          parameters: floats(),
          slotIndices: integers(),
          state: {
            counts: integers(),
            sums: integers(),
            moments: integers(),
            extrema: floats(),
            status: integers()
          }
        }
      },
      {
        make: (props: unknown) =>
          new GPURasterClimateRunAccumulator(
            props as ConstructorParameters<typeof GPURasterClimateRunAccumulator>[0]
          ),
        props: {
          id: 'climate',
          cellCount: 1,
          chunkSliceCount: 2,
          values: floats(),
          mask: integers(),
          parameters: floats(),
          state: {runs: integers(), degreeDays: integers()}
        }
      },
      {
        make: (props: unknown) =>
          new GPURasterStandardizedIndexCalibration(
            props as ConstructorParameters<typeof GPURasterStandardizedIndexCalibration>[0]
          ),
        props: {
          id: 'calibrate',
          cellCount: 1,
          chunkSliceCount: 2,
          distribution: 'gamma',
          values: floats(),
          mask: integers(),
          parameters: floats(),
          slotIndices: integers(),
          state: {counts: integers(), moments: integers()},
          fit: floats()
        }
      },
      {
        make: (props: unknown) =>
          new GPURasterStandardizedIndex(
            props as ConstructorParameters<typeof GPURasterStandardizedIndex>[0]
          ),
        props: {
          id: 'evaluate',
          cellCount: 1,
          chunkSliceCount: 2,
          distribution: 'gamma',
          values: floats(),
          mask: integers(),
          parameters: floats(),
          slotIndices: integers(),
          fit: floats(),
          output: {index: floats(), status: integers()}
        }
      },
      {
        make: (props: unknown) =>
          new GPURasterRollingMean(props as ConstructorParameters<typeof GPURasterRollingMean>[0]),
        props: {
          id: 'rolling',
          cellCount: 1,
          chunkSliceCount: 2,
          windowLength: 1,
          values: floats(),
          mask: integers(),
          parameters: floats(),
          ring: floats(),
          seenCount: integers(),
          output: floats()
        }
      }
    ];
    for (const {make, props} of cases) {
      expect(() => make(props)).not.toThrow();
      const nodes = make(props).getCommandNodes(graph);
      for (const source of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(source.source)).not.toThrow();
        expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
      }
      const fields: Record<string, unknown> = {...props};
      for (const group of ['state', 'output']) {
        const nested = fields[group];
        if (nested && typeof nested === 'object' && !('format' in nested)) {
          delete fields[group];
          for (const [key, value] of Object.entries(nested)) fields[`${group}.${key}`] = value;
        }
      }
      for (const [path, value] of Object.entries(fields)) {
        if (!value || typeof value !== 'object' || !('format' in value)) continue;
        const malformed: Record<string, unknown> = {...props};
        const [group, name] = path.split('.');
        if (name)
          malformed[group] = {
            ...(malformed[group] as Record<string, unknown>),
            [name]: view('unorm8x4')
          };
        else malformed[path] = view('unorm8x4');
        expect(() => make(malformed), `${props.id} ${path}`).toThrow(
          new RegExp(`${props.id}.*${path}.*(float32|uint32).*unorm8x4`)
        );
      }
      // A vector claiming float32 still cannot contain a substituted physical chunk.
      const badChunk = view('unorm8x4', 2);
      const vector = new GraphVectorView({
        id: `vector-${serial++}`,
        name: 'bad-chunk',
        format: 'unorm8x4',
        length: 2,
        valueLength: 2,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: [badChunk]
      });
      Object.defineProperty(vector, 'format', {value: 'float32'});
      expect(() => make({...props, values: vector})).toThrow(/values.*(float32|unorm8x4)/);
    }
  } finally {
    device.destroy();
  }
});

import {decodeGPURasterStandardizedIndexStatus} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {createRasterTimeSeriesChunkCase} from './raster-time-series-chunks-b-cases';

it('decodes all standardized-index codes and rejects absent or unknown codes', () => {
  expect(decodeGPURasterStandardizedIndexStatus([0])).toEqual({
    complete: true,
    invalidInput: false,
    calibrationUnavailable: false,
    iterationLimit: false
  });
  expect(decodeGPURasterStandardizedIndexStatus([1])).toEqual({
    complete: false,
    invalidInput: true,
    calibrationUnavailable: false,
    iterationLimit: false
  });
  expect(decodeGPURasterStandardizedIndexStatus([2])).toEqual({
    complete: false,
    invalidInput: false,
    calibrationUnavailable: true,
    iterationLimit: false
  });
  expect(() => decodeGPURasterStandardizedIndexStatus([])).toThrow();
  expect(() => decodeGPURasterStandardizedIndexStatus([4])).toThrow();
});

it('accepts byte buffers, unaligned bounded reads and signed numeric-word triples', () => {
  const allocation = new Uint8Array(19).fill(77);
  const words = new DataView(allocation.buffer, 3, 12);
  words.setUint32(0, 1, true);
  words.setUint32(4, 0, true);
  words.setUint32(8, 0, true);
  expect([...getGPURasterFixedPointValues(allocation.subarray(3, 15), 0)]).toEqual([1]);
  expect([...getGPURasterFixedPointValues(Uint32Array.of(1, 0, 0).buffer, 0)]).toEqual([1]);
  expect([...getGPURasterFixedPointValues([0xffffffff, 0xffffffff, 0xffffffff], 0)]).toEqual([-1]);
  expect(() => getGPURasterFixedPointValues(new Uint8Array(16), 0)).toThrow(
    /multiple of 3.*received 4/
  );
});

it('preserves bytes after invalid location or threshold updates', () => {
  for (const [length, pack] of [
    [
      4,
      (target: Float32Array) =>
        getGPURasterClimateRunAccumulatorParameterValues(
          {reset: true, threshold: NaN, maximumIncrement: 1},
          2,
          target
        )
    ],
    [
      7,
      (target: Float32Array) =>
        getGPURasterStandardizedIndexCalibrationParameterValues(
          {reset: true, maximumValueMagnitude: 1, location: NaN},
          2,
          target
        )
    ],
    [
      2,
      (target: Float32Array) =>
        getGPURasterStandardizedIndexParameterValues({location: NaN}, 2, target)
    ]
  ] as const) {
    const target = new Float32Array(length).fill(77);
    expect(() => pack(target)).toThrow();
    expect([...target]).toEqual(Array(length).fill(77));
  }
});

it('keeps valid packed and unequal complete-cell vector combinations including short chunks', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    for (const chunked of [false, true]) {
      for (const name of [
        'streaming',
        'calibration',
        'standardized',
        'rolling',
        'rolling-short',
        'rolling-short-mask'
      ] as const) {
        const fixture = createRasterTimeSeriesChunkCase(name, {
          cellCount: 3,
          field(fieldName, format, entriesPerCell) {
            const chunks = (chunked ? [1, 2] : [3]).map(count =>
              createTransientView(graph, `${fieldName}-${serial++}`, format, count * entriesPerCell)
            );
            return chunked
              ? new GraphVectorView({
                  id: `valid-${serial++}`,
                  name: fieldName,
                  format,
                  length: 3 * entriesPerCell,
                  valueLength: 3 * entriesPerCell,
                  stride: 1,
                  byteStride: 4,
                  rowByteLength: 4,
                  data: chunks
                })
              : chunks[0];
          },
          shared(values) {
            return createTransientView(graph, `shared-${serial++}`, 'float32', values.length);
          },
          dates() {
            throw new Error('this fixture uses no date axis');
          }
        });
        const nodes = fixture.contributor.getCommandNodes(graph);
        expect(nodes).toHaveLength(fixture.contributor.estimatedCommandNodeCount);
        for (const source of getGPUCommandNodeKernelSources(nodes))
          expect(() => new WgslReflect(source.source)).not.toThrow();
      }
    }
  } finally {
    device.destroy();
  }
});
