// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {scaleThreshold} from 'd3-scale';
import {afterAll, expect, test, vi} from 'vitest';
import {
  createNativeStylePlan,
  type NativeStyleLayer,
  type NativeStylePlanContext,
  type NativeJenksThresholds
} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import {computeNativeJenksBreaks} from '../../../examples/deck/kepler-compute/src/native-jenks-breaks';
import {
  readNativeJenksThresholds,
  type NativeJenksReadProps
} from '../../../examples/deck/kepler-compute/src/native-jenks-reader';
import {getResidentStyleLegendRows} from '../../../examples/deck/kepler-compute/src/resident-style-legend';
import {getKeplerResidentStyleKey, type KeplerResidentStylePlan} from '../src/styles/style-plan';

vi.hoisted(() => vi.resetModules());
afterAll(() => {
  vi.doUnmock('@kepler.gl/components');
  vi.resetModules();
});

vi.mock('@kepler.gl/components', () => ({LayerColorLegendFactory: () => () => null}));

const require = createRequire(import.meta.url);
const {default: nativeJenks} = require(
  join(dirname(require.resolve('@kepler.gl/utils')), 'jenks.js')
) as {
  default(values: number[], classCount: number): number[];
};
const source = {datasetId: 'jenks', revision: 4};
const palette = ['#ff0000', '#00ff00', '#0000ff'];
const layer: NativeStyleLayer = {
  id: 'points',
  type: 'point',
  config: {
    dataId: source.datasetId,
    color: [0, 0, 0],
    colorField: {name: 'value'},
    colorScale: 'jenks',
    // Native resident table bridges intentionally publish an empty placeholder here.
    colorDomain: [],
    visConfig: {colorRange: {colors: palette}}
  }
};
const context: NativeStylePlanContext = {source, columns: {value: {format: 'float32'}}, zoom: 12};
function controls(
  values: readonly number[],
  overrides: Partial<NativeJenksThresholds> = {}
): NativeJenksThresholds {
  return {source, column: 'value', classCount: 3, values, ...overrides};
}
function translated(values: readonly number[], options = context): KeplerResidentStylePlan {
  const result = createNativeStylePlan(layer, {
    ...options,
    jenksThresholds: {fillColor: controls(values)}
  });
  if (!result.supported) throw new Error(result.reasons.join(' '));
  const channel = result.plan.channels.fillColor;
  if (channel?.kind !== 'resident') throw new Error('Expected resident color channel');
  return channel.plan;
}

test('pinned Jenks is seeded candidate search, not the existing exact Fisher-Jenks objective', () => {
  const values = Array.from({length: 20}, (_, index) => index);
  const breaks = nativeJenks(values, 5);
  expect(breaks).toEqual([4, 8, 11, 16]);
  const cost = (thresholds: number[]) => {
    let total = 0;
    for (let index = 0; index <= thresholds.length; index++) {
      const group = values.filter(
        value =>
          value >= (thresholds[index - 1] ?? -Infinity) && value < (thresholds[index] ?? Infinity)
      );
      const mean = group.reduce((sum, value) => sum + value, 0) / group.length;
      total += group.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    }
    return total;
  };
  expect(cost(breaks)).toBe(27);
  expect(cost([4, 8, 12, 16])).toBe(25);
});

test.each([
  [7, 7, 7],
  [1, 1, 2, 2],
  [0, 1, 2, 3, 10, 11, 12, 13],
  []
])('explicit pinned Jenks boundaries preserve ties, threshold equality and unused colors for %j', (...values: number[]) => {
  const boundaries = nativeJenks(values, 3);
  const plan = translated(boundaries);
  expect(plan.kind).toBe('color');
  if (plan.kind !== 'color') throw new Error('Expected color plan');
  expect(plan.scale).toBe('threshold');
  expect(plan.domain).toEqual({kind: 'fixed', values: boundaries});
  expect(plan.palette.length).toBe(boundaries.length + 1);
  const original = scaleThreshold<number, string>().domain(boundaries).range(palette);
  const resident = scaleThreshold<number, number>()
    .domain(plan.domain.kind === 'fixed' ? [...plan.domain.values] : [])
    .range([...plan.palette]);
  const colors = [0xff0000ff, 0xff00ff00, 0xffff0000];
  for (const value of [-100, ...values, ...boundaries, 100])
    expect(resident(value)).toBe(colors[palette.indexOf(original(value))]);
  const rows = getResidentStyleLegendRows({
    styleId: plan.id,
    revision: 1,
    source,
    scale: plan.scale,
    domain: boundaries,
    domainStatus: 'valid',
    finiteCount: null,
    palette: plan.palette,
    numericalPolicy: 'float32'
  });
  if (boundaries.length) {
    expect(rows[0].label).toBe(`< ${boundaries[0]}`);
    expect(rows.at(-1)?.label).toBe(`≥ ${boundaries.at(-1)}`);
  } else expect(rows).toEqual([{color: colors[0], label: 'All finite values'}]);
});

test('Jenks controls reject missing, stale and nonrepresentable domains before a resident plan publishes', () => {
  expect(createNativeStylePlan(layer, context)).toMatchObject({
    supported: false,
    reasons: [expect.stringContaining('explicit or host-derived thresholds')]
  });
  for (const changed of [
    controls([1, 2], {source: {...source, revision: 3}}),
    controls([1, 2], {column: 'other'}),
    controls([1, 2], {classCount: 2}),
    controls([1, 2], {selectionRevision: 2}),
    controls([1, 0]),
    controls([0.1, 2]),
    controls([NaN, 2]),
    controls([1, Infinity]),
    controls([1, 2, 3])
  ])
    expect(
      createNativeStylePlan(layer, {...context, jenksThresholds: {fillColor: changed}}).supported
    ).toBe(false);
  expect(
    createNativeStylePlan(layer, {
      ...context,
      domains: {fillColor: {kind: 'selection', revision: 3}},
      jenksThresholds: {fillColor: controls([1, 2], {selectionRevision: 2})}
    }).supported
  ).toBe(false);
  const result = createNativeStylePlan(layer, {
    ...context,
    domains: {fillColor: {kind: 'selection', revision: 3}},
    jenksThresholds: {fillColor: controls([1, 2], {selectionRevision: 3})}
  });
  expect(result).toMatchObject({
    supported: true,
    plan: {
      channels: {
        fillColor: {
          kind: 'resident',
          plan: {domain: {kind: 'fixed', values: [1, 2], selectionRevision: 3}}
        }
      }
    }
  });
});

test('Jenks enforces the declared bounded color capacity instead of silently truncating classes', () => {
  const colors = Array.from({length: 64}, () => '#000000');
  const result = createNativeStylePlan(
    {...layer, config: {...layer.config, visConfig: {colorRange: {colors}}}},
    {...context, jenksThresholds: {fillColor: controls([], {classCount: 64})}}
  );
  expect(result).toMatchObject({supported: false, reasons: [expect.stringContaining('1 to 63')]});
});

test('editing derived Jenks breaks as native custom breaks yields a new matching resident plan', () => {
  const derived = translated([4, 9]);
  const edited = createNativeStylePlan(
    {
      ...layer,
      config: {
        ...layer.config,
        colorScale: 'custom',
        visConfig: {
          colorRange: {
            colors: [],
            colorMap: [
              [4, palette[0]],
              [10, palette[1]],
              [null, palette[2]]
            ]
          }
        }
      }
    },
    context
  );
  if (!edited.supported) throw new Error(edited.reasons.join(' '));
  const channel = edited.plan.channels.fillColor;
  if (channel?.kind !== 'resident' || channel.plan.kind !== 'color')
    throw new Error('Expected resident color channel');
  expect(channel.plan.domain).toMatchObject({kind: 'fixed', values: [4, 4, 10, 10]});
  expect(getKeplerResidentStyleKey(channel.plan)).not.toBe(getKeplerResidentStyleKey(derived));
  expect(channel.plan.palette).toHaveLength(derived.kind === 'color' ? derived.palette.length : 0);
});

function makeVector<T extends 'float32' | 'uint32'>(
  chunks: ArrayLike<number>[],
  Constructor: Float32ArrayConstructor | Uint32ArrayConstructor
) {
  return {
    length: chunks.reduce((total, chunk) => total + chunk.length, 0),
    data: chunks.map(chunk => {
      const array = Constructor.from(chunk as ArrayLike<number>);
      return {
        byteOffset: 0,
        byteStride: 4,
        length: array.length,
        buffer: {readAsync: async () => new Uint8Array(array.buffer.slice(0))}
      };
    })
  } as unknown as NativeJenksReadProps['values'] & {__format?: T};
}
function readProps(
  chunks: number[][],
  overrides: Partial<NativeJenksReadProps> = {}
): NativeJenksReadProps {
  return {
    source,
    column: 'value',
    classCount: 3,
    values: makeVector(chunks, Float32Array),
    reserveMemory: () => ({release() {}}) as never,
    computeBreaks: async (values, classCount) =>
      computeNativeJenksBreaks({values, classCount}, nativeJenks).breaks as number[],
    ...overrides
  };
}

test('Jenks respects validity and mask and equals native across chunks', async () => {
  const values = Array.from({length: 1000}, (_, index) => (index * 37) % 101);
  const validity = values.map((_, index) => (index % 10 ? 1 : 0));
  const mask = values.map((_, index) => (index % 7 ? 1 : 0));
  const expected = nativeJenks(
    values.filter((_, index) => validity[index] && mask[index]),
    3
  );
  const result = await readNativeJenksThresholds(
    readProps([values.slice(0, 400), values.slice(400)], {
      validity: makeVector([validity.slice(0, 400), validity.slice(400)], Uint32Array),
      mask: makeVector([mask.slice(0, 400), mask.slice(400)], Uint32Array)
    })
  );
  expect(result.values).toEqual(expected);
});

test('Jenks has no row cap and equals native above 262,144 rows, masked and with non-finite values', async () => {
  const total = 262144 + 50000;
  const values = Array.from({length: total}, (_, index) => {
    const cluster = index % 3;
    return Math.fround(cluster * 1000 + ((index * 7919) % 1000) / 10);
  });
  values[5] = NaN;
  values[6] = Infinity;
  const mask = values.map((_, index) => (index % 11 ? 1 : 0));
  const half = Math.floor(total / 2);
  const reserved: number[] = [];
  const chunks = [values.slice(0, half), values.slice(half)];
  const maskChunks = [mask.slice(0, half), mask.slice(half)];
  for (const masked of [false, true]) {
    const result = await readNativeJenksThresholds(
      readProps(chunks, {
        ...(masked ? {mask: makeVector(maskChunks, Uint32Array)} : {}),
        reserveMemory: byteLength => {
          reserved.push(byteLength);
          return {release() {}} as never;
        }
      })
    );
    const expected = nativeJenks(
      values.filter((value, index) => Number.isFinite(value) && (!masked || mask[index])),
      3
    );
    expect(result.values).toEqual(expected);
  }
  // The reservation covers the full list; there is no refusal other than the registry's.
  expect(reserved[0]).toBeGreaterThanOrEqual(total * 4);
});

test('Jenks releases its reservation and propagates a registry refusal or worker failure', async () => {
  const release = vi.fn();
  await expect(
    readNativeJenksThresholds(
      readProps([[1, 2, 3]], {
        reserveMemory: () => ({release}) as never,
        computeBreaks: async () => {
          throw new Error('worker failed');
        }
      })
    )
  ).rejects.toThrow('worker failed');
  expect(release).toHaveBeenCalledTimes(1);
  await expect(
    readNativeJenksThresholds(
      readProps([[1, 2, 3]], {
        reserveMemory: () => {
          throw new Error('over budget');
        }
      })
    )
  ).rejects.toThrow('over budget');
});

test('pure Jenks core wraps native results and errors for the worker', () => {
  expect(
    computeNativeJenksBreaks(
      {values: new Float32Array([0, 1, 2, 3, 10, 11, 12, 13]), classCount: 3},
      nativeJenks
    )
  ).toEqual({breaks: nativeJenks([0, 1, 2, 3, 10, 11, 12, 13], 3)});
  expect(
    computeNativeJenksBreaks({values: new Float32Array(1), classCount: 2}, () => {
      throw new Error('boom');
    })
  ).toEqual({error: 'boom'});
});
