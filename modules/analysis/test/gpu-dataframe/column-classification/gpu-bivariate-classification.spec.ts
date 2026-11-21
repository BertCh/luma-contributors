import {selectBivariateOutput} from './select-classification-outputs.test-utils';
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUBivariateClassification,
  type GPUBivariateClassificationProps
} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import {
  getGPUBivariateClassificationParameterValues,
  type GPUBivariateClassificationSettings
} from '../../../src/dataframe-analysis/column-classification/bivariate-classification-parameters';
import {packGPUColor} from '../../../src/dataframe-analysis/column-classification/color-scale-parameters';
import {computeBivariateClassificationOnCPU} from './bivariate-classification-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  BIVARIATE_OPTIONAL_FIXTURE,
  CLASSIFICATION_OUTPUT_SUBSETS
} from './optional-output-fixtures';

const NONE = 0xffffffff;
const NO_DATA = 0x0a0b0c0d;

for (const includeAlpha of [false, true]) {
  it.each(
    CLASSIFICATION_OUTPUT_SUBSETS
  )(`GPUBivariateClassification optional outputs match hand-derived literals (alpha=${includeAlpha}): %j`, async subset => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const expected = BIVARIATE_OPTIONAL_FIXTURE;
    const fixture = createFixture(
      device,
      {
        valuesX: Float32Array.from(expected.valuesX),
        valuesY: Float32Array.from(expected.valuesY),
        mask: Uint32Array.from(expected.mask),
        alphaValues: includeAlpha ? Float32Array.from(expected.alphaValues) : undefined,
        maximumClassCount: 3,
        mutateOutputSelection: true
      },
      {colors: subset.colors, classIds: subset.classes, classCounts: subset.counts}
    );
    try {
      for (let frame = 0; frame < 2; frame++) {
        fixture.buffers.classIds.write(new Uint32Array(expected.valuesX.length).fill(0xabcdef));
        fixture.buffers.colors.write(new Uint32Array(expected.valuesX.length).fill(0xabcdef));
        fixture.buffers.classCounts.write(new Uint32Array(expected.counts.length).fill(0xabcdef));
        const actual = await fixture.run({
          breaksX: expected.breaksX,
          breaksY: expected.breaksY,
          palette: expected.palette,
          options: {
            classCountX: 2,
            classCountY: 2,
            noDataColor: 7,
            valueByAlpha: includeAlpha ? {domain: [0, 1], minimumAlpha: 0.5} : undefined
          }
        });
        if (subset.colors)
          expect(Array.from(actual.colors)).toEqual(
            includeAlpha ? expected.fadedColors : expected.colors
          );
        if (subset.classes) expect(Array.from(actual.classIds)).toEqual(expected.classes);
        if (subset.counts) expect(Array.from(actual.classCounts)).toEqual(expected.counts);
      }
      expect(fixture.compileCount).toBe(1);
    } finally {
      fixture.destroy();
    }
  });
}

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

type Scene = {
  valuesX: Float32Array;
  valuesY: Float32Array;
  mask?: Uint32Array;
  alphaValues?: Float32Array;
  maximumClassCount: number;
  mutateOutputSelection?: boolean;
};

type Frame = {
  breaksX: number[];
  breaksY: number[];
  palette: number[];
  options: GPUBivariateClassificationSettings;
};

type Result = {classIds: Uint32Array; colors: Uint32Array; classCounts: Uint32Array};

type Fixture = {
  compileCount: number;
  /** Output and input buffers a case may poison or rewrite between runs. */
  buffers: {classCounts: Buffer; classIds: Buffer; colors: Buffer; mask?: Buffer};
  run(frame: Frame): Promise<Result>;
  destroy(): void;
};

/** Omitted fields retain the full-output fixture default. */
type FixtureOutputs = {colors?: boolean; classIds?: boolean; classCounts?: boolean};

function createFixture(device: Device, scene: Scene, wired: FixtureOutputs = {}): Fixture {
  const includeColors = wired.colors ?? true;
  const includeClassIds = wired.classIds ?? true;
  const includeClassCounts = wired.classCounts ?? true;
  const rows = scene.valuesX.length;
  const maximum = scene.maximumClassCount;
  const graph = new GPUCommandGraph(device, {id: 'bivariate-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'bivariate-parameters',
    format: 'float32',
    length: 8
  });
  const breaksXBuffer = track(createInputBuffer(device, new Float32Array(maximum + 1)));
  const breaksYBuffer = track(createInputBuffer(device, new Float32Array(maximum + 1)));
  const paletteBuffer = track(createInputBuffer(device, new Uint32Array(maximum * maximum)));
  const maskBuffer = scene.mask ? track(createInputBuffer(device, scene.mask)) : undefined;
  const outputs = {
    classIds: track(createOutputBuffer(device, rows)),
    colors: track(createOutputBuffer(device, rows)),
    classCounts: track(createOutputBuffer(device, maximum * maximum))
  };
  outputs.classIds.write(new Uint32Array(rows).fill(0xabcdef));
  outputs.colors.write(new Uint32Array(rows).fill(0xabcdef));
  outputs.classCounts.write(new Uint32Array(maximum * maximum).fill(0xabcdef));
  const contributorProps: GPUBivariateClassificationProps = {
    id: 'bivariate',
    valuesX: importGraphBuffer(
      graph,
      'x',
      track(createInputBuffer(device, scene.valuesX)),
      'float32',
      rows
    ),
    valuesY: importGraphBuffer(
      graph,
      'y',
      track(createInputBuffer(device, scene.valuesY)),
      'float32',
      rows
    ),
    mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
    breaksX: importGraphBuffer(graph, 'breaks-x', breaksXBuffer, 'float32', maximum + 1),
    breaksY: importGraphBuffer(graph, 'breaks-y', breaksYBuffer, 'float32', maximum + 1),
    palette: importGraphBuffer(graph, 'palette', paletteBuffer, 'uint32', maximum * maximum),
    alphaValues: scene.alphaValues
      ? importGraphBuffer(
          graph,
          'alpha',
          track(createInputBuffer(device, scene.alphaValues)),
          'float32',
          rows
        )
      : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    maximumClassCount: maximum,
    output: selectBivariateOutput({
      classIds: includeClassIds
        ? importGraphBuffer(graph, 'o-ids', outputs.classIds, 'uint32', rows)
        : undefined,
      colors: includeColors
        ? importGraphBuffer(graph, 'o-colors', outputs.colors, 'uint32', rows)
        : undefined,
      classCounts: includeClassCounts
        ? importGraphBuffer(graph, 'o-counts', outputs.classCounts, 'uint32', maximum * maximum)
        : undefined
    })
  };
  const contributor = new GPUBivariateClassification(contributorProps);
  // R4: removing the caller's count selection must not remove its GPU writers.
  const callerOutput: Partial<GPUBivariateClassificationProps['output']> = contributorProps.output;
  if (scene.mutateOutputSelection) delete callerOutput.classCounts;
  graph.add(contributor);
  const fixture: Fixture = {
    compileCount: 0,
    buffers: {...outputs, mask: maskBuffer},
    async run(frame) {
      const breaksX = new Float32Array(maximum + 1);
      breaksX.set(frame.breaksX);
      const breaksY = new Float32Array(maximum + 1);
      breaksY.set(frame.breaksY);
      const palette = new Uint32Array(maximum * maximum);
      palette.set(frame.palette);
      breaksXBuffer.write(breaksX);
      breaksYBuffer.write(breaksY);
      paletteBuffer.write(palette);
      parameterBuffer.write(getGPUBivariateClassificationParameterValues(frame.options));
      submitGraph(device, compiled, undefined);
      return {
        classIds: includeClassIds
          ? Uint32Array.from(await readUint32(outputs.classIds, rows))
          : new Uint32Array(0),
        colors: includeColors
          ? Uint32Array.from(await readUint32(outputs.colors, rows))
          : new Uint32Array(0),
        classCounts: includeClassCounts
          ? Uint32Array.from(await readUint32(outputs.classCounts, maximum * maximum))
          : new Uint32Array(0)
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
  const compiled = graph.compile();
  fixture.compileCount++;
  return fixture;
}

/** Compares a frame with the brute-force oracle; alpha may differ by 1 when value-by-alpha is on. */
function expectParity(actual: Result, scene: Scene, frame: Frame, label: string) {
  const palette = new Uint32Array(scene.maximumClassCount ** 2);
  palette.set(frame.palette);
  const expected = computeBivariateClassificationOnCPU({
    valuesX: scene.valuesX,
    valuesY: scene.valuesY,
    mask: scene.mask,
    breaksX: frame.breaksX,
    breaksY: frame.breaksY,
    classCountX: frame.options.classCountX,
    classCountY: frame.options.classCountY,
    maximumClassCount: scene.maximumClassCount,
    palette,
    alphaValues: scene.alphaValues,
    noDataColor: frame.options.noDataColor,
    valueByAlpha: frame.options.valueByAlpha
  });
  expect(Array.from(actual.classIds), `${label} classIds`).toEqual(Array.from(expected.classIds));
  expect(Array.from(actual.classCounts), `${label} counts`).toEqual(
    Array.from(expected.classCounts)
  );
  const fadeActive = Boolean(frame.options.valueByAlpha && scene.alphaValues);
  for (let row = 0; row < expected.colors.length; row++) {
    if (fadeActive) {
      expect(actual.colors[row] & 0x00ffffff, `${label} rgb row ${row}`).toBe(
        expected.colors[row] & 0x00ffffff
      );
      expect(
        Math.abs((actual.colors[row] >>> 24) - (expected.colors[row] >>> 24)),
        `${label} alpha row ${row}`
      ).toBeLessThanOrEqual(1);
    } else {
      expect(actual.colors[row], `${label} color row ${row}`).toBe(expected.colors[row]);
    }
  }
  return expected;
}

function createGridPalette(size: number): number[] {
  return Array.from({length: size * size}, (_, index) =>
    packGPUColor(40 + index * 11, 200 - index * 9, (index * 53) & 255, 220)
  );
}

function createScene(seed: number, rows: number, maximumClassCount: number): Scene {
  const random = createRandom(seed);
  const valuesX = new Float32Array(rows);
  const valuesY = new Float32Array(rows);
  const mask = new Uint32Array(rows);
  const alphaValues = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    // Integer-valued data on a coarse grid puts many values exactly on break edges.
    valuesX[row] = random() < 0.03 ? NaN : Math.floor(random() * 24) - 2;
    valuesY[row] = random() < 0.03 ? NaN : Math.floor(random() * 24) - 2;
    mask[row] = random() < 0.05 ? 0 : 1;
    alphaValues[row] = random() < 0.05 ? NaN : random() * 3 - 0.5;
  }
  valuesX[0] = Infinity;
  valuesY[0] = -Infinity;
  valuesX[1] = -0;
  valuesY[1] = 0;
  return {valuesX, valuesY, mask, alphaValues, maximumClassCount};
}

it('GPUBivariateClassification classifies 3x3 and 4x4 grids and changes frames without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(3, 3000, 4);
  const fixture = createFixture(device, scene);
  const palette9 = createGridPalette(3);
  const palette16 = createGridPalette(4);
  const frames: Frame[] = [
    {
      breaksX: [0, 5, 12, 20],
      breaksY: [0, 8, 16, 22],
      palette: palette9,
      options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
    },
    {
      breaksX: [-Infinity, 0, 6, 12, Infinity],
      breaksY: [-5, 4, 8, 16, 30],
      palette: palette16,
      options: {classCountX: 4, classCountY: 4, noDataColor: NO_DATA}
    },
    // Unequal axes use classY * classCountX + classX.
    {
      breaksX: [0, 10, 22],
      breaksY: [0, 3, 9, 12, 22],
      palette: palette16,
      options: {classCountX: 2, classCountY: 4, noDataColor: 0xff112233}
    },
    // Duplicate inner edges leave an empty class.
    {
      breaksX: [0, 7, 7, 20],
      breaksY: [0, 7, 7, 20],
      palette: palette9,
      options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
    },
    // Zero classes on an axis makes every row no-data.
    {
      breaksX: [0, 7, 14],
      breaksY: [0, 7, 14],
      palette: palette9,
      options: {classCountX: 0, classCountY: 2, noDataColor: NO_DATA}
    },
    {
      breaksX: [0, 5, 12, 20],
      breaksY: [0, 8, 16, 22],
      palette: palette9,
      options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
    }
  ];
  for (const [index, frame] of frames.entries()) {
    const actual = await fixture.run(frame);
    const expected = expectParity(actual, scene, frame, `frame ${index}`);
    if (index === 0) {
      // Row 0 is (+Inf, -Inf): last X class, first Y class. Row 1 is (-0, 0) on the first edge.
      expect(actual.classIds[0]).toBe(0 * 3 + 2);
      expect(actual.classIds[1]).toBe(0);
      const counted = Array.from(expected.classCounts).reduce((sum, count) => sum + count, 0);
      expect(counted).toBeGreaterThan(2000);
    }
    if (index === 4) {
      expect(actual.classIds.every(classId => classId === NONE)).toBe(true);
      expect(actual.colors.every(color => color === NO_DATA)).toBe(true);
    }
  }
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUBivariateClassification applies value-by-alpha per frame and treats NaN alpha as minimum', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(8, 2500, 3);
  // Hand-checkable rows: the first five use a class-0 palette entry with alpha 200.
  scene.valuesX.set([1, 1, 1, 1, 1]);
  scene.valuesY.set([1, 1, 1, 1, 1]);
  scene.mask!.set([1, 1, 1, 1, 1]);
  scene.alphaValues!.set([0, 1, 0.5, NaN, 2]);
  const fixture = createFixture(device, scene);
  const palette = createGridPalette(3).map(color => ((color & 0x00ffffff) | (200 << 24)) >>> 0);
  const base: Frame = {
    breaksX: [0, 5, 12, 20],
    breaksY: [0, 8, 16, 22],
    palette,
    options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
  };
  const off = await fixture.run(base);
  expectParity(off, scene, base, 'vba off');
  expect(Array.from(off.colors.slice(0, 5), color => color >>> 24)).toEqual([
    200, 200, 200, 200, 200
  ]);
  const on: Frame = {
    ...base,
    options: {...base.options, valueByAlpha: {domain: [0, 1], minimumAlpha: 0.25}}
  };
  const faded = await fixture.run(on);
  expectParity(faded, scene, on, 'vba on');
  // factor = 0.25 + 0.75 * clamp(a): alpha 0 -> 50, 1 -> 200, 0.5 -> 125, NaN -> 50, 2 -> 200.
  expect(Array.from(faded.colors.slice(0, 5), color => color >>> 24)).toEqual([
    50, 200, 125, 50, 200
  ]);
  // Class counts are unaffected by the fade.
  expect(Array.from(faded.classCounts)).toEqual(Array.from(off.classCounts));
  const other: Frame = {
    ...on,
    options: {...on.options, valueByAlpha: {domain: [-1, 2], minimumAlpha: 0.5}}
  };
  expectParity(await fixture.run(other), scene, other, 'vba other');
  // Back to off on the same graph.
  expectParity(await fixture.run(base), scene, base, 'vba off again');
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUBivariateClassification ignores value-by-alpha without an alphaValues view', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(4, 500, 3);
  delete scene.alphaValues;
  const fixture = createFixture(device, scene);
  const frame: Frame = {
    breaksX: [0, 5, 12, 20],
    breaksY: [0, 8, 16, 22],
    palette: createGridPalette(3),
    options: {
      classCountX: 3,
      classCountY: 3,
      valueByAlpha: {domain: [0, 1], minimumAlpha: 0}
    }
  };
  expectParity(await fixture.run(frame), scene, frame, 'no alpha view');
  fixture.destroy();
});

it('GPUBivariateClassification is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(21, 4000, 4);
  const fixture = createFixture(device, scene);
  const frame: Frame = {
    breaksX: [-Infinity, 0, 6, 12, Infinity],
    breaksY: [-5, 4, 8, 16, 30],
    palette: createGridPalette(4),
    options: {
      classCountX: 4,
      classCountY: 4,
      noDataColor: NO_DATA,
      valueByAlpha: {domain: [0, 2], minimumAlpha: 0.3}
    }
  };
  const first = await fixture.run(frame);
  const second = await fixture.run(frame);
  expect(Array.from(second.colors)).toEqual(Array.from(first.colors));
  expect(Array.from(second.classIds)).toEqual(Array.from(first.classIds));
  expect(Array.from(second.classCounts)).toEqual(Array.from(first.classCounts));
  fixture.destroy();
});

it('DPC-122: GPUBivariateClassification writes all seven output subsets with literal results on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // TOOL-114's hand-classified 2x2 scene. Axis tie 5 is upper; id=2*classY+classX.
  const scene: Scene = {
    valuesX: Float32Array.of(1, 6, 1, 6, 7, NaN, 2, 5, 6),
    valuesY: Float32Array.of(1, 1, 6, 6, 8, 1, 3, 5, 6),
    mask: Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 1, 0),
    maximumClassCount: 2
  };
  const frame: Frame = {
    breaksX: [0, 5, 10],
    breaksY: [0, 5, 10],
    palette: [0xff010203, 0xff112233, 0xff445566, 0xff778899],
    options: {classCountX: 2, classCountY: 2, noDataColor: NO_DATA}
  };
  const firstIds = [0, 1, 2, 3, 3, NONE, 0, 3, NONE];
  const firstColors = [
    0xff010203,
    0xff112233,
    0xff445566,
    0xff778899,
    0xff778899,
    NO_DATA,
    0xff010203,
    0xff778899,
    NO_DATA
  ];
  for (let subset = 1; subset < 8; subset++) {
    const wired = {
      colors: Boolean(subset & 1),
      classIds: Boolean(subset & 2),
      classCounts: Boolean(subset & 4)
    };
    const fixture = createFixture(device, scene, wired);
    const poison = () => {
      fixture.buffers.colors.write(new Uint32Array(9).fill(0xdeadbeef));
      fixture.buffers.classIds.write(new Uint32Array(9).fill(0xdeadbeef));
      fixture.buffers.classCounts.write(new Uint32Array(4).fill(0xdeadbeef));
    };
    const check = (result: Result, ids: number[], colors: number[], counts: number[]) => {
      expect(result.colors.length).toBe(wired.colors ? 9 : 0);
      expect(result.classIds.length).toBe(wired.classIds ? 9 : 0);
      expect(result.classCounts.length).toBe(wired.classCounts ? 4 : 0);
      if (wired.colors)
        expect(Array.from(result.colors), `subset ${subset} colors`).toEqual(colors);
      if (wired.classIds) expect(Array.from(result.classIds), `subset ${subset} ids`).toEqual(ids);
      if (wired.classCounts)
        expect(Array.from(result.classCounts), `subset ${subset} counts`).toEqual(counts);
    };
    try {
      poison();
      check(await fixture.run(frame), firstIds, firstColors, [2, 1, 1, 3]);
      // Only rows 2 and 7 remain selected; one active class per axis, three inactive count slots.
      fixture.buffers.mask!.write(Uint32Array.of(0, 0, 1, 0, 0, 0, 0, 1, 0));
      poison();
      check(
        await fixture.run({
          breaksX: [0, 10],
          breaksY: [0, 10],
          palette: [0xff010203],
          options: {classCountX: 1, classCountY: 1, noDataColor: NO_DATA}
        }),
        [NONE, NONE, 0, NONE, NONE, NONE, NONE, 0, NONE],
        [NO_DATA, NO_DATA, 0xff010203, NO_DATA, NO_DATA, NO_DATA, NO_DATA, 0xff010203, NO_DATA],
        [2, 0, 0, 0]
      );
      expect(fixture.compileCount).toBe(1);
    } finally {
      fixture.destroy();
    }
  }
});

it('DPC-110: GPUBivariateClassification agrees with producer counts for signed zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, {
    valuesX: Float32Array.of(-0, +0),
    valuesY: Float32Array.of(4, 4),
    maximumClassCount: 2
  });
  try {
    const result = await fixture.run({
      breaksX: [-1, +0, 1],
      breaksY: [3, 5],
      palette: [0xff010203, 0xff112233],
      options: {classCountX: 2, classCountY: 1}
    });
    expect(Array.from(result.classIds)).toEqual([1, 1]);
    expect(Array.from(result.classCounts)).toEqual([0, 2, 0, 0]);
    expect(Array.from(result.colors)).toEqual([0xff112233, 0xff112233]);
  } finally {
    fixture.destroy();
  }
});

it('GPUBivariateClassification with only a classCounts output counts a hand-classified 2x2 scene', async () => {
  // TOOL-114: neither colors nor classIds is wired, so the class ids live in a transient view.
  // Two classes per axis with the inner edge at 5: an axis class is 1 when the value is >= 5.
  // Class id = classY * 2 + classX. Rows: (1,1) 0, (6,1) 1, (1,6) 2, (6,6) 3, (7,8) 3, (NaN,1)
  // none, (2,3) 0, (5,5) 3 (an equal edge belongs to the upper class), (6,6) masked out.
  // Expected counts by hand: [2, 1, 1, 3].
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: Scene = {
    valuesX: Float32Array.from([1, 6, 1, 6, 7, NaN, 2, 5, 6]),
    valuesY: Float32Array.from([1, 1, 6, 6, 8, 1, 3, 5, 6]),
    mask: Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 0]),
    maximumClassCount: 2
  };
  const frame: Frame = {
    breaksX: [0, 5, 10],
    breaksY: [0, 5, 10],
    palette: createGridPalette(2),
    options: {classCountX: 2, classCountY: 2, noDataColor: NO_DATA}
  };
  const fixture = createFixture(device, scene, {colors: false, classIds: false});
  const poison = () => fixture.buffers.classCounts.write(new Uint32Array(4).fill(0xdeadbeef));
  poison();
  const counted = await fixture.run(frame);
  expect(Array.from(counted.classCounts)).toEqual([2, 1, 1, 3]);
  // Every row masked out on the same compiled graph: the counts are cleared, not accumulated.
  poison();
  fixture.buffers.mask!.write(new Uint32Array(scene.valuesX.length));
  const masked = await fixture.run(frame);
  expect(Array.from(masked.classCounts)).toEqual([0, 0, 0, 0]);
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUBivariateClassification without a colors output writes the same class ids and counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createScene(23, 4000, 4);
  const frame: Frame = {
    breaksX: [-Infinity, 0, 6, 12, Infinity],
    breaksY: [-5, 4, 8, 16, 30],
    palette: createGridPalette(4),
    options: {classCountX: 4, classCountY: 4, noDataColor: NO_DATA}
  };
  const full = createFixture(device, scene);
  const expected = await full.run(frame);
  full.destroy();
  const countsOnly = createFixture(device, scene, {colors: false});
  const actual = await countsOnly.run(frame);
  countsOnly.destroy();
  expect(Array.from(expected.classCounts).reduce((sum, count) => sum + count, 0)).toBeGreaterThan(
    0
  );
  expect(Array.from(actual.classIds)).toEqual(Array.from(expected.classIds));
  expect(Array.from(actual.classCounts)).toEqual(Array.from(expected.classCounts));
});
