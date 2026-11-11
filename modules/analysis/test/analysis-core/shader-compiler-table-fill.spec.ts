// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getAdapterLabel, getRawTestDevice, runRawComputeShader} from './shader-compiler-raw-webgpu';

// The reduced reproduction of the "table reads back as zeros" fault of the macOS test GPU
// (contributor contract section 11; dev-docs/research/math-correctness-review-2026-11-10/
// shader-compiler-findings.md has the measurements, the generated Metal source and the cause).
//
// The shape, in plain WebGPU with no library code: a local `array<u32, N>` (or `array<i32, N>`)
// that WGSL zero-initialises, then a counted loop with a constant trip count that stores a value
// to every entry, then a read at an index the compiler cannot resolve (an index or a loop bound
// from a uniform). For 33 to 96 entries, except 35, 39, ... 95, the stores of the loop are lost
// and the read returns the zero initialisation. The Metal source that Tint generates is right and
// so is the intermediate code of the Metal front end; the Apple GPU back end loses the stores
// when it unrolls the loop (`#pragma clang loop unroll(disable)` on the loop cures it in Metal).
// It does not depend on fast math.
//
// Every expectation here is the WGSL meaning of the shader (entry `e` holds `e + 1`), not a
// kernel formula. The test that pins the fault skips on an adapter that does not show it.

const SIZES = [16, 32, 33, 35, 36, 64, 96, 97, 128];

/** Sizes of {@link SIZES} at which the reduced shader loses its fill on the test GPU (measured 2026-11-11). */
const MEASURED_LOST_SIZES = [33, 36, 64, 96];

type TableVariant = {
  /** What is varied, for messages. */
  name: string;
  /** Module-scope declarations (a table declared there, a helper function). */
  getModuleScope?: (size: number) => string;
  /** Statements that declare `table` (unless it is at module scope) and fill it. */
  getFill: (size: number) => string;
  /** `u32` WGSL expression that is `entry + 1u` when `table[entry]` holds what the fill stored and `0u` when it holds zero. */
  read?: string;
};

const getCountedFill = (size: number, value = 'entry + 1u'): string =>
  `for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = ${value}; }`;

const getLiteral = (size: number): string =>
  `array<u32, ${size}>(${Array.from({length: size}, (_, entry) => `${entry + 1}u`).join(', ')})`;

/** The reduced shader: the whole table is read through a loop whose bound comes from a uniform. */
const REDUCED: TableVariant = {
  name: 'function-scope array<u32, N>, counted fill',
  getFill: size => `var table: array<u32, ${size}>;
  ${getCountedFill(size)}`
};

function getTableShader(variant: TableVariant, size: number): string {
  return `struct Parameters { count: u32, index: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
${variant.getModuleScope?.(size) ?? ''}
@compute @workgroup_size(1)
fn main() {
  ${variant.getFill(size)}
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = ${variant.read ?? 'table[entry]'}; }
}`;
}

/** Runs one variant at one size and returns the entries whose stored value was lost (read as zero). */
async function getLostEntries(
  device: GPUDevice,
  variant: TableVariant,
  size: number
): Promise<number[]> {
  const words = await runRawComputeShader(device, {
    code: getTableShader(variant, size),
    outputLength: size,
    uniformWords: [size, 0]
  });
  const lostEntries: number[] = [];
  words.forEach((word, entry) => {
    if (word === 0) {
      lostEntries.push(entry);
    } else {
      // An entry is either what the loop stored or the zero initialisation, never anything else.
      expect(word, `${variant.name}, ${size} entries, entry ${entry}`).toBe(entry + 1);
    }
  });
  return lostEntries;
}

/** Sizes of {@link SIZES} at which a variant loses any entry. */
async function getLostSizes(device: GPUDevice, variant: TableVariant): Promise<number[]> {
  const lostSizes: number[] = [];
  for (const size of SIZES) {
    if ((await getLostEntries(device, variant, size)).length > 0) {
      lostSizes.push(size);
    }
  }
  return lostSizes;
}

it('a counted fill of a zero-initialised u32 table is lost at 33 to 96 entries on the test GPU', async context => {
  const device = await getRawTestDevice();
  const lostEntriesBySize = new Map<number, number[]>();
  for (const size of SIZES) {
    lostEntriesBySize.set(size, await getLostEntries(device, REDUCED, size));
  }
  const lostSizes = SIZES.filter(size => (lostEntriesBySize.get(size) ?? []).length > 0);

  // The shader of the bug report: one read at an index from a uniform. 38 by the language.
  const [single] = await runRawComputeShader(device, {
    code: `struct Parameters { index: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@compute @workgroup_size(1)
fn main() {
  var table: array<u32, 64>;
  for (var entry = 0u; entry < 64u; entry++) { table[entry] = entry + 1u; }
  output[0] = table[parameters.index];
}`,
    outputLength: 1,
    uniformWords: [37]
  });

  if (lostSizes.length === 0) {
    expect(single).toBe(38);
    // Not the test GPU, or a compiler that no longer has the fault: then contributor contract
    // section 11 and upstream-workarounds.md section 6 can drop the rule for this adapter.
    context.skip(`${getAdapterLabel(device)} keeps the fill at every size tried`);
  }
  // Measured on the test GPU (Apple M3 Pro, macOS 14.1, Chromium 151). A different list means
  // the compiler changed: measure again before trusting any table size.
  expect(lostSizes).toEqual(MEASURED_LOST_SIZES);
  expect(single).toBe(0);
  const getEntries = (count: number): number[] => Array.from({length: count}, (_, entry) => entry);
  // 33 entries: the first 32 are lost and the last one is right. 36, 64, 96: all are lost.
  expect(lostEntriesBySize.get(33)).toEqual(getEntries(32));
  expect(lostEntriesBySize.get(36)).toEqual(getEntries(36));
  expect(lostEntriesBySize.get(64)).toEqual(getEntries(64));
  expect(lostEntriesBySize.get(96)).toEqual(getEntries(96));
});

/** Shapes that lose the fill wherever the reduced shader does: none of these is a cure. */
const NOT_A_CURE: TableVariant[] = [
  {
    name: 'var<private> at module scope',
    getModuleScope: size => `var<private> table: array<u32, ${size}>;`,
    getFill: size => getCountedFill(size)
  },
  {
    name: 'array<i32, N>',
    getFill: size => `var table: array<i32, ${size}>;
  ${getCountedFill(size, 'i32(entry) + 1')}`,
    read: 'u32(table[entry])'
  },
  {
    name: 'constant fill (7)',
    getFill: size => `var table: array<u32, ${size}>;
  ${getCountedFill(size, '7u')}`,
    read: 'select(table[entry], entry + 1u, table[entry] == 7u)'
  },
  {
    name: 'sentinel fill (0xffffffff)',
    getFill: size => `var table: array<u32, ${size}>;
  ${getCountedFill(size, '0xffffffffu')}`,
    read: 'select(table[entry], entry + 1u, table[entry] == 0xffffffffu)'
  },
  {
    name: 'fill in a helper function through a pointer',
    getModuleScope: size => `fn fillTable(table: ptr<function, array<u32, ${size}>>) {
  for (var entry = 0u; entry < ${size}u; entry++) { (*table)[entry] = entry + 1u; }
}`,
    getFill: size => `var table: array<u32, ${size}>;
  fillTable(&table);`
  },
  {
    name: 'a store at a constant index after the fill',
    getFill: size => `var table: array<u32, ${size}>;
  ${getCountedFill(size)}
  table[3] = 4u;`
  }
];

it('module scope, the element sign, the stored value and a later constant store do not change it', async () => {
  const device = await getRawTestDevice();
  const lostSizes = await getLostSizes(device, REDUCED);
  for (const variant of NOT_A_CURE) {
    // On an adapter without the fault both lists are empty.
    expect(await getLostSizes(device, variant), variant.name).toEqual(lostSizes);
  }
});

/** Shapes that read right at every size on the test GPU, and must on any adapter. */
const KEEPS_THE_FILL: TableVariant[] = [
  {
    name: 'literal initializer',
    getFill: size => `var table = ${getLiteral(size)};`
  },
  {
    name: 'module-scope const',
    getModuleScope: size => `const table = ${getLiteral(size)};`,
    getFill: () => ''
  },
  {
    name: 'fill loop bounded by a uniform',
    getFill: size => `var table: array<u32, ${size}>;
  for (var entry = 0u; entry < parameters.count; entry++) { table[entry] = entry + 1u; }`
  },
  {
    name: 'a store at an index from a uniform after the fill',
    getFill: size => `var table: array<u32, ${size}>;
  ${getCountedFill(size)}
  table[parameters.index] = parameters.index + 1u;`
  },
  {
    name: 'array<f32, N>',
    getFill: size => `var table: array<f32, ${size}>;
  ${getCountedFill(size, 'f32(entry + 1u)')}`,
    read: 'u32(table[entry])'
  },
  {
    name: 'array<vec2<u32>, N>',
    getFill: size => `var table: array<vec2<u32>, ${size}>;
  ${getCountedFill(size, 'vec2<u32>(entry + 1u, entry + 1u)')}`,
    read: 'table[entry].y'
  }
];

it('a literal initializer, a fill bounded at run time, a later run-time store and float entries keep the fill', async () => {
  const device = await getRawTestDevice();
  for (const variant of KEEPS_THE_FILL) {
    expect(await getLostSizes(device, variant), variant.name).toEqual([]);
  }
});
