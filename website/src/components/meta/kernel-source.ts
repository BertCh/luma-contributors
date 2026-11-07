// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A kernel of the library, shown on the kernels page: GPUTerrainRGBDecode on a packed-buffer input,
// Terrarium encoding. The lines are the complete WGSL that createWGSLKernelNode hands to the GPU,
// captured from the shipped operation on real WebGPU (the module source, nothing abridged). Only
// the workgroup size and the two numbers derived from it change with the slider, and the tile is
// shown at 32 x 32 pixels with the default valid range of -11000 to 9000 m.

export type KernelLineTag =
  | 'setup'
  | 'shape'
  | 'index'
  | 'guard'
  | 'read'
  | 'math'
  | 'write'
  | 'plain';

export type KernelLine = {text: string; tag: KernelLineTag};

export const KERNEL_SOURCE_PATH =
  'modules/analysis/src/gpu-terrain/terrain-decode/gpu-terrain-rgb-decode.ts';

/** The WGSL lines for a given workgroup size. */
export function getKernelLines(workgroupSize: number): KernelLine[] {
  // The kernel returns early for workgroups beyond what a 32-bit linear index can address.
  const maximumLinearWorkgroupCount = Math.floor(0xffffffff / workgroupSize) + 1;
  return [
    {text: 'override INVOCATION_COUNT: u32;', tag: 'setup'},
    {text: 'override WGSL_KERNEL_DISPATCH_X: u32;', tag: 'setup'},
    {text: 'override WGSL_KERNEL_DISPATCH_Y: u32;', tag: 'setup'},
    {text: 'const inputWordsOffset: u32 = 0u;', tag: 'setup'},
    {text: '@group(0) @binding(0) var<storage, read> inputWords: array<u32>;', tag: 'setup'},
    {text: 'const outputValuesOffset: u32 = 0u;', tag: 'setup'},
    {
      text: '@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;',
      tag: 'setup'
    },
    {text: 'const outputValidityOffset: u32 = 0u;', tag: 'setup'},
    {
      text: '@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;',
      tag: 'setup'
    },
    {text: 'const WIDTH: u32 = 32u;', tag: 'setup'},
    {text: 'const HEIGHT: u32 = 32u;', tag: 'setup'},
    {
      text: 'fn isEncodedPixelValid(red: u32, green: u32, blue: u32, alpha: u32) -> bool {',
      tag: 'math'
    },
    {text: '  if (alpha == 0u) { return false; }', tag: 'math'},
    {text: '  ', tag: 'math'},
    {text: '  return true;', tag: 'math'},
    {text: '}', tag: 'math'},
    {
      text: '// Integer code N = R * 65536 + G * 256 + B < 2^24, then exactly one float operation sequence.',
      tag: 'plain'
    },
    {text: 'fn decodeHeight(red: u32, green: u32, blue: u32) -> f32 {', tag: 'math'},
    {text: '  let code = (red << 16u) | (green << 8u) | blue;', tag: 'math'},
    {text: '  return f32(code) * 0.00390625 - 32768.0;', tag: 'math'},
    {text: '}', tag: 'math'},
    {text: 'fn isHeightInRange(height: f32) -> bool {', tag: 'math'},
    {text: '  return height >= -11000.0 && height <= 9000.0;', tag: 'math'},
    {text: '}', tag: 'math'},
    {text: 'fn applySeaClamp(height: f32) -> f32 {', tag: 'math'},
    {text: '  ', tag: 'math'},
    {text: '  return height;', tag: 'math'},
    {text: '}', tag: 'math'},
    {text: '', tag: 'plain'},
    {text: `@compute @workgroup_size(${workgroupSize})`, tag: 'shape'},
    {text: 'fn main(', tag: 'plain'},
    {text: '  @builtin(local_invocation_index) localInvocationIndex: u32,', tag: 'index'},
    {text: '  @builtin(workgroup_id) workgroupId: vec3<u32>', tag: 'index'},
    {text: ') {', tag: 'plain'},
    {text: '  _ = &inputWords;', tag: 'plain'},
    {text: '  _ = &outputValues;', tag: 'plain'},
    {text: '  _ = &outputValidity;', tag: 'plain'},
    {text: '  _ = INVOCATION_COUNT;', tag: 'plain'},
    {text: '  let workgroupIndex =', tag: 'index'},
    {
      text: '    (workgroupId.z * WGSL_KERNEL_DISPATCH_Y + workgroupId.y) * WGSL_KERNEL_DISPATCH_X +',
      tag: 'index'
    },
    {text: '    workgroupId.x;', tag: 'index'},
    {text: `  if (workgroupIndex >= ${maximumLinearWorkgroupCount}u) { return; }`, tag: 'index'},
    {
      text: `  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;`,
      tag: 'index'
    },
    {text: '  if (index >= INVOCATION_COUNT) {', tag: 'guard'},
    {text: '    return;', tag: 'guard'},
    {text: '  }', tag: 'guard'},
    {text: '  let pixelIndex = index;', tag: 'math'},
    {text: '  let word = inputWords[inputWordsOffset + pixelIndex];', tag: 'read'},
    {text: '  let red = word & 0xffu;', tag: 'math'},
    {text: '  let green = (word >> 8u) & 0xffu;', tag: 'math'},
    {text: '  let blue = (word >> 16u) & 0xffu;', tag: 'math'},
    {text: '  let alpha = word >> 24u;', tag: 'math'},
    {text: '  var isValid = isEncodedPixelValid(red, green, blue, alpha);', tag: 'math'},
    {text: '  ', tag: 'plain'},
    {text: '  let height = decodeHeight(red, green, blue);', tag: 'math'},
    {text: '  isValid = isValid && isHeightInRange(height);', tag: 'math'},
    {
      text: '  outputValidity[outputValidityOffset + pixelIndex] = select(0u, 1u, isValid);',
      tag: 'write'
    },
    {text: '  outputValues[outputValuesOffset + pixelIndex] =', tag: 'write'},
    {
      text: '    select(bitcast<f32>(0x7fc00000u | (pixelIndex & 0u)), applySeaClamp(height), isValid);',
      tag: 'write'
    },
    {text: '}', tag: 'plain'}
  ];
}

/** Workgroup sizes offered by the slider. */
export const WORKGROUP_SIZES = [8, 16, 32, 64, 128, 256] as const;

/** One pixel per invocation: a 32 x 32 tile of a Terrarium elevation image. */
export const KERNEL_GRID = {columns: 32, rows: 32, invocationCount: 1000} as const;

/** What the kernel does for one workgroup of a given size. */
export function describeWorkgroup(workgroupSize: number, workgroupIndex: number) {
  const first = workgroupIndex * workgroupSize;
  const last = first + workgroupSize - 1;
  const lastActive = Math.min(last, KERNEL_GRID.invocationCount - 1);
  const guarded = Math.max(0, last - lastActive);
  return {first, last: lastActive, guarded, activeCount: Math.max(0, lastActive - first + 1)};
}
