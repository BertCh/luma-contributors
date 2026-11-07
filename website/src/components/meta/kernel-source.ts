// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A kernel of the library, shown on the kernels page: GPUTerrainRGBDecode (Terrarium encoding,
// packed-buffer input, default options), simplified from what createWGSLKernelNode emits. Unused
// option branches, offset constants, `_ = &binding;` keep-alive lines and the extra workgroup-count
// early return are left out, and the validity helpers are inlined into one `isValid` line.

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
  return [
    {text: 'override INVOCATION_COUNT: u32;', tag: 'setup'},
    {text: 'override WGSL_KERNEL_DISPATCH_X: u32;', tag: 'setup'},
    {text: 'override WGSL_KERNEL_DISPATCH_Y: u32;', tag: 'setup'},
    {text: '@group(0) @binding(0) var<storage, read> inputWords: array<u32>;', tag: 'setup'},
    {
      text: '@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;',
      tag: 'setup'
    },
    {
      text: '@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;',
      tag: 'setup'
    },
    {text: '', tag: 'plain'},
    {text: 'fn decodeHeight(red: u32, green: u32, blue: u32) -> f32 {', tag: 'math'},
    {text: '  let code = (red << 16u) | (green << 8u) | blue;', tag: 'math'},
    {text: '  return f32(code) * 0.00390625 - 32768.0;', tag: 'math'},
    {text: '}', tag: 'math'},
    {text: '', tag: 'plain'},
    {text: `@compute @workgroup_size(${workgroupSize})`, tag: 'shape'},
    {text: 'fn main(', tag: 'plain'},
    {text: '  @builtin(local_invocation_index) localInvocationIndex: u32,', tag: 'index'},
    {text: '  @builtin(workgroup_id) workgroupId: vec3<u32>', tag: 'index'},
    {text: ') {', tag: 'plain'},
    {
      text: '  let workgroupIndex = (workgroupId.z * WGSL_KERNEL_DISPATCH_Y + workgroupId.y) *',
      tag: 'index'
    },
    {text: '    WGSL_KERNEL_DISPATCH_X + workgroupId.x;', tag: 'index'},
    {
      text: `  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;`,
      tag: 'index'
    },
    {text: '  if (index >= INVOCATION_COUNT) { return; }', tag: 'guard'},
    {text: '  let word = inputWords[index];', tag: 'read'},
    {text: '  let red = word & 0xffu;', tag: 'math'},
    {text: '  let green = (word >> 8u) & 0xffu;', tag: 'math'},
    {text: '  let blue = (word >> 16u) & 0xffu;', tag: 'math'},
    {text: '  let height = decodeHeight(red, green, blue);', tag: 'math'},
    {
      text: '  let isValid = (word >> 24u) != 0u && height >= -11000.0 && height <= 9000.0;',
      tag: 'math'
    },
    {text: '  outputValidity[index] = select(0u, 1u, isValid);', tag: 'write'},
    {
      text: '  outputValues[index] = select(bitcast<f32>(0x7fc00000u), height, isValid);',
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
