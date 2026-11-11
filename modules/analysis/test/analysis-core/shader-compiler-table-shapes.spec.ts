// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GET_NAN_WGSL} from '../../src/analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createOptimizationHarness} from '../gpu-network/network-optimization/optimization-test-harness';
import {requireTestDevice} from '../utils/require-test-device';
import {getAdapterLabel, getRawTestDevice, runRawComputeShader} from './shader-compiler-raw-webgpu';

// The "lost table fill" fault of the macOS test GPU (contributor contract section 11;
// dev-docs/research/math-correctness-review-2026-11-10/shader-compiler-findings.md) in the shapes
// that the kernels of this library use, reduced to plain WebGPU, and in the one kernel of the
// library that was found to lose a table (GPUNetworkLouvain as it was before 2026-11-10).
//
// `shader-compiler-table-fill.spec.ts` holds the minimal reproduction. This file answers two
// questions the library audit of 2026-11-11 needed:
//
// 1. Which ways of filling an integer table are exposed? Only a loop that does nothing but store
//    to local tables (a constant, the index, a value copied from a buffer), when no store at a
//    data-dependent index follows. A loop that also writes storage, a store that depends on the
//    data, a sorting network, an insertion and a bit mask all read right at every size.
// 2. Is a table below 33 entries safe? No. Inside the old Louvain kernel a 12-entry table was
//    lost, and the lower limit moved back to 33 when the calls to a helper that declares two
//    float arrays were removed. The size window of the reduced shader is not a rule.
//
// Every expectation is the WGSL meaning of the shader. The tests that pin a loss skip on an
// adapter that does not show it; the shapes that must be right are asserted on every adapter.

const SIZES = [8, 12, 16, 20, 24, 32, 33, 36, 40, 48, 64, 96, 128, 200, 250];

type TableShape = {
  /** What the shape is, for messages. */
  name: string;
  /** Module-scope declarations. */
  getModuleScope?: (size: number) => string;
  /** Body of `main`: fills the table and writes one word per entry to `output`. */
  getBody: (size: number) => string;
  /** Expected word of an entry; `entry + 1` when omitted. */
  getExpected?: (entry: number, size: number) => number;
};

const READ_BACK =
  'for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = table[entry]; }';

const getLiteral = (size: number): string =>
  `array<u32, ${size}>(${Array.from({length: size}, (_, entry) => `${entry + 1}u`).join(', ')})`;

/** Shapes whose only stores to the table are those of a loop that stores nothing else. */
const EXPOSED_SHAPES: (TableShape & {measuredLostSizes: number[]})[] = [
  {
    name: 'counted fill, nothing else',
    measuredLostSizes: [33, 36, 40, 48, 64, 96],
    getBody: size => `var table: array<u32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = entry + 1u; }
  ${READ_BACK}`
  },
  {
    name: 'a u32 sentinel and an f32 fill in one loop, no later store',
    measuredLostSizes: [33, 36, 40, 48],
    getBody: size => `var table: array<u32, ${size}>;
  var keys: array<f32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = 0xffffffffu; keys[entry] = 3.0e38; }
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = select(table[entry], entry + 1u, table[entry] == 0xffffffffu && keys[entry] > 1.0); }`
  },
  {
    name: 'values copied from a storage buffer by a counted loop',
    measuredLostSizes: [36, 40, 48, 64, 96],
    getBody: size => `var table: array<u32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = u32(input[entry]); }
  ${READ_BACK}`
  }
];

/** Shapes that keep what was stored, at every size. */
const KEPT_SHAPES: TableShape[] = [
  {
    name: 'var<workgroup> table',
    getModuleScope: size => `var<workgroup> table: array<u32, ${size}>;`,
    getBody: size => `for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = entry + 1u; }
  ${READ_BACK}`
  },
  {
    name: 'var<private> table with a literal initializer',
    getModuleScope: size => `var<private> table: array<u32, ${size}> = ${getLiteral(size)};`,
    getBody: () => READ_BACK
  },
  {
    name: 'member of a local struct',
    getModuleScope: size =>
      `struct Holder { scale: array<f32, ${size}>, table: array<u32, ${size}>, size: u32 }`,
    getBody: size => `var holder: Holder;
  for (var entry = 0u; entry < ${size}u; entry++) { holder.table[entry] = entry + 1u; }
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = holder.table[entry]; }`
  },
  {
    name: 'table returned from a function',
    getModuleScope: size => `fn getTable() -> array<u32, ${size}> {
  var table: array<u32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = entry + 1u; }
  return table;
}`,
    getBody: () => `let table = getTable();
  ${READ_BACK}`
  },
  {
    // The cure for a kernel without a uniform: every binding has at least one element, so the
    // bound equals the size, and the compiler cannot know it.
    name: 'fill bounded by size * min(arrayLength(&binding), 1u)',
    getBody: size => `var table: array<u32, ${size}>;
  let bound = ${size}u * min(arrayLength(&input), 1u);
  for (var entry = 0u; entry < bound; entry++) { table[entry] = entry + 1u; }
  ${READ_BACK}`
  },
  {
    name: 'fill bounded by min(size, arrayLength(&binding))',
    getBody: size => `var table: array<u32, ${size}>;
  let bound = min(${size}u, arrayLength(&output));
  for (var entry = 0u; entry < bound; entry++) { table[entry] = entry + 1u; }
  ${READ_BACK}`
  },
  {
    name: 'fill loop that also writes storage, read in a loop nest',
    getBody: size => `var table: array<u32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { output[entry] = 0u; table[entry] = entry + 1u; }
  for (var visit = 0u; visit < ${size}u; visit++) {
    var found = 0u;
    for (var entry = 0u; entry < ${size}u; entry++) { if (table[entry] == visit + 1u) { found = entry + 1u; } }
    output[visit] = found;
  }`
  },
  {
    name: 'store at the loop index under a data condition',
    getBody: size => `var table: array<u32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) {
    if (input[entry] < 0.0) { continue; }
    table[entry] = entry + 1u;
  }
  ${READ_BACK}`
  },
  {
    name: 'sums accumulated by a counted loop inside a run-time loop',
    getBody: size => `var table: array<u32, ${size}>;
  for (var sweep = 0u; sweep < parameters.sweeps; sweep++) {
    for (var entry = 0u; entry < ${size}u; entry++) { table[entry] += u32(input[entry]); }
  }
  ${READ_BACK}`,
    getExpected: entry => 2 * (entry + 1)
  },
  {
    name: 'identity fill, then a compare-exchange network at loop indices',
    getBody: size => `var table: array<u32, ${size}>;
  var keys: array<f32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = entry; keys[entry] = input[entry]; }
  for (var round = 0u; round < ${size}u; round++) {
    for (var first = round % 2u; first + 1u < ${size}u; first += 2u) {
      let left = table[first];
      let right = table[first + 1u];
      if (keys[left] < keys[right]) { table[first] = right; table[first + 1u] = left; }
    }
  }
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = table[entry] + 1u; }`,
    // The keys ascend with the entry, so the order by key descending is the reverse.
    getExpected: (entry, size) => size - entry
  },
  {
    name: 'sentinel fill, then insertion at a data position',
    getBody: size => `var table: array<u32, ${size}>;
  var keys: array<f32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = 0xffffffffu; keys[entry] = 3.0e38; }
  for (var sample = 0u; sample < ${size}u; sample++) {
    let key = -input[sample];
    if (key < keys[${size - 1}u]) {
      var slot = ${size - 1}u;
      while (slot > 0u && key < keys[slot - 1u]) { keys[slot] = keys[slot - 1u]; table[slot] = table[slot - 1u]; slot--; }
      keys[slot] = key; table[slot] = sample;
    }
  }
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = table[entry] + 1u; }`,
    getExpected: (entry, size) => size - entry
  },
  {
    name: 'i32 sentinel fill, then stores at data indices',
    getBody: size => `var table: array<i32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { table[entry] = -1; }
  table[parameters.sweeps] = 0;
  for (var entry = 0u; entry + 1u < ${size}u; entry++) {
    let neighbor = u32(input[entry]) % ${size}u;
    if (table[neighbor] < 0 && neighbor == 5u) { table[neighbor] = 1; }
  }
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = u32(table[entry] + i32(entry) + 2); }`,
    getExpected: entry => entry + 1 + (entry === 2 ? 1 : 0) + (entry === 5 ? 2 : 0)
  },
  {
    name: 'limbs copied between regions of one workspace through a pointer',
    getModuleScope:
      size => `fn copyLimbs(work: ptr<function, array<u32, ${2 * size}>>, destination: u32, source: u32, limbs: u32) {
  for (var limb = 0u; limb < limbs; limb++) { (*work)[destination + limb] = (*work)[source + limb]; }
}
fn loadLimbs(work: ptr<function, array<u32, ${2 * size}>>, destination: u32, limbs: u32) {
  for (var limb = 0u; limb < limbs; limb++) {
    var value = 0u;
    if (input[limb] > 0.0) { value = u32(input[limb]); }
    (*work)[destination + limb] = value & 0xffffu;
  }
}`,
    getBody: size => `var work: array<u32, ${2 * size}>;
  loadLimbs(&work, 0u, ${size}u);
  copyLimbs(&work, ${size}u, 0u, ${size}u);
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = work[${size}u + entry]; }`
  },
  {
    name: 'bit mask set by a counted loop',
    getBody: size => `var table: array<u32, ${size}>;
  for (var bit = 0u; bit < ${32 * size}u; bit++) { if (input[bit / 32u] > 0.0) { table[bit / 32u] |= 1u << (bit % 32u); } }
  for (var entry = 0u; entry < parameters.count; entry++) { output[entry] = select(0u, entry + 1u, table[entry] == 0xffffffffu); }`
  }
];

function getShapeShader(shape: TableShape, size: number): string {
  return `struct Parameters { count: u32, sweeps: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@group(0) @binding(2) var<storage, read> input: array<f32>;
${shape.getModuleScope?.(size) ?? ''}
@compute @workgroup_size(1)
fn main() {
  _ = &input;
  _ = parameters.count;
  ${shape.getBody(size)}
}`;
}

/** Sizes of {@link SIZES} at which a shape reads back anything but what the WGSL stores. */
async function getWrongSizes(device: GPUDevice, shape: TableShape): Promise<number[]> {
  const wrongSizes: number[] = [];
  for (const size of SIZES) {
    const words = await runRawComputeShader(device, {
      code: getShapeShader(shape, size),
      outputLength: size,
      // `count` entries are read back; `sweeps` is 2 (two accumulation sweeps, and index 2).
      uniformWords: [size, 2],
      inputValues: Float32Array.from({length: size}, (_, entry) => entry + 1)
    });
    if (words.some((word, entry) => word !== (shape.getExpected?.(entry, size) ?? entry + 1))) {
      wrongSizes.push(size);
    }
  }
  return wrongSizes;
}

it('a loop that only stores to local tables loses an integer table when no data store follows', async context => {
  const device = await getRawTestDevice();
  const wrongSizesByShape = new Map<string, number[]>();
  for (const shape of EXPOSED_SHAPES) {
    wrongSizesByShape.set(shape.name, await getWrongSizes(device, shape));
  }
  if ([...wrongSizesByShape.values()].every(wrongSizes => wrongSizes.length === 0)) {
    context.skip(`${getAdapterLabel(device)} keeps every fill`);
  }
  // Measured on the test GPU (Apple M3 Pro, macOS 14.1, Chromium 151).
  for (const shape of EXPOSED_SHAPES) {
    expect(wrongSizesByShape.get(shape.name), shape.name).toEqual(shape.measuredLostSizes);
  }
});

it('the table shapes of the library kernels keep their contents at every size', async () => {
  const device = await getRawTestDevice();
  for (const shape of KEPT_SHAPES) {
    expect(await getWrongSizes(device, shape), shape.name).toEqual([]);
  }
});

/** How the pre-fix Louvain kernel declares and fills its visit order. */
type VisitOrderShape = 'counted-fill' | 'literal' | 'private-counted-fill' | 'run-time-bound';

/**
 * GPUNetworkLouvain as it was before the review of 2026-11-10 (ascending visits through an order
 * table filled by an identity loop), with the table read out to a `probe` binding as it is used.
 * `callsObjective: false` replaces the two calls to `objective`, which declares two float arrays.
 */
function getOldLouvainKernel(
  nodeCount: number,
  visitOrderShape: VisitOrderShape,
  callsObjective: boolean
): {declarations: string; body: string} {
  const visitOrderSource = {
    'counted-fill': `var order: array<u32, ${nodeCount}>;
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { order[vertex] = vertex; }`,
    literal: `var order = array<u32, ${nodeCount}>(${Array.from({length: nodeCount}, (_, vertex) => `${vertex}u`).join(', ')});`,
    'private-counted-fill':
      'for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { order[vertex] = vertex; }',
    'run-time-bound': `var order: array<u32, ${nodeCount}>;
for (var vertex = 0u; vertex < NODE_COUNT * min(arrayLength(&weights), 1u); vertex++) { order[vertex] = vertex; }`
  }[visitOrderShape];
  return {
    declarations: `${GET_NAN_WGSL}
const NODE_COUNT: u32 = ${nodeCount}u;
${visitOrderShape === 'private-counted-fill' ? `var<private> order: array<u32, ${nodeCount}>;` : ''}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn objective(total: f32, resolution: f32) -> f32 {
  var internal = 0.0;
  var strengths: array<f32, ${nodeCount}>;
  var incoming: array<f32, ${nodeCount}>;
  for (var source = 0u; source < NODE_COUNT; source++) {
    let sourceCommunity = communities[communitiesOffset + source];
    for (var destination = 0u; destination < NODE_COUNT; destination++) {
      let weight = weights[weightsOffset + source * NODE_COUNT + destination];
      strengths[sourceCommunity] += weight;
      incoming[communities[communitiesOffset + destination]] += weight;
      if (sourceCommunity == communities[communitiesOffset + destination]) { internal += weight; }
    }
  }
  var penalty = 0.0;
  for (var community = 0u; community < NODE_COUNT; community++) { penalty += (strengths[community] / total) * (incoming[community] / total); }
  return internal / total - resolution * penalty;
}`,
    body: `for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
for (var entry = 0u; entry < 129u; entry++) { trace[traceOffset + entry] = getNaN(); }
var groups: array<u32, ${nodeCount}>;
${visitOrderSource}
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { communities[communitiesOffset + vertex] = vertex; groups[vertex] = vertex; }
let resolution = parameters[parametersOffset];
let tolerance = parameters[parametersOffset + 1u];
if (!finite(resolution) || resolution < 0.0 || !finite(tolerance) || tolerance < 0.0) { return; }
var total = 0.0;
for (var source = 0u; source < NODE_COUNT; source++) {
  for (var destination = 0u; destination < NODE_COUNT; destination++) {
    let weight = weights[weightsOffset + source * NODE_COUNT + destination];
    if (!finite(weight) || weight < 0.0 || weight != weights[weightsOffset + destination * NODE_COUNT + source]) { return; }
    total += weight;
  }
}
if (!finite(total)) { return; }
summary[summaryOffset] = 1u;
if (total == 0.0) { trace[traceOffset] = 0.0; summary[summaryOffset + 1u] = 1u; return; }
var current = ${callsObjective ? 'objective(total, resolution)' : 'resolution'};
if (!finite(current)) { summary[summaryOffset] = 0u; return; }
trace[traceOffset] = current;
var sweeps = 0u;
for (var level = 0u; level < 8u; level++) {
  var levelChanged = false;
  var stable = false;
  for (var round = 0u; round < 16u; round++) {
    var changed = false;
    for (var visit = 0u; visit < NODE_COUNT; visit++) {
      let group = order[visit];
      probe[probeOffset + visit] = group + 1u;
      var present = false;
      var original = 0u;
      for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { present = true; original = communities[communitiesOffset + vertex]; } }
      if (!present) { continue; }
      var best = original;
      var bestObjective = current;
      for (var candidate = 0u; candidate < NODE_COUNT; candidate++) {
        var adjacent = false;
        for (var source = 0u; source < NODE_COUNT; source++) {
          if (groups[source] != group) { continue; }
          for (var destination = 0u; destination < NODE_COUNT; destination++) {
            if (groups[destination] != group && communities[communitiesOffset + destination] == candidate && (weights[weightsOffset + source * NODE_COUNT + destination] > 0.0 || weights[weightsOffset + destination * NODE_COUNT + source] > 0.0)) { adjacent = true; }
          }
        }
        if (!adjacent || candidate == original) { continue; }
        for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = candidate; } }
        let candidateObjective = ${callsObjective ? 'objective(total, resolution)' : 'f32(candidate)'};
        if (candidateObjective > bestObjective + tolerance) { best = candidate; bestObjective = candidateObjective; }
        for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = original; } }
      }
      for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = best; } }
      if (best != original) { changed = true; current = bestObjective; }
    }
    sweeps++;
    trace[traceOffset + sweeps] = current;
    levelChanged = levelChanged || changed;
    if (!changed) { stable = true; break; }
  }
  summary[summaryOffset + 2u] = level + 1u;
  summary[summaryOffset + 3u] = sweeps;
  if (!stable) { return; }
  if (!levelChanged) { summary[summaryOffset + 1u] = 1u; return; }
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { groups[vertex] = communities[communitiesOffset + vertex]; }
}`
  };
}

const LOUVAIN_SIZES = [4, 8, 10, 12, 16, 24, 32, 40];

/** Sizes of {@link LOUVAIN_SIZES} at which the old Louvain kernel reads a wrong visit order. */
async function getOldLouvainWrongSizes(
  visitOrderShape: VisitOrderShape,
  callsObjective = true
): Promise<number[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wrongSizes: number[] = [];
  for (const nodeCount of LOUVAIN_SIZES) {
    // Disjoint cliques of four vertices; the weights only have to be a valid symmetric matrix.
    const weights = new Float32Array(nodeCount * nodeCount);
    for (let source = 0; source < nodeCount; source++) {
      for (let destination = 0; destination < nodeCount; destination++) {
        if (source !== destination && Math.floor(source / 4) === Math.floor(destination / 4)) {
          weights[source * nodeCount + destination] = 1;
        }
      }
    }
    const harness = createOptimizationHarness(device);
    const views = {
      weights: harness.input('float32', weights),
      parameters: harness.input('float32', Float32Array.of(1, 0)),
      communities: harness.output('uint32', nodeCount),
      trace: harness.output('float32', 129),
      summary: harness.output('uint32', 4),
      probe: harness.output('uint32', nodeCount)
    };
    const kernel = getOldLouvainKernel(nodeCount, visitOrderShape, callsObjective);
    const result = await harness.run({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: `old-louvain-${visitOrderShape}-${nodeCount}`,
          operation: 'GPUNetworkLouvain',
          invocationCount: 1,
          workgroupSize: 2,
          bindings: [
            {name: 'weights', view: views.weights, type: 'f32', access: 'read'},
            {name: 'parameters', view: views.parameters, type: 'f32', access: 'read'},
            {name: 'communities', view: views.communities, type: 'u32', access: 'read_write'},
            {name: 'trace', view: views.trace, type: 'f32', access: 'read_write'},
            {name: 'summary', view: views.summary, type: 'u32', access: 'read_write'},
            {name: 'probe', view: views.probe, type: 'u32', access: 'read_write'}
          ],
          declarations: kernel.declarations,
          body: kernel.body
        })
      ]
    });
    // Visit `v` must read group `v`: the probe holds `v + 1`.
    if (result(views.probe).some((word, visit) => word !== visit + 1)) {
      wrongSizes.push(nodeCount);
    }
  }
  return wrongSizes;
}

it('inside the old Louvain kernel a counted fill is lost from 12 entries, not 33', async context => {
  const wrongSizes = await getOldLouvainWrongSizes('counted-fill');
  if (wrongSizes.length === 0) {
    context.skip('this adapter keeps the visit order of the old Louvain kernel');
  }
  // Measured on the test GPU: every size tried from 12 up, 8 and 10 right. The reduced shader
  // keeps a table of up to 32 entries, so a size below its window is not a safe size.
  expect(wrongSizes).toEqual([12, 16, 24, 32, 40]);
  // Without the helper that declares two float arrays of the same length the window is the one
  // of the reduced shader again: what else the kernel holds decides the sizes.
  expect(await getOldLouvainWrongSizes('counted-fill', false)).toEqual([40]);
}, 300_000);

it('a literal initializer and a fill bounded at run time cure the old Louvain kernel', async () => {
  expect(await getOldLouvainWrongSizes('literal')).toEqual([]);
  expect(await getOldLouvainWrongSizes('run-time-bound')).toEqual([]);
  // `var<private>` with the same counted fill also reads right in this kernel, which is why the
  // first pass took it for a cure. The reduced shader loses a private table
  // (shader-compiler-table-fill.spec.ts), so this line is an observation about one kernel.
  expect(await getOldLouvainWrongSizes('private-counted-fill')).toEqual([]);
}, 300_000);
