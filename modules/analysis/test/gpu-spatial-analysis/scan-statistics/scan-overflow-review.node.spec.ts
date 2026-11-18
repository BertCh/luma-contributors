// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  findStatisticsNonFiniteConstants,
  findStatisticsBarrierRisks
} from '../spatial-weights/statistics-wgsl-audit';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect, WgslExec, WgslParser} from 'wgsl_reflect';
import {SCAN_DRAW_INTERVAL_WGSL} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-draw-interval';
import {createSpatialScanStatisticForTesting} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import type {GPUSpatialScanModel} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

/** WgslExec lacks these builtins. Supply normal-scalar IEEE semantics for the test inputs. */
function createScaledInterpreter(source: string): WgslExec {
  new WgslParser().parse(source);
  const executableSource = source
    .replace(/\bfrexp\(/g, 'probeFrexp(')
    .replace(/\bldexp\(/g, 'probeLdexp(');
  return new WgslExec(
    new WgslParser().parse(`${executableSource}
struct ProbeFraction { fract: f32, exp: i32 }
fn probeFrexp(value: f32) -> ProbeFraction {
  if (value == 0.0) { return ProbeFraction(0.0, 0); }
  let bits = bitcast<u32>(value);
  return ProbeFraction(bitcast<f32>((bits & 0x807fffffu) | 0x3f000000u), i32((bits >> 23u) & 255u) - 126);
}
fn probeLdexp(value: f32, exponent: i32) -> f32 { return value * exp2(f32(exponent)); }
`)
  );
}

function createScene(model: GPUSpatialScanModel = 'bernoulli', maximumPermutations = 3, zones = 3) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scan-review-${serial++}`, format, length);
  const props = {
    model,
    positions: view('float32x2', zones),
    cases: view('uint32', zones),
    ...(model === 'poisson'
      ? {baseline: view('float32', zones)}
      : model === 'bernoulli'
        ? {controls: view('uint32', zones)}
        : {}),
    parameters: view('uint32', 8),
    timeBuckets: 1,
    maximumPermutations,
    maximumWindowZones: zones,
    maximumClusters: 1,
    batchReplicates: 1,
    output: {
      clusterIndices: view('uint32', 4),
      clusterStatistics: view('float32', 8),
      statistics: view('float32', maximumPermutations + 1),
      summary: view('uint32', 4),
      status: {invalidCount: view('uint32', 1)}
    }
  };
  return {device, graph, view, props};
}

it('Poisson likelihood preserves baseline units at large finite scales', () => {
  const {device, graph, props} = createScene('poisson');
  try {
    const sources = getGPUCommandNodeKernelSources(
      createSpatialScanStatisticForTesting(props).getCommandNodes(graph)
    );
    const evaluate = sources.find(kernel => kernel.id.endsWith('-evaluate'))!.source;
    const likelihood = evaluate.match(
      /fn getEntropyExcess[\s\S]*?fn getLogLikelihoodRatio[\s\S]*?\n}/
    )![0];
    const scales = Float32Array.of(1, 1e30, 1e38);
    const output = new Float32Array(scales.length);
    const source = `${likelihood}
@group(0) @binding(0) var<storage, read> scales: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
 let index = invocation.x;
 output[index] = getLogLikelihoodRatio(10.0, scales[index], 10.0, 2.0 * scales[index], 0u);
}`;
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [scales.length, 1, 1], {
      0: {0: scales, 1: output}
    });
    for (const value of output) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(10 * Math.log(2), 5);
    }
  } finally {
    device.destroy();
  }
});

it('Poisson sums normalized baselines before totals, including overflowing raw sums and zeros', () => {
  const {device, graph, props} = createScene('poisson');
  try {
    const sources = getGPUCommandNodeKernelSources(
      createSpatialScanStatisticForTesting(props).getCommandNodes(graph)
    );
    for (const scale of [0, 1e-30, 1, 1e38, 3e38]) {
      const memory: Record<string, Float32Array | Uint32Array> = {
        rawBaseline: Float32Array.of(scale / 2, scale, scale / 4),
        normalizedBaseline: new Float32Array(3),
        baselineMaximum: new Uint32Array(1),
        blockBaseline: new Float32Array(3),
        blockCases: new Uint32Array(3),
        cases: Uint32Array.of(10, 0, 0),
        totals: new Uint32Array(7)
      };
      memory['baseline'] = memory['normalizedBaseline'];
      memory['fillTarget'] = memory['baselineMaximum'];
      for (const kernel of sources.filter(kernel =>
        /-baseline-(block-maximum|maximum-clear|normalize)$|-block-totals$|-totals$/.test(kernel.id)
      )) {
        const reflection = new WgslReflect(kernel.source);
        const bindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
        for (const binding of reflection.storage) {
          expect(memory[binding.name], binding.name).toBeDefined();
          bindings[binding.group][binding.binding] = memory[binding.name];
        }
        // The interpreter does not support phony assignments. Keep the kernel arithmetic intact
        // and supply the same invocation/dispatch overrides used by the graph for these sizes.
        const interpretedSource = kernel.source.replace(/^\s*_ = [^;]+;/gm, '');
        createScaledInterpreter(interpretedSource).dispatchWorkgroups('main', [1, 1, 1], bindings, {
          constants: {
            INVOCATION_COUNT:
              /-baseline-maximum-clear$|-totals$/.test(kernel.id) &&
              !kernel.id.endsWith('-block-totals')
                ? 1
                : 3,
            WGSL_KERNEL_DISPATCH_X: 1,
            WGSL_KERNEL_DISPATCH_Y: 1
          }
        });
      }
      const roundedScale = Math.fround(scale);
      const scaledMaximum =
        scale === 0 ? 0 : roundedScale / 2 ** Math.floor(Math.log2(roundedScale));
      expect(Array.from(memory['normalizedBaseline'])).toEqual([
        scaledMaximum / 2,
        scaledMaximum,
        scaledMaximum / 4
      ]);
      expect(new Float32Array(memory['totals'].buffer)[1]).toBeCloseTo(1.75 * scaledMaximum, 6);
      expect(memory['totals'][0]).toBe(10);
    }
  } finally {
    device.destroy();
  }
});

it('Poisson binary normalization avoids raw reciprocals across baseline scales', () => {
  for (const scale of [1e-30, 1, 1e30, 1e38, 3e38]) {
    const result = interpretBaselineReview(Float32Array.of(scale / 2, scale, scale / 4), true);
    expect(result.weights[0] / result.weights[1]).toBe(0.5);
    expect(result.weights[2] / result.weights[1]).toBe(0.25);
  }
});

function interpretBaselineReview(values: Float32Array, flushSubnormalLoads = false) {
  const zones = values.length;
  const {device, graph, props} = createScene('poisson', 1, zones);
  try {
    const sources = getGPUCommandNodeKernelSources(
      createSpatialScanStatisticForTesting({...props, maximumWindowZones: 1}).getCommandNodes(graph)
    );
    const memory: Record<string, Float32Array | Uint32Array> = {
      rawBaseline: new Uint32Array(values.buffer),
      baselineMaximum: new Uint32Array(1),
      normalizedBaseline: new Float32Array(zones),
      blockBaseline: new Float32Array(zones),
      blockCases: new Uint32Array(zones),
      cases: Uint32Array.from({length: zones}, (_, row) => (row === 0 ? 10 : 0)),
      totals: new Uint32Array(7),
      parameters: Uint32Array.of(
        1,
        0,
        1,
        new Uint32Array(Float32Array.of(0.5).buffer)[0],
        1,
        1,
        0,
        0
      ),
      neighbors: Uint32Array.from(
        {length: zones * 2},
        (_, index) => (Math.floor(index / 2) + (index % 2)) % zones
      ),
      centerBest: new Uint32Array(zones * 32)
    };
    memory['baseline'] = memory['normalizedBaseline'];
    memory['fillTarget'] = memory['baselineMaximum'];
    let windowInfoStart = 0;
    for (const kernel of sources.filter(kernel =>
      /-baseline-(block-maximum|maximum-clear|normalize)$|-block-totals$|-totals$|-window-limit$/.test(
        kernel.id
      )
    )) {
      let source = kernel.source.replace(/^\s*_ = [^;]+;/gm, '');
      if (flushSubnormalLoads && /rawBaseline: array<f32>/.test(source))
        source =
          `fn probeFlush(value: f32) -> f32 { if ((bitcast<u32>(value) & 0x7f800000u) == 0u) { return 0.0; } return value; }\n${source}`.replace(
            /rawBaseline\[rawBaselineOffset \+ (cell|index)\]/g,
            'probeFlush($&)'
          );
      const bindings: Record<number, Record<number, Float32Array | Uint32Array>> = {0: {}};
      for (const binding of new WgslReflect(source).storage)
        bindings[binding.group][binding.binding] = memory[binding.name];
      createScaledInterpreter(source).dispatchWorkgroups('main', [1, 1, 1], bindings, {
        constants: kernel.constants
      });
      if (kernel.id.endsWith('-window-limit'))
        windowInfoStart = Number(source.match(/const WINDOW_INFO_START: u32 = (\d+)u/)![1]);
    }
    return {
      maximumBits: memory['baselineMaximum'][0],
      weights: Array.from(memory['normalizedBaseline']),
      total: new Float32Array(memory['totals'].buffer)[1],
      slots: memory['centerBest'][windowInfoStart]
    };
  } finally {
    device.destroy();
  }
}

it('binary baseline scaling preserves a window at exactly half the study population', () => {
  const result = interpretBaselineReview(Float32Array.of(6, 1, 4, 1));
  expect(result.slots).toBe(1);
  expect(result.weights[0] / result.total).toBe(0.5);
});

it('positive subnormal baselines survive float-load flushing and admit a half-population window', () => {
  const result = interpretBaselineReview(new Float32Array(Uint32Array.of(1, 1).buffer), true);
  expect(result.maximumBits).toBe(1);
  expect(result.weights).toEqual([1, 1]);
  expect(result.slots).toBe(1);
});

for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const)
  it(`${model} uses saturated counts and audits every generated identifier`, () => {
    const {device, graph, props} = createScene(model);
    try {
      const nodes = createSpatialScanStatisticForTesting(props).getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(nodes.some(node => node.id.endsWith('invalid-count-status'))).toBe(true);
      expect(sources.find(item => item.id.endsWith('-totals'))?.source).toContain(
        'getSaturatedScanSum'
      );
      if (model !== 'space-time-permutation')
        expect(nodes.some(node => node.id.endsWith('cumulative-saturate'))).toBe(true);
      for (const {source} of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
        expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
        if (/workgroupBarrier|storageBarrier/.test(source))
          expect(findStatisticsBarrierRisks(source)).toEqual([]);
        expect(() => new WgslReflect(source)).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });

it('scan rejects invalid caller usage before declarations and rejects writable aliases', () => {
  const {device, graph, props} = createScene();
  const buffer = device.createBuffer({byteLength: 12, usage: Buffer.COPY_DST});
  try {
    props.cases = importGraphBuffer(graph, 'invalid-cases', buffer, 'uint32', 3);
    const contributor = createSpatialScanStatisticForTesting(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declare = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declare).not.toHaveBeenCalled();
    props.output.status.invalidCount = graph.createDataView(props.output.summary.buffer, {
      format: 'uint32',
      length: 1
    });
    expect(() => createSpatialScanStatisticForTesting(props)).toThrow(/overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('scan pipeline count stays constant as replicate batches grow', () => {
  const record = (permutations: number) => {
    const {device, graph, props} = createScene('poisson', permutations);
    try {
      return getGPUCommandNodePipelineKeys(
        createSpatialScanStatisticForTesting(props).getCommandNodes(graph)
      ).size;
    } finally {
      device.destroy();
    }
  };
  expect(record(3)).toBe(record(9));
});

it('production scan intervals cover upper-bound totals exactly with no wrapped endpoints', () => {
  const totals = Uint32Array.of(0, 1, 1023, 1024, 1025, 0xffffff00, 0xfffffffd, 0xfffffffe);
  const output = new Uint32Array(totals.length * 1024 * 2);
  const source = `${SCAN_DRAW_INTERVAL_WGSL}
@group(0) @binding(0) var<storage, read> totals: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
 let index = invocation.x;
 let interval = getScanDrawInterval(totals[index / 1024u], 1024u, index % 1024u);
 output[index * 2u] = interval.first; output[index * 2u + 1u] = interval.stop;
}`;
  expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
  if (/workgroupBarrier|storageBarrier/.test(source))
    expect(findStatisticsBarrierRisks(source)).toEqual([]);
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    [totals.length * 1024, 1, 1],
    {0: {0: totals, 1: output}}
  );
  for (const [domain, total] of totals.entries()) {
    let previous = 0;
    let covered = 0;
    for (let blockIndex = 0; blockIndex < 1024; blockIndex++) {
      const offset = (domain * 1024 + blockIndex) * 2;
      expect(output[offset]).toBe(Math.min(blockIndex * Math.ceil(total / 1024), total));
      expect(output[offset + 1]).toBe(Math.min((blockIndex + 1) * Math.ceil(total / 1024), total));
      expect(output[offset]).toBe(previous);
      expect(output[offset + 1]).toBeGreaterThanOrEqual(previous);
      expect(output[offset + 1]).toBeLessThanOrEqual(total);
      covered += output[offset + 1] - output[offset];
      previous = output[offset + 1];
    }
    expect(previous).toBe(total);
    expect(covered).toBe(total);
  }
});
