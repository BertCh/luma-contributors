// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {GPUCommandGraph, GraphDataView, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {
  GPUSpatialScanStatistic,
  type GPUSpatialScanStatisticProps
} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {GPUExpectationBasedSpaceTimeScan} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {computeScanOracle} from './scan-statistic-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `round1-${serial++}`, format, length);
  const props: GPUSpatialScanStatisticProps = {
    id: 'round1-scan',
    positions: view('float32x2', 2),
    cases: view('uint32', 2),
    baseline: view('float32', 2),
    parameters: view('uint32', 8),
    maximumPermutations: 19,
    maximumClusters: 1,
    output: {
      clusterIndices: view('uint32', 4),
      clusterStatistics: view('float32', 8),
      statistics: view('float32', 20),
      summary: view('uint32', 4)
    }
  };
  return {device, graph, props};
}

it('device fixtures retain buffers that implement host writes (B1)', () => {
  const source = readFileSync(new URL('./scan-inference-w3.spec.ts', import.meta.url), 'utf8');
  const tree = ts.createSourceFile('fixture.ts', source, ts.ScriptTarget.Latest, true);
  const declarations: ts.VariableDeclaration[] = [];
  const visit = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      ['cases', 'parameters'].includes(node.name.text) &&
      node.initializer &&
      ts.isCallExpression(node.initializer) &&
      node.initializer.expression.getText(tree).startsWith('rig.')
    )
      declarations.push(node);
    ts.forEachChild(node, visit);
  };
  visit(tree);
  const device = createNullWebGPUDevice();
  const rig = new GraphRig(device);
  try {
    expect(declarations.length).toBeGreaterThanOrEqual(3);
    for (const declaration of declarations) {
      // Execute the real fixture's host allocation expression, without running a GPU test.
      const allocate = new Function(
        'rig',
        'settings',
        `return ${declaration.initializer!.getText(tree)};`
      );
      const writable = allocate(rig, () => Uint32Array.of(0, 0, 2, 0));
      expect(typeof writable.buffer.write).toBe('function');
      expect(writable.view.buffer).not.toBe(writable.buffer);
    }
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('uses bracket access for kernel override records (B2)', () => {
  const source = readFileSync(
    new URL('./scan-overflow-review.node.spec.ts', import.meta.url),
    'utf8'
  );
  expect(/kernel\.constants\.INVOCATION_COUNT/.test(source)).toBe(false);
});

it('publishes the complete invalid-valid-invalid summary and zero invalid dispatch widths (B3)', () => {
  const {device, graph, props} = createScene();
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPUSpatialScanStatistic(props).getCommandNodes(graph)
    );
    const memory: Record<string, Uint32Array> = {
      parameters: getGPUSpatialScanStatisticParameterValues({seed: 0, permutationCount: 19}),
      state: new Uint32Array(8),
      totals: new Uint32Array(7),
      overflowPrefixes: new Uint32Array(2),
      summary: new Uint32Array(4).fill(77),
      dispatch: new Uint32Array(9).fill(77)
    };
    for (const invalid of [true, false, true]) {
      memory['totals'].fill(0);
      memory['totals'][0] = invalid ? 0 : 3;
      memory['totals'][5] = Number(invalid);
      memory['totals'][6] = 3;
      for (const kernel of kernels.filter(kernel =>
        /-batch-control-0$|-summary$/.test(kernel.id)
      )) {
        const bindings: Record<number, Record<number, Uint32Array>> = {0: {}};
        for (const binding of new WgslReflect(kernel.source).storage) {
          expect(memory[binding.name], binding.name).toBeDefined();
          bindings[binding.group][binding.binding] = memory[binding.name];
        }
        new WgslExec(
          new WgslParser().parse(kernel.source.replace(/^\s*_ = [^;]+;/gm, ''))
        ).dispatchWorkgroups('main', 1, bindings, {constants: kernel.constants});
      }
      expect(
        [memory['dispatch'][0], memory['dispatch'][3], memory['dispatch'][6]].every(width =>
          invalid ? width === 0 : width > 0
        )
      ).toBe(true);
      expect(Array.from(memory['summary'])).toEqual([0, 3, invalid ? 0 : 19, 0]);
    }
  } finally {
    device.destroy();
  }
});

it('refuses impossible Poisson oracle frames and retains zero-case/zero-exposure controls (B4)', () => {
  const fixture = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0),
    cases: Uint32Array.of(5, 0, 12),
    baseline: Float32Array.of(0, 100, 1),
    timeBuckets: 1,
    maximumPopulationFraction: 0.5,
    windowShape: 'circle' as const,
    maximumClusters: 1,
    seed: 0,
    permutationCount: 1
  };
  expect(() => computeScanOracle(fixture)).toThrow(/positive cases.*zero exposure/);
  const control = computeScanOracle({...fixture, cases: Uint32Array.of(0, 0, 12)});
  // Exact conditional singleton score: all 12 cases in exposure 1 of total exposure 101.
  expect(control.clusters).toHaveLength(1);
  expect(Number.isFinite(control.clusters[0].logLikelihoodRatio)).toBe(true);
  expect(control.clusters[0].logLikelihoodRatio).toBeCloseTo(55.38144620209512, 10);
  expect(() =>
    computeScanOracle({
      ...fixture,
      cases: Uint32Array.of(1, 0, 0),
      baseline: Float32Array.of(2 ** -70, 2 ** 70, 0)
    })
  ).toThrow(/exposure scaling/);
});

it('caps count bindings at the producer limit and includes selected radix bindings (B6)', () => {
  const shape = {
    zones: 2,
    timeBuckets: 1,
    maximumNeighbors: 1,
    maximumWindowLength: 1,
    maximumReplicates: 1
  };
  const large = GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, {
    maxBufferSize: 268435456,
    maxStorageBufferBindingSize: 268435456
  });
  expect(large.constraints.find(constraint => constraint.name === 'count table')!.maximum).toBe(
    134217728
  );
  const small = GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, {
    maxBufferSize: 32,
    maxStorageBufferBindingSize: 32
  });
  expect(small.feasible).toBe(false);
  expect(
    small.constraints.some(constraint => constraint.received === 64 && !constraint.satisfied)
  ).toBe(true);
  expect(
    GPUExpectationBasedSpaceTimeScan.getExecutionPlan(
      {...shape, nearestZoneSearch: 'full-scan'},
      {maxBufferSize: 32, maxStorageBufferBindingSize: 32}
    ).feasible
  ).toBe(true);
});

it('retains actual selector values and allowed alternatives in structured errors (S1)', () => {
  for (const [field, accepted] of [
    ['windowShape', ['circle', 'nearest']],
    ['rateDirection', ['high', 'low', 'both']]
  ] as const) {
    try {
      getGPUSpatialScanStatisticParameterValues({
        seed: 0,
        permutationCount: 1,
        [field]: 'constructor'
      });
      throw Error('expected rejection');
    } catch (error) {
      expect(error).toMatchObject({
        code: 'INVALID_OPTION',
        path: field,
        received: 'constructor',
        expected: {values: accepted}
      });
    }
  }
  const {device, props} = createScene();
  try {
    for (const [field, accepted] of [
      ['model', ['poisson', 'bernoulli', 'space-time-permutation']],
      ['temporalMode', ['retrospective', 'prospective']]
    ] as const) {
      try {
        new GPUSpatialScanStatistic({...props, [field]: 'constructor'});
        throw Error('expected rejection');
      } catch (error) {
        expect(error).toMatchObject({
          code: 'INVALID_OPTION',
          path: field,
          received: 'constructor',
          expected: {values: accepted}
        });
      }
    }
  } finally {
    device.destroy();
  }
});

it('declares an explicit command context without a generic instantiation expression (S3)', () => {
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(/ReturnType<\s*typeof createSpatialScanCommandContext<Parameters>/.test(source)).toBe(
    false
  );
});

it('device ordering fixture checks both centers and the tie-breaking winner (B5)', () => {
  const source = readFileSync(
    new URL('./stx07-expectation-based-scan.spec.ts', import.meta.url),
    'utf8'
  );
  const fixture = source.slice(
    source.indexOf("it('selects the higher overflowing-ratio singleton for the reviewer fixture")
  );
  expect(fixture.includes('zoneStatistics')).toBe(true);
  expect(fixture.includes('centerZone')).toBe(true);
});

it('matches selected-route radix binding bounds to actual published sort resources (B6)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const zones of [1, 2, 3, 257, 4097]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `radix-${serial++}`, format, length);
      const shape = {
        zones,
        timeBuckets: 1,
        maximumNeighbors: 1,
        maximumWindowLength: 1,
        maximumReplicates: 1,
        nearestZoneSearch: 'indexed' as const
      };
      const plan = GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, {
        maxBufferSize: 268435456,
        maxStorageBufferBindingSize: 268435456
      });
      const scan = new GPUExpectationBasedSpaceTimeScan({
        ...shape,
        positions: view('float32x2', zones),
        cases: view('uint32', zones),
        baseline: view('float32', zones),
        parameters: view('uint32', 4),
        output: {
          clusterZones: view('uint32', 1),
          clusterStatistics: view('float32', 4),
          statistics: view('float32', 2),
          summary: view('uint32', 5)
        }
      });
      const resources = scan
        .getCommandNodes(graph)
        .filter(node => /-index-[xy]/.test(node.id))
        .flatMap(node => node.resources ?? []);
      const bytes = resources.flatMap(resource => {
        if (!('buffer' in resource)) return [];
        const bound = resource.buffer;
        return [
          bound instanceof GraphDataView
            ? bound.byteOffset +
              Math.max(0, bound.length - 1) * bound.byteStride +
              bound.rowByteLength
            : bound.byteLength
        ];
      });
      expect(bytes.length).toBeGreaterThan(0);
      expect(Math.max(...bytes)).toBe(
        plan.constraints.find(constraint => constraint.name === 'indexed radix bindings')!.received
      );
    }
  } finally {
    device.destroy();
  }
});
