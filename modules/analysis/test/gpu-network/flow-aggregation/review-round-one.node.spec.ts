// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUKernelFlowSmoothing} from '../../../src/gpu-network/flow-aggregation/gpu-kernel-flow-smoothing';
import {GPURadiationFlowModel} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model';
import {GPUNetworkPotentialPathArea} from '../../../src/gpu-network/network-time-geography/gpu-network-potential-path-area';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

function execute(source: string, values: Record<string, Uint32Array | Float32Array>) {
  source = source
    .replace(/^\s*_ = .*;$/gm, '')
    .replace('override INVOCATION_COUNT: u32;', 'const INVOCATION_COUNT: u32 = 4u;')
    .replace('override WGSL_KERNEL_DISPATCH_X: u32;', 'const WGSL_KERNEL_DISPATCH_X: u32 = 1u;')
    .replace('override WGSL_KERNEL_DISPATCH_Y: u32;', 'const WGSL_KERNEL_DISPATCH_Y: u32 = 1u;');
  const bindings: Record<number, Uint32Array | Float32Array> = {};
  for (const match of source.matchAll(/@binding\((\d+)\)\s+var<storage,\s*\w+>\s+(\w+)/g)) {
    bindings[Number(match[1])] = values[match[2]] ?? new Uint32Array(4096);
  }
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: bindings});
}

it('B2 B3 B4 S3 S4 preserve producer evidence and compatible count views', () => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {graph, input, output} = context;
  const count = input('count', 'uint32', [1, 99]);
  const overflow = input('overflow', 'uint32', [0]);
  const candidate = input('candidate', 'uint32', [1]);
  const pairRows = {
    kind: 'pairs' as const,
    originZoneIds: input('origin', 'uint32', [0, 1]),
    destinationZoneIds: input('destination', 'uint32', [1, 0]),
    weights: input('weights', 'float32', [2, 8])
  };
  const props = {
    positions: input('positions', 'float32x2', [0, 0, 1, 0]),
    locationWeights: input('location', 'float32', [1, 1]),
    parameters: input('parameters', 'float32', [0, 0.5, 0, 0]),
    flows: {...pairRows, completeness: 'known-complete' as const, count},
    output: {
      smoothed: output('smoothed', 'float32', 4),
      status: {complete: output('complete', 'uint32', 1)}
    }
  };
  try {
    expect(() => new GPUKernelFlowSmoothing(props)).not.toThrow();
    expect(
      () =>
        new GPUKernelFlowSmoothing({...props, flows: {...pairRows, count, status: null} as never})
    ).toThrow(/status|completeness/);
    const status = Object.create({
      get candidateOverflow() {
        return candidate;
      }
    });
    Object.assign(status, {count, overflow});
    const inherited = new GPUKernelFlowSmoothing({...props, flows: {...pairRows, status}});
    expect(
      inherited.props.flows.kind === 'pairs' && inherited.props.flows.status?.candidateOverflow
    ).toBe(candidate);
    const nodes = new GPUKernelFlowSmoothing({
      ...props,
      id: 'pair-loss',
      flows: {...pairRows, status: {count, overflow, pairOverflow: candidate} as never}
    }).getCommandNodes(graph);
    const source = getParsedKernelSources(nodes).get('pair-loss-completeness')!;
    const complete = new Uint32Array(1);
    execute(source, {
      count: Uint32Array.of(1, 99),
      overflow: Uint32Array.of(0),
      pairOverflow: Uint32Array.of(1),
      complete
    });
    expect(Array.from(complete)).toEqual([0]);
    expect(source).not.toMatch(/var<storage, read_write> smoothed/);
  } finally {
    context.destroy();
    device.destroy();
  }
});

it('B3 captures inherited prism certificates and radiation normalization', () => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {input, output} = context;
  try {
    const certificate = input('certificate', 'uint32', [0]);
    const prism = Object.assign(
      Object.create({
        get forwardComplete() {
          return certificate;
        }
      }),
      {
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 1]),
          neighbors: input('neighbors', 'uint32', [1]),
          weights: input('edge-weights', 'float32', [1])
        },
        forwardCosts: input('forward', 'float32', [0, 1]),
        reverseCosts: input('reverse', 'float32', [1, 0]),
        budget: input('budget', 'float32', [2]),
        output: {
          edgeSlack: output('slack', 'float32', 1),
          edgeInPrism: output('edges', 'uint32', 1),
          nodeInPrism: output('nodes', 'uint32', 2)
        }
      }
    );
    expect(new GPUNetworkPotentialPathArea(prism).props.forwardComplete).toBe(certificate);
    const radiation = Object.assign(
      Object.create({
        get normalization() {
          return 'finite-size';
        }
      }),
      {
        positions: input('positions', 'float32x2', [0, 0, 1, 0]),
        populations: input('populations', 'float32', [5, 1]),
        outflows: input('outflows', 'float32', [12, 9]),
        output: {flows: output('flows', 'float32', 4)}
      }
    );
    expect(new GPURadiationFlowModel(radiation).props.normalization).toBe('finite-size');
  } finally {
    context.destroy();
    device.destroy();
  }
});

it('B5 S5 and earlier radiation overflow preserve the zero diagonal and finite-size outflow', () => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {graph, input, output} = context;
  try {
    const nodes = new GPURadiationFlowModel({
      positions: input('positions', 'float32x2', [0, 0, 1, 0]),
      populations: input('populations', 'float32', [2 ** 64, 1]),
      outflows: input('outflows', 'float32', [8, 6]),
      normalization: 'finite-size',
      output: {flows: output('flows', 'float32', 4)}
    }).getCommandNodes(graph);
    const source = getParsedKernelSources(nodes).get('radiation-flow-model-flows')!;
    const flows = new Float32Array(4);
    execute(source, {
      positions: Float32Array.of(0, 0, 1, 0),
      populations: Float32Array.of(2 ** 64, 1),
      outflows: Float32Array.of(8, 6),
      flows
    });
    expect(Array.from(flows)).toEqual([0, 8, 6, 0]);
    execute(source, {
      positions: Float32Array.of(0, 0, Infinity, 0),
      populations: Float32Array.of(1, 1),
      outflows: Float32Array.of(8, 6),
      invalid: Uint32Array.of(1),
      flows
    });
    expect(Array.from(flows)).toEqual([0, NaN, NaN, 0]);
    expect(source).not.toContain('for (var zone = 0u; zone < 2u;');
  } finally {
    context.destroy();
    device.destroy();
  }
});

it('B2 S1 certifies complete partial prefixes and reads each loss flag independently', () => {
  for (const lossName of ['pairOverflow', 'candidateOverflow', 'none', 'guaranteed'] as const) {
    for (const withStatus of [false, true]) {
      const device = createNullWebGPUDevice();
      const context = createOracleContext(device);
      const {graph, input, output} = context;
      const count = input('count', 'uint32', [2, 77]);
      const overflow = input('overflow', 'uint32', [0]);
      const candidate = input('candidate', 'uint32', [0]);
      try {
        const nodes = new GPUKernelFlowSmoothing({
          positions: input('positions', 'float32x2', [0, 0, 1, 0]),
          locationWeights: input('locations', 'float32', [1, 1]),
          parameters: input('parameters', 'float32', [0, 0.5, 0, 0]),
          flows: {
            kind: 'pairs',
            originZoneIds: input('origin', 'uint32', [0, 1]),
            destinationZoneIds: input('destination', 'uint32', [1, 0]),
            weights: input('weights', 'float32', [2, 8]),
            status: {
              count,
              overflow,
              ...(lossName === 'pairOverflow' ? {pairOverflow: candidate} : {}),
              ...(lossName === 'candidateOverflow' ? {candidateOverflow: candidate} : {}),
              ...(lossName === 'guaranteed' ? {completeness: 'no-candidate-loss' as const} : {})
            }
          },
          output: {
            smoothed: output('smoothed', 'float32', 4),
            ...(withStatus ? {status: {complete: output('complete', 'uint32', 1)}} : {})
          }
        }).getCommandNodes(graph);
        const source = getParsedKernelSources(nodes).get('kernel-flow-smoothing-completeness')!;
        const known = lossName !== 'none';
        for (const [prefix, finalLoss, candidateLoss] of [
          [2, 0, 0],
          [1, 0, 0],
          [2, 1, 0],
          [2, 0, 1]
        ]) {
          const complete = Uint32Array.of(99);
          const smoothed = Float32Array.of(0, 2, 8, 0);
          execute(source, {
            count: Uint32Array.of(prefix, 77),
            overflow: Uint32Array.of(finalLoss),
            candidateOverflow: Uint32Array.of(candidateLoss),
            pairOverflow: Uint32Array.of(candidateLoss),
            complete,
            smoothed
          });
          const expected =
            known &&
            !finalLoss &&
            (!(lossName === 'pairOverflow' || lossName === 'candidateOverflow') || !candidateLoss);
          if (withStatus) expect(Array.from(complete)).toEqual([Number(expected)]);
          else expect(Array.from(smoothed)).toEqual(expected ? [0, 2, 8, 0] : [NaN, NaN, NaN, NaN]);
        }
      } finally {
        context.destroy();
        device.destroy();
      }
    }
  }
});
