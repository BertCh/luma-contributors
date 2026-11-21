// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {buildSync} from 'esbuild';
import {createRequire} from 'node:module';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {GPUSpatialRegressionDiagnostics} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {snapshotRegressionConstructionProps} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-construction-snapshot';

for (const location of ['whole', 'root', 'later-child'] as const) {
  for (const kind of ['traces', 'collinearity', 'diagnostics', 'hac'] as const)
    for (const withFlags of [false, true]) {
      if ((kind === 'diagnostics' || kind === 'hac') && location !== 'whole') continue;
      it(`W5-27 ${kind} preserves ${location} positive certificates with flags=${withFlags} and 1/0/1 recovery`, () => {
        const device = createNullWebGPUDevice();
        try {
          const graph = new GPUCommandGraph(device);
          let serial = 0;
          const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
            createTransientView(graph, `certificate-${serial++}`, format, length);
          const complete = view('uint32', 1);
          const clear = view('uint32', 1);
          const status = withFlags
            ? {complete, overflow: clear, candidateOverflow: clear}
            : {complete};
          const csr = (rows: number, withStatus: boolean) => ({
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', rows),
            weights: view('float32', rows),
            ...(withStatus ? {status} : {})
          });
          const whole = csr(4, location === 'whole');
          const weights =
            location === 'whole'
              ? whole
              : {
                  rowCount: 4,
                  ...(location === 'root' ? {status} : {}),
                  chunks: [0, 2].map(rowStart => ({
                    rowStart,
                    rowCount: 2,
                    weights: csr(2, location === 'later-child' && rowStart === 2)
                  }))
                };
          const contributor =
            kind === 'traces'
              ? new GPUSpatialWeightsTraces({
                  weights,
                  maximumPower: 2,
                  mode: 'exact',
                  output: {
                    powerTraces: view('float32', 2),
                    powerSums: view('float32', 2),
                    infinityNorm: view('float32', 1)
                  }
                })
              : kind === 'collinearity'
                ? new GPULocalCollinearity({
                    weights,
                    predictors: view('float32', 4),
                    predictorCount: 1,
                    output: {
                      varianceInflationFactors: view('float32', 4),
                      conditionNumbers: view('float32', 4),
                      status: view('uint32', 4)
                    }
                  })
                : kind === 'diagnostics'
                  ? new GPUSpatialRegressionDiagnostics({
                      weights: whole,
                      predictors: view('float32', 4),
                      response: view('float32', 4),
                      residuals: view('float32', 4),
                      predictorCount: 1,
                      output: {
                        tests: view('float32', 18),
                        summary: view('float32', 13),
                        status: {code: view('uint32', 1)}
                      }
                    })
                  : new GPUInstrumentalVariablesCovariance({
                      kernel: whole,
                      method: 'HAC',
                      design: view('float32', 4),
                      instruments: view('float32', 4),
                      residuals: view('float32', 4),
                      coefficientCount: 1,
                      instrumentCount: 1,
                      output: {covariance: view('float32', 4), status: {code: view('uint32', 1)}}
                    });
          const allNodes = contributor.getCommandNodes(graph);
          const nodes = allNodes.filter(
            node => node.id.includes('-weights-loss-') || node.id.includes('-weights-complete-')
          );
          expect(nodes.some(node => node.id.includes('-weights-complete-'))).toBe(true);
          const memory = createCPUGraphMemory();
          memory.write(clear, Uint32Array.of(0));
          const fold = nodes.find(node => node.id.includes('-weights-complete-'))!;
          const loss = fold
            .resources!.flatMap(resource =>
              'buffer' in resource ? [resource.buffer as GraphDataView] : []
            )
            .find(resource => resource !== complete)!;
          for (const value of [1, 0, 1]) {
            memory.write(complete, Uint32Array.of(value));
            executeCommandNodesOnCPU(nodes, memory);
            expect(Array.from(memory.readUint32(loss))).toEqual([1 - value]);
            if (contributor instanceof GPULocalCollinearity) {
              memory.write(contributor.props.output.status, new Uint32Array(4));
              executeCommandNodesOnCPU(
                allNodes.filter(node => node.id.endsWith('-incomplete-status')),
                memory
              );
              expect(Array.from(memory.readUint32(contributor.props.output.status))).toEqual(
                new Array(4).fill(1 - value)
              );
            }
          }
        } finally {
          device.destroy();
        }
      });
    }
}

it('W5-28 count-only foreign view rejects before mutation and permits a same-ID retry', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const foreign = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `count-${serial++}`, format, length);
    const props = {
      id: 'count-retry',
      predictorCount: 1,
      entityCount: 4,
      periodCount: 2,
      maximumPower: 2,
      traceMode: 'exact' as const,
      predictors: view('float32', 8),
      response: view('float32', 8),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 4),
        weights: view('float32', 4),
        status: {count: createTransientView(foreign, 'foreign-count', 'uint32', 1)}
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      },
      diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
    };
    expect(() => graph.add(new GPUSpatialPanelErrorMaximumLikelihood(props))).toThrow(/graph/);
    expect(
      () =>
        new GPUSpatialPanelErrorMaximumLikelihood({
          ...props,
          weights: {...props.weights, status: {count: props.output.status.code}}
        })
    ).toThrow(/share buffers|overlap|alias/);
    const packed = view('uint32', 65);
    const packedOutput = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    const packedCount = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    expect(
      () =>
        new GPUSpatialPanelErrorMaximumLikelihood({
          ...props,
          output: {...props.output, status: {code: packedOutput}},
          weights: {...props.weights, status: {count: packedCount}}
        })
    ).toThrow(/share buffers/);
    const oversized = view('uint32', Math.floor(device.limits.maxStorageBufferBindingSize / 4) + 1);
    expect(() =>
      graph.add(
        new GPUSpatialPanelErrorMaximumLikelihood({
          ...props,
          weights: {...props.weights, status: {count: oversized}}
        })
      )
    ).toThrow(/one uint32 row/);
    graph.add(
      new GPUSpatialPanelErrorMaximumLikelihood({
        ...props,
        weights: {...props.weights, status: {count: view('uint32', 1)}}
      })
    );
  } finally {
    device.destroy();
  }
});

it('W5-29 preserves ESM/CJS scalar identities, named strings and code maps', () => {
  const require = createRequire(import.meta.url);
  const commonjs = require('@luma.gl/graph/gpu-core') as typeof import('@luma.gl/graph/gpu-core');
  // Compile this one helper and its local dependencies in memory, with published packages
  // external, so its graph import genuinely selects the CommonJS package condition.
  const compiled = buildSync({
    entryPoints: [
      new URL(
        '../../../src/gpu-spatial-analysis/spatial-regression/regression-construction-snapshot.ts',
        import.meta.url
      ).pathname
    ],
    bundle: true,
    packages: 'external',
    platform: 'node',
    format: 'cjs',
    write: false,
    logLevel: 'silent'
  });
  const commonjsModule = {
    exports:
      {} as typeof import('../../../src/gpu-spatial-analysis/spatial-regression/regression-construction-snapshot')
  };
  new Function('require', 'module', 'exports', compiled.outputFiles[0].text)(
    require,
    commonjsModule,
    commonjsModule.exports
  );
  const device = createNullWebGPUDevice();
  try {
    for (const snapshot of [
      snapshotRegressionConstructionProps,
      commonjsModule.exports.snapshotRegressionConstructionProps
    ])
      for (const api of [{GPUCommandGraph, createTransientView}, commonjs]) {
        const graph = new api.GPUCommandGraph(device);
        let serial = 0;
        const view = (name: string) =>
          api.createTransientView(graph, `${name}-${serial++}`, 'uint32', 1);
        const scalar = view('scalar');
        for (const [status, statusShape] of [
          [scalar, 'scalar'],
          [{code: scalar}, 'map'],
          ['named', 'scalar']
        ] as const) {
          const captured = snapshot(
            {
              weights: {
                offsets: view('offsets'),
                neighbors: view('neighbors'),
                weights: api.createTransientView(graph, `weights-${serial++}`, 'float32', 1)
              },
              output: {status}
            },
            ['weights', 'output'],
            ['status'],
            'cross-condition',
            ['code'],
            statusShape
          );
          if (typeof status === 'object' && 'code' in status)
            expect(captured.output.status).toEqual({code: scalar});
          else expect(captured.output.status).toBe(status);
        }
      }
  } finally {
    device.destroy();
  }
});

it('CCK-01 / W5-33 declared node estimates cover the parallel reference and centering passes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `budget-${serial++}`, format, length);
    const certificate = () => ({complete: view('uint32', 1), overflow: view('uint32', 1)});
    const weights = {
      offsets: view('uint32', 65),
      neighbors: view('uint32', 4096),
      weights: view('float32', 4096),
      status: {...certificate(), requiredCount: view('uint32', 1), count: view('uint32', 1)}
    };
    const diagnostics = new GPUSpatialRegressionDiagnostics({
      weights,
      predictors: view('float32', 64),
      response: view('float32', 64),
      residuals: view('float32', 64),
      predictorCount: 1,
      restandardizeRows: true,
      output: {
        tests: view('float32', 18),
        summary: view('float32', 13),
        status: {code: view('uint32', 1)}
      }
    });
    const diagnosticsNodes = diagnostics.getCommandNodes(graph);
    expect(diagnosticsNodes.length).toBeLessThanOrEqual(diagnostics.estimatedCommandNodeCount);
    // The selected-row reference is reduced over CSR slots by parallel tiles, not one invocation.
    const reference = diagnosticsNodes.filter(node => node.id.includes('observed-origin'));
    expect(reference.length).toBeGreaterThanOrEqual(4);
    for (const kernel of [false, true]) {
      const covariance = new GPUInstrumentalVariablesCovariance({
        id: `budget-covariance-${kernel}`,
        ...(kernel ? {kernel: weights} : {}),
        method: kernel ? 'HAC' : 'HC0',
        design: view('float32', 128),
        instruments: view('float32', 192),
        residuals: view('float32', 64),
        coefficientCount: 2,
        instrumentCount: 3,
        output: {covariance: view('float32', 9), status: {code: view('uint32', 1)}}
      });
      const nodes = covariance.getCommandNodes(graph);
      expect(nodes.some(node => node.id.endsWith('-centering-zero'))).toBe(true);
      expect(nodes.length).toBeLessThanOrEqual(covariance.estimatedCommandNodeCount);
    }
  } finally {
    device.destroy();
  }
});
