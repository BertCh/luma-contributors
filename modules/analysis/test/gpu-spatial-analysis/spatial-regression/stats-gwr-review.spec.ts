// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {GPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {
  runOperationGraph,
  makeCPUVector,
  getCPUVectorRows
} from '../../utils/mixed-execution-test-utils';
import {GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import type {MultiscaleGeographicallyWeightedRegressionOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';

it('rejects output float32 overflow on both backends instead of publishing an infinite limit iterate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    for (const overflow of ['coefficient', 'prediction'] as const) {
      const rowCount = overflow === 'coefficient' ? 2 : 3;
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(2 * rowCount)},
          predictors: {
            format: 'float32',
            values:
              overflow === 'coefficient'
                ? Float32Array.of(-1e-12, 1e-12)
                : Float32Array.of(-0.5, 0, 1)
          },
          response: {
            format: 'float32',
            values:
              overflow === 'coefficient'
                ? Float32Array.of(-1e28, 1e28)
                : Float32Array.of(-1e38, 1e38, 3.1e38)
          },
          parameters: {format: 'float32', values: Float32Array.of(1e-6, 1, 1)}
        },
        operations: [
          new Operation({
            rowCount,
            predictorCount: 1,
            kernel: 'gaussian',
            maximumIterations: 1,
            positions: 'positions',
            predictors: 'predictors',
            response: 'response',
            parameters: 'parameters',
            inference: {
              output: {
                standardErrors: 'standardErrors',
                effectiveParameters: 'effectiveParameters',
                summary: 'summary',
                status: 'inferenceStatus'
              }
            },
            output: {
              coefficients: 'coefficients',
              fitted: 'fitted',
              status: {codes: 'codes', iterationCount: 'iterationCount'}
            }
          })
        ],
        outputs: [
          'coefficients',
          'fitted',
          'codes',
          'iterationCount',
          'standardErrors',
          'inferenceStatus'
        ]
      });
      expect(Array.from(result['codes'])).toEqual(new Array(rowCount).fill(4));
      for (const port of ['coefficients', 'fitted', 'standardErrors'])
        expect(Array.from(result[port]).every(Number.isNaN)).toBe(true);
      expect(result['inferenceStatus'][0]).toBe(1);
      expect(result['iterationCount'][0]).toBe(1);
    }
  }
});

it('preserves finite search selections, backfit convergence and independent component-search exhaustion on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    for (const searchMethod of ['golden-section', 'finite-grid'] as const) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)},
          predictors: {format: 'float32', values: Float32Array.of(1, -1, 1, -1)},
          response: {format: 'float32', values: Float32Array.of(1, 6, 2, 8)},
          parameters: {format: 'float32', values: Float32Array.of(100, 1, 1)},
          candidates: {format: 'float32', values: Float32Array.of(1, 4, 1, 4)}
        },
        operations: [
          new Operation({
            rowCount: 4,
            predictorCount: 1,
            kernel: 'gaussian',
            maximumIterations: 1,
            positions: 'positions',
            predictors: 'predictors',
            response: 'response',
            parameters: 'parameters',
            bandwidthCandidates: 'candidates',
            bandwidthCandidateCount: 2,
            bandwidthSelection: {
              searchMethod,
              maximumSearchIterations: searchMethod === 'golden-section' ? 1 : undefined
            },
            inference: {
              output: {
                standardErrors: 'standardErrors',
                effectiveParameters: 'effectiveParameters',
                summary: 'summary',
                status: 'inferenceStatus'
              }
            },
            output: {
              coefficients: 'coefficients',
              fitted: 'fitted',
              selectedBandwidths: 'bandwidths',
              status: {
                codes: 'codes',
                iterationCount: 'iterationCount',
                componentSearchIterationLimitReached: 'searchLimits'
              }
            }
          })
        ],
        outputs: [
          'coefficients',
          'fitted',
          'bandwidths',
          'codes',
          'iterationCount',
          'inferenceStatus',
          'searchLimits'
        ]
      });
      expect(Array.from(result['codes'])).toEqual([0, 0, 0, 0]);
      expect(result['inferenceStatus'][0]).toBe(0);
      expect(Array.from(result['bandwidths'])).toEqual([4, 4]);
      expect(Array.from(result['searchLimits'])).toEqual(
        searchMethod === 'golden-section' ? [1, 1] : [0, 0]
      );
      for (const port of ['coefficients', 'fitted'])
        expect(Array.from(result[port]).every(Number.isFinite)).toBe(true);
    }
  }
});

it('keeps one validated sweep when the same operation executes twice after caller mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    const props: MultiscaleGeographicallyWeightedRegressionOperationProps = {
      rowCount: 3,
      predictorCount: 1,
      kernel: 'bisquare',
      maximumIterations: 1,
      positions: 'positions',
      predictors: 'predictors',
      response: 'response',
      parameters: 'parameters',
      output: {
        coefficients: 'coefficients',
        fitted: 'fitted',
        status: {codes: 'codes', iterationCount: 'iterationCount'}
      }
    };
    const operation = new Operation(props);
    props.maximumIterations = 2;
    const graph = new GPUGraph();
    const inputs = {
      positions: makeCPUVector('float32x2', Float32Array.of(0, 0, 1, 0, 2, 0)),
      predictors: makeCPUVector('float32', Float32Array.of(1, -1, 1)),
      response: makeCPUVector('float32', Float32Array.of(2, 4, 8)),
      parameters: makeCPUVector('float32', Float32Array.of(1e-6, 1, 1))
    };
    for (const name of ['positions', 'predictors', 'response', 'parameters'] as const)
      graph.addInput(
        name,
        gpuGraph.vector(name === 'positions' ? 'float32x2' : 'float32', {
          length: inputs[name].length
        })
      );
    graph.add(operation);
    const outputs = ['coefficients', 'fitted', 'codes', 'iterationCount'];
    const compiled = graph.compile(device, {
      outputs,
      outputLocations: Object.fromEntries(outputs.map(name => [name, 'cpu' as const])),
      lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
    });
    try {
      for (let execution = 0; execution < 2; execution++) {
        const result: Record<string, unknown> = await compiled.run({inputs});
        const rows = (name: string) => {
          const vector = result[name];
          if (!(vector instanceof CPUVector)) throw new Error(`${name} must be a CPU output`);
          return getCPUVectorRows(vector);
        };
        expect(Array.from(rows('codes'))).toEqual([3, 3, 3]);
        expect(Array.from(rows('iterationCount'))).toEqual([1]);
        expect(Array.from(rows('coefficients')).every(Number.isFinite)).toBe(true);
        expect(Array.from(rows('fitted')).every(Number.isFinite)).toBe(true);
        props.maximumIterations = 3;
      }
    } finally {
      compiled.destroy();
      if ('destroy' in operation) operation.destroy();
    }
  }
});

it('clears component search-limit flags when a second submission reaches search tolerance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    const graph = new GPUGraph();
    const inputs = {
      positions: makeCPUVector('float32x2', Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)),
      predictors: makeCPUVector('float32', Float32Array.of(1, -1, 1, -1)),
      response: makeCPUVector('float32', Float32Array.of(1, 6, 2, 8)),
      parameters: makeCPUVector('float32', Float32Array.of(100, 1, 1)),
      candidates: makeCPUVector('float32', Float32Array.of(1, 4, 1, 4))
    };
    for (const name of ['positions', 'predictors', 'response', 'parameters', 'candidates'] as const)
      graph.addInput(
        name,
        gpuGraph.vector(name === 'positions' ? 'float32x2' : 'float32', {
          length: inputs[name].length
        })
      );
    const operation = new Operation({
      rowCount: 4,
      predictorCount: 1,
      kernel: 'gaussian',
      maximumIterations: 1,
      positions: 'positions',
      predictors: 'predictors',
      response: 'response',
      parameters: 'parameters',
      bandwidthCandidates: 'candidates',
      bandwidthCandidateCount: 2,
      bandwidthSelection: {searchMethod: 'golden-section', maximumSearchIterations: 1},
      inference: {
        output: {
          standardErrors: 'standardErrors',
          effectiveParameters: 'effectiveParameters',
          summary: 'summary',
          status: 'inferenceStatus'
        }
      },
      output: {
        coefficients: 'coefficients',
        fitted: 'fitted',
        status: {
          codes: 'codes',
          iterationCount: 'iterationCount',
          componentSearchIterationLimitReached: 'searchLimits'
        }
      }
    });
    graph.add(operation);
    const outputs = ['codes', 'fitted', 'searchLimits'];
    const compiled = graph.compile(device, {
      outputs,
      outputLocations: Object.fromEntries(outputs.map(name => [name, 'cpu' as const])),
      lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
    });
    try {
      for (let execution = 0; execution < 2; execution++) {
        const result: Record<string, unknown> = await compiled.run({inputs});
        const flags = result['searchLimits'];
        const codes = result['codes'];
        const fitted = result['fitted'];
        if (
          !(flags instanceof CPUVector) ||
          !(codes instanceof CPUVector) ||
          !(fitted instanceof CPUVector)
        )
          throw new Error('Expected CPU outputs');
        expect(Array.from(getCPUVectorRows(flags))).toEqual(execution === 0 ? [1, 1] : [0, 0]);
        expect(Array.from(getCPUVectorRows(codes))).toEqual([0, 0, 0, 0]);
        expect(Array.from(getCPUVectorRows(fitted)).every(Number.isFinite)).toBe(true);
        inputs.response = makeCPUVector('float32', new Float32Array(4).fill(5));
      }
    } finally {
      compiled.destroy();
      if ('destroy' in operation) operation.destroy();
    }
  }
});
import {requireTestDevice} from '../../utils/require-test-device';

for (const fullInference of [false, true]) {
  for (const invalidBandwidth of [false, true]) {
    it(`MGWR publishes the finite limit iterate and withholds unsuccessful inference: full=${fullInference}, invalid=${invalidBandwidth}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const Operation of [
        GPUMultiscaleGeographicallyWeightedRegressionOperation,
        CPUMultiscaleGeographicallyWeightedRegressionOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)},
            predictors: {format: 'float32', values: Float32Array.of(1, -1, 1, 2)},
            response: {format: 'float32', values: Float32Array.of(2, 4, 8, 100)},
            parameters: {
              format: 'float32',
              values: Float32Array.of(1e-6, invalidBandwidth ? 0 : 1, 1)
            },
            mask: {format: 'uint32', values: Uint32Array.of(1, 1, 1, 0)},
            noiseVariance: {format: 'float32', values: Float32Array.of(1)}
          },
          operations: [
            new Operation({
              rowCount: 4,
              predictorCount: 1,
              kernel: 'bisquare',
              maximumIterations: 1,
              positions: 'positions',
              predictors: 'predictors',
              response: 'response',
              parameters: 'parameters',
              mask: 'mask',
              noiseVariance: 'noiseVariance',
              inference: fullInference
                ? {
                    output: {
                      standardErrors: 'standardErrors',
                      effectiveParameters: 'effectiveParameters',
                      summary: 'summary',
                      status: 'inferenceStatus'
                    }
                  }
                : undefined,
              output: {
                coefficients: 'coefficients',
                fitted: 'fitted',
                selectedBandwidths: 'selectedBandwidths',
                conditionalStandardErrors: 'conditionalStandardErrors',
                status: {codes: 'codes', iterationCount: 'iterationCount'}
              }
            })
          ],
          outputs: [
            'coefficients',
            'fitted',
            'codes',
            'iterationCount',
            'selectedBandwidths',
            'conditionalStandardErrors',
            ...(fullInference
              ? ['standardErrors', 'effectiveParameters', 'summary', 'inferenceStatus']
              : [])
          ]
        });
        expect(Array.from(result['codes'])).toEqual(invalidBandwidth ? [2, 2, 2, 1] : [3, 3, 3, 1]);
        expect(Array.from(result['iterationCount'])).toEqual([1]);
        expect(Array.from(result['selectedBandwidths'])).toEqual([invalidBandwidth ? 0 : 1, 1]);
        // h=1 retains self only. First-sweep OLS [4.5,0.5] yields [y-x/2,1/2];
        // full inference starts from zero, yielding [y,0]. Both predict y exactly.
        const expectedCoefficients = fullInference
          ? [2, 0, 4, 0, 8, 0]
          : [1.5, 0.5, 4.5, 0.5, 7.5, 0.5];
        for (const [port, expected] of [
          ['coefficients', expectedCoefficients],
          ['fitted', [2, 4, 8]]
        ] as const) {
          for (let element = 0; element < result[port].length; element++) {
            const actual = result[port][element];
            if (invalidBandwidth || element >= expected.length) expect(actual).toBeNaN();
            else {
              expect(Number.isFinite(actual)).toBe(true);
              expect(Math.abs(actual - expected[element])).toBeLessThan(1e-5);
            }
          }
        }
        expect(Array.from(result['conditionalStandardErrors']).every(Number.isNaN)).toBe(true);
        if (fullInference) {
          expect(result['inferenceStatus'][0]).toBe(1);
          for (const port of ['standardErrors', 'effectiveParameters', 'summary'])
            expect(Array.from(result[port]).every(Number.isNaN), port).toBe(true);
        }
      }
    });
  }
}
