// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  expectVecchiaMarginalTotals,
  expectVecchiaSuppressedTotals
} from './vecchia-marginal-expectations';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeLikelihoodCPUInputs} from './likelihoods-operation-test-utils';
import {CPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {CPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {CPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {
  GPUVecchiaLogLikelihood,
  decodeGPUVecchiaLogLikelihoodResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {
  GPUVecchiaFisherScoring,
  decodeGPUVecchiaFisherScoringResult
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {GPUVecchiaSpaceTimeLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {computeVecchiaLogLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {computeVecchiaSpaceTimeLogLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {computeVecchiaFisherScoringOnCPU} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {getVecchiaLogLikelihoodResult} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {getVecchiaFisherEvaluation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getGPUVecchiaSpaceTimeLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';

const covariance = {model: 'exponential' as const, range: 3, sill: 1, nugget: 0.1};
const model = {type: 'metric' as const, joint: covariance, anisotropy: 1};
const createInput = () => ({
  positions: Float32Array.of(0, 0, 1, 0),
  observations: Float32Array.of(1, 0, 0, 1, 1, 0, 0, 1),
  neighborhoodOffsets: Uint32Array.of(0, 0, 0),
  neighborhoods: new Uint32Array(),
  parameters: getGPUVecchiaLogLikelihoodParameterValues(covariance),
  neighborhoodStatus: {candidateOverflow: Uint32Array.of(0)}
});

describe('Vecchia loss and readback regressions', () => {
  it('CPL-01 rejects complete certificates on every CPU evaluator, including contradictory flags', () => {
    for (const complete of [0, 1]) {
      for (const candidateOverflow of [undefined, Uint32Array.of(0), Uint32Array.of(1)]) {
        const input = {
          ...createInput(),
          id: 'certificate-fit',
          neighborhoodStatus: {complete: Uint32Array.of(complete), candidateOverflow}
        };
        for (const evaluate of [
          () => computeVecchiaLogLikelihoodOnCPU(input),
          () => computeVecchiaFisherScoringOnCPU(input),
          () =>
            computeVecchiaSpaceTimeLogLikelihoodOnCPU({
              ...input,
              times: Float32Array.of(0, 0),
              parameters: getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model)
            })
        ])
          expect(evaluate).toThrow(/certificate-fit.*complete.*completeness-aware/);
      }
    }
  });

  it('CPL-01 rejects unsupported certificates in all six operation constructors', () => {
    const output = {
      conditionals: 'conditionals',
      conditionalStatus: 'conditional-status',
      siteTerms: 'terms',
      siteStatus: 'sites',
      logLikelihood: 'likelihood',
      status: 'counts',
      totals: 'totals'
    };
    for (const candidateOverflow of [undefined, 'candidate-loss']) {
      const props = {
        id: 'certificate-fit',
        rowCount: 2,
        neighborCount: 1,
        positions: 'positions',
        observations: 'observations',
        neighborhoodOffsets: 'offsets',
        neighborhoods: 'neighbors',
        parameters: 'parameters',
        neighborhoodStatus: {complete: 'complete', candidateOverflow},
        output
      };
      const temporal = {
        ...props,
        times: 'times',
        temporalContext: {
          encoding: 'float32-relative' as const,
          unit: 'second' as const,
          epoch: 0n,
          axis: {kind: 'irregular' as const}
        }
      };
      for (const construct of [
        () => new CPUVecchiaLogLikelihoodOperation(props),
        () => new GPUVecchiaLogLikelihoodOperation(props),
        () => new CPUVecchiaFisherScoringOperation(props),
        () => new GPUVecchiaFisherScoringOperation(props),
        () => new CPUVecchiaSpaceTimeLogLikelihoodOperation(temporal),
        () => new GPUVecchiaSpaceTimeLogLikelihoodOperation(temporal)
      ])
        expect(construct).toThrow(/certificate-fit.*complete.*completeness-aware/);
    }
  });

  it('CPL-01 captures each declared certificate getter once on all three GPU contributors', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let next = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `certificate-${next++}`, format, length);
    const output = {
      conditionals: view('float32x4', 2),
      conditionalStatus: view('uint32', 2),
      siteTerms: view('float32', 2),
      siteStatus: view('uint32', 2),
      logLikelihood: view('float32', 2),
      totals: view('float32', 20),
      status: view('uint32', 3)
    };
    const common = {
      id: 'certificate-fit',
      positions: view('float32x2', 2),
      observations: view('float32x4', 2),
      neighborhoodOffsets: view('uint32', 3),
      neighborhoods: view('uint32', 1),
      parameters: view('float32', 8),
      neighborCount: 1,
      output
    };
    try {
      for (const kind of ['spatial', 'temporal', 'fisher'] as const) {
        for (const hasComplete of [false, true]) {
          const reads = {overflow: 0, candidateOverflow: 0, complete: 0};
          const loss = view('uint32', 1);
          const neighborhoodStatus = {
            get overflow() {
              reads.overflow++;
              return loss;
            },
            get candidateOverflow() {
              reads.candidateOverflow++;
              return loss;
            },
            get complete() {
              reads.complete++;
              return hasComplete ? loss : undefined;
            }
          };
          const construct = () =>
            kind === 'fisher'
              ? new GPUVecchiaFisherScoring({
                  ...common,
                  neighborhoodStatus,
                  output: {...output, siteTerms: view('float32', 20)}
                })
              : kind === 'temporal'
                ? new GPUVecchiaSpaceTimeLogLikelihood({
                    ...common,
                    neighborhoodStatus,
                    parameters: view('float32', 28),
                    times: view('float32', 2),
                    temporalContext: {
                      encoding: 'float32-relative',
                      unit: 'second',
                      epoch: 0n,
                      axis: {kind: 'irregular'}
                    }
                  })
                : new GPUVecchiaLogLikelihood({...common, neighborhoodStatus});
          if (hasComplete)
            expect(construct).toThrow(/certificate-fit.*complete.*completeness-aware/);
          else expect(construct).not.toThrow();
          expect(reads).toEqual({overflow: 1, candidateOverflow: 1, complete: 1});
        }
      }
    } finally {
      device.destroy();
    }
  });

  it('suppresses all three totals on candidate loss and recovers on reuse (CPL-01)', () => {
    const input = createInput();
    const evaluate = () => [
      computeVecchiaLogLikelihoodOnCPU(input).logLikelihood,
      computeVecchiaSpaceTimeLogLikelihoodOnCPU({
        ...input,
        times: Float32Array.of(0, 0),
        parameters: getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model)
      }).logLikelihood,
      computeVecchiaFisherScoringOnCPU(input).totals
    ];
    for (const [index, totals] of evaluate().entries())
      expectVecchiaMarginalTotals(totals, index === 2);
    input.neighborhoodStatus.candidateOverflow[0] = 1;
    expect(Array.from(computeVecchiaLogLikelihoodOnCPU(input).neighborhoodStatus)).toEqual([
      0, 0, 1
    ]);
    expect(Array.from(computeVecchiaLogLikelihoodOnCPU(input).status)).toEqual([0, 0, 2]);
    for (const [index, totals] of evaluate().entries())
      expectVecchiaSuppressedTotals(totals, index === 2);
    input.neighborhoodStatus.candidateOverflow[0] = 0;
    for (const [index, totals] of evaluate().entries())
      expectVecchiaMarginalTotals(totals, index === 2);
    const overflow = computeVecchiaLogLikelihoodOnCPU({
      ...input,
      neighborhoodStatus: {overflow: Uint32Array.of(1)}
    });
    expect(Array.from(overflow.neighborhoodStatus)).toEqual([0, 1, 0]);
    expect(Array.from(overflow.logLikelihood).every(Number.isNaN)).toBe(true);
    expect(
      Number.isFinite(
        computeVecchiaLogLikelihoodOnCPU({...input, neighborhoodStatus: undefined}).logLikelihood[0]
      )
    ).toBe(true);
  });

  it('decodes named typed ports and separates neighborhood loss from solve failures', () => {
    const status = Uint32Array.of(0, 0, 1);
    const logLikelihood = Float32Array.of(-4, 0.25);
    const result = decodeGPUVecchiaLogLikelihoodResult({
      logLikelihood: logLikelihood.buffer,
      status: status.buffer
    });
    expect(result.logLikelihood).toBe(-3.75);
    expect(result.status).toEqual({
      failedSiteCount: 0,
      excludedSiteCount: 0,
      usedSiteCount: 1,
      complete: true,
      overflow: false,
      candidateOverflow: false
    });
    const fisher = decodeGPUVecchiaFisherScoringResult({
      totals: new Float32Array(20),
      status,
      neighborhoodStatus: Uint32Array.of(0, 0, 1)
    });
    expect(fisher.status.candidateOverflow).toBe(true);
    expect(fisher.status.failedSiteCount).toBe(0);
    expect(fisher.score.every(Number.isNaN)).toBe(true);
    expect(
      decodeGPUVecchiaLogLikelihoodResult({logLikelihood: Float32Array.of(NaN, NaN), status}).status
        .complete
    ).toBe(false);
    expect(() =>
      decodeGPUVecchiaLogLikelihoodResult({logLikelihood, status: new Uint8Array(11)})
    ).toThrow(/status/);
    expect(() => decodeGPUVecchiaFisherScoringResult({totals: new Uint8Array(79), status})).toThrow(
      /totals/
    );
  });

  it('carries loss through all CPU operations and GPU lowering with captured port maps', async () => {
    const output = {
      conditionals: 'conditional-result',
      conditionalStatus: 'conditional-state',
      siteTerms: 'terms',
      siteStatus: 'sites',
      logLikelihood: 'likelihood',
      status: 'counts',
      neighborhoodStatus: 'loss-evidence',
      totals: 'fisher-totals'
    };
    const common = {
      id: 'custom-fit',
      rowCount: 2,
      neighborCount: 1,
      positions: 'coordinates',
      observations: 'measurements',
      neighborhoodOffsets: 'starts',
      neighborhoods: 'earlier-rows',
      parameters: 'covariance',
      neighborhoodStatus: {candidateOverflow: 'lost-candidates'},
      output
    };
    const temporalContext = {
      encoding: 'float32-relative' as const,
      unit: 'second' as const,
      epoch: 0n,
      axis: {kind: 'irregular' as const}
    };
    for (const kind of ['spatial', 'temporal', 'fisher'] as const) {
      const props = {
        ...common,
        neighborhoodStatus: {...common.neighborhoodStatus},
        output:
          kind === 'fisher'
            ? {
                siteTerms: output.siteTerms,
                siteStatus: output.siteStatus,
                totals: output.totals,
                status: output.status,
                neighborhoodStatus: output.neighborhoodStatus
              }
            : {
                conditionals: output.conditionals,
                conditionalStatus: output.conditionalStatus,
                siteTerms: output.siteTerms,
                siteStatus: output.siteStatus,
                logLikelihood: output.logLikelihood,
                status: output.status,
                neighborhoodStatus: output.neighborhoodStatus
              }
      };
      const temporal = {...props, times: 'relative-times', temporalContext};
      // Each branch retains its precise output props type for construction.
      const operation =
        kind === 'fisher'
          ? new CPUVecchiaFisherScoringOperation({
              ...props,
              output: {
                siteTerms: output.siteTerms,
                siteStatus: output.siteStatus,
                totals: output.totals,
                status: output.status,
                neighborhoodStatus: output.neighborhoodStatus
              }
            })
          : kind === 'temporal'
            ? new CPUVecchiaSpaceTimeLogLikelihoodOperation({...temporal, output})
            : new CPUVecchiaLogLikelihoodOperation({...props, output});
      const gpu =
        kind === 'fisher'
          ? new GPUVecchiaFisherScoringOperation({
              ...props,
              output: {
                siteTerms: output.siteTerms,
                siteStatus: output.siteStatus,
                totals: output.totals,
                status: output.status,
                neighborhoodStatus: output.neighborhoodStatus
              }
            })
          : kind === 'temporal'
            ? new GPUVecchiaSpaceTimeLogLikelihoodOperation({...temporal, output})
            : new GPUVecchiaLogLikelihoodOperation({...props, output});
      props.neighborhoodStatus.candidateOverflow = 'replaced';
      expect(gpu.inputs['neighborhoodCandidateOverflow']).toBe('lost-candidates');
      const input = createInput();
      const parameters =
        kind === 'temporal'
          ? getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model)
          : input.parameters;
      const inputs = makeLikelihoodCPUInputs({
        positions: {format: 'float32x2', values: input.positions},
        observations: {format: 'float32x4', values: input.observations},
        neighborhoodOffsets: {format: 'uint32', values: input.neighborhoodOffsets},
        neighborhoods: {format: 'uint32', values: input.neighborhoods},
        parameters: {format: 'float32', values: parameters},
        times: {format: 'float32', values: new Float32Array(2)},
        neighborhoodCandidateOverflow: {format: 'uint32', values: Uint32Array.of(1)}
      });
      const result = await operation.execute({inputs});
      expect(Array.from(result['loss-evidence'] as Uint32Array)).toEqual([0, 0, 1]);
      expect(
        Array.from(
          result[kind === 'fisher' ? 'fisher-totals' : 'likelihood'] as Float32Array
        ).every(Number.isNaN)
      ).toBe(true);
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      let serial = 0;
      const views = Object.fromEntries(
        Object.entries(inputs).map(([name, inputVector]) => [
          name,
          createTransientView(
            graph,
            `input-${serial++}`,
            inputVector.format,
            Math.max(inputVector.length, 1)
          )
        ])
      );
      for (const [name, port] of Object.entries(gpu.outputs))
        views[name] = createTransientView(graph, `output-${serial++}`, port.format, port.length);
      const contributor = gpu.createContributor(
        (name => views[name]) as Parameters<typeof gpu.createContributor>[0]
      ) as GPUVecchiaLogLikelihood;
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.some(node => node.id.endsWith('-neighborhood-status'))).toBe(true);
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(sources.length).toBeGreaterThan(0);
      graph.device.destroy();
    }
  });

  it('rejects wrong declared port formats in all three constructors (VAL-01)', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `format-${serial++}`, format, length);
    const common = {
      id: 'typed-fit',
      neighborCount: 1,
      positions: view('float32x2', 2),
      observations: view('float32x4', 2),
      neighborhoodOffsets: view('uint32', 3),
      neighborhoods: view('uint32', 1),
      parameters: view('float32', 8)
    };
    const output = {
      conditionals: view('float32x4', 2),
      conditionalStatus: view('uint32', 2),
      siteTerms: view('float32', 2),
      siteStatus: view('uint32', 2),
      logLikelihood: view('float32', 2),
      status: view('uint32', 3)
    };
    const wrong = view('uint32', 2) as unknown as typeof common.observations;
    expect(() => new GPUVecchiaLogLikelihood({...common, observations: wrong, output})).toThrow(
      /typed-fit.*observations.*float32x4.*uint32/
    );
    expect(
      () =>
        new GPUVecchiaSpaceTimeLogLikelihood({
          ...common,
          observations: wrong,
          times: view('float32', 2),
          temporalContext: {
            encoding: 'float32-relative',
            unit: 'second',
            epoch: 0n,
            axis: {kind: 'irregular'}
          },
          output
        })
    ).toThrow(/typed-fit.*observations.*float32x4.*uint32/);
    expect(
      () =>
        new GPUVecchiaFisherScoring({
          ...common,
          observations: wrong,
          output: {...output, totals: view('float32', 20)}
        })
    ).toThrow(/typed-fit.*observations.*float32x4.*uint32/);
    graph.device.destroy();
  });

  it('reads byte ranges as float32 totals and uint32 status (EGR-01)', () => {
    const totals = Float32Array.of(77, -4, 0.25, 77);
    const status = Uint32Array.of(77, 0, 0, 1, 77);
    const result = getVecchiaLogLikelihoodResult(
      new Uint8Array(totals.buffer, 4, 8),
      new Uint8Array(status.buffer, 4, 12)
    );
    expect(result.logLikelihood).toBe(-3.75);
    expect(result.usedSiteCount).toBe(1);
    const unaligned = new Uint8Array(10);
    unaligned.set(new Uint8Array(totals.buffer, 4, 8), 1);
    expect(
      getVecchiaLogLikelihoodResult(unaligned.subarray(1, 9), status.subarray(1, 4)).logLikelihood
    ).toBe(-3.75);
    const fisher = new Float32Array(22).fill(0.25);
    fisher[1] = -4;
    expect(
      getVecchiaFisherEvaluation(new Uint8Array(fisher.buffer, 4, 80), status.subarray(1, 4))
        .logLikelihood
    ).toBe(-3.75);
  });

  it('rejects short totals, missing status and partial elements (EGR-06)', () => {
    expect(() => getVecchiaLogLikelihoodResult([10, 0.5], [])).toThrow(/status/);
    expect(() => getVecchiaLogLikelihoodResult([10], [0, 0, 1])).toThrow(/logLikelihood/);
    expect(() => getVecchiaFisherEvaluation([10, 0.5], [0, 0, 1])).toThrow(/totals/);
    expect(() => getVecchiaLogLikelihoodResult(new Uint8Array(9), [0, 0, 1])).toThrow(
      /logLikelihood/
    );
  });

  it('names the caller and malformed port in CPU shape errors (PTS-137)', () => {
    expect(() =>
      computeVecchiaLogLikelihoodOnCPU({
        ...createInput(),
        id: 'my-fit',
        observations: new Float32Array(4)
      })
    ).toThrow(/my-fit.*observations.*8.*4/);
  });

  it('preserves trailing words and leaves failed destinations byte-identical (UPT-02)', () => {
    for (const [length, pack, invalid] of [
      [
        8,
        (target: Float32Array) => getGPUVecchiaLogLikelihoodParameterValues(covariance, target),
        (target: Float32Array) =>
          getGPUVecchiaLogLikelihoodParameterValues(
            {...covariance, model: 'matern-3/2', nugget: 0},
            target
          )
      ],
      [
        28,
        (target: Float32Array) => getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target),
        (target: Float32Array) =>
          getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
            {...model, joint: {...covariance, model: 'matern-3/2', nugget: 0}},
            target
          )
      ]
    ] as const) {
      const target = new Float32Array(length + 2).fill(77);
      pack(target);
      expect(Array.from(target.subarray(length))).toEqual([77, 77]);
      const before = new Uint8Array(target.buffer).slice();
      expect(() => invalid(target)).toThrow(/nugget/);
      expect(new Uint8Array(target.buffer)).toEqual(before);
    }
  });
});
