// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  validateGWRBindings,
  validateGWROutputs
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-scalability-utils';
import {validateRegressionAllocations} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-likelihood-allocations';
import {GPUSpatialLagImpacts} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {GPUAreaOfApplicability} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-area-of-applicability';
import {GPUConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-conformal-prediction-intervals';
import {GPUGeoConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GPUSpatialCrossValidatedOLS} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';

type ViewFactory = <Format extends 'float32' | 'uint32' | 'float32x2'>(
  format: Format,
  length: number
) => GraphDataView<Format>;
type Factory = (view: ViewFactory) => GPUCommandNodeProducer;
function weights(view: ViewFactory, rows = 8) {
  return {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows),
    weights: view('float32', rows)
  };
}
function fitOutput(view: ViewFactory, rows = 8, converged = false) {
  return {
    table: view('float32', 12),
    summary: view('float32', 22),
    residuals: view('float32', rows),
    status: {code: view('uint32', 1), ...(converged ? {converged: view('uint32', 1)} : {})}
  };
}
function folds(view: ViewFactory, includeBlocks = true) {
  const mask = view('uint32', 8);
  return {
    positions: view('float32x2', 8),
    blocks: includeBlocks ? view('uint32', 8) : mask,
    mask,
    foldCount: 2,
    output: {
      folds: view('uint32', 8),
      training: view('uint32', 16),
      testing: view('uint32', 16),
      counts: view('uint32', 6),
      status: view('uint32', 2)
    }
  };
}
function conformal(view: ViewFactory) {
  return {
    calibration: {
      observations: view('float32', 8),
      predictions: view('float32', 8),
      scales: view('float32', 8),
      mask: view('uint32', 8)
    },
    query: {predictions: view('float32', 4), scales: view('float32', 4)},
    parameters: view('float32', 4),
    output: {
      intervals: view('float32', 8),
      status: view('uint32', 4),
      calibrationCount: view('uint32', 1)
    }
  };
}
const factories: [string, Factory][] = [
  [
    'lag impacts',
    view =>
      new GPUSpatialLagImpacts({
        coefficients: view('float32', 12),
        coefficientStride: 4,
        rhoIndex: 2,
        predictorCount: 1,
        rowCount: 8,
        maximumPower: 2,
        traces: {
          powerTraces: view('float32', 2),
          powerSums: view('float32', 2),
          infinityNorm: view('float32', 1)
        },
        status: view('uint32', 1),
        output: {impacts: view('float32', 3)}
      })
  ],
  ...(['kelejian-prucha', 'homoskedastic'] as const).map(
    estimator =>
      [
        `error GM ${estimator}`,
        (view: ViewFactory) =>
          new GPUSpatialErrorGM({
            weights: weights(view),
            predictors: view('float32', 8),
            response: view('float32', 8),
            predictorCount: 1,
            estimator,
            ...(estimator === 'homoskedastic'
              ? {
                  traceMode: 'stochastic' as const,
                  traceProbeCount: 2,
                  traceParameters: view('uint32', 1)
                }
              : {}),
            output: fitOutput(view)
          })
      ] as [string, Factory]
  ),
  [
    'lag ML',
    view =>
      new GPUSpatialLagMaximumLikelihood({
        weights: weights(view),
        predictors: view('float32', 8),
        response: view('float32', 8),
        predictorCount: 1,
        maximumPower: 2,
        traceMode: 'stochastic',
        traceProbeCount: 2,
        traceParameters: view('uint32', 1),
        logDeterminant: 'exact',
        eigenvalues: view('float32', 8),
        output: {
          ...fitOutput(view, 8, true),
          powerTraces: view('float32', 2),
          powerSums: view('float32', 2),
          infinityNorm: view('float32', 1),
          powerTraceStandardErrors: view('float32', 2)
        }
      })
  ],
  [
    'weights traces',
    view =>
      new GPUSpatialWeightsTraces({
        weights: weights(view),
        maximumPower: 2,
        mode: 'stochastic',
        probeCount: 2,
        probeBatchSize: 1,
        parameters: view('uint32', 1),
        output: {
          powerTraces: view('float32', 2),
          powerSums: view('float32', 2),
          infinityNorm: view('float32', 1),
          gramTraces: view('float32', 4),
          powerTraceStandardErrors: view('float32', 2),
          status: {approximation: view('uint32', 1)}
        }
      })
  ],
  [
    '2SLS robust',
    view =>
      new GPUSpatialTwoStageLeastSquares({
        weights: weights(view),
        predictors: view('float32', 8),
        response: view('float32', 8),
        predictorCount: 1,
        instrumentOrder: 2,
        output: fitOutput(view),
        robust: {
          method: 'CLUSTER',
          clusterCount: 2,
          clusters: view('uint32', 8),
          output: {covariance: view('float32', 9), status: {code: view('uint32', 1)}}
        }
      })
  ],
  [
    'panel regression',
    view =>
      new GPUSpatialPanelRegression({
        weights: {...weights(view, 4), status: {overflow: view('uint32', 1)}},
        periodCount: 2,
        effects: 'unit',
        predictorCount: 1,
        predictors: view('float32', 8),
        response: view('float32', 8),
        output: fitOutput(view)
      })
  ],
  [
    'regime regression',
    view =>
      new GPUSpatialRegimeRegression({
        predictorCount: 1,
        predictors: view('float32', 8),
        response: view('float32', 8),
        regimes: view('uint32', 8),
        mask: view('uint32', 8),
        regimeCount: 2,
        output: {
          coefficients: view('float32', 4),
          fitted: view('float32', 8),
          status: view('uint32', 2),
          counts: view('uint32', 2)
        }
      })
  ],
  [
    'IV covariance',
    view =>
      new GPUInstrumentalVariablesCovariance({
        coefficientCount: 2,
        instrumentCount: 2,
        design: view('float32', 16),
        instruments: view('float32', 16),
        residuals: view('float32', 8),
        method: 'HAC',
        kernel: {...weights(view), status: {overflow: view('uint32', 1)}},
        output: {covariance: view('float32', 9), status: {code: view('uint32', 1)}}
      })
  ],
  ...([false, true] as const).map(
    inference =>
      [
        `MGWR inference=${inference}`,
        (view: ViewFactory) =>
          new GPUMultiscaleGeographicallyWeightedRegression({
            positions: view('float32x2', 8),
            predictors: view('float32', 8),
            response: view('float32', 8),
            predictorCount: 1,
            parameters: view('float32', 3),
            mask: view('uint32', 8),
            kernel: 'bisquare',
            indexGridSize: [2, 2],
            maximumIterations: 1,
            bandwidthCandidates: view('float32', 4),
            bandwidthCandidateCount: 2,
            bandwidthSelection: {searchMethod: 'golden-section', maximumSearchIterations: 1},
            noiseVariance: view('float32', 1),
            ...(inference
              ? {
                  inference: {
                    output: {
                      standardErrors: view('float32', 16),
                      effectiveParameters: view('float32', 2),
                      summary: view('float32', 4),
                      status: view('uint32', 1)
                    }
                  }
                }
              : {}),
            output: {
              coefficients: view('float32', 16),
              fitted: view('float32', 8),
              status: {codes: view('uint32', 8), iterationCount: view('uint32', 1)},
              selectedBandwidths: view('float32', 2),
              conditionalStandardErrors: view('float32', 16)
            }
          })
      ] as [string, Factory]
  ),
  [
    'area of applicability',
    view =>
      new GPUAreaOfApplicability({
        training: view('float32', 16),
        query: view('float32', 8),
        predictorCount: 2,
        weights: view('float32', 2),
        mask: view('uint32', 8),
        crossValidation: {foldCount: 2, folds: view('uint32', 8), training: view('uint32', 16)},
        output: {
          trainDissimilarity: view('float32', 8),
          queryDissimilarity: view('float32', 4),
          applicable: view('uint32', 4),
          summary: view('float32', 6),
          status: view('uint32', 1)
        }
      })
  ],
  ['conformal intervals', view => new GPUConformalPredictionIntervals(conformal(view))],
  [
    'geo conformal intervals',
    view =>
      new GPUGeoConformalPredictionIntervals({
        ...conformal(view),
        neighbors: {
          offsets: view('uint32', 5),
          indices: view('uint32', 8),
          weights: view('float32', 8)
        }
      })
  ],
  ['cross validation', view => new GPUSpatialCrossValidation(folds(view))],
  [
    'cross validation generated design',
    view =>
      new GPUSpatialCrossValidation({
        ...folds(view, false),
        design: {
          mode: 'knndm',
          predictionPositions: view('float32x2', 3),
          candidateCount: 2,
          diagnostics: view('float32', 4)
        }
      })
  ],
  [
    'cross validated OLS',
    view =>
      new GPUSpatialCrossValidatedOLS({
        folds: folds(view),
        predictorCount: 1,
        predictors: view('float32', 8),
        response: view('float32', 8),
        output: {
          coefficients: view('float32', 4),
          predictions: view('float32', 16),
          fitStatus: view('uint32', 2),
          metrics: view('float32', 12),
          metricStatus: view('uint32', 2)
        }
      })
  ]
];

function fixture(factory: Factory, offset = 0) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const views: GraphDataView[] = [];
  const view: ViewFactory = (format, length) => {
    const result = createTransientView(
      graph,
      `caller-${views.length}`,
      format,
      length + (offset ? 1 : 0)
    );
    const caller = offset
      ? graph.createDataView(result.buffer, {format, length, byteOffset: offset})
      : result;
    views.push(caller);
    return caller;
  };
  return {device, graph, views, contributor: factory(view)};
}
function mutationSpies(graph: GPUCommandGraph) {
  return [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'add')
  ];
}
for (const [name, factory] of factories) {
  it(`${name}: every caller STORAGE usage fails before graph mutation`, () => {
    const context = fixture(factory);
    try {
      const spies = mutationSpies(context.graph);
      for (const view of context.views) {
        const usage = view.buffer.usage;
        Object.defineProperty(view.buffer, 'usage', {value: Buffer.COPY_DST});
        expect(() => context.contributor.getCommandNodes(context.graph), view.buffer.id).toThrow(
          /usage must include Buffer.STORAGE/
        );
        for (const spy of spies) expect(spy).not.toHaveBeenCalled();
        Object.defineProperty(view.buffer, 'usage', {value: usage});
      }
    } finally {
      context.device.destroy();
    }
  });
  it(`${name}: aligned caller ranges fail before graph mutation`, () => {
    const context = fixture(factory, 4);
    try {
      const largest = Math.max(...context.views.map(view => getViewBindingRange(view).size));
      Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {
        value: largest - 4
      });
      const spies = mutationSpies(context.graph);
      expect(() => context.contributor.getCommandNodes(context.graph)).toThrow(
        /needs .* bytes in one binding/
      );
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      context.device.destroy();
    }
  });
  it(`${name}: emitted identifiers use the complete WGSL reserved-word list`, () => {
    const context = fixture(factory);
    try {
      const sources = getGPUCommandNodeKernelSources(
        context.contributor.getCommandNodes(context.graph)
      );
      expect(sources.length).toBeGreaterThan(0);
      for (const source of sources)
        expect(findReservedIdentifiers(stripWGSLComments(source.source)), source.id).toEqual([]);
    } finally {
      context.device.destroy();
    }
  });
  it(`${name}: the complete allocation plan fails before graph mutation`, () => {
    const reference = fixture(factory);
    try {
      const allocations = vi.spyOn(reference.graph, 'createTransientBuffer');
      expect(() => reference.contributor.getCommandNodes(reference.graph)).not.toThrow();
      const maximumScratchBytes = Math.max(
        0,
        ...allocations.mock.results.map(result => result.value.byteLength)
      );
      if (!maximumScratchBytes) return; // Impacts is a single dispatch without scratch.
      const context = fixture(factory);
      try {
        Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {
          value: maximumScratchBytes - 4
        });
        const spies = mutationSpies(context.graph);
        expect(() => context.contributor.getCommandNodes(context.graph)).toThrow();
        for (const spy of spies) expect(spy).not.toHaveBeenCalled();
      } finally {
        context.device.destroy();
      }
    } finally {
      reference.device.destroy();
    }
  });
}

const packedFields: Record<string, readonly [string, string, number]> = {
  'error GM kelejian-prucha': ['table', 'residuals', 48],
  'error GM homoskedastic': ['table', 'residuals', 48],
  'lag ML': ['table', 'residuals', 48],
  'weights traces': ['powerSums', 'status.approximation', 8],
  '2SLS robust': ['table', 'residuals', 48],
  'panel regression': ['table', 'residuals', 48],
  'regime regression': ['coefficients', 'counts', 256],
  'IV covariance': ['covariance', 'status.code', 256],
  'MGWR inference=false': ['coefficients', 'fitted', 256],
  'MGWR inference=true': ['coefficients', 'fitted', 256],
  'area of applicability': ['queryDissimilarity', 'applicable', 256],
  'conformal intervals': ['intervals', 'calibrationCount', 32],
  'geo conformal intervals': ['intervals', 'calibrationCount', 32],
  'cross validation': ['training', 'testing', 256],
  'cross validation generated design': ['training', 'testing', 256],
  'cross validated OLS': ['coefficients', 'fitStatus', 256]
};
function getOutputField(contributor: GPUCommandNodeProducer, path: string): GraphDataView {
  const {props} = contributor as unknown as {props: {output: Record<string, unknown>}};
  let result: unknown = props.output;
  for (const part of path.split('.')) result = (result as Record<string, unknown>)[part];
  return result as GraphDataView;
}
function replaceViews(value: unknown, replacements: Map<GraphDataView, GraphDataView>): unknown {
  if (!value || typeof value !== 'object') return value;
  if ('buffer' in value && 'format' in value)
    return replacements.get(value as GraphDataView) ?? value;
  if (Array.isArray(value)) return value.map(item => replaceViews(item, replacements));
  return Object.fromEntries(
    Object.entries(value).map(([name, item]) => [name, replaceViews(item, replacements)])
  );
}
function reconstruct(
  contributor: GPUCommandNodeProducer,
  replacements: Map<GraphDataView, GraphDataView>
): GPUCommandNodeProducer {
  const original = contributor as unknown as {
    props: unknown;
    constructor: new (props: unknown) => GPUCommandNodeProducer;
  };
  return new original.constructor(replaceViews(original.props, replacements));
}
for (const [name, factory] of factories) {
  it(`${name}: accepts nonzero binding prefixes and valid packed outputs`, () => {
    const context = fixture(factory, 4);
    try {
      const fields = packedFields[name];
      if (!fields) {
        expect(() => context.contributor.getCommandNodes(context.graph)).not.toThrow();
        return;
      }
      const [firstPath, secondPath, secondOffset] = fields;
      const first = getOutputField(context.contributor, firstPath);
      const second = getOutputField(context.contributor, secondPath);
      const buffer = context.graph.createTransientBuffer({
        id: 'packed-outputs',
        byteLength: secondOffset + second.length * 4,
        usage: Buffer.STORAGE
      });
      const replacements = new Map<GraphDataView, GraphDataView>([
        [first, context.graph.createDataView(buffer, {format: first.format, length: first.length})],
        [
          second,
          context.graph.createDataView(buffer, {
            format: second.format,
            length: second.length,
            byteOffset: secondOffset
          })
        ]
      ]);
      expect(() =>
        reconstruct(context.contributor, replacements).getCommandNodes(context.graph)
      ).not.toThrow();
    } finally {
      context.device.destroy();
    }
  });
  if (packedFields[name])
    it(`${name}: rejects overlapping logical output bytes at construction`, () => {
      const context = fixture(factory);
      try {
        const [firstPath, secondPath] = packedFields[name];
        const first = getOutputField(context.contributor, firstPath);
        const second = getOutputField(context.contributor, secondPath);
        const buffer = context.graph.createTransientBuffer({
          id: 'overlapping-outputs',
          byteLength: Math.max(first.length, second.length) * 4,
          usage: Buffer.STORAGE
        });
        const replacements = new Map<GraphDataView, GraphDataView>([
          [
            first,
            context.graph.createDataView(buffer, {format: first.format, length: first.length})
          ],
          [
            second,
            context.graph.createDataView(buffer, {format: second.format, length: second.length})
          ]
        ]);
        const spies = mutationSpies(context.graph);
        expect(() => reconstruct(context.contributor, replacements)).toThrow(
          /logical writable byte ranges/
        );
        for (const spy of spies) expect(spy).not.toHaveBeenCalled();
      } finally {
        context.device.destroy();
      }
    });
}

it('rejects aligned co-bound output overlap, even when logical bytes are disjoint', () => {
  const factory = factories.find(([name]) => name === 'regime regression')![1];
  const context = fixture(factory);
  try {
    const coefficients = getOutputField(context.contributor, 'coefficients');
    const counts = getOutputField(context.contributor, 'counts');
    const buffer = context.graph.createTransientBuffer({
      id: 'aligned-overlap',
      byteLength: 24,
      usage: Buffer.STORAGE
    });
    expect(() =>
      reconstruct(
        context.contributor,
        new Map<GraphDataView, GraphDataView>([
          [coefficients, context.graph.createDataView(buffer, {format: 'float32', length: 4})],
          [
            counts,
            context.graph.createDataView(buffer, {format: 'uint32', length: 2, byteOffset: 16})
          ]
        ])
      )
    ).toThrow(/aligned binding byte ranges/);
  } finally {
    context.device.destroy();
  }
});
it('rejects one physical buffer bound read-only and writable, even with disjoint aligned ranges', () => {
  const factory = factories.find(([name]) => name === 'regime regression')![1];
  const context = fixture(factory);
  try {
    const coefficients = getOutputField(context.contributor, 'coefficients');
    const fitted = getOutputField(context.contributor, 'fitted');
    const buffer = context.graph.createTransientBuffer({
      id: 'incompatible-access',
      byteLength: 288,
      usage: Buffer.STORAGE
    });
    expect(() =>
      reconstruct(
        context.contributor,
        new Map<GraphDataView, GraphDataView>([
          [coefficients, context.graph.createDataView(buffer, {format: 'float32', length: 4})],
          [
            fitted,
            context.graph.createDataView(buffer, {format: 'float32', length: 8, byteOffset: 256})
          ]
        ])
      )
    ).toThrow(/one physical buffer read-only and writable/);
  } finally {
    context.device.destroy();
  }
});

for (const kind of ['lag ML', 'weights traces', 'error GM'] as const) {
  it(`${kind}: oversized nested trace tables fail before the first resource`, () => {
    const context = fixture(view => {
      const common = {
        weights: weights(view, 2),
        predictors: view('float32', 2),
        response: view('float32', 2),
        predictorCount: 1
      };
      if (kind === 'error GM')
        return new GPUSpatialErrorGM({
          ...common,
          estimator: 'homoskedastic',
          traceMode: 'stochastic',
          traceProbeCount: 8388609,
          output: fitOutput(view, 2)
        });
      if (kind === 'lag ML')
        return new GPUSpatialLagMaximumLikelihood({
          ...common,
          maximumPower: 256,
          traceMode: 'stochastic',
          traceProbeCount: 131073,
          output: fitOutput(view, 2)
        });
      return new GPUSpatialWeightsTraces({
        weights: common.weights,
        maximumPower: 256,
        mode: 'stochastic',
        probeCount: 131073,
        output: {powerTraces: view('float32', 256), powerSums: view('float32', 256)}
      });
    });
    try {
      const spies = mutationSpies(context.graph);
      expect(() => context.contributor.getCommandNodes(context.graph)).toThrow(
        /probe dots.*needs .* bytes in one binding/
      );
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      context.device.destroy();
    }
  });
}
it('trace preflight includes padding and batch-information tables when probeBatchSize is explicit', () => {
  const context = fixture(
    view =>
      new GPUSpatialWeightsTraces({
        weights: weights(view, 2),
        maximumPower: 1,
        mode: 'stochastic',
        probeCount: 5,
        probeBatchSize: 1,
        output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
      })
  );
  try {
    // Five 256-byte batch slots. Vectors, caller CSR, and unpadded probe counts all fit.
    Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {value: 1024});
    const spies = mutationSpies(context.graph);
    expect(() => context.contributor.getCommandNodes(context.graph)).toThrow(
      /probe batch information.*1280/
    );
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});

for (const rhoBound of [Infinity, NaN, 1e39, 1e-100]) {
  it(`lag impacts rejects an unrepresentable rhoBound ${rhoBound} at construction`, () => {
    const factory = factories.find(([name]) => name === 'lag impacts')![1];
    const context = fixture(factory);
    try {
      const props = (context.contributor as GPUSpatialLagImpacts).props;
      expect(() => new GPUSpatialLagImpacts({...props, rhoBound})).toThrow(
        /spatial-lag-impacts rhoBound/
      );
    } finally {
      context.device.destroy();
    }
  });
}
it('lag impacts formats finite scientific-notation bounds as valid WGSL literals', () => {
  const factory = factories.find(([name]) => name === 'lag impacts')![1];
  const context = fixture(factory);
  try {
    const props = (context.contributor as GPUSpatialLagImpacts).props;
    const nodes = new GPUSpatialLagImpacts({...props, rhoBound: 1e30}).getCommandNodes(
      context.graph
    );
    const [kernel] = getGPUCommandNodeKernelSources(nodes);
    expect(kernel.source).toContain('const RHO_BOUND: f32 = 1.0000000150474662e+30;');
    expect(kernel.source).not.toContain('e+30.0');
  } finally {
    context.device.destroy();
  }
});

for (const secondPath of ['powerSums', 'powerTraceStandardErrors', 'powerTraces']) {
  it(`exact lag ML accepts table packed with separately dispatched ${secondPath}`, () => {
    const factory = factories.find(([name]) => name === 'lag ML')![1];
    const context = fixture(factory);
    try {
      const first = getOutputField(context.contributor, 'table');
      const second = getOutputField(context.contributor, secondPath);
      const buffer = context.graph.createTransientBuffer({
        id: 'packed-trace-publication',
        byteLength: 56,
        usage: Buffer.STORAGE
      });
      const replacements = new Map<GraphDataView, GraphDataView>([
        [first, context.graph.createDataView(buffer, {format: 'float32', length: first.length})],
        [
          second,
          context.graph.createDataView(buffer, {
            format: 'float32',
            length: second.length,
            byteOffset: 48
          })
        ]
      ]);
      expect(() =>
        reconstruct(context.contributor, replacements).getCommandNodes(context.graph)
      ).not.toThrow();
    } finally {
      context.device.destroy();
    }
  });
}
it('MGWR ignores an unused optional noise-variance buffer aliasing an output', () => {
  const factory = factories.find(([name]) => name === 'MGWR inference=false')![1];
  const context = fixture(factory);
  try {
    const props = (context.contributor as GPUMultiscaleGeographicallyWeightedRegression).props;
    const noiseVariance = context.graph.createDataView(props.output.coefficients.buffer, {
      format: 'float32',
      length: 1
    });
    const contributor = new GPUMultiscaleGeographicallyWeightedRegression({
      ...props,
      noiseVariance,
      output: {...props.output, conditionalStandardErrors: undefined}
    });
    expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});
it('non-group fold designs ignore unused blocks aliasing their generated fold output', () => {
  const factory = factories.find(([name]) => name === 'cross validation generated design')![1];
  const context = fixture(factory);
  try {
    const props = (context.contributor as GPUSpatialCrossValidation).props;
    const contributor = new GPUSpatialCrossValidation({...props, blocks: props.output.folds});
    expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

it('the shared GWR preflight checks STORAGE usage before any graph mutation', () => {
  const factory = factories.find(([name]) => name === 'lag impacts')![1];
  const context = fixture(factory);
  try {
    const view = context.views[0];
    Object.defineProperty(view.buffer, 'usage', {value: Buffer.COPY_DST});
    const spies = mutationSpies(context.graph);
    expect(() => validateGWRBindings(context.device, 'gwr-contract', [view], [])).toThrow(
      /usage must include Buffer.STORAGE/
    );
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});
it('shared GWR dispatch validation accepts packed separate writes and rejects mixed physical access', () => {
  const factory = factories.find(([name]) => name === 'lag impacts')![1];
  const context = fixture(factory);
  try {
    const buffer = context.graph.createTransientBuffer({
      id: 'gwr-packed',
      byteLength: 264,
      usage: Buffer.STORAGE
    });
    const first = context.graph.createDataView(buffer, {format: 'float32', length: 2});
    const consecutive = context.graph.createDataView(buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 8
    });
    const aligned = context.graph.createDataView(buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 256
    });
    expect(() =>
      validateGWROutputs(
        'gwr-contract',
        [first, consecutive],
        [{writes: [first]}, {writes: [consecutive]}]
      )
    ).not.toThrow();
    expect(() =>
      validateGWROutputs('gwr-contract', [first, aligned], [{reads: [first], writes: [aligned]}])
    ).toThrow(/one physical buffer read-only and writable/);
  } finally {
    context.device.destroy();
  }
});
it('derived scratch preflight checks maxBufferSize and the backing row of an empty view', () => {
  const factory = factories.find(([name]) => name === 'lag impacts')![1];
  const context = fixture(factory);
  try {
    const spies = mutationSpies(context.graph);
    Object.defineProperty(context.device.limits, 'maxBufferSize', {value: 128});
    expect(() =>
      validateRegressionAllocations(context.device, 'allocation-contract', {scratch: 33})
    ).toThrow(/maxBufferSize/);
    Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {value: 3});
    expect(() =>
      validateRegressionAllocations(context.device, 'allocation-contract', {empty: 0})
    ).toThrow(/empty needs 4 bytes/);
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});

for (const traceProbeCount of [0, 1.5, Infinity]) {
  it(`homoskedastic GM validates nested traceProbeCount ${traceProbeCount} at construction`, () => {
    const factory = factories.find(([name]) => name === 'error GM homoskedastic')![1];
    const context = fixture(factory);
    try {
      const props = (context.contributor as GPUSpatialErrorGM).props;
      const spies = mutationSpies(context.graph);
      expect(() => new GPUSpatialErrorGM({...props, traceProbeCount})).toThrow(
        /spatial-error-gm traceProbeCount/
      );
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      context.device.destroy();
    }
  });
}
it('homoskedastic GM validates its nested traceMode at construction', () => {
  const factory = factories.find(([name]) => name === 'error GM homoskedastic')![1];
  const context = fixture(factory);
  try {
    const props = (context.contributor as GPUSpatialErrorGM).props;
    const spies = mutationSpies(context.graph);
    expect(() => new GPUSpatialErrorGM({...props, traceMode: 'invalid' as never})).toThrow(
      /spatial-error-gm traceMode/
    );
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});
