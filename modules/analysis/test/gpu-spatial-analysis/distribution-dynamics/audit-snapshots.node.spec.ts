// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCPUArray,
  type GPUCPUVector
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUClassAssignment} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {GPUClassificationFit} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit';
import {GPUTransitionMatrix} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix';
import {GPUTransitionDiagnostics} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {GPURankMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {GPULISAMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov';
import {GPULocationChangePoints} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';
import {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import {CPUTransitionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation';
import {
  GPUTransitionDiagnosticsOperation,
  type TransitionDiagnosticsOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics-operation';
import {CPURankMarkovOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation';
import {
  GPURankMarkovOperation,
  type RankMarkovOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation';
import {CPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {
  GPULocationChangePointsOperation,
  type LocationChangePointsOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import {CPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {
  GPUPettittTestOperation,
  type PettittTestOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {
  getGPUClassBreaksParameterLength,
  type GPUClassBreaksMethod
} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {AUDIT_RANK_CASES, AUDIT_RANK_VALUES, AUDIT_RANK_MASK} from './audit-rank-markov-fixture';
import {
  AUDIT_CHANGE_CASES,
  AUDIT_CHANGE_SERIES,
  AUDIT_PETTITT_EXPECTED,
  AUDIT_PETTITT_SERIES,
  packAuditSeries
} from '../change-points/audit-change-points-fixture';

function vector(values: Float32Array | Uint32Array): GPUCPUVector {
  return {
    format: values instanceof Float32Array ? 'float32' : 'uint32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

it('snapshots all contributor topology, output routes, weights and method lists at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `snapshot-${serial++}`, format, length);
  const weights = () => ({
    offsets: view('uint32', 9),
    neighbors: view('uint32', 8),
    weights: view('float32', 8),
    information: {rowCount: 8, transform: 'raw' as const, symmetric: false},
    status: {overflow: view('uint32', 1)}
  });
  try {
    const assignmentProps = {
      values: view('float32', 8),
      breaks: view('float32', 3),
      classCount: view('uint32', 1),
      intervalClosure: 'left' as 'left' | 'right',
      output: view('uint32', 8)
    };
    const assignment = new GPUClassAssignment(assignmentProps);
    assignmentProps.intervalClosure = 'right';
    expect(assignment.props.intervalClosure).toBe('left');

    const fitProps = {
      values: view('float32', 8),
      classes: view('uint32', 8),
      classCount: 2,
      output: {
        counts: view('uint32', 3),
        medians: view('float32', 3),
        absoluteDeviations: view('float32', 3),
        squaredDeviations: view('float32', 3)
      }
    };
    const fit = new GPUClassificationFit(fitProps);
    const fitCounts = fitProps.output.counts;
    fitProps.classCount = 3;
    fitProps.output.counts = view('uint32', 4);
    expect(fit.props.classCount).toBe(2);
    expect(fit.props.output.counts).toBe(fitCounts);

    const matrixProps = {
      classes: view('uint32', 16),
      periodCount: 2,
      classCount: 2,
      output: {counts: view('uint32', 4)}
    };
    const matrix = new GPUTransitionMatrix(matrixProps);
    matrixProps.periodCount = 4;
    expect(matrix.props.periodCount).toBe(2);

    const diagnosticProps = {
      counts: view('uint32', 4),
      classCount: 2,
      output: {steadyStates: view('float32', 2), status: {converged: view('uint32', 1)}}
    };
    const diagnostic = new GPUTransitionDiagnostics(diagnosticProps);
    const converged = diagnosticProps.output.status.converged;
    diagnosticProps.classCount = 3;
    diagnosticProps.output.status.converged = view('uint32', 1);
    expect(diagnostic.props.classCount).toBe(2);
    expect(diagnostic.props.output.status!.converged).toBe(converged);

    const rankProps = {
      values: view('float32', 16),
      periodCount: 2,
      mode: 'full-rank' as 'full-rank' | 'geographic-rank',
      output: {ranks: view('uint32', 16), counts: view('uint32', 64)}
    };
    const rank = new GPURankMarkov(rankProps);
    rankProps.mode = 'geographic-rank';
    expect(rank.props.mode).toBe('full-rank');

    const lagMethods: GPUClassBreaksMethod[] = ['quantile'];
    const spatialProps = {
      values: view('float32', 16),
      classes: view('uint32', 16),
      periodCount: 2,
      classCount: 2,
      weights: weights(),
      maximumLagClassCount: 2,
      lagMethods,
      lagParameters: view('float32', getGPUClassBreaksParameterLength(2)),
      output: {
        counts: view('uint32', 8),
        lagBreaks: view('float32', 3),
        lagClassCount: view('uint32', 1)
      }
    };
    const spatial = new GPUSpatialMarkov(spatialProps);
    const spatialOverflow = spatialProps.weights.status.overflow;
    lagMethods[0] = 'equal-interval';
    spatialProps.weights.status.overflow = view('uint32', 1);
    expect(spatial.props.lagMethods).toEqual(['quantile']);
    expect(spatial.props.weights.status!.overflow).toBe(spatialOverflow);

    const lisaProps = {
      values: view('float32', 16),
      weights: weights(),
      periodCount: 2,
      parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
      output: {counts: view('uint32', 25)}
    };
    const lisa = new GPULISAMarkov(lisaProps);
    lisaProps.weights.neighbors = view('uint32', 9);
    expect(lisa.props.weights.neighbors.length).toBe(8);

    const changeProps = {
      values: view('float32', 8),
      layout: 'cell-major' as const,
      cellCount: 1,
      sliceCount: 8,
      cost: 'mean' as const,
      maximumChangePoints: 1,
      parameters: view('float32', 2),
      output: {changePointCount: view('uint32', 1), changePoints: view('uint32', 1)}
    };
    const change = new GPULocationChangePoints(changeProps);
    changeProps.sliceCount = 7;
    expect(change.props.sliceCount).toBe(8);

    const pettittProps = {
      values: view('float32', 8),
      layout: 'cell-major' as 'cell-major' | 'period-major',
      cellCount: 1,
      sliceCount: 8,
      output: {
        statistic: view('uint32', 1),
        changeSlice: view('uint32', 1),
        pValues: view('float32', 1)
      }
    };
    const pettitt = new GPUPettittTest(pettittProps);
    pettittProps.layout = 'period-major';
    expect(pettitt.props.layout).toBe('cell-major');
  } finally {
    device.destroy();
  }
});

it('keeps diagnostic operation forms on the declared options and oracle-backed routes', async () => {
  const props: TransitionDiagnosticsOperationProps = {
    counts: 'counts',
    classCount: 2,
    output: {
      steadyStates: 'steadyStates',
      stateSparsity: 'stateSparsity',
      status: {converged: 'converged'}
    }
  };
  const gpu = new GPUTransitionDiagnosticsOperation(props);
  const cpu = new CPUTransitionDiagnosticsOperation(props);
  props.classCount = 3;
  props.output.status!.converged = 'late-converged';
  const contributor = constructNetworkOperationContributor(gpu, {
    counts: {format: 'uint32', length: 4}
  }) as GPUTransitionDiagnostics;
  expect(contributor.props.classCount).toBe(2);
  const actual = await cpu.execute({inputs: {counts: vector(Uint32Array.of(0, 1, 1, 0))}});
  // Periodic two-cycle: invariant mass [1/2,1/2], both rows sparse, solver succeeds.
  expectNumericArrayClose(actual['steadyStates'] as GPUCPUArray, [0.5, 0.5]);
  expectNumericArrayClose(actual['stateSparsity'] as GPUCPUArray, [2, 0]);
  expectNumericArrayClose(actual['converged'] as GPUCPUArray, [1]);
});

it('keeps rank operation forms on the declared options and SciPy literals', async () => {
  const {mode, tieOrder, expected} = AUDIT_RANK_CASES[0];
  const props: RankMarkovOperationProps = {
    values: 'values',
    mask: 'mask',
    rowCount: 8,
    periodCount: 4,
    mode,
    tieOrder,
    output: {
      ranks: 'ranks',
      counts: 'counts',
      probabilities: 'probabilities',
      rowTotals: 'rowTotals',
      ignored: 'ignored',
      mobility: 'mobility',
      steadyStates: 'steadyStates',
      stateSparsity: 'stateSparsity',
      status: {converged: 'converged'}
    }
  };
  const gpu = new GPURankMarkovOperation(props);
  const cpu = new CPURankMarkovOperation(props);
  props.periodCount = 2;
  props.tieOrder = 'row-id';
  props.output.status!.converged = 'late-converged';
  const contributor = constructNetworkOperationContributor(gpu, {
    values: {format: 'float32', length: 32},
    mask: {format: 'uint32', length: 8}
  }) as GPURankMarkov;
  expect(contributor.props.periodCount).toBe(4);
  expect(contributor.props.tieOrder).toBe('giddy');
  const actual = await cpu.execute({
    inputs: {
      values: vector(Float32Array.from(AUDIT_RANK_VALUES)),
      mask: vector(Uint32Array.from(AUDIT_RANK_MASK))
    }
  });
  for (const port of Object.keys(expected) as (keyof typeof expected)[])
    expectNumericArrayClose(actual[port] as GPUCPUArray, expected[port], {
      absoluteTolerance: 1e-7,
      label: port
    });
});

it('keeps location and Pettitt operation forms on their original independent oracle fixtures', async () => {
  const locationProps: LocationChangePointsOperationProps = {
    values: 'values',
    parameters: 'parameters',
    layout: 'cell-major',
    cellCount: 5,
    sliceCount: 8,
    cost: 'mean',
    maximumChangePoints: 1,
    output: {
      changePointCount: 'changePointCount',
      changePoints: 'changePoints',
      requiredChangePointCounts: 'requiredChangePointCounts',
      optimalCost: 'optimalCost',
      status: {overflow: 'overflow'}
    }
  };
  const gpu = new GPULocationChangePointsOperation(locationProps);
  const cpu = new CPULocationChangePointsOperation(locationProps);
  locationProps.cost = 'poisson';
  locationProps.output.optimalCost = 'late-cost';
  const contributor = constructNetworkOperationContributor(gpu, {
    values: {format: 'float32', length: 40},
    parameters: {format: 'float32', length: 2}
  }) as GPULocationChangePoints;
  expect(contributor.props.cost).toBe('mean');
  const actual = await cpu.execute({
    inputs: {
      values: vector(packAuditSeries(AUDIT_CHANGE_SERIES, 'cell-major')),
      parameters: vector(Float32Array.of(3, 1))
    }
  });
  const expected = AUDIT_CHANGE_CASES[0].expected;
  for (const port of Object.keys(expected) as (keyof typeof expected)[])
    expectNumericArrayClose(actual[port] as GPUCPUArray, expected[port], {
      absoluteTolerance: 1e-5,
      label: port
    });

  const pettittProps: PettittTestOperationProps = {
    values: 'values',
    layout: 'cell-major',
    cellCount: 5,
    sliceCount: 8,
    output: {statistic: 'statistic', changeSlice: 'changeSlice', pValues: 'pValues'}
  };
  const gpuPettitt = new GPUPettittTestOperation(pettittProps);
  const cpuPettitt = new CPUPettittTestOperation(pettittProps);
  pettittProps.layout = 'period-major';
  pettittProps.output.pValues = 'late-p-values';
  const pettitt = constructNetworkOperationContributor(gpuPettitt, {
    values: {format: 'float32', length: 40}
  }) as GPUPettittTest;
  expect(pettitt.props.layout).toBe('cell-major');
  const pettittActual = await cpuPettitt.execute({
    inputs: {values: vector(packAuditSeries(AUDIT_PETTITT_SERIES, 'cell-major'))}
  });
  for (const port of Object.keys(AUDIT_PETTITT_EXPECTED) as (keyof typeof AUDIT_PETTITT_EXPECTED)[])
    expectNumericArrayClose(pettittActual[port] as GPUCPUArray, AUDIT_PETTITT_EXPECTED[port], {
      absoluteTolerance: 1e-7,
      label: port
    });
});
