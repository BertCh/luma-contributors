// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {snapshotSpaceTimeTemporalContext} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-construction';
import {snapshotSpaceTimeHawkesProps} from '../../../src/gpu-spatial-analysis/space-time-hawkes/space-time-hawkes-construction';
import {GPUKnoxTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-knox-test';
import {GPUJacquezTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-jacquez-test';
import {GPULocalKnox} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-local-knox';
import {GPUSpaceTimeCrossCorrelation} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation';
import {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('PTS-133 captures nested local Knox construction fields', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `capture-${serial++}`, format, length);
  try {
    const pairs = {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 6),
      weights: view('float32', 6),
      distances: view('float32', 6),
      status: {
        overflow: view('uint32', 1),
        count: view('uint32', 1),
        invalidCount: view('uint32', 1)
      }
    };
    const temporalContext = {
      encoding: 'float32-relative' as const,
      unit: 'second' as const,
      epoch: 0n,
      axis: {kind: 'irregular' as const}
    };
    const localProps = {
      pairs,
      temporalContext,
      times: view('float32', 3),
      parameters: view('uint32', 8),
      maximumPermutations: 1,
      output: {summary: view('float32', 24)}
    };
    const local = new GPULocalKnox(localProps);
    expect(local.props.pairs.status?.count).toBe(pairs.status.count);
    expect(local.props.pairs.status?.invalidCount).toBe(pairs.status.invalidCount);
    const originalSummary = localProps.output.summary;
    const originalOverflow = pairs.status.overflow;
    localProps.maximumPermutations = 5;
    localProps.output.summary = view('float32', 1);
    pairs.status.overflow = view('uint32', 1);
    temporalContext.epoch = 100n;
    expect(local.props.maximumPermutations).toBe(1);
    expect(local.props.output.summary).toBe(originalSummary);
    expect(local.props.pairs.status?.overflow).toBe(originalOverflow);
    expect(local.temporalContext?.epoch).toBe(0n);
    expect(local.props.parameters).toBe(localProps.parameters); // buffer stays live
    expect(local.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('PTS-133 captures cross-correlation lags and optional outputs before declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `correlation-capture-${serial++}`, format, length);
  try {
    const lags = [0, 1];
    const props = {
      first: view('float32', 8),
      second: view('float32', 8),
      cellCount: 1,
      periodCount: 8,
      layout: 'cell-major' as const,
      lags,
      output: {
        correlations: view('float32', 2),
        pValues: view('float32', 2),
        adjustedPValues: view('float32', 2),
        testedCount: view('uint32', 1)
      }
    };
    const contributor = new GPUSpaceTimeCrossCorrelation(props);
    const original = props.output.adjustedPValues;
    lags[1] = 100;
    props.output.adjustedPValues = view('float32', 1);
    props.periodCount = 1;
    expect(contributor.props.lags).toEqual([0, 1]);
    expect(contributor.props.periodCount).toBe(8);
    expect(contributor.props.output.adjustedPValues).toBe(original);
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.some(kernel => kernel.source.includes('erfcValue'))).toBe(true);
    expect(kernels.some(kernel => kernel.id.endsWith('publish-tested-count'))).toBe(true);
  } finally {
    device.destroy();
  }
});

it('PTS-133 captures Hawkes output/status and context declarations while retaining views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `hawkes-capture-${serial++}`, format, length);
  try {
    const temporalContext = {
      encoding: 'float32-relative' as const,
      unit: 'second' as const,
      epoch: 0n,
      axis: {kind: 'irregular' as const}
    };
    const props = {
      positions: view('float32x2', 2),
      times: view('float32', 2),
      parameters: view('float32', 9),
      temporalContext,
      maximumIterations: 1,
      output: {
        estimates: view('float32', 4),
        fittedLogLikelihood: view('float32', 1),
        status: {converged: view('uint32', 1)}
      }
    };
    const contributor = new GPUSpaceTimeHawkes(props);
    const estimates = props.output.estimates;
    const converged = props.output.status.converged;
    props.output.estimates = view('float32', 1);
    props.output.status.converged = view('uint32', 2);
    temporalContext.epoch = 200n;
    props.maximumIterations = 2;
    expect(contributor.props.output.estimates).toBe(estimates);
    expect(contributor.props.output.status?.converged).toBe(converged);
    expect(contributor.props.maximumIterations).toBe(1);
    expect(contributor.temporalContext?.epoch).toBe(0n);
    expect(contributor.props.parameters).toBe(props.parameters);
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('PTS-137 names the contributor and violated construction requirement', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `requirements-${serial++}`, format, length);
  try {
    const pairs = {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 6),
      weights: view('float32', 6)
    };
    const common = {
      pairs,
      times: view('float32', 2),
      parameters: view('uint32', 4),
      maximumPermutations: 1
    };
    expect(
      () => new GPULocalKnox({...common, id: 'named-local', output: {summary: view('float32', 24)}})
    ).toThrow(/^named-local times needs one row per pairs row/);
    expect(
      () =>
        new GPUKnoxTest({
          ...common,
          id: 'named-knox',
          output: {statistics: view('uint32', 2), summary: view('float32', 13)}
        })
    ).toThrow(/^named-knox times needs one row per pairs row/);
    expect(
      () =>
        new GPUJacquezTest({
          ...common,
          id: 'named-jacquez',
          neighbors: pairs,
          neighborCount: 3,
          times: view('float32', 3),
          output: {statistics: view('uint32', 2), summary: view('float32', 13)}
        })
    ).toThrow(/^named-jacquez neighbors rowCount needs more rows than neighborCount/);
  } finally {
    device.destroy();
  }
});

it('S1 captures provenance before caller mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `provenance-${serial++}`, format, length);
  try {
    const provenance = {source: 'upload', criterion: 'distance-band', radius: 2};
    const contributor = new GPULocalKnox({
      pairs: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 6),
        weights: view('float32', 6),
        information: {
          rowCount: 3,
          transform: 'raw',
          symmetric: true,
          hasSelfWeights: false,
          provenance
        }
      },
      times: view('float32', 3),
      parameters: view('uint32', 4),
      maximumPermutations: 1,
      output: {summary: view('float32', 24)}
    });
    provenance.radius = 500;
    expect(contributor.props.pairs.information?.provenance?.radius).toBe(2);
    expect(Object.isFrozen(contributor.props.pairs.information?.provenance)).toBe(true);
  } finally {
    device.destroy();
  }
});

it('S1 reads temporal step and Hawkes status getters once', () => {
  let stepReads = 0;
  let kindReads = 0;
  const context = snapshotSpaceTimeTemporalContext({
    encoding: 'float32-relative',
    unit: 'second',
    epoch: 0n,
    axis: {
      get kind() {
        kindReads++;
        return 'regular' as const;
      },
      get step() {
        stepReads++;
        return {calendarUnit: 'day' as const, interval: stepReads};
      }
    }
  });
  expect(stepReads).toBe(1);
  expect(kindReads).toBe(1);
  expect(context.axis).toEqual({kind: 'regular', step: {calendarUnit: 'day', interval: 1}});
  let statusReads = 0;
  const captured = snapshotSpaceTimeHawkesProps({
    rowCount: 2,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    output: {
      estimates: 'estimates',
      get status() {
        statusReads++;
        return {iterationCount: `iterations-${statusReads}`};
      }
    }
  });
  expect(statusReads).toBe(1);
  expect(captured.output.status?.iterationCount).toBe('iterations-1');
});
