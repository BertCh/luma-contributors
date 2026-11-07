// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {GPUNetworkKernelDensity} from '../../../src/gpu-network/gpu-network-kernel-density';
import {GPUInhomogeneousPointPattern} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {PHILOX_PURPOSES, getPhilox4x32} from '../../../src/analysis-core/kernels/philox-random';

it('point process kernels declare parseable GPU stages within eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `point-process-${serial++}`, format, length);
  const positions = view('float32x2', 8);
  const optics = new GPUOPTICS({
    positions,
    minimumSamples: 2,
    maximumDistance: 2,
    extractionDistance: 1,
    output: {
      ordering: view('uint32', 8),
      coreDistances: view('float32', 8),
      reachability: view('float32', 8),
      predecessors: view('uint32', 8),
      labels: view('uint32', 8),
      status: view('uint32', 3)
    }
  });
  const envelope = new GPUPointPatternEnvelope({
    observed: view('float32', 3),
    simulations: view('float32', 12),
    reference: view('float32', 3),
    integrationWeights: view('float32', 3),
    simulationCount: 4,
    nullModel: 'fixed supplied null',
    fittedParameters: 'fixed-external',
    output: {
      lower: view('float32', 3),
      upper: view('float32', 3),
      statistics: view('float32', 4),
      status: view('uint32', 1)
    }
  });
  const sample = new GPUSpatialBalancedSample({
    positions,
    frameIds: view('uint32', 8),
    strata: view('uint32', 8),
    allocations: [
      {stratumId: 0, sampleCount: 2},
      {stratumId: 1, sampleCount: 2}
    ],
    bounds: [0, 0, 10, 10],
    seed: 42,
    output: {
      frameIds: view('uint32', 4),
      inclusionProbabilities: view('float32', 4),
      designWeights: view('float32', 4),
      status: view('uint32', 4)
    }
  });
  const network = new GPUNetworkKernelDensity({
    distances: view('float32', 12),
    eventCount: 3,
    quadratureLengths: view('float32', 4),
    bandwidth: 2,
    excludedEventIds: view('uint32', 4),
    output: {
      intensity: view('float32', 4),
      kernelMass: view('float32', 3),
      status: view('uint32', 4)
    }
  });
  const studyWindow = new GPUStudyWindowPreparation({
    positions: view('float32x2', 5),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2),
    samplingGrid: [4, 4]
  });
  graph.add(studyWindow);
  const inhomogeneous = new GPUInhomogeneousPointPattern({
    positions,
    studyWindow,
    radii: view('float32', 3),
    bandwidth: 1,
    output: {
      intensity: view('float32', 8),
      k: view('float32', 3),
      pairCorrelation: view('float32', 3),
      status: view('uint32', 3)
    }
  });
  for (const contributor of [
    optics,
    envelope,
    sample,
    new GPUSpatialBalancedSample({
      ...sample.props,
      id: 'unequal',
      inclusionTickets: view('uint32', 8),
      parameters: view('uint32', 2)
    }),
    network,
    inhomogeneous
  ]) {
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels) {
      const reflection = new WgslReflect(kernel.source);
      expect(reflection.storage.length).toBeLessThanOrEqual(8);
    }
  }
  expect(() => new GPUOPTICS({...optics.props, minimumSamples: 0})).toThrow();
  expect(() => new GPUOPTICS({...optics.props, maximumIterations: 9})).toThrow();
  expect(() => new GPUPointPatternEnvelope({...envelope.props, rank: 3})).toThrow();
  expect(() => new GPUSpatialBalancedSample({...sample.props, seed: -1})).toThrow();
  expect(() => new GPUSpatialBalancedSample({...sample.props, bounds: [0, 0, 0, 1]})).toThrow();
  expect(() => new GPUNetworkKernelDensity({...network.props, bandwidth: 0})).toThrow();
  device.destroy();
});

it('circular systematic design has exact inclusion probabilities for every frame size and sample size', () => {
  for (let frameCount = 1; frameCount <= 48; frameCount++) {
    for (let sampleCount = 1; sampleCount <= frameCount; sampleCount++) {
      const frequencies = new Array<number>(frameCount).fill(0);
      for (let shift = 0; shift < frameCount; shift++) {
        const selected = new Set<number>();
        for (let sample = 0; sample < sampleCount; sample++) {
          selected.add((shift + Math.floor((sample * frameCount) / sampleCount)) % frameCount);
        }
        expect(selected.size).toBe(sampleCount);
        for (const rank of selected) {
          frequencies[rank]++;
        }
      }
      expect(frequencies).toEqual(new Array(frameCount).fill(sampleCount));
    }
  }
});

it('independent Philox survey replicates retain empirical inclusion and spatial spread', () => {
  const frameCount = 64;
  const sampleCount = 8;
  const frequencies = new Array<number>(frameCount).fill(0);
  for (let replicate = 0; replicate < 8192; replicate++) {
    const draw = getPhilox4x32(
      [0, 0x80000000, replicate, 0],
      [123, PHILOX_PURPOSES.spatialBalancedSampling.id]
    )[0];
    const shift = draw % frameCount;
    for (let sample = 0; sample < sampleCount; sample++) {
      frequencies[(shift + sample * 8) % frameCount]++;
    }
  }
  for (const count of frequencies) {
    expect(Math.abs(count / 8192 - 0.125)).toBeLessThan(0.02);
  }
  // Compare coverage gaps with simple random sampling without replacement on a spatial line.
  let randomMaximumGap = 0;
  for (let replicate = 0; replicate < 256; replicate++) {
    const randomOrder = Array.from({length: frameCount}, (_, row) => ({
      row,
      priority: getPhilox4x32(
        [row, replicate, 0, 0],
        [991, PHILOX_PURPOSES.spatialBalancedSampling.id]
      )[0]
    }));
    const selected = randomOrder
      .sort((left, right) => left.priority - right.priority)
      .slice(0, sampleCount)
      .map(value => value.row)
      .sort((left, right) => left - right);
    const gaps = selected.map(
      (rank, index) =>
        (index + 1 < sampleCount ? selected[index + 1] : selected[0] + frameCount) - rank
    );
    randomMaximumGap += Math.max(...gaps);
  }
  expect(randomMaximumGap / 256).toBeGreaterThan((2 * frameCount) / sampleCount);
});

it('integer systematic unequal inclusion has exact first-order probabilities without replacement', () => {
  const tickets = [1, 1, 2, 2];
  const sampleCount = 2;
  const step = 3;
  const frequencies = [0, 0, 0, 0];
  for (let shift = 0; shift < step; shift++) {
    const selected = new Set<number>();
    for (let sample = 0; sample < sampleCount; sample++) {
      const targetMass = shift + sample * step;
      let cumulative = 0;
      for (let row = 0; row < tickets.length; row++) {
        if (cumulative <= targetMass && targetMass < cumulative + tickets[row]) {
          selected.add(row);
          frequencies[row]++;
        }
        cumulative += tickets[row];
      }
    }
    expect(selected.size).toBe(sampleCount);
  }
  expect(frequencies).toEqual(tickets);
});
