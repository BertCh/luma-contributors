// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUGeographicalDetector} from '../../../src/gpu-spatial-analysis/geographical-detector/gpu-geographical-detector';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPU_REGION_PARTITION_EVALUATION_LAYOUT} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {executeSpatialAuditKernel} from '../spatial-validation/spatial-design-audit.test-utils';
import {watchConstructionFields} from '../spatial-validation/construction-capture-audit.test-utils';
import {
  getGeographicalDetectorPValue,
  getNonCentralFUpperTail,
  readGeographicalDetectorPValue
} from '../../../src/gpu-spatial-analysis/geographical-detector/geographical-detector-p-value';
import {DETECTOR_NORMALIZATION_WGSL} from '../../../src/gpu-spatial-analysis/geographical-detector/geographical-detector-normalization';
import {findReservedIdentifiers} from '../../../../../examples/deck/spatial-analysis-showcase/scripts/check-wgsl-identifiers.mjs';

it('F6 detector normalization uses valid WGSL identifiers', () => {
  expect(findReservedIdentifiers(DETECTOR_NORMALIZATION_WGSL)).toEqual([]);
});

it('B3 detector ratios retain within variation under power-of-two rescaling', () => {
  for (const power of [0, -76, -145]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `rescaling-${serial++}`, format, length);
    try {
      const contributor = new GPUGeographicalDetector({
        values: view('float32', 4),
        labels: view('uint32', 4),
        labelCapacity: 2,
        output: {summary: view('float32', 8)}
      });
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      const values = Float32Array.from([0, 2, 4, 8], value => value * 2 ** power);
      const magnitudeRows = new Uint32Array(4);
      const magnitude = new Uint32Array(1);
      const normalized = Float32Array.from(values);
      for (const kernel of kernels.filter(kernel => kernel.id.endsWith('-magnitude-rows')))
        executeSpatialAuditKernel(kernel, {
          values,
          labels: Uint32Array.of(0, 0, 1, 1),
          magnitudeRows
        });
      magnitude[0] = Math.max(...magnitudeRows);
      for (const kernel of kernels.filter(kernel => kernel.id.endsWith('-normalize-values')))
        executeSpatialAuditKernel(kernel, {values, magnitude, normalized});
      // Independent two-group float32 moment transcription. Hand values before scale:
      // SSW=10h², SSB=25h², SST=35h², q=5/7, F=5, noncentrality=10/7.
      const firstMean = Math.fround((normalized[0] + normalized[1]) / 2);
      const secondMean = Math.fround((normalized[2] + normalized[3]) / 2);
      const mean = Math.fround((firstMean + secondMean) / 2);
      const sumSquares = (numbers: number[]) =>
        numbers.reduce((sum, value) => Math.fround(sum + Math.fround(value * value)), 0);
      const within = sumSquares(
        Array.from(normalized, (value, row) =>
          Math.fround(value - (row < 2 ? firstMean : secondMean))
        )
      );
      const between = sumSquares([
        firstMean - mean,
        firstMean - mean,
        secondMean - mean,
        secondMean - mean
      ]);
      const total = sumSquares(Array.from(normalized, value => Math.fround(value - mean)));
      const evaluation = new Float32Array(GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
      evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount] = 2;
      evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = within;
      evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = between;
      evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = total;
      evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = (2 * between) / within;
      const summary = new Float32Array(8);
      for (const kernel of kernels.filter(kernel => kernel.id === 'geographical-detector-summary'))
        executeSpatialAuditKernel(kernel, {
          evaluation,
          magnitude,
          tiles: Float32Array.of(
            firstMean * firstMean + secondMean * secondMean,
            Math.SQRT2 * (firstMean + secondMean)
          ),
          summary
        });
      expect(Number.isFinite(summary[0])).toBe(true);
      expect(Math.abs(summary[0] - 5 / 7)).toBeLessThan(1e-6);
      expect(summary[1]).toBe(5);
      expect(Number.isFinite(summary[2])).toBe(true);
      expect(Math.abs(summary[2] - 10 / 7)).toBeLessThan(1e-6);
      expect(summary[7]).toBe(Math.fround(35 * 2 ** (2 * power)));
    } finally {
      device.destroy();
    }
  }
});

it('W5-26 perfect partitions publish exact q one while constants remain undefined', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `perfect-q-${serial++}`, format, length);
  try {
    const contributor = new GPUGeographicalDetector({
      values: view('float32', 5),
      labels: view('uint32', 5),
      labelCapacity: 2,
      output: {summary: view('float32', 8)}
    });
    const kernels = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id === 'geographical-detector-summary')
    );
    const evaluation = new Float32Array(GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount] = 2;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 0;
    // [0] and [1,1,1,1]: mean=4/5, SSB=SST=4/5, SSW=0 exactly.
    // Independently reduced float32 SSB/SST can differ by two ulps, as in R2.
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = 0.7999999523162842;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = 0.8000000715255737;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = Infinity;
    const summary = new Float32Array(8);
    for (const kernel of kernels)
      executeSpatialAuditKernel(kernel, {
        evaluation,
        magnitude: new Uint32Array(1),
        tiles: Float32Array.of(1, 2),
        summary
      });
    expect(summary[0]).toBe(1);
    expect(summary[1]).toBe(Infinity);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = 0;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = 0;
    for (const kernel of kernels)
      executeSpatialAuditKernel(kernel, {
        evaluation,
        magnitude: new Uint32Array(1),
        tiles: Float32Array.of(0, 0),
        summary
      });
    expect(summary[0]).toBeNaN();
    expect(summary[1]).toBeNaN();
  } finally {
    device.destroy();
  }
});

it('CPL-14 retains complete and incumbent partition provenance without requiring categorical certificates', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `provenance-${serial++}`, format, length);
  try {
    const props = {
      values: view('float32', 4),
      labels: view('uint32', 4),
      output: {summary: view('float32', 8)}
    };
    const ordinary = new GPUGeographicalDetector(props).getCommandNodes(graph);
    expect(ordinary.some(node => node.id.endsWith('-partition-provenance'))).toBe(false);
    const certified = new GPUGeographicalDetector({
      ...props,
      id: 'certified',
      labelsComplete: view('uint32', 1),
      output: {...props.output, labelsComplete: view('uint32', 1)}
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      certified.getCommandNodes(graph).filter(node => node.id.endsWith('-partition-provenance'))
    );
    const complete = new Uint32Array(1);
    for (const word of [1, 0, 1]) {
      executeSpatialAuditKernel(kernel, {labelsComplete: Uint32Array.of(word), complete});
      expect(Array.from(complete)).toEqual([word]);
    }
    expect(
      () => new GPUGeographicalDetector({...props, labelsComplete: view('uint32', 1)})
    ).toThrow(/both be supplied/);
  } finally {
    device.destroy();
  }
});

it('review 2 R2 detector retains directly accumulated small between-stratum variation', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `small-between-${nextView++}`, format, length);
  try {
    const contributor = new GPUGeographicalDetector({
      values: view('float32', 8),
      labels: view('uint32', 8),
      labelCapacity: 2,
      output: {summary: view('float32', 8)}
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id === 'geographical-detector-summary')
    );
    const evaluation = new Float32Array(9),
      summary = new Float32Array(8);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount] = 2;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 8;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = 2 ** -25;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = 8;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = 6 * 2 ** -28;
    executeSpatialAuditKernel(kernel, {
      evaluation,
      magnitude: new Uint32Array(1),
      tiles: Float32Array.of(2 ** -26, 2 ** -12),
      summary
    });
    // Two means 0 and 2^-13 give SSB=8*(2^-14)^2=2^-25; SSW=8.
    expect(Array.from(summary)).toEqual([2 ** -28, 6 * 2 ** -28, 2 ** -27, 2, 8, 1, 6, 8]);
    expect(Math.abs(getGeographicalDetectorPValue(summary) - 0.999885559082955)).toBeLessThan(
      2e-12
    );
  } finally {
    device.destroy();
  }
});

it('review 2 R3 detector keeps exponent-safe representable F from the partition', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `large-between-${nextView++}`, format, length);
  try {
    const contributor = new GPUGeographicalDetector({
      values: view('float32', 8),
      labels: view('uint32', 8),
      labelCapacity: 7,
      output: {summary: view('float32', 8)}
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id === 'geographical-detector-summary')
    );
    const evaluation = new Float32Array(9),
      summary = new Float32Array(8);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount] = 7;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 2 ** -101;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = 6 * 2 ** 26;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = 6 * 2 ** 26;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = 2 ** 127;
    executeSpatialAuditKernel(kernel, {
      evaluation,
      magnitude: new Uint32Array(1),
      tiles: Float32Array.of(6 * 2 ** 26, 0),
      summary
    });
    // SSB=6*2^26, SSW=2^-101 and df=(6,1): F=2^127, not overflow.
    expect(Array.from(summary)).toEqual([1, 2 ** 127, 8, 7, 8, 6, 1, 6 * 2 ** 26]);
    expect(getGeographicalDetectorPValue(summary)).toBeGreaterThan(0);
    expect(Math.abs(getGeographicalDetectorPValue(summary) / 9.0780105121473e-20 - 1)).toBeLessThan(
      2e-8
    );
  } finally {
    device.destroy();
  }
});

it.each([
  200000000, 199999984
])('R5 nearly perfect factor strata retain finite F and a positive noncentral tail (packed SSB=%s)', packedBetween => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `near-perfect-${nextView++}`, format, length);
  try {
    const contributor = new GPUGeographicalDetector({
      values: view('float32', 8),
      labels: view('uint32', 8),
      labelCapacity: 2,
      output: {summary: view('float32', 8)}
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id === 'geographical-detector-summary')
    );
    const evaluation = new Float32Array(16),
      summary = new Float32Array(8);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount] = 2;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 1.5;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = 200000001.5;
    // Direct float32 SSD reduction may be one ulp below the exact SSB=200000000.
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = packedBetween;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = 800000000;
    executeSpatialAuditKernel(
      kernel,
      {
        evaluation,
        magnitude: new Uint32Array(1),
        tiles: Float32Array.of(100005000.125, 20001),
        summary
      },
      {reciprocalDivision: true}
    );
    // NumPy/SciPy oracle: SSB=200000000, SSW=1.5, df=(1,6), lambda rounds to 2.
    // Exact q=200000000/200000001.5 rounds to 1, but independently accumulated
    // float32 SSDs and division need not. Two epsilons at unity cover those errors.
    expect(Number.isFinite(summary[0])).toBe(true);
    expect(Math.abs(summary[0] - 1)).toBeLessThanOrEqual(2 ** -22);
    expect(summary[1]).toBe(800000000);
    const probability = getGeographicalDetectorPValue(summary);
    expect(probability).toBeGreaterThan(0);
    expect(Math.abs(probability / 1.5205077624435442e-24 - 1)).toBeLessThan(2e-8);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 0;
    executeSpatialAuditKernel(kernel, {
      evaluation,
      magnitude: new Uint32Array(1),
      tiles: Float32Array.of(100005000.125, 20001),
      summary
    });
    expect(summary[1]).toBe(Infinity);
    // Positive within SS with unrepresentable finite F must not impersonate an exact split.
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 1e-30;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = Infinity;
    executeSpatialAuditKernel(kernel, {
      evaluation,
      magnitude: new Uint32Array(1),
      tiles: Float32Array.of(100005000.125, 20001),
      summary
    });
    expect(summary[1]).toBeNaN();
  } finally {
    device.destroy();
  }
});

it('captures geographical detector getters and its summary map once before encoding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `detector-capture-${nextView++}`, format, length);
  try {
    const outputValues = {summary: view('float32', 8)};
    const output = watchConstructionFields(outputValues, 'output');
    const values = {
      id: 'detector-capture',
      values: view('float32', 8),
      labels: view('uint32', 8),
      labelCapacity: 3,
      output: output.props
    };
    const originalSummary = outputValues.summary,
      originalLabels = values.labels;
    const watched = watchConstructionFields(values, 'props');
    const contributor = new GPUGeographicalDetector(watched.props);
    watched.expectReadOnce();
    output.expectReadOnce();
    values.labelCapacity = 1;
    values.values = view('float32', 1);
    values.labels = view('uint32', 1);
    outputValues.summary = view('float32', 1);
    expect(contributor.props.labelCapacity).toBe(3);
    expect(contributor.props.labels).toBe(originalLabels);
    expect(contributor.props.output.summary).toBe(originalSummary);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    watched.expectReadOnce();
    output.expectReadOnce();
  } finally {
    device.destroy();
  }
});

it('R5 bounded summary merge retains the final partial above 4096 tiles', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `detector-merge-${nextView++}`, format, length);
  try {
    const contributor = new GPUGeographicalDetector({
      values: view('float32', 8),
      labels: view('uint32', 8),
      labelCapacity: 4096 * 256 + 1,
      output: {summary: view('float32', 8)}
    });
    const kernels = getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(node => node.id.startsWith('geographical-detector-summary'))
    );
    expect(kernels).toHaveLength(2);
    const groups = new Float32Array(4);
    // 4097 pairs (1,0) have exact integer sums (4097,0); losing the final tile fails.
    executeSpatialAuditKernel(kernels[0], {
      source: Float32Array.from({length: 4097 * 2}, (_, index) => (index % 2 === 0 ? 1 : 0)),
      groups
    });
    expect(Array.from(groups)).toEqual([4096, 0, 1, 0]);
    const evaluation = new Float32Array(16),
      summary = new Float32Array(8);
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount] = 2;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares] = 2;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSumOfSquares] = 8;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.betweenSumOfSquares] = 6;
    evaluation[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF] = 18;
    executeSpatialAuditKernel(kernels[1], {
      evaluation,
      magnitude: new Uint32Array(1),
      tiles: groups,
      summary
    });
    expect(Array.from(summary)).toEqual([0.75, 18, 4097, 2, 8, 1, 6, 8]);
  } finally {
    device.destroy();
  }
});

it('audit-validation noncentral F tails match independent SciPy literals', () => {
  // geographical-detector-audit.oracle.py; relative 2e-10 covers binary64 beta/Poisson sums.
  for (const [value, numerator, denominator, noncentrality, expected] of [
    [0.1, 1.5, 40, 0, 0.8498329955931397],
    [3, 5, 40, 8, 0.35681241324281293],
    [1.25, 1, 7, 2, 0.6347949893164966],
    [25, 3, 50, 20, 0.0011778737406814518],
    [5000, 2, 300, 10000, 0.5108717881837139]
  ]) {
    const actual = getNonCentralFUpperTail(value, numerator, denominator, noncentrality);
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs(actual - expected)).toBeLessThan(2e-10 * expected);
  }
});

it('audit-validation summary p-values retain the documented shift-dependent noncentrality', () => {
  for (const [noncentrality, total, expected] of [
    [7.7618241754832376, 258.22222222222223, 0.021629971159284507],
    [8.580640170465687, 1032.888888888889, 0.024860157244734598]
  ]) {
    const summary = new Float32Array([
      0.9412650602409639,
      26.709401709401735,
      noncentrality,
      4,
      9,
      3,
      5,
      total
    ]);
    const probability = getGeographicalDetectorPValue(summary);
    expect(Number.isFinite(probability)).toBe(true);
    expect(Math.abs(probability - expected)).toBeLessThan(1e-12);
    expect(readGeographicalDetectorPValue(summary)).toEqual({
      pValue: probability,
      unavailableReason: null
    });
  }
});
