// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUClassificationFit} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit';
import {GPULISAMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {getGPUClassBreaksParameterLength} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {
  GPUMaxPRegions,
  GPU_MAX_P_STATUS_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPUAZPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getGraphOutputViews} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUClassBreaksMethod} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';

type ContributorName =
  | 'classification-fit'
  | 'lisa-markov'
  | 'spatial-markov'
  | 'continuous-sample'
  | 'balanced-sample'
  | 'max-p'
  | 'azp';

function createFixture(
  name: ContributorName,
  options: {
    offset?: number;
    rows?: number;
    improvement?: boolean;
    natural?: boolean;
    packed?: boolean;
    lagMethods?: readonly GPUClassBreaksMethod[];
    maximumLagClassCount?: number;
    unusedWeights?: boolean;
  } = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = options.rows ?? 8;
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `binding-${serial++}`, format, length);
  const input =
    name === 'continuous-sample' || name === 'balanced-sample'
      ? view('float32x2', rows + 64)
      : view('float32', rows * (name.includes('markov') ? 2 : 1) + 64);
  const source = graph.createDataView(input.buffer, {
    format: input.format,
    length: rows * (name.includes('markov') ? 2 : 1),
    byteOffset: options.offset ?? 0
  });
  const values = source as GraphDataView<'float32'>;
  const positions = source as GraphDataView<'float32x2'>;
  const weights = {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows),
    weights: view('float32', rows)
  };
  if (options.unusedWeights) {
    const distances = view('float32', rows);
    Object.defineProperty(distances.buffer, 'usage', {value: Buffer.COPY_DST});
    Object.assign(weights, {distances});
  }
  const fields: GraphDataView[] = [source];
  let contributor: GPUCommandNodeProducer;
  if (name === 'classification-fit') {
    const classes = view('uint32', rows);
    const mask = view('uint32', rows);
    const storage = view('uint32', 193);
    const output = {
      counts: options.packed
        ? graph.createDataView(storage.buffer, {format: 'uint32', length: 3})
        : view('uint32', 3),
      medians: options.packed
        ? graph.createDataView(storage.buffer, {format: 'float32', length: 3, byteOffset: 256})
        : view('float32', 3),
      absoluteDeviations: view('float32', 3),
      squaredDeviations: view('float32', 3),
      summary: options.packed
        ? graph.createDataView(storage.buffer, {format: 'float32', length: 5, byteOffset: 512})
        : view('float32', 5)
    };
    contributor = new GPUClassificationFit({values, classes, mask, classCount: 2, output});
    fields.push(classes, mask, ...Object.values(output));
  } else if (name === 'lisa-markov' || name === 'spatial-markov') {
    const maximumLagClassCount = options.maximumLagClassCount ?? 2;
    const classCount = name === 'lisa-markov' ? 5 : 2;
    const conditionCount = name === 'lisa-markov' ? 1 : maximumLagClassCount;
    const output = {
      counts: view('uint32', name === 'lisa-markov' ? 25 : 4 * maximumLagClassCount),
      steadyStates: view(
        'float32',
        name === 'lisa-markov' ? 5 : 2 * (maximumLagClassCount > 1 ? maximumLagClassCount + 1 : 1)
      ),
      probabilities: view('float32', conditionCount * classCount * classCount),
      rowTotals: view('uint32', conditionCount * classCount),
      ignored: view('uint32', 1),
      stateSparsity: view('uint32', 2),
      status: {converged: view('uint32', 1)}
    };
    const mask = view('uint32', rows);
    fields.push(
      mask,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      ...getGraphOutputViews(output)
    );
    if (name === 'lisa-markov') {
      const parameters = view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH);
      const quadrants = view('uint32', rows * 2);
      contributor = new GPULISAMarkov({
        values,
        weights,
        periodCount: 2,
        parameters,
        mask,
        output: {...output, quadrants}
      });
      fields.push(parameters, quadrants);
    } else {
      const classes = view('uint32', rows * 2);
      const lagParameters = view('float32', getGPUClassBreaksParameterLength(maximumLagClassCount));
      const lagBreaks = view('float32', maximumLagClassCount + 1);
      const lagClassCount = view('uint32', 1);
      const lagValues = view('float32', rows * 2);
      const lagClasses = view('uint32', rows * 2);
      contributor = new GPUSpatialMarkov({
        values,
        weights,
        classes,
        classCount: 2,
        periodCount: 2,
        maximumLagClassCount,
        lagMethods: options.lagMethods ?? (options.natural ? ['natural-breaks'] : ['quantile']),
        lagParameters,
        mask,
        output: {...output, lagBreaks, lagClassCount, lagValues, lagClasses}
      });
      fields.push(classes, lagParameters, lagBreaks, lagClassCount, lagValues, lagClasses);
    }
  } else if (name === 'continuous-sample') {
    const legacySites = view('float32', 2);
    const parameters = view('uint32', 2);
    const output = {
      positions: view('float32x2', 2),
      designWeights: view('float32', 2),
      provenance: view('uint32x4', 2),
      status: view('uint32x4', 1)
    };
    contributor = new GPUContinuousSpatialSample({
      frame: {type: 'line', positions, legacySites},
      seed: 1,
      spacing: 1,
      capacity: 2,
      parameters,
      output
    });
    fields.push(legacySites, parameters, ...Object.values(output));
  } else if (name === 'balanced-sample') {
    const frameIds = view('uint32', rows);
    const strata = view('uint32', rows);
    const inclusionTickets = view('uint32', rows);
    const parameters = view('uint32', 2);
    const storage = view('uint32', 193);
    const output = {
      frameIds: options.packed
        ? graph.createDataView(storage.buffer, {format: 'uint32', length: 2})
        : view('uint32', 2),
      inclusionProbabilities: options.packed
        ? graph.createDataView(storage.buffer, {format: 'float32', length: 2, byteOffset: 256})
        : view('float32', 2),
      designWeights: options.packed
        ? graph.createDataView(storage.buffer, {format: 'float32', length: 2, byteOffset: 512})
        : view('float32', 2),
      status: options.packed
        ? graph.createDataView(storage.buffer, {format: 'uint32', length: 2, byteOffset: 520})
        : view('uint32', 2)
    };
    contributor = new GPUSpatialBalancedSample({
      positions,
      frameIds,
      strata,
      inclusionTickets,
      parameters,
      allocations: [{stratumId: 0, sampleCount: 2}],
      bounds: [0, 0, 1, 1],
      seed: 1,
      output
    });
    fields.push(frameIds, strata, inclusionTickets, parameters, ...Object.values(output));
  } else {
    const parameters = view('uint32', 1);
    const extensiveValues = view('float32', rows);
    const threshold = view('float32', 1);
    const storage = view('uint32', 129);
    const output = {
      labels: options.packed
        ? graph.createDataView(storage.buffer, {format: 'uint32', length: rows})
        : view('uint32', rows),
      objective: options.packed
        ? graph.createDataView(storage.buffer, {format: 'float32', length: 1, byteOffset: rows * 4})
        : view('float32', 1),
      status: view('uint32', GPU_MAX_P_STATUS_LAYOUT.length)
    };
    if (name === 'max-p')
      contributor = new GPUMaxPRegions({
        values,
        weights,
        extensiveValues,
        threshold,
        parameters,
        startCount: 4,
        maximumGrowthSteps: 1,
        maximumIterations: options.improvement ? 1 : 0,
        output
      });
    else {
      const initialLabels = view('uint32', rows);
      contributor = new GPUAZPRegions({
        values,
        weights,
        extensiveValues,
        extensiveThreshold: threshold,
        parameters,
        initialLabels,
        regionCapacity: rows,
        maximumIterations: 1,
        output
      });
      fields.push(initialLabels);
    }
    fields.push(
      parameters,
      extensiveValues,
      threshold,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      ...Object.values(output)
    );
  }
  return {device, graph, contributor, source, fields};
}

function assertNoGraphMutation(fixture: ReturnType<typeof createFixture>, expected: RegExp) {
  const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
  const createView = vi.spyOn(fixture.graph, 'createDataView');
  const add = vi.spyOn(fixture.graph, 'add');
  const createBuffer = vi.spyOn(fixture.device, 'createBuffer');
  expect(() => fixture.contributor.getCommandNodes(fixture.graph)).toThrow(expected);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
  expect(createBuffer).not.toHaveBeenCalled();
}

const contributors: ContributorName[] = [
  'classification-fit',
  'lisa-markov',
  'spatial-markov',
  'continuous-sample',
  'balanced-sample',
  'max-p',
  'azp'
];
for (const name of contributors) {
  it(`${name} checks aligned caller ranges before any resource, view or node declaration`, () => {
    const fixture = createFixture(name, {offset: 64});
    try {
      Object.assign(fixture.device.limits, {
        maxStorageBufferBindingSize: fixture.source.length * fixture.source.rowByteLength
      });
      assertNoGraphMutation(fixture, /needs .* bytes.*limit/);
    } finally {
      fixture.device.destroy();
    }
  });

  it(`${name} checks STORAGE usage of every caller binding before graph mutation`, () => {
    const fixture = createFixture(name);
    try {
      for (const field of fixture.fields) {
        const previousUsage = field.buffer.usage;
        Object.defineProperty(field.buffer, 'usage', {value: Buffer.COPY_DST, configurable: true});
        assertNoGraphMutation(fixture, /STORAGE/);
        Object.defineProperty(field.buffer, 'usage', {value: previousUsage, configurable: true});
      }
    } finally {
      fixture.device.destroy();
    }
  });
}

for (const [name, options, limit, expected] of [
  ['lisa-markov', {}, 256, /padded period scratch.*512.*256/],
  // Four lag classes need three inner quantiles: 2 * 3 * 256 uint32 histogram words = 6144 bytes.
  [
    'spatial-markov',
    {maximumLagClassCount: 4},
    2048,
    /lag classifier quantile histogram.*6144.*2048/
  ],
  ['spatial-markov', {natural: true}, 2048, /lag classifier natural histogram.*40980.*2048/],
  ['max-p', {}, 256, /region records.*512.*256/],
  ['max-p', {improvement: true}, 256, /region records.*512.*256/],
  ['azp', {}, 64, /AZP region records.*96.*64/]
] as const) {
  it(`${name} preflights nested scratch at binding limit ${limit} (${JSON.stringify(options)})`, () => {
    const fixture = createFixture(name, options);
    try {
      Object.assign(fixture.device.limits, {maxStorageBufferBindingSize: limit});
      assertNoGraphMutation(fixture, expected);
    } finally {
      fixture.device.destroy();
    }
  });
}

for (const name of ['classification-fit', 'balanced-sample', 'max-p', 'azp'] as const) {
  it(`${name} accepts valid packed outputs and every derived binding fits the preflight limit`, () => {
    const fixture = createFixture(name, {packed: true});
    try {
      const limit = name === 'max-p' ? 512 : name === 'azp' ? 128 : 64;
      Object.assign(fixture.device.limits, {maxStorageBufferBindingSize: limit});
      const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
      const nodes = fixture.contributor.getCommandNodes(fixture.graph);
      expect(nodes.length).toBeGreaterThan(0);
      for (const [descriptor] of allocate.mock.calls) {
        expect(descriptor.byteLength).toBeLessThanOrEqual(limit);
        expect(descriptor.usage! & Buffer.STORAGE).not.toBe(0);
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    } finally {
      fixture.device.destroy();
    }
  });
}

for (const name of ['lisa-markov', 'spatial-markov'] as const) {
  it(`${name} accepts nonaligned period lengths and preflights every allocated nested plane`, () => {
    const fixture = createFixture(name, {
      rows: 65,
      maximumLagClassCount: 6,
      unusedWeights: true,
      lagMethods: [
        'quantile',
        'box-plot',
        'equal-interval',
        'standard-deviation',
        'head-tail',
        'natural-breaks',
        'maximum-breaks',
        'custom'
      ]
    });
    try {
      const limit = 262144;
      Object.assign(fixture.device.limits, {maxStorageBufferBindingSize: limit});
      const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
      const nodes = fixture.contributor.getCommandNodes(fixture.graph);
      expect(nodes.length).toBeGreaterThan(0);
      for (const [descriptor] of allocate.mock.calls) {
        expect(descriptor.byteLength).toBeLessThanOrEqual(limit);
        expect(descriptor.usage! & Buffer.STORAGE).not.toBe(0);
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    } finally {
      fixture.device.destroy();
    }
  });
}
