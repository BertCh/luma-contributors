// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {GPULocalCategoryDependence} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {GPUCategoryPairCounts} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-category-pair-counts';
import {GPUSpatialEntropy} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-spatial-entropy';
import {createNullWebGPUDevice, createInputBuffer} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {watchConstructionFields} from '../spatial-validation/construction-capture-audit.test-utils';

it.each([
  'pair-counts',
  'entropy',
  'colocation',
  'dependence'
] as const)('captures %s categorical, CSR metadata and output getters once', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `categorical-capture-${nextView++}`, format, length);
  try {
    const statusValues = {requiredCount: undefined as GraphDataView<'uint32'> | undefined};
    const status = watchConstructionFields(statusValues, 'weights.status');
    const provenanceValues = {source: 'upload', criterion: 'knn', k: 2, radius: 4, order: 1};
    const provenance = watchConstructionFields(provenanceValues, 'weights.information.provenance');
    const informationValues = {
      rowCount: 8,
      transform: 'binary' as const,
      symmetric: false,
      hasSelfWeights: false,
      provenance: provenance.props
    };
    const information = watchConstructionFields(informationValues, 'weights.information');
    const weightsValues = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 8),
      weights: view('float32', 8),
      distances: view('float32', 8),
      status: status.props,
      information: information.props
    };
    const weights = watchConstructionFields(weightsValues, 'weights');
    const outputValues: Record<string, GraphDataView> =
      kind === 'pair-counts'
        ? {pairCounts: view('uint32', 16), categoryRowCounts: view('uint32', 4)}
        : kind === 'entropy'
          ? {summary: view('float32', 8), pairCounts: view('uint32', 16), counts: view('uint32', 2)}
          : kind === 'colocation'
            ? {
                globalQuotients: view('float32', 16),
                localQuotients: view('float32', 32),
                categoryRowCounts: view('uint32', 4)
              }
            : {statistics: view('float32', 48), categoryRowCounts: view('uint32', 4)};
    const output = watchConstructionFields(outputValues, 'output');
    const values = {
      id: 'categorical-capture',
      weights: weights.props,
      categories: view('uint32', 8),
      categoryCount: 4,
      output: output.props
    };
    const watched = watchConstructionFields(values, 'props');
    const contributor =
      kind === 'pair-counts'
        ? new GPUCategoryPairCounts(watched.props as never)
        : kind === 'entropy'
          ? new GPUSpatialEntropy(watched.props as never)
          : kind === 'colocation'
            ? new GPUColocationQuotient(watched.props as never)
            : new GPULocalCategoryDependence(watched.props as never);
    const watchers = [watched, weights, status, information, provenance, output];
    for (const watcher of watchers) watcher.expectReadOnce();
    const originalWeightView = weightsValues.weights;
    const originalOutputs = {...outputValues};
    values.categoryCount = 1;
    values.categories = view('uint32', 1);
    weightsValues.offsets = view('uint32', 2);
    weightsValues.weights = view('float32', 1);
    weightsValues.status = {requiredCount: view('uint32', 1)};
    provenanceValues.radius = 99;
    for (const field of Object.keys(outputValues)) outputValues[field] = view('uint32', 1);
    expect(contributor.props.categoryCount).toBe(4);
    expect(contributor.props.weights.weights).toBe(originalWeightView);
    expect(contributor.props.weights.information!.provenance!.radius).toBe(4);
    for (const [field, original] of Object.entries(originalOutputs))
      expect((contributor.props.output as Record<string, GraphDataView>)[field]).toBe(original);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    for (const watcher of watchers) watcher.expectReadOnce();
  } finally {
    device.destroy();
  }
});

it('audit-validation refuses loss-bearing weights when the categorical output cannot propagate loss', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `loss-audit-${nextView++}`, format, length);
  try {
    const weights = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    };
    const categories = view('uint32', 8);
    for (const name of [
      'overflow',
      'candidateOverflow',
      'requiredCount',
      'converged',
      'iterationLimitReached',
      'unresolvedCount'
    ] as const) {
      const props = {
        id: 'categorical-loss-audit',
        weights: {...weights, status: {[name]: view('uint32', 1)}},
        categories,
        categoryCount: 2
      };
      for (const construct of [
        () =>
          new GPUCategoryPairCounts({
            ...props,
            output: {pairCounts: view('uint32', 4), categoryRowCounts: view('uint32', 2)}
          }),
        () => new GPUSpatialEntropy({...props, output: {summary: view('float32', 8)}})
      ])
        expect(construct).toThrow(
          new RegExp(`^categorical-loss-audit weights\\.status.*${name}.*complete CSR`)
        );
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'pair-counts',
  'entropy',
  'colocation',
  'dependence'
] as const)('R4 %s refuses a required-count-only truncated two-way link', kind => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  let nextView = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, `truncated-${nextView++}`, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `truncated-${nextView++}`, format, length);
  try {
    // Complete links 0->1 and 1->0 require two slots, but only 0->1 survived.
    // Its pair table [0,1,0,0] would differ from the complete hand oracle [0,1,1,0].
    const props = {
      weights: {
        offsets: input('uint32', Uint32Array.of(0, 1, 1)),
        neighbors: input('uint32', Uint32Array.of(1)),
        weights: input('float32', Float32Array.of(1)),
        status: {requiredCount: input('uint32', Uint32Array.of(2))}
      },
      categories: input('uint32', Uint32Array.of(0, 1)),
      categoryCount: 2
    };
    expect(() =>
      kind === 'pair-counts'
        ? new GPUCategoryPairCounts({
            ...props,
            output: {pairCounts: output('uint32', 4), categoryRowCounts: output('uint32', 2)}
          })
        : kind === 'entropy'
          ? new GPUSpatialEntropy({...props, output: {summary: output('float32', 8)}})
          : kind === 'colocation'
            ? new GPUColocationQuotient({...props, output: {globalQuotients: output('float32', 4)}})
            : new GPULocalCategoryDependence({
                ...props,
                output: {statistics: output('float32', 12)}
              })
    ).toThrow(/weights.status.*complete CSR/);
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
});
