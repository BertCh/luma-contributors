// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  createGeneralizedGWRReviewScene,
  computeGeneralizedGWRReviewExpectations
} from './generalized-gwr-scalability-oracle';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {
  GENERALIZED_GWR_FOCUS_BATCH_ROWS,
  GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT,
  GENERALIZED_GWR_GRID_NODE_ALLOWANCE,
  GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES,
  getGeneralizedGWRScoreMergeLayout
} from '../../../src/gpu-spatial-analysis/spatial-regression/generalized-gwr-limits';

function setup(rows: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scale-${serial++}`, format, length);
  const props = {
    id: 'generalized-scale',
    family: 'poisson' as const,
    kernel: 'bisquare' as const,
    positions: view('float32x2', rows),
    predictors: view('float32', rows),
    predictorCount: 1,
    response: view('float32', rows),
    parameters: view('float32', 2),
    output: {
      coefficients: view('float32', rows * 2),
      fitted: view('float32', rows),
      status: {codes: view('uint32', rows), iterationCount: view('uint32', rows)}
    }
  };
  const selection = (maximumIterations: number) => ({
    method: 'golden-section' as const,
    parameters: view('float32', 3),
    maximumIterations,
    output: {
      scores: view('float32', maximumIterations * 2),
      bandwidth: view('float32', 1),
      status: view('uint32', 1)
    }
  });
  return {device, graph, view, props, selection};
}

it('accepts the full indexed envelope and retains dense/exact Gaussian reference caps', () => {
  for (const rows of [1048577, GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT]) {
    const context = setup(rows);
    try {
      const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
        ...context.props,
        ...(rows === GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT
          ? {indexGridSize: [4096, 8191] as const}
          : {})
      });
      expect(contributor.props.indexGridSize).toBeDefined();
      const nodes = contributor.getCommandNodes(context.graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      // Account for every node returned by the shared grid, including W2-GWR's extent reduction.
      expect(nodes.findIndex(node => node.id.endsWith('-focus-init'))).toBeLessThanOrEqual(
        GENERALIZED_GWR_GRID_NODE_ALLOWANCE
      );
      expect(nodes.filter(node => /-fit-tile\d+$/.test(node.id))).toHaveLength(
        Math.ceil(rows / GENERALIZED_GWR_FOCUS_BATCH_ROWS)
      );
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...context.props,
            indexGridSize: false
          })
      ).toThrow(/response rows.*65536/);
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...context.props,
            kernel: 'gaussian',
            indexGridSize: [64, 64]
          })
      ).toThrow(/response rows.*1048576/);
    } finally {
      context.device.destroy();
    }
  }
  const context = setup(GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT + 1);
  try {
    expect(() => new GPUGeneralizedGeographicallyWeightedRegression(context.props)).toThrow(
      /response rows.*16777216/
    );
  } finally {
    context.device.destroy();
  }
});

it('fit and publish pipeline counts stay constant as focal tiles grow, with full reserved-word auditing', () => {
  const counts: number[] = [];
  for (const rows of [65537, 1048577]) {
    const context = setup(rows);
    try {
      const contributor = new GPUGeneralizedGeographicallyWeightedRegression(context.props);
      const nodes = contributor
        .getCommandNodes(context.graph)
        .filter(node => /-(fit|publish|focus-advance)-tile\d+$/.test(node.id));
      counts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
      // Audit every distinct generated identifier, including the cursor/header paths.
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const source of new Map(sources.map(source => [source.source, source])).values()) {
        expect(auditWGSLModule(source.source, source.constants), source.id).toEqual([]);
        const reserved = WGSL_RESERVED_WORDS;
        const identifiers =
          source.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_]\w*/g) ?? [];
        expect(
          identifiers.filter(identifier => reserved.has(identifier)),
          source.id
        ).toEqual([]);
        expect(new WgslReflect(source.source).storage.length).toBeLessThanOrEqual(8);
      }
    } finally {
      context.device.destroy();
    }
  }
  expect(counts).toEqual([3, 3]);
});

it('candidate and golden-round pipeline counts stay constant and indexed AICc uses every focal tile', () => {
  const counts: number[] = [];
  for (const maximumIterations of [1, 4]) {
    const context = setup(1048577);
    try {
      const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
        ...context.props,
        bandwidthSelection: context.selection(maximumIterations)
      });
      const nodes = contributor.getCommandNodes(context.graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const searchNodes = nodes.filter(node => /-candidate-|search-\d+-iterate/.test(node.id));
      counts.push(getGPUCommandNodePipelineKeys(searchNodes, {includeShaderLayout: true}).size);
      const sources = getGPUCommandNodeKernelSources(searchNodes);
      for (const source of new Map(sources.map(source => [source.source, source])).values()) {
        expect(auditWGSLModule(source.source, source.constants), source.id).toEqual([]);
        const reserved = WGSL_RESERVED_WORDS;
        const identifiers =
          source.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_]\w*/g) ?? [];
        expect(
          identifiers.filter(identifier => reserved.has(identifier)),
          source.id
        ).toEqual([]);
        expect(new WgslReflect(source.source).storage.length).toBeLessThanOrEqual(8);
      }
      expect(nodes.filter(node => /-candidate-0-fit-tile/.test(node.id))).toHaveLength(17);
    } finally {
      context.device.destroy();
    }
  }
  expect(counts[0]).toBe(counts[1]);
});

it('checks aligned caller bindings and grid scratch before declaring any graph resources', () => {
  for (const failure of ['aligned-view', 'grid-scratch'] as const) {
    const context = setup(64);
    try {
      const parameters =
        failure === 'aligned-view'
          ? context.graph.createDataView(context.view('float32', 1025).buffer, {
              format: 'float32',
              byteOffset: 4,
              length: 1024
            })
          : context.props.parameters;
      const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
        ...context.props,
        parameters,
        indexGridSize: failure === 'grid-scratch' ? [1024, 1024] : undefined
      });
      Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {value: 4096});
      const declarations = vi.spyOn(context.graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(context.graph)).toThrow(
        failure === 'aligned-view' ? /view binding.*4100/ : /grid cell counts/
      );
      expect(declarations).not.toHaveBeenCalled();
    } finally {
      context.device.destroy();
    }
  }
});

it.each([
  {rows: 65536, fractional: false},
  {rows: 65537, fractional: false},
  {rows: 1048576, fractional: false},
  {rows: 1048577, fractional: false},
  {rows: 1048600, fractional: false},
  {rows: 1048576, fractional: true},
  {rows: 1048584, fractional: true}
])('bounded/parallel score merging matches global AICc at $rows rows (fractional=$fractional)', ({
  rows,
  fractional
}) => {
  const context = setup(rows);
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      indexGridSize: [64, 64],
      bandwidthSelection: context.selection(1)
    });
    const sources = getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(context.graph)
        .filter(node => /candidate-0-(merge-scores|score)$/.test(node.id))
    );
    const tileCount = Math.ceil(rows / 256);
    const tiles = new Float32Array(tileCount * 4);
    for (let tile = 0; tile < tileCount; tile++) {
      const count = Math.min(256, rows - tile * 256);
      tiles.set(
        fractional
          ? [count * (0.7 + (tile % 7) / 100), count * 0.08, count, 0]
          : [3 * count, count / 4, count, 0],
        tile * 4
      );
    }
    const totals = new Float32Array(Math.ceil(tileCount / 256) * 4);
    const hasMerge = sources.some(source => source.id.includes('merge-scores'));
    const scores = new Float32Array(2);
    const parameters = new Float32Array(3);
    for (const source of sources) {
      const isMerge = source.id.includes('merge-scores');
      const buffers: Record<string, Float32Array> = {
        tiles: isMerge || !hasMerge ? tiles : totals,
        totals,
        scores,
        parameters
      };
      const bindings = Object.fromEntries(
        new WgslReflect(source.source).storage.map(binding => [
          binding.binding,
          buffers[binding.name]
        ])
      );
      // The interpreter parses a single multi-digit workgroup size as separate digits;
      // spelling out all three dimensions preserves the emitted dispatch shape.
      new WgslExec(
        new WgslParser().parse(
          source.source
            .replace(/^\s*_\s*=.*;$/gm, '')
            .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)')
        )
      ).dispatchWorkgroups('main', 1, {0: bindings}, {constants: source.constants});
    }
    // Independent float64 oracle sums the supplied float32 partials once, with no staged merge.
    let deviance = 0;
    let trace = 0;
    let historicalDeviance = 0;
    let historicalTrace = 0;
    for (let tile = 0; tile < tileCount; tile++) {
      deviance += tiles[4 * tile];
      trace += tiles[4 * tile + 1];
      historicalDeviance = Math.fround(historicalDeviance + tiles[4 * tile]);
      historicalTrace = Math.fround(historicalTrace + tiles[4 * tile + 1]);
    }
    const expected = deviance + 2 * trace + (2 * trace * (trace + 1)) / (rows - trace - 1);
    if (fractional) {
      // These inputs detect the long float32 running-sum error that exact integer totals hide.
      const historicalScore =
        historicalDeviance +
        2 * historicalTrace +
        (2 * historicalTrace * (historicalTrace + 1)) / (rows - historicalTrace - 1);
      expect(Math.abs(historicalScore - expected)).toBeGreaterThan(2);
    }
    expect(Math.abs(scores[0] - expected)).toBeLessThan(1);
    expect(parameters[2]).toBe(1);
    expect(
      (hasMerge ? totals : tiles)
        .filter((_, index) => index % 4 === 2)
        .reduce((sum, count) => sum + count, 0)
    ).toBe(rows);
  } finally {
    context.device.destroy();
  }
});

it('reuses the focus cursor and publishes only the valid tail of a partial final tile', () => {
  const rows = 65544;
  const context = setup(rows);
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression(context.props);
    const nodes = contributor
      .getCommandNodes(context.graph)
      .filter(node => /focus-init$|focus-advance-tile0$|publish-tile1$/.test(node.id));
    const sources = getGPUCommandNodeKernelSources(nodes);
    const fits = new Float32Array(65536 * 5 + 1);
    const buffers: Record<string, Float32Array | Uint32Array> = {
      fits,
      dispatch: new Uint32Array(3),
      coefficients: new Float32Array(rows * 2),
      fitted: new Float32Array(rows),
      status: new Uint32Array(rows),
      iterations: new Uint32Array(rows)
    };
    const execute = (suffix: string) => {
      const source = sources.find(source => source.id.endsWith(suffix))!;
      const bindings = Object.fromEntries(
        new WgslReflect(source.source).storage.map(binding => [
          binding.binding,
          buffers[binding.name]
        ])
      );
      new WgslExec(
        new WgslParser().parse(
          source.source
            .replace(/^\s*_\s*=.*;$/gm, '')
            .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)')
        )
      ).dispatchWorkgroups('main', 1, {0: bindings}, {constants: source.constants});
    };
    execute('-focus-init');
    execute('-focus-advance-tile0');
    expect(fits[0]).toBe(65536);
    expect(Array.from(buffers['dispatch'])).toEqual([1, 1, 1]);
    for (let row = 0; row < 8; row++) fits.set([2, -3, row + 10, 0, 4], 1 + row * 5);
    execute('-publish-tile1');
    expect(Array.from(buffers['coefficients'].slice(65536 * 2))).toEqual(
      Array.from({length: 8}, () => [2, -3]).flat()
    );
    expect(Array.from(buffers['fitted'].slice(65536))).toEqual([10, 11, 12, 13, 14, 15, 16, 17]);
    expect(Array.from(buffers['iterations'].slice(65536))).toEqual(Array(8).fill(4));
    expect(buffers['fitted'][65535]).toBe(0);
    execute('-focus-advance-tile0');
    expect(buffers['dispatch'][0]).toBe(0);
  } finally {
    context.device.destroy();
  }
});

it('gates every candidate focus-control node, including 864 controls after invalid bounds', () => {
  const context = setup(1048584);
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      bandwidthSelection: context.selection(24)
    });
    const controls = contributor
      .getCommandNodes(context.graph)
      .filter(node => /candidate-.*-focus-(init|advance)/.test(node.id));
    expect(controls).toHaveLength(864);
    for (const node of controls) {
      expect(node.condition?.source).toBe('gpu');
      if (node.condition?.source === 'gpu') expect(node.condition.byteOffset).toBe(12);
    }
  } finally {
    context.device.destroy();
  }
});

it('rejects aligned co-bound output overlaps while accepting aligned adjacent fields', () => {
  const context = setup(16);
  try {
    for (const byteOffset of [64, 256]) {
      const storage = context.view('uint32', 80);
      const fitted = context.graph.createDataView(storage.buffer, {format: 'float32', length: 16});
      const codes = context.graph.createDataView(storage.buffer, {
        format: 'uint32',
        byteOffset,
        length: 16
      });
      const construct = () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...context.props,
          output: {...context.props.output, fitted, status: {...context.props.output.status, codes}}
        });
      if (byteOffset === 64) expect(construct).toThrow(/output.*overlap/);
      else expect(construct).not.toThrow();
    }
  } finally {
    context.device.destroy();
  }
});

it('independent paired-population fixtures have opposing tile preferences and distinctive tails', () => {
  for (const family of ['poisson', 'logistic'] as const)
    for (const rows of [65536, 65552, 1048584]) {
      const scene = createGeneralizedGWRReviewScene(rows, family, true);
      const narrow = computeGeneralizedGWRReviewExpectations(scene, 0.5);
      const wide = computeGeneralizedGWRReviewExpectations(scene, 2);
      expect(narrow.score(0, 65536)).toBeLessThan(wide.score(0, 65536));
      if (rows <= 65552) expect(narrow.score()).toBeLessThan(wide.score());
      else {
        expect(wide.score()).toBeLessThan(narrow.score());
        for (const expected of [narrow, wide])
          expect(Math.abs(expected.score() - expected.score(0, 1048576))).toBeGreaterThan(8);
      }
    }
});

it('audits all owned generated kernels against the W3C reserved-word list', () => {
  for (const family of ['poisson', 'logistic'] as const)
    for (const method of ['grid', 'golden'] as const) {
      const context = setup(32);
      try {
        const selection = context.selection(2);
        const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
          ...context.props,
          family,
          indexGridSize: [4, 4],
          mask: context.view('uint32', 32),
          offset: context.view('float32', 32),
          inference: {
            standardErrors: context.view('float32', 64),
            influence: context.view('float32', 32)
          },
          bandwidthSelection:
            method === 'golden' ? selection : {bandwidths: [1, 2, 3, 4], output: selection.output}
        });
        const nodes = contributor
          .getCommandNodes(context.graph)
          .filter(node => !/-grid-(?!positions)/.test(node.id));
        const sources = getGPUCommandNodeKernelSources(nodes);
        for (const source of new Map(sources.map(source => [source.source, source])).values()) {
          expect(auditWGSLModule(source.source, source.constants), source.id).toEqual([]);
          const identifiers =
            source.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_]\w*/g) ?? [];
          expect(
            identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier)),
            source.id
          ).toEqual([]);
        }
      } finally {
        context.device.destroy();
      }
    }
});

it('preflights the shared aligned extent scratch before allocating any focal or grid resource', () => {
  const context = setup(1);
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      indexGridSize: [1, 1]
    });
    Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {value: 512});
    const declarations = vi.spyOn(context.graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(context.graph)).toThrow(/grid tile bounds.*1024/);
    expect(declarations).not.toHaveBeenCalled();
  } finally {
    context.device.destroy();
  }
});

it('rejects non-finite and out-of-float32-range shader props at construction', () => {
  const context = setup(32);
  try {
    for (const value of [NaN, Infinity, -Infinity, Number.MAX_VALUE, 3.5e38]) {
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...context.props,
            gaussianCutoff: value
          })
      ).toThrow(/generalized-scale.*gaussianCutoff/);
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...context.props,
            bandwidthSelection: {bandwidths: [1, value], output: context.selection(1).output}
          })
      ).toThrow(/generalized-scale.*bandwidths\[1\]/);
      expect(
        () =>
          new GPUGeneralizedGeographicallyWeightedRegression({
            ...context.props,
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              units: 'meters',
              sphereRadius: value
            }
          })
      ).toThrow(/generalized-scale.*sphereRadius/);
    }
  } finally {
    context.device.destroy();
  }
});

it('bounds actual final-score bindings at the maximum supported rows and covers every partial', () => {
  for (const rows of [
    1,
    65536,
    65537,
    1048576,
    1048584,
    GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT
  ]) {
    const layout = getGeneralizedGWRScoreMergeLayout(rows);
    expect(layout.finalTileCount).toBeLessThanOrEqual(GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES);
    // Adjacent groups partition the complete tile range, including a partial final group.
    const groupWidth = layout.usesParallelMerge ? 256 : 1;
    let covered = 0;
    for (let group = 0; group < layout.finalTileCount; group++)
      covered += Math.min(groupWidth, layout.tileCount - group * groupWidth);
    expect(covered).toBe(Math.ceil(rows / 256));
  }
  const context = setup(GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT);
  try {
    const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
      ...context.props,
      bandwidthSelection: context.selection(1)
    });
    const nodes = contributor.getCommandNodes(context.graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const scoreNode = nodes.find(node => node.id.endsWith('-candidate-0-score'))!;
    const readBinding = scoreNode.resources!.find(
      resource => 'buffer' in resource && resource.usage === 'storage-read'
    );
    expect(readBinding && 'buffer' in readBinding && 'length' in readBinding.buffer).toBe(true);
    if (readBinding && 'buffer' in readBinding && 'length' in readBinding.buffer) {
      expect(readBinding.buffer.length / 4).toBe(256);
      expect(readBinding.buffer.length / 4).toBeLessThanOrEqual(
        GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES
      );
    }
    const mergeNode = nodes.find(node => node.id.endsWith('-candidate-0-merge-scores'))!;
    expect(mergeNode).toBeDefined();
    const mergedWrite = mergeNode.resources!.find(
      resource => 'buffer' in resource && resource.usage === 'storage-read-write'
    );
    expect(
      mergedWrite &&
        'buffer' in mergedWrite &&
        readBinding &&
        'buffer' in readBinding &&
        mergedWrite.buffer === readBinding.buffer
    ).toBe(true);
  } finally {
    context.device.destroy();
  }
});
