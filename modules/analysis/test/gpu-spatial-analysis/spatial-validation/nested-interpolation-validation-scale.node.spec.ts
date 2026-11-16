// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, getViewBindingRange} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expect, it, vi} from 'vitest';
import {GPUNestedInterpolationValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import {
  auditWGSLModule,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

function createFixture(rows: number, folds = 2) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `nested-scale-${serial++}`, format, length);
  const props = {
    id: 'nested-scale',
    positions: view('float32x2', rows),
    response: view('float32', rows),
    training: view('uint32', folds * rows),
    testing: view('uint32', folds * rows),
    innerBlocks: view('uint32', rows),
    foldCount: folds,
    innerFoldCount: 2,
    neighborCount: 8,
    candidates: [
      {mode: 'rbf' as const, kernel: 'thin-plate-spline' as const, degree: 1 as const, epsilon: 1}
    ],
    output: {
      selected: view('uint32', folds),
      scores: view('float32', folds * 3),
      predictions: view('float32', folds * rows),
      metrics: view('float32', folds * 6),
      status: view('uint32', folds)
    }
  };
  return {device, graph, props, view};
}

it('tiles nested scores above 256 rows and retains the explicit reference envelope', () => {
  const fixture = createFixture(513);
  try {
    expect(() => new GPUNestedInterpolationValidation(fixture.props)).not.toThrow();
    expect(
      () => new GPUNestedInterpolationValidation({...fixture.props, mode: 'reference'})
    ).toThrow(/rows.*256/);
    const contributor = new GPUNestedInterpolationValidation(fixture.props);
    const nodes = contributor.getCommandNodes(fixture.graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.some(kernel => kernel.id.endsWith('-score-tiles'))).toBe(true);
    for (const kernel of sources) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    }
  } finally {
    fixture.device.destroy();
  }
});

it('keeps shader and pipeline counts constant as outer fold count grows', () => {
  const counts: [number, number][] = [];
  for (const folds of [2, 5, 16]) {
    const fixture = createFixture(513, folds);
    try {
      const nodes = new GPUNestedInterpolationValidation(fixture.props).getCommandNodes(
        fixture.graph
      );
      counts.push([
        new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)).size,
        new Set(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true})).size
      ]);
    } finally {
      fixture.device.destroy();
    }
  }
  expect(counts).toEqual([counts[0], counts[0], counts[0]]);
});

it('preflights nested candidate scratch before declaring any resources or subviews', () => {
  const fixture = createFixture(400);
  try {
    const props = {
      ...fixture.props,
      candidates: [...fixture.props.candidates, ...fixture.props.candidates],
      output: {...fixture.props.output, scores: fixture.view('float32', 12)}
    };
    const contributor = new GPUNestedInterpolationValidation(props);
    Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {value: 4096});
    const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
    const createView = vi.spyOn(fixture.graph, 'createDataView');
    expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(/candidate inner predictions/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    fixture.device.destroy();
  }
});

it('merges a constant extreme-valued metric tile without an overflowing zero-weight correction', () => {
  const fixture = createFixture(513);
  try {
    recordWGSLKernelNodes(fixture.graph);
    const nodes = new GPUNestedInterpolationValidation(fixture.props).getCommandNodes(
      fixture.graph
    );
    const node = nodes.find(candidate => candidate.id.endsWith('-metric-merge'))!;
    const record = getWGSLKernelNodeRecord(fixture.graph, node.id)!;
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => {
        const length = getViewBindingRange(binding.view).size / 4;
        const values = binding.type === 'u32' ? new Uint32Array(length) : new Float32Array(length);
        // One nonempty tile with an exactly constant response and perfect predictions.
        if (binding.name === 'partial') {
          values[0] = 64;
          values[5] = 1e20;
        }
        return [index, values];
      })
    );
    new WgslExec(
      new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    ).dispatchWorkgroups(
      'main',
      [...record.workgroups],
      {0: bindings},
      {constants: record.constants}
    );
    const metrics = bindings[record.bindings.findIndex(binding => binding.name === 'metrics')];
    const status = bindings[record.bindings.findIndex(binding => binding.name === 'status')];
    expect(Array.from(metrics.slice(0, 5))).toEqual([64, 0, 0, 0, 0]);
    expect(Number.isNaN(metrics[5])).toBe(true);
    expect(status[0]).toBe(0);
  } finally {
    fixture.device.destroy();
  }
});

it('selects a finite candidate even when its score is the largest float32 value', () => {
  const fixture = createFixture(513);
  try {
    recordWGSLKernelNodes(fixture.graph);
    const nodes = new GPUNestedInterpolationValidation(fixture.props).getCommandNodes(
      fixture.graph
    );
    const node = nodes.find(candidate => candidate.id.endsWith('-select'))!;
    const record = getWGSLKernelNodeRecord(fixture.graph, node.id)!;
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => {
        const length = getViewBindingRange(binding.view).size / 4;
        const values = binding.type === 'u32' ? new Uint32Array(length) : new Float32Array(length);
        if (binding.name === 'scores') values[0] = Math.fround(3.4028234663852886e38);
        return [index, values];
      })
    );
    new WgslExec(
      new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    ).dispatchWorkgroups(
      'main',
      [...record.workgroups],
      {0: bindings},
      {constants: record.constants}
    );
    const selected = bindings[record.bindings.findIndex(binding => binding.name === 'selected')];
    const status = bindings[record.bindings.findIndex(binding => binding.name === 'status')];
    expect(selected[0]).toBe(0);
    expect(status[0]).toBe(0);
  } finally {
    fixture.device.destroy();
  }
});
