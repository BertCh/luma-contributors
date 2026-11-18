// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, getViewBindingRange} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it, vi} from 'vitest';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
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

function createFixture(rows = 64) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `f-reg-${serial++}`, format, length);
  const sur = {
    id: 'sur-improvement',
    equationCount: 2,
    predictorCount: 1,
    rowCount: rows,
    predictors: view('float32', 2 * rows),
    responses: view('float32', 2 * rows),
    output: {
      coefficients: view('float32', 4),
      covariance: view('float32', 16),
      residualCovariance: view('float32', 4),
      residuals: view('float32', 2 * rows),
      status: view('uint32', 1)
    }
  };
  const panel = {
    id: 'panel-improvement',
    periodCount: 4,
    predictorCount: 1,
    predictors: view('float32', 16),
    response: view('float32', 16),
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    output: {
      coefficients: view('float32', 4),
      covariance: view('float32', 9),
      summary: view('float32', 6),
      status: view('uint32', 1),
      fitted: view('float32', 16),
      residuals: view('float32', 16)
    }
  };
  return {device, graph, view, sur, panel};
}

it('tiled panel profile retains the variance minimum when a nonzero floor hides rho differences', () => {
  const fixture = createFixture();
  try {
    recordWGSLKernelNodes(fixture.graph);
    const contributor = new GPUSpatialPanelLagMaximumLikelihood({
      ...fixture.panel,
      effects: 'unit',
      fitMode: 'tiled',
      eigenvalues: fixture.view('float32', 4)
    });
    const node = contributor
      .getCommandNodes(fixture.graph)
      .find(node => node.id.endsWith('-profile-rho'))!;
    const record = getWGSLKernelNodeRecord(fixture.graph, node.id)!;
    const memory = createCPUGraphMemory();
    const model = record.bindings.find(binding => binding.name === 'model')!.view;
    const modelValues = new Float32Array(model.length);
    // With a zero spectrum, 16 residual squares and 16 lag squares, the objective is
    // 8*log(1 + (rho - 0.5)^2). Its unique minimizer is exactly 0.5. Evaluating the
    // rounded variance first loses the squared offset whenever it is below half an ulp.
    modelValues[1] = 0.5;
    modelValues[2] = 16;
    memory.write(model, modelValues);
    memory.write(
      record.bindings.find(binding => binding.name === 'partial')!.view,
      Float32Array.of(16, 0)
    );
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => {
        const range = getViewBindingRange(binding.view);
        return [
          index,
          new Float32Array(
            memory.getBuffer(binding.view.buffer).backing,
            range.offset,
            range.size / 4
          ).slice()
        ];
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
    const actual = bindings[record.bindings.findIndex(binding => binding.name === 'model')];
    expect(Number.isFinite(actual[1])).toBe(true);
    expect(Math.abs(actual[1] - 0.5)).toBeLessThan(1e-6);
    expect(actual[2]).toBe(1);
    expect(actual[3]).toBeCloseTo(-8 * (1 + Math.log(2 * Math.PI) + Math.log(16)), 4);
  } finally {
    fixture.device.destroy();
  }
});

it('reference panel objective preserves the dense determinant likelihood and its constants', () => {
  const fixture = createFixture();
  try {
    recordWGSLKernelNodes(fixture.graph);
    const node = new GPUSpatialPanelLagMaximumLikelihood(fixture.panel)
      .getCommandNodes(fixture.graph)
      .find(node => node.id.endsWith('-random-fit'))!;
    const record = getWGSLKernelNodeRecord(fixture.graph, node.id)!;
    // Zero dense weights isolate the log-variance term. This checks the generated likelihood
    // function without claiming these weights satisfy the full fitter's normalization checks.
    const source =
      record.source.slice(0, record.source.indexOf('@compute')) +
      `
@compute @workgroup_size(1) fn main() {
  let direct = vec4<f32>(32.0, 16.0, 0.5, 0.0);
  results[resultsOffset] = objective(0.5, direct);
  results[resultsOffset + 1u] = objective(0.7, direct);
  results[resultsOffset + 2u] = objective(0.5, vec4<f32>(16.0, 16.0, 0.5, 0.0));
}`;
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => {
        const range = getViewBindingRange(binding.view);
        return [index, new Float32Array(range.size / 4)];
      })
    );
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: bindings},
      {constants: record.constants}
    );
    const actual = bindings[record.bindings.findIndex(binding => binding.name === 'results')];
    expect(actual[0]).toBeCloseTo(8 * Math.log(2), 5);
    expect(actual[1]).toBeCloseTo(8 * Math.log(2 + (Math.fround(0.7) - 0.5) ** 2), 5);
    expect(actual[2]).toBe(0);
  } finally {
    fixture.device.destroy();
  }
});

it('panel-lag uses shared LU for its determinant and inverse and audits generated identifiers', () => {
  const fixture = createFixture();
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPUSpatialPanelLagMaximumLikelihood(fixture.panel).getCommandNodes(fixture.graph)
    );
    const fit = kernels.find(kernel => kernel.id.endsWith('-random-fit'))!;
    expect(fit.source).toContain('factorDenseLU_4(');
    expect(fit.source).toContain('invertDenseLU_4(');
    expect(fit.source).toContain('factorization.logAbsDeterminant');
    for (const kernel of kernels) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    }
  } finally {
    fixture.device.destroy();
  }
});

it('panel-lag rejects unrepresentable shader tolerances at construction', () => {
  const fixture = createFixture();
  try {
    for (const tolerance of [NaN, Infinity, 1e100, 1e-100, 0, -1]) {
      expect(() => new GPUSpatialPanelLagMaximumLikelihood({...fixture.panel, tolerance})).toThrow(
        /panel-improvement.*tolerance/
      );
    }
  } finally {
    fixture.device.destroy();
  }
});

it('panel-lag preflights all derived scratch before graph mutation', () => {
  const fixture = createFixture();
  try {
    const contributor = new GPUSpatialPanelLagMaximumLikelihood(fixture.panel);
    Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {value: 256});
    const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
    const createView = vi.spyOn(fixture.graph, 'createDataView');
    expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(/packed panel.*512 bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    fixture.device.destroy();
  }
});

it('accepts disjoint packed outputs across dispatches and rejects co-bound aligned prefixes', () => {
  const fixture = createFixture();
  try {
    const storage = fixture.view('float32', 24);
    const coefficients = fixture.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 4
    });
    const fitted = fixture.graph.createDataView(storage.buffer, {
      format: 'float32',
      byteOffset: 16,
      length: 16
    });
    const props = {...fixture.panel, output: {...fixture.panel.output, coefficients, fitted}};
    expect(() => new GPUSpatialPanelLagMaximumLikelihood(props)).not.toThrow();
    expect(
      new GPUSpatialPanelLagMaximumLikelihood(props).getCommandNodes(fixture.graph).length
    ).toBeGreaterThan(0);
    const summary = fixture.graph.createDataView(storage.buffer, {
      format: 'float32',
      byteOffset: 16,
      length: 6
    });
    expect(
      () =>
        new GPUSpatialPanelLagMaximumLikelihood({
          ...fixture.panel,
          output: {...fixture.panel.output, coefficients, summary}
        })
    ).toThrow(/aligned binding.*overlap/);
    const overlappingFitted = fixture.graph.createDataView(storage.buffer, {
      format: 'float32',
      byteOffset: 12,
      length: 16
    });
    expect(
      () =>
        new GPUSpatialPanelLagMaximumLikelihood({
          ...props,
          output: {...props.output, fitted: overlappingFitted}
        })
    ).toThrow(/logical writable.*overlap/);
  } finally {
    fixture.device.destroy();
  }
});

it('uses a constant set of pipelines for a growing storage-backed pivoted inverse', () => {
  const counts: number[] = [];
  for (const entities of [4, 8, 40]) {
    const fixture = createFixture();
    try {
      const periods = 65;
      const rows = entities * periods;
      const contributor = new GPUSpatialPanelLagMaximumLikelihood({
        ...fixture.panel,
        effects: 'unit',
        fitMode: 'tiled',
        periodCount: periods,
        eigenvalues: fixture.view('float32', entities),
        weights: {
          offsets: fixture.view('uint32', entities + 1),
          neighbors: fixture.view('uint32', entities * 2),
          weights: fixture.view('float32', entities * 2)
        },
        predictors: fixture.view('float32', rows),
        response: fixture.view('float32', rows),
        output: {
          ...fixture.panel.output,
          fitted: fixture.view('float32', rows),
          residuals: fixture.view('float32', rows)
        }
      });
      const nodes = contributor.getCommandNodes(fixture.graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const pivots = nodes.filter(node =>
        /-pivot-(select|swap|factors|normalize|eliminate)-/.test(node.id)
      );
      expect(pivots).toHaveLength(5 * entities);
      counts.push(new Set(getGPUCommandNodePipelineKeys(pivots, {includeShaderLayout: true})).size);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
    } finally {
      fixture.device.destroy();
    }
  }
  expect(counts).toEqual([5, 5, 5]);
});

it('the emitted pivot passes compute an independently known inverse with a row swap', () => {
  const fixture = createFixture();
  try {
    recordWGSLKernelNodes(fixture.graph);
    const contributor = new GPUSpatialPanelLagMaximumLikelihood({
      ...fixture.panel,
      effects: 'unit',
      fitMode: 'tiled',
      eigenvalues: fixture.view('float32', 4)
    });
    const nodes = contributor.getCommandNodes(fixture.graph);
    const pivots = nodes.filter(node =>
      /-pivot-(select|swap|factors|normalize|eliminate)-/.test(node.id)
    );
    const memory = createCPUGraphMemory();
    const first = getWGSLKernelNodeRecord(fixture.graph, pivots[0].id)!;
    memory.write(
      first.bindings.find(binding => binding.name === 'matrix')!.view,
      Float32Array.of(0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 3, 1, 0, 0, 1, 2)
    );
    const swap = getWGSLKernelNodeRecord(fixture.graph, pivots[1].id)!;
    memory.write(
      swap.bindings.find(binding => binding.name === 'sourceInverse')!.view,
      Float32Array.of(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    );
    for (const node of pivots) {
      const record = getWGSLKernelNodeRecord(fixture.graph, node.id)!;
      const pivot = Number(node.id.match(/-(\d+)$/)![1]);
      memory.write(
        record.bindings.find(binding => binding.name === 'parameters')!.view,
        Uint32Array.of(pivot)
      );
    }
    for (const node of pivots) {
      const record = getWGSLKernelNodeRecord(fixture.graph, node.id)!;
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const range = getViewBindingRange(binding.view);
          const buffer = memory.getBuffer(binding.view.buffer).backing;
          return [
            index,
            binding.type === 'u32'
              ? new Uint32Array(buffer, range.offset, range.size / 4).slice()
              : new Float32Array(buffer, range.offset, range.size / 4).slice()
          ];
        })
      );
      // wgsl_reflect ignores TypedArray.byteOffset. Bind copies at offset zero and write the
      // writable ranges back, so runtime pivot slots exercise the actual aligned bindings.
      new WgslExec(
        new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
      ).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: record.constants}
      );
      record.bindings.forEach((binding, index) => {
        if (binding.access === 'read_write') memory.write(binding.view, bindings[index]);
      });
    }
    const final = getWGSLKernelNodeRecord(fixture.graph, pivots[pivots.length - 1].id)!;
    const actual = memory.readFloat32(
      final.bindings.find(binding => binding.name === 'inverse')!.view
    );
    const expected = [-2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0.4, -0.2, 0, 0, -0.2, 0.6];
    expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 6));
  } finally {
    fixture.device.destroy();
  }
});
