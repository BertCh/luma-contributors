// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {extractLensExample} from '../../../src/computation-lenses/lens-example';
import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../../analysis/src/testing/gpu-contributor-test-utils';
import {runExample as runAccessibility} from '../../../src/computation-lenses/scenarios/gpu-network/network-accessibility.example';
import {runExample as runWeights} from '../../../src/computation-lenses/scenarios/gpu-network/network-cost-weights.example';
import {runExample as runMatrix} from '../../../src/computation-lenses/scenarios/gpu-network/network-cost-matrix.example';

it.each([
  'null-complete',
  'wrong-complete-format',
  'short-complete',
  'null-limit',
  'foreign-complete'
])('weights rejects invalid evidence before allocation and permits same-graph retry (%s, B7)', async invalid => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const matrixComplete = createTransientView(graph, 'complete', 'uint32', 1);
    const matrixCostLimit = createTransientView(graph, 'limit', 'float32', 1);
    const wrongFormat = createTransientView(graph, 'wrong-format', 'float32', 1);
    const short = createTransientView(graph, 'short', 'uint32', 0);
    const foreign = createTransientView(new GPUCommandGraph(device), 'foreign', 'uint32', 1);
    const input = {
      graph,
      matrixCosts: createTransientView(graph, 'costs', 'float32', 3),
      parameters: createTransientView(graph, 'decay', 'float32', 4),
      facilityCount: 1,
      nodeCount: 3,
      matrixComplete,
      matrixCostLimit
    };
    const bad = {
      ...input,
      matrixComplete:
        invalid === 'null-complete'
          ? null
          : invalid === 'wrong-complete-format'
            ? wrongFormat
            : invalid === 'short-complete'
              ? short
              : invalid === 'foreign-complete'
                ? foreign
                : matrixComplete,
      matrixCostLimit: invalid === 'null-limit' ? null : matrixCostLimit
    };
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    await expect(
      runWeights(device, bad as unknown as Parameters<typeof runWeights>[1])
    ).rejects.toThrow(
      /network-cost-weights-example.*(matrixComplete|matrixCostLimit|target graph)/
    );
    expect(allocate).not.toHaveBeenCalled();
    const retry = await runWeights(device, input);
    expect(retry.status.complete).toBeDefined();
    allocate.mockRestore();
  } finally {
    device.destroy();
  }
});

it('the facility-origin matrix example admits source intersections without changing forward travel', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const example = await runMatrix(device, {
      graph,
      roads: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
        weights: createTransientView(graph, 'costs', 'float32', 2),
        nodePositions: createTransientView(graph, 'positions', 'float32x2', 3)
      },
      facilityPositions: createTransientView(graph, 'facilities', 'float32x2', 1)
    });
    expect(example.snapping.props.startDirection).toBe('forward');
    expect(example.snapping.props.intersectionAware).toBe(true);
  } finally {
    device.destroy();
  }
});

it.each([
  'omitted',
  'complete-only',
  'bounded'
])('weights accepts %s evidence and rejects a lone horizon before allocation', async evidence => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const input = {
      graph,
      matrixCosts: createTransientView(graph, 'costs', 'float32', 3),
      parameters: createTransientView(graph, 'decay', 'float32', 4),
      facilityCount: 1,
      nodeCount: 3
    };
    const matrixComplete = createTransientView(graph, 'complete', 'uint32', 1);
    const matrixCostLimit = createTransientView(graph, 'limit', 'float32', 1);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    await expect(
      runWeights(device, {...input, matrixCostLimit} as Parameters<typeof runWeights>[1])
    ).rejects.toThrow(/network-cost-weights-example.*matrixCostLimit.*matrixComplete.*received/i);
    expect(allocate).not.toHaveBeenCalled();
    allocate.mockRestore();
    const result = await runWeights(
      device,
      evidence === 'omitted'
        ? input
        : {
            ...input,
            matrixComplete,
            matrixCostLimit: evidence === 'bounded' ? matrixCostLimit : undefined
          }
    );
    expect(Boolean(result.status.complete)).toBe(evidence !== 'omitted');
  } finally {
    device.destroy();
  }
});

it('executable accessibility and weights examples forward the matrix certificate and horizon', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const matrixCosts = createTransientView(graph, 'matrix', 'float32', 5);
    const matrixComplete = createTransientView(graph, 'matrix-complete', 'uint32', 1);
    const matrixCostLimit = createTransientView(graph, 'matrix-limit', 'float32', 1);
    const parameters = createTransientView(graph, 'parameters', 'float32', 4);
    const input = {
      graph,
      matrixCosts,
      matrixComplete,
      matrixCostLimit,
      parameters,
      nodeCount: 5,
      facilityCount: 1
    };
    const accessibility = await runAccessibility(device, {
      ...input,
      opportunityWeights: createTransientView(graph, 'opportunities', 'float32', 1)
    });
    expect(accessibility.accessibility.props.costsComplete).toBe(matrixComplete);
    expect(accessibility.accessibility.props.costLimit).toBe(matrixCostLimit);
    expect(accessibility.accessibility.props.status?.complete).toBeDefined();
    const weights = await runWeights(device, input);
    expect(weights.costWeights.props.costsComplete).toBe(matrixComplete);
    expect(weights.costWeights.props.costLimit).toBe(matrixCostLimit);
    expect(weights.status.complete).toBeDefined();
  } finally {
    device.destroy();
  }
});

it.each([
  'accessibility',
  'cost-weights'
])('network-%s lens refuses incomplete capture and recovers', name => {
  const source = readFileSync(
    `modules/lenses/src/computation-lenses/scenarios/gpu-network/network-${name}.lens.ts`,
    'utf8'
  );
  const start = source.indexOf('  capture(final) {');
  const end = source.indexOf('\n};', start);
  expect(start).toBeGreaterThan(0);
  const body = transpileModule(`const scenario = {${source.slice(start, end)}};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const capture = new Function(`
    const facilities = [{name: 'facility', lon: 0, lat: 0}];
    const BETA = 0, THRESHOLD_SECONDS = 4, COST_LIMIT_SECONDS = 4;
    const createNodeValuePoints = () => ({}), gatherRows = () => new Float32Array(0);
    const sliceCapture = (capture, start, end, options) => ({...capture, ...options, values: capture.values.slice(start, end)});
    ${body} return scenario.capture;
  `)();
  let complete = 0;
  const final = (key: string) => ({
    id: key,
    name: key,
    values:
      key === 'accessibilityComplete' || key === 'weightComplete'
        ? Uint32Array.of(complete)
        : key === 'converged'
          ? Uint32Array.of(1)
          : key === 'weightOverflow'
            ? Uint32Array.of(0)
            : key === 'weightOffsets'
              ? Uint32Array.of(0, 5)
              : new Float32Array(10).fill(1)
  });
  expect(() => capture(final)).toThrow(/incomplete/i);
  complete = 1;
  expect(capture(final).length).toBeGreaterThan(0);
});

it.each([
  'accessibility',
  'cost-weights'
])('network-%s lens records the certificate and keeps its shown example short', async name => {
  const device = createNullWebGPUDevice();
  try {
    const lens =
      name === 'accessibility'
        ? await import(
            '../../../src/computation-lenses/scenarios/gpu-network/network-accessibility.lens'
          )
        : await import(
            '../../../src/computation-lenses/scenarios/gpu-network/network-cost-weights.lens'
          );
    const record = await lens.describeStructure(device);
    expect(validateGPUComputationRecord(record).issues).toEqual([]);
    const example = extractLensExample(
      readFileSync(
        `modules/lenses/src/computation-lenses/scenarios/gpu-network/network-${name}.example.ts`,
        'utf8'
      )
    );
    expect(example.lineCount).toBeLessThanOrEqual(25);
    const certificate = name === 'accessibility' ? 'accessibilityComplete' : 'weightComplete';
    expect(JSON.stringify(record)).toContain(certificate);
  } finally {
    device.destroy();
  }
});
