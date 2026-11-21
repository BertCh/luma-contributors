// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUAnalysisOperator} from '../../../src/gpu-spatial-analysis/ontology/index';
import {
  GPUHDBSCAN,
  getGPUHDBSCANParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import fixture from '../../oracles/fixtures/space-time-hdbscan.json';
import {
  computeHierarchicalDensityModel,
  getAdjustedRandIndex
} from './hierarchical-density-clustering-oracle';

function createCase(rows = 257, minimumSamples = 32) {
  const context = createKernelAuditContext('space-time-hdbscan');
  const {view} = context;
  const props = {
    id: 'GPUHDBSCAN-test',
    positions: view('float32x2', rows),
    times: view('float32', rows),
    parameters: view('float32', 4),
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    } as const,
    minimumSamples,
    minimumClusterSize: 2,
    output: {
      labels: view('uint32', rows),
      coreDistances: view('float32', rows),
      status: {count: view('uint32', 1)}
    }
  };
  return {...context, props};
}

it('validates space-time column presence, encoding, layouts and output aliases at construction', () => {
  const {device, view, props} = createCase();
  try {
    expect(() => new GPUHDBSCAN({...props, temporalContext: undefined})).toThrow(/temporalContext/);
    expect(() => new GPUHDBSCAN({...props, parameters: undefined})).toThrow(/parameters/);
    expect(() => new GPUHDBSCAN({...props, times: undefined})).toThrow(/require times/);
    expect(() => new GPUHDBSCAN({...props, times: view('float32', 1)})).toThrow(/times length/);
    expect(() => new GPUHDBSCAN({...props, parameters: view('float32', 3)})).toThrow(
      /parameters length.*4/
    );
    expect(
      () =>
        new GPUHDBSCAN({
          ...props,
          temporalContext: {...props.temporalContext, encoding: 'int64-words'}
        })
    ).toThrow(/int64-words.*float32-relative/);
    expect(
      () =>
        new GPUHDBSCAN({
          ...props,
          output: {...props.output, coreDistances: props.times}
        })
    ).toThrow();
    expect(() => new GPUHDBSCAN({...props, minimumSamples: 33})).toThrow(/minimumSamples.*32/);
  } finally {
    device.destroy();
  }
});

it('packs the per-frame scale after float32 rounding and rejects invalid scales', () => {
  expect(Array.from(getGPUHDBSCANParameterValues({timeScale: 2}))).toEqual([2, 0, 0, 0]);
  expect(Array.from(getGPUHDBSCANParameterValues({timeScale: 0}))).toEqual([0, 0, 0, 0]);
  for (const timeScale of [-1, -1e-50, NaN, Infinity, 1e40]) {
    expect(() => getGPUHDBSCANParameterValues({timeScale})).toThrow(/^GPUHDBSCAN.*timeScale/);
  }
  expect(() => getGPUHDBSCANParameterValues({timeScale: 1}, new Float32Array(3))).toThrow(
    /target length.*4/
  );
});

it('the generated scaled metric shares the BVH and tree, respects eight bindings and its node estimate', () => {
  for (const rows of [1, 257, 1025, 1048576]) {
    for (const minimumSamples of [1, 32]) {
      const {device, graph, props} = createCase(rows, minimumSamples);
      try {
        const contributor = new GPUHDBSCAN(props);
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        expect(nodes.some(node => node.id.endsWith('-space-time-core'))).toBe(minimumSamples > 1);
        // Avoid retaining thousands of parsed kernels for the maximum-row cases.
        if (rows === 257) {
          for (const {id: identifier, source} of getParsedKernelSources(nodes).configurations) {
            expect(getStorageBindingCount(source), identifier).toBeLessThanOrEqual(8);
          }
        }
      } finally {
        device.destroy();
      }
    }
  }
});

it('scaled-coordinate HDBSCAN model agrees with independent sklearn fixtures', () => {
  for (const scene of fixture.scenes) {
    const coordinates = scene.positions.map((position, row) => [
      ...position,
      scene.times[row] * scene.timeScale
    ]);
    const actual = computeHierarchicalDensityModel(
      coordinates,
      scene.minimumClusterSize,
      scene.minimumSamples
    );
    actual.coreDistances.forEach((distance, row) =>
      expect(distance).toBeCloseTo(scene.coreDistances[row], 10)
    );
    expect(getAdjustedRandIndex(actual.labels, scene.labels), scene.name).toBeGreaterThanOrEqual(
      0.97
    );
    expect(Math.max(...actual.labels) + 1, scene.name).toBe(scene.clusterCount);
  }
});

it('catalogues the existing contributor as planar scaled space-time HDBSCAN with named limits', () => {
  const operator = getGPUAnalysisOperator('GPUHDBSCAN');
  expect(operator?.dimensionality).toBe('2d+time');
  expect(operator?.coordinateModels).toEqual(['planar']);
  expect(operator?.tier).toBe('scale');
  expect(operator?.limits).toEqual(
    expect.arrayContaining([
      expect.objectContaining({name: 'positions.length', maximum: 4194304}),
      expect.objectContaining({name: 'minimumSamples', maximum: 32})
    ])
  );
});
