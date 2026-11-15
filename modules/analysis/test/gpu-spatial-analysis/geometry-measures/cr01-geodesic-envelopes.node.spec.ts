// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  computeGeodesicPairsOnCPU,
  CPUGeodesicPairsOperation
} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {GPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getBearingErrorDegrees, getPositionErrorMeters} from './cr01-geodesic-assertions';
import fixture from './cr01-geodesic-fixture.json';
import {GEOMETRY_CASES} from '../../analysis-core/kernel-audit/cases-geometry';
import {auditKernelCase} from '../../analysis-core/kernel-audit/kernel-audit-runner';

it('keeps the CPU twin consistent with PROJ at poles, antipodes and across the exclusion boundary', () => {
  for (const metric of ['great-circle', 'ellipsoidal'] as const) {
    const rows = fixture.pairs.filter(row => row.model === metric);
    const result = computeGeodesicPairsOnCPU(
      {
        origins: new Float32Array(rows.flatMap(row => row.origin)),
        targets: new Float32Array(rows.flatMap(row => row.target))
      },
      {coordinateSpace: 'longitude-latitude', metric, units: 'meters'}
    );
    const sphereRows = fixture.pairs.filter(row => row.model === 'great-circle');
    expect(Array.from(result.iterationLimitReached)).toEqual(Array(rows.length).fill(0));
    for (const [index, row] of rows.entries()) {
      const message = JSON.stringify({index, row});
      expect(result.distances[index], message).toBe(Math.fround(row.distance));
      expect(result.converged[index], message).toBe(1);
      if (row.distance > 0) {
        expect(
          getBearingErrorDegrees(result.initialBearings[index], row.initial),
          message
        ).toBeLessThan(1.6e-5);
        expect(
          getBearingErrorDegrees(result.finalBearings[index], row.final),
          message
        ).toBeLessThan(1.6e-5);
      }
      // Ellipsoidal midpoints intentionally use the sphere. Compare physical position;
      // longitude has no unique value at a pole, and antipodal midpoints are rejected.
      if (Number.isNaN(result.midpoints[2 * index])) {
        expect(sphereRows[index].distance, message).toBeGreaterThan(Math.PI * 6371008.8 - 7);
        expect(result.midpoints[2 * index + 1], message).toBeNaN();
        expect(result.approximation[index], message).toBe(1);
      } else {
        expect(
          getPositionErrorMeters(
            Array.from(result.midpoints.slice(2 * index, 2 * index + 2)),
            row.midpoint
          ),
          message
        ).toBeLessThan(2);
      }
    }
    expect(result.midpoints[2 * 5]).toBeNaN();
    expect(result.approximation[5]).toBe(1);
    expect(result.approximation[14]).toBe(0);
  }
  expect(
    computeGeodesicPairsOnCPU(
      {origins: new Float32Array(), targets: new Float32Array()},
      {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal'
      }
    ).approximation.length
  ).toBe(0);
});

it('uses longitude zero for coincident negative-longitude poles, including signed zeros', () => {
  const origins = new Float32Array([-130, 90, -130, -90, -180, 90, 180, -90]);
  for (const metric of ['great-circle', 'ellipsoidal'] as const) {
    const result = computeGeodesicPairsOnCPU(
      {origins, targets: origins.slice()},
      {coordinateSpace: 'longitude-latitude', metric, units: 'meters'}
    );
    expect(Array.from(result.midpoints)).toEqual([0, 90, 0, -90, 0, 90, 0, -90]);
    expect(Array.from(result.distances)).toEqual([0, 0, 0, 0]);
    expect(Array.from(result.approximation)).toEqual([0, 0, 0, 0]);
  }
});

it('exposes approximation on both operations, including status-only output', () => {
  for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation]) {
    const operation = new Operation({
      origins: 'origins',
      targets: 'targets',
      pairCount: 1,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      output: {status: {approximation: 'approximation'}}
    });
    expect(Object.keys(operation.outputs)).toHaveLength(1);
    const output = Object.values(operation.outputs)[0];
    if (operation instanceof GPUGeodesicPairsOperation)
      expect(operation.outputs['approximation'].value).toBe('approximation');
    else expect(Object.keys(operation.outputs)).toEqual(['approximation']);
    expect(output.format).toBe('uint32');
    expect(output.length).toBe(1);
  }
});

it('validates named iteration limits and per-row approximation output shape at construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const origins = createTransientView(graph, 'origins', 'float32x2', 2);
  const targets = createTransientView(graph, 'targets', 'float32x2', 2);
  const approximation = createTransientView(graph, 'approximation', 'uint32', 2);
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
  const pairs = {
    id: 'cr01-pairs',
    spatialContext,
    origins,
    targets,
    output: {status: {approximation}}
  };
  const destination = {
    id: 'cr01-direct',
    spatialContext,
    origins,
    bearings: createTransientView(graph, 'bearings', 'float32', 2),
    distances: createTransientView(graph, 'distances', 'float32', 2),
    output: {destinations: createTransientView(graph, 'destinations', 'float32x2', 2)}
  };
  for (const maximumIterations of [1, 16, 200]) {
    expect(() => new GPUGeodesicPairs({...pairs, maximumIterations})).not.toThrow();
    expect(() => new GPUGeodesicDestination({...destination, maximumIterations})).not.toThrow();
  }
  for (const maximumIterations of [0, 201, 1.5, NaN]) {
    expect(() => new GPUGeodesicPairs({...pairs, maximumIterations})).toThrow(
      /^cr01-pairs maximumIterations.*\[1, 200\]/
    );
    expect(() => new GPUGeodesicDestination({...destination, maximumIterations})).toThrow(
      /^cr01-direct maximumIterations.*\[1, 200\]/
    );
  }
  expect(
    () =>
      new GPUGeodesicPairs({
        ...pairs,
        output: {
          status: {
            approximation: createTransientView(graph, 'short-status', 'uint32', 1)
          }
        }
      })
  ).toThrow(/^cr01-pairs output.status.approximation must have 2 rows/);
});

it.each([
  0, 4
])('rejects cross-dispatch output aliases at byte offset %i during construction', limitOffset => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const shared = createTransientView(graph, 'shared-status', 'uint32', 6);
  const converged = graph.createDataView(shared.buffer, {format: 'uint32', length: 2});
  const iterationLimitReached = graph.createDataView(shared.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: limitOffset
  });
  expect(converged).not.toBe(iterationLimitReached);
  expect(
    () =>
      new GPUGeodesicPairs({
        id: 'cr01-output-alias',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
        origins: createTransientView(graph, 'origins', 'float32x2', 2),
        targets: createTransientView(graph, 'targets', 'float32x2', 2),
        output: {
          distances: createTransientView(graph, 'distances', 'float32', 2),
          initialBearings: createTransientView(graph, 'initial', 'float32', 2),
          finalBearings: createTransientView(graph, 'final', 'float32', 2),
          midpoints: createTransientView(graph, 'midpoints', 'float32x2', 2),
          status: {
            converged,
            approximation: createTransientView(graph, 'approximation', 'uint32', 2),
            iterationLimitReached
          }
        }
      })
  ).toThrow(
    /^cr01-output-alias output\.status\.iterationLimitReached must not overlap output\.status\.converged/
  );
});

it.each([
  {pairCount: 2, limitOffset: 8},
  {pairCount: 0, limitOffset: 0}
])('validates shared-buffer outputs while retaining the empty-input limit: %j', ({
  pairCount,
  limitOffset
}) => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const shared = createTransientView(graph, 'shared-status', 'uint32', 6);
  const construct = () =>
    new GPUGeodesicPairs({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
      origins: createTransientView(graph, 'origins', 'float32x2', pairCount),
      targets: createTransientView(graph, 'targets', 'float32x2', pairCount),
      output: {
        status: {
          converged: graph.createDataView(shared.buffer, {format: 'uint32', length: pairCount}),
          iterationLimitReached: graph.createDataView(shared.buffer, {
            format: 'uint32',
            length: pairCount,
            byteOffset: limitOffset
          })
        }
      }
    });
  if (pairCount === 0) expect(construct).toThrow(/origins.length must be an integer in \[1,/);
  else expect(construct).not.toThrow();
});

it('keeps every optional column within the portable eight-storage-binding budget', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const contributor = new GPUGeodesicPairs({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
    origins: createTransientView(graph, 'origins', 'float32x2', 3),
    targets: createTransientView(graph, 'targets', 'float32x2', 3),
    output: {
      distances: createTransientView(graph, 'distances', 'float32', 3),
      initialBearings: createTransientView(graph, 'initial', 'float32', 3),
      finalBearings: createTransientView(graph, 'final', 'float32', 3),
      midpoints: createTransientView(graph, 'midpoints', 'float32x2', 3),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 3),
        approximation: createTransientView(graph, 'approximation', 'uint32', 3),
        iterationLimitReached: createTransientView(graph, 'limit', 'uint32', 3)
      }
    }
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes).toHaveLength(2);
  expect(nodes.map(node => getWGSLKernelNodeRecord(graph, node.id)?.bindings.length)).toEqual([
    8, 3
  ]);
});

for (const auditCase of GEOMETRY_CASES.filter(
  entry =>
    entry.exportName === 'GPUGeodesicPairs' ||
    entry.exportName === 'GPUGeodesicDestination' ||
    entry.exportName === 'GPUGeometryMeasures'
)) {
  it(`audits generated geodesic WGSL: ${auditCase.exportName} ${auditCase.name}`, () => {
    expect(auditKernelCase(auditCase).findings).toEqual([]);
  });
}
