// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMinimumBoundsOnCPU} from '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation';
import {computeLabelPointOnCPU} from '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';
import {
  computeMaximumInscribedCircleOnCPU,
  CPUMaximumInscribedCircleOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {makeCPUInputRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';

function polygon(positions: number[]) {
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array([0, positions.length / 2]),
    polygonOffsets: new Uint32Array([0, 1]),
    featureOffsets: new Uint32Array([0, 1])
  };
}

it('finds the inscribed circle in a named resident polygon window', async () => {
  const props = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    featureCount: 1,
    tolerance: 0.01,
    output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
  };
  const inputs = {
    positions: makeCPUInputRows(
      'float32x2',
      Float32Array.of(99, 99, 0, 0, 6, 0, 6, 2, 0, 2, 88, 88)
    ),
    ringOffsets: makeCPUInputRows('uint32', Uint32Array.of(1, 5)),
    polygonOffsets: makeCPUInputRows('uint32', Uint32Array.of(0, 1)),
    featureOffsets: makeCPUInputRows('uint32', Uint32Array.of(0, 1))
  };
  const result = await new CPUMaximumInscribedCircleOperation(props).execute({inputs});
  // The 6-by-2 rectangle has maximum radius 1.
  expect((result['circles'] as Float32Array)[2]).toBe(1);
  expect(result['status']).toEqual(Uint32Array.of(0));
  expect(() =>
    new CPUMaximumInscribedCircleOperation({...props, storage: 'compact'}).execute({inputs})
  ).toThrow(/ringOffsets initial offset/);
});

it('UGE-04 preserves source envelopes and diagnoses lattice precision loss', () => {
  const alone = computeMinimumBoundsOnCPU({
    positions: new Float32Array([0, 0, 0.01, 0, 0.01, 0.01, 0, 0.01]),
    labels: new Uint32Array(4),
    groupCount: 1
  });
  const together = computeMinimumBoundsOnCPU({
    positions: new Float32Array([0, 0, 0.01, 0, 0.01, 0.01, 0, 0.01, 10000000, 0]),
    labels: new Uint32Array([0, 0, 0, 0, 1]),
    groupCount: 2
  });
  // The envelope is the componentwise extent of the four stored float32 corners.
  expect(Array.from(together.sourceBounds.slice(0, 4))).toEqual([
    0,
    0,
    Math.fround(0.01),
    Math.fround(0.01)
  ]);
  expect(together.precisionLimited[0]).toBe(1);
  expect(alone.hullSizes[0]).toBe(4);
  expect(Number.isFinite(alone.diameters[0])).toBe(true);
  expect(alone.diameters[0]).toBe(Math.fround(Math.SQRT2 * Math.fround(0.01)));
});

it('UGE-05 evaluates clearance at the published float32 center', () => {
  for (const origin of [0, 1048576]) {
    const input = polygon([
      origin,
      origin,
      origin + 0.125,
      origin,
      origin + 0.125,
      origin + 0.125,
      origin,
      origin + 0.125
    ]);
    const label = computeLabelPointOnCPU(input);
    const circle = computeMaximumInscribedCircleOnCPU(input, 0.000002);
    // Midpoint .0625 is exactly representable locally, but rounds to the boundary at 2^20.
    const clearance = origin === 0 ? 0.0625 : 0;
    expect(label.publishedCenterClearance[0]).toBe(clearance);
    expect(label.publishedCenterClearance[0]).toBe(clearance);
    expect(label.precisionLimited[0]).toBe(origin === 0 ? 0 : 1);
    expect(circle.circles[2]).toBe(clearance);
    expect(circle.publishedCenterClearance[0]).toBe(clearance);
    expect(circle.precisionLimited[0]).toBe(origin === 0 ? 0 : 1);
    if (origin !== 0)
      expect(Array.from(circle.witnesses)).toEqual(Array.from(circle.circles.slice(0, 2)));
  }
});

it('DGN-09 preserves zero parents and one empty parent through host geometry', async () => {
  const {computeNormalizePolygonsOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation'
  );
  const {computeLineSimplificationOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation'
  );
  const {computeSegmentSnapOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation'
  );
  const {computeShapeDescriptorsOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation'
  );
  const {computePolygonTriangulationOnCPU} = await import(
    '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation'
  );
  const {computeConstrainedDelaunayTriangulationOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation'
  );
  const {computeSpatialPredicateJoinOnCPU, computeNearestFeatureJoinOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join'
  );
  const {computePointInPolygonJoinOnCPU} = await import(
    '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation'
  );
  for (const count of [0, 1]) {
    const input = {
      positions: new Float32Array(),
      ringOffsets: new Uint32Array([0]),
      polygonOffsets: new Uint32Array([0]),
      featureOffsets: new Uint32Array(count + 1)
    };
    expect(computeLabelPointOnCPU(input).points.length).toBe(count * 2);
    expect(computeMaximumInscribedCircleOnCPU(input, 0.01).status.length).toBe(count);
    expect(computeMinimumBoundsOnCPU(input).sourceBounds.length).toBe(count * 4);
    const normalized = computeNormalizePolygonsOnCPU(input);
    expect(Array.from(normalized.ringOffsets)).toEqual([0]);
    expect(Array.from(normalized.polygonOffsets)).toEqual([0]);
    expect(
      computeShapeDescriptorsOnCPU(input, new Float32Array([0.05, 0, 0, 0])).areas.length
    ).toBe(count);
    const line = {positions: input.positions, lineOffsets: new Uint32Array(count + 1)};
    expect(computeLineSimplificationOnCPU(line).lineCounts.length).toBe(count);
    expect(
      computeSegmentSnapOnCPU(
        {...line, referencePositions: input.positions},
        {tolerance: 0.1, maximumVerticesPerLine: 4}
      ).counts.length
    ).toBe(count * 4);
    // Controls retain per-polygon triangulation rows, while joins retain feature rows.
    for (const triangulate of [
      computePolygonTriangulationOnCPU,
      computeConstrainedDelaunayTriangulationOnCPU
    ]) {
      const result = triangulate(input, {indexCapacity: 0});
      expect(result.triangleCount.length).toBe(0);
      expect(Array.from(result.triangleOffsets)).toEqual([0]);
    }
    const geometry = {...input, kind: 'polygons' as const};
    expect(
      computeSpatialPredicateJoinOnCPU(geometry, geometry, {
        predicate: 'intersects',
        candidateCapacity: 1,
        pairCapacity: 1
      }).status.count
    ).toBe(0);
    expect(computeNearestFeatureJoinOnCPU(geometry, geometry).neighborDistances.length).toBe(count);
    const pointJoin = computePointInPolygonJoinOnCPU(
      {
        points: {kind: 'points', positions: input.positions, featureOffsets: input.featureOffsets},
        polygons: geometry
      },
      {candidateCapacity: 1, matchCapacity: 0}
    );
    expect(pointJoin['pointMatchCounts'].length).toBe(count);
    expect(pointJoin.polygonAssignmentCounts.length).toBe(count);
    expect(pointJoin.featurePolygonMembership.length).toBe(count * count);
  }
});

it('DGN-09 rejects malformed compact terminals before label/circle searching', () => {
  const input = {...polygon([0, 0, 1, 0, 1, 1, 0, 1]), ringOffsets: new Uint32Array([0, 999])};
  expect(() => computeLabelPointOnCPU(input)).toThrow(/ringOffsets/);
  expect(() => computeMaximumInscribedCircleOnCPU(input, 0.01)).toThrow(/ringOffsets/);
});

it('DGN-08 resolves exact collinearity before either pole search', () => {
  for (const vertices of [
    [0, 0, 1, 0, 2, 0],
    [0, 0, 1, 1, 2, 2]
  ]) {
    const input = polygon(vertices);
    // A straight segment has no interior, regardless of its rotation or work budget.
    const bounded = computeMaximumInscribedCircleOnCPU(input, 0.000002, {maximumIterations: 0});
    expect(bounded.circles[2]).toBe(0);
    expect(bounded.circles[3]).toBe(0);
    expect(bounded.status[0]).toBe(0);
    const label = computeLabelPointOnCPU(input);
    expect(label.publishedCenterClearance[0]).toBe(0);
    expect(label.degenerate[0]).toBe(1);
    const circle = computeMaximumInscribedCircleOnCPU(input, 0.000002);
    expect(circle.circles[2]).toBe(0);
    expect(circle.circles[3]).toBe(0);
  }
  // Two unit squares with opposite winding have cancelling signed area, but both have interior.
  const input = {
    ...polygon([0, 0, 1, 0, 1, 1, 0, 1, 3, 0, 3, 1, 4, 1, 4, 0]),
    ringOffsets: new Uint32Array([0, 4, 8]),
    polygonOffsets: new Uint32Array([0, 1, 2]),
    featureOffsets: new Uint32Array([0, 2])
  };
  expect(computeLabelPointOnCPU(input).publishedCenterClearance[0]).toBe(0.5);
  const circle = computeMaximumInscribedCircleOnCPU(input, 0.01);
  expect(Number.isFinite(circle.circles[2])).toBe(true);
  expect(circle.circles[2]).toBeGreaterThan(0.49);
  expect(0.5 - circle.circles[2]).toBeLessThanOrEqual(circle.circles[3]);
});

it('UGE-11 proves the 2 by 1 rectangle optimum from its envelope', () => {
  const input = polygon([0, 0, 2, 0, 2, 1, 0, 1]);
  const bounded = computeMaximumInscribedCircleOnCPU(input, 0.000002, {maximumIterations: 0});
  // Any enclosed disk projects into the unit-height envelope, so radius <= 1/2.
  expect(bounded.circles[2]).toBe(0.5);
  expect(bounded.circles[3]).toBe(0);
  expect(bounded.status[0]).toBe(0);
  expect(computeLabelPointOnCPU(input).publishedCenterClearance[0]).toBe(0.5);
  const circle = computeMaximumInscribedCircleOnCPU(input, 0.000002);
  expect(circle.circles[2]).toBe(0.5);
  expect(circle.circles[3]).toBe(0);
});

it('CPL-14 keeps partition provenance independent of geometric overflow', () => {
  for (const complete of [0, 1]) {
    const result = computeMinimumBoundsOnCPU({
      positions: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
      labels: new Uint32Array([1, 1, 3, 3]),
      groupCount: 4,
      labelsComplete: new Uint32Array([complete])
    });
    expect(result.labelsComplete[0]).toBe(complete);
    expect(result.overflow[0]).toBe(0);
    expect(result.groupOverflow[0]).toBe(0);
    expect(Array.from(result.sourceBounds.slice(12))).toEqual([0, 1, 1, 1]);
  }
  expect(
    computeMinimumBoundsOnCPU({
      positions: new Float32Array([0, 0]),
      labels: new Uint32Array([0]),
      groupCount: 1
    }).labelsComplete[0]
  ).toBe(1);
});

it('DGN-09 preserves declared cardinality in owned named CPU operation forms', async () => {
  const {CPULabelPointOperation} = await import(
    '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation'
  );
  const {CPUMinimumBoundsOperation} = await import(
    '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation'
  );
  const {CPUShapeDescriptorsOperation} = await import(
    '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation'
  );
  const rows = (
    format: 'float32' | 'float32x2' | 'uint32',
    values: Float32Array | Uint32Array
  ) => ({
    format,
    length: values.length / (format === 'float32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'float32x2' ? 2 : 1)}],
    pack: () => values
  });
  for (const featureCount of [0, 1]) {
    const props = {
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      featureCount
    };
    const inputs = {
      positions: rows('float32x2', new Float32Array()),
      ringOffsets: rows('uint32', new Uint32Array([0])),
      polygonOffsets: rows('uint32', new Uint32Array([0])),
      featureOffsets: rows('uint32', new Uint32Array(featureCount + 1)),
      parameters: rows('float32', new Float32Array([0.05, 0, 0, 0]))
    };
    const label = new CPULabelPointOperation({
      ...props,
      output: {points: 'points', publishedCenterClearance: 'clearance'}
    });
    const bounds = new CPUMinimumBoundsOperation({
      ...props,
      rowCount: 0,
      output: {sourceBounds: 'bounds', status: {overflow: 'overflow'}}
    });
    const shape = new CPUShapeDescriptorsOperation({
      ...props,
      parameters: 'parameters',
      output: {areas: 'areas'}
    });
    expect(((await label.execute({inputs}))['points'] as Float32Array).length).toBe(
      featureCount * 2
    );
    expect(((await bounds.execute({inputs}))['bounds'] as Float32Array).length).toBe(
      featureCount * 4
    );
    expect(((await shape.execute({inputs}))['areas'] as Float32Array).length).toBe(featureCount);
  }
});

it('UGE-04 UGE-05 and CPL-14 reject retired ports and capture provenance in named forms', async () => {
  const {CPUMinimumBoundsOperation} = await import(
    '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation'
  );
  const {GPUMinimumBoundsOperation} = await import(
    '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation'
  );
  const {GPULabelPointOperation} = await import(
    '../../../src/gpu-spatial-analysis/label-point/gpu-label-point-operation'
  );
  const {CPULabelPointOperation} = await import(
    '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation'
  );
  for (const Operation of [CPUMinimumBoundsOperation, GPUMinimumBoundsOperation]) {
    const operation = new Operation({
      positions: 'positions',
      labels: 'labels',
      labelsComplete: 'inputCompletion',
      rowCount: 1,
      groupCount: 1,
      output: {
        sourceBounds: 'bounds',
        labelsComplete: 'partitionComplete',
        status: {overflow: 'overflow'}
      }
    });
    expect(operation.partitionProvenance.labelsComplete).toBe('partitionComplete');
    expect(
      operation instanceof CPUMinimumBoundsOperation
        ? Object.keys(operation.outputs).filter(name => name === 'bounds').length
        : Object.values(operation.outputs).filter(output => output.value === 'bounds').length
    ).toBe(1);
    const legacyOutput = {sourceBounds: 'new', bounds: 'old', status: {overflow: 'overflow'}};
    expect(
      () =>
        new Operation({
          positions: 'positions',
          labels: 'labels',
          rowCount: 1,
          groupCount: 1,
          output: legacyOutput
        })
    ).toThrow(/bounds.*sourceBounds/);
  }
  for (const Operation of [CPULabelPointOperation, GPULabelPointOperation]) {
    const input = {
      positions: 'positions',
      ringOffsets: 'rings',
      polygonOffsets: 'polygons',
      featureOffsets: 'features',
      featureCount: 1
    };
    const output = {
      points: 'points',
      publishedCenterClearance: 'clearance'
    };
    const operation = new Operation({...input, output});
    expect(
      operation instanceof CPULabelPointOperation
        ? Object.keys(operation.outputs).filter(name => name === 'clearance').length
        : Object.values(operation.outputs).filter(column => column.value === 'clearance').length
    ).toBe(1);
    expect(
      () =>
        new Operation({
          ...input,
          // The retired name is rejected at run time, so the type is bypassed on purpose.
          output: {...output, distances: 'old'} as unknown as typeof output
        })
    ).toThrow(/distances.*publishedCenterClearance/);
  }
});
