// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, existsSync} from 'node:fs';
import {expect, it} from 'vitest';
import {
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';

it('advertises candidate loss on the audited compact producer ports', () => {
  for (const exportName of [
    'GPUPairGather',
    'GPUNearestFeatureJoin',
    'GPUBufferSelection',
    'GPURegionStatistics',
    'GPUDotDensity',
    'GPURandomPointsInPolygon',
    'GPURandomPointsOnLine',
    'GPUZoneEvents',
    'GPUTrajectoryEncounters',
    'GPUTrajectoryClosestApproach',
    'GPUBoundsFilter',
    'addFleetDwellZoneEventsRecipe',
    'GPUNetworkPathExtraction',
    'GPUNetworkCoarsening',
    'GPUFlowAggregation'
  ]) {
    expect(
      queryGPUAnalysisOperators({status: 'candidateOverflow'}).some(
        operator => operator.exportName === exportName
      ),
      exportName
    ).toBe(true);
  }
});

it('uses ring requirements only for polygonize and make-valid', () => {
  for (const exportName of ['GPUPolygonize', 'GPUMakeValid']) {
    expect(getGPUAnalysisOperator(exportName)!.status).toEqual([
      'count',
      'requiredRingCount',
      'overflow',
      'candidateOverflow',
      'uncertainCount'
    ]);
  }
  expect(getGPUAnalysisOperator('GPUPairGather')!.status).toContain('candidateOverflow');
  expect(getGPUAnalysisOperator('GPUPairGather')!.status).not.toContain('requiredCount');
  expect(getGPUAnalysisOperator('GPUNearestFeatureJoin')!.status).toContain('requiredCount');
});

it('records construction admission and attainable SKATER cuts without inventing a row cap', () => {
  for (const exportName of [
    'GPUSkaterRegions',
    'GPUWardRegions',
    'GPUAZPRegions',
    'GPUMaxPRegions'
  ]) {
    const operator = getGPUAnalysisOperator(exportName)!;
    expect(
      operator.limits?.find(limit => limit.name === 'estimatedCommandNodeCount')?.maximum,
      exportName
    ).toBe(32768);
    expect(operator.limitation, exportName).toContain('construction');
  }
  expect(getGPUAnalysisOperator('GPUSkaterRegions')!.limitation).toContain(
    'min(maximumRegionCount - 1, rows - 1)'
  );
});

it('records the derived tile count and command budget while leaving unaudited metadata unknown', () => {
  const operator = getGPUAnalysisOperator('GPURasterTileReprojection')!;
  expect(operator.limits?.find(limit => limit.name === 'sources.length')?.maximum).toBe(16382);
  expect(operator.limits?.find(limit => limit.name === 'estimatedCommandNodeCount')?.maximum).toBe(
    32768
  );
  expect(operator.limitation).toContain('3 + 2 * sources.length');
  expect(operator.status).toBe('unknown');
  expect(operator.readback).toBe('unknown');
  expect(operator.tier).toBe('unknown');
});

it('states terrain axis restrictions and pyramid support semantics', () => {
  const terrain = getGPUAnalysisOperator('GPUTerrainDerivatives')!;
  expect(terrain.summary).toContain('orthogonal');
  expect(terrain.limitation).toContain('shear');
  expect(terrain.limitation).toContain('uniform');
  const pyramid = getGPUAnalysisOperator('GPURasterEmbeddingPyramidLevel')!;
  expect(pyramid.summary).toContain('sums');
  expect(pyramid.limitation).toContain('output.vectorSums');
  expect(pyramid.limitation).toContain('cancelling');
  expect(pyramid.limitation).toContain('2^32 - 1');
});

it('documents fractional area modes, exclusions, geographic axes and the snap cap', () => {
  const fractional = getGPUAnalysisOperator('GPURasterFractionalZonalStatistics')!;
  expect(fractional.summary).toContain('uniform');
  expect(fractional.limitation).toContain('cellAreas');
  expect(fractional.limitation).toContain('axis-aligned');
  expect(fractional.evidence.tests).toBe(
    'modules/analysis/test/gpu-raster/fractional-cell-size-mode.spec.ts'
  );
  expect(fractional.evidence.additionalTests).toContain(
    'modules/analysis/test/gpu-raster/raster-fractional-zonal-statistics-operation.spec.ts'
  );
  expect(getGPUAnalysisOperator('GPURasterZonalStatistics')!.limitation).toContain('axis-aligned');
  expect(getGPUAnalysisOperator('GPURasterFractionalCoverage')!.limitation).toContain('1e-5');
});

it('keeps point-cloud source evidence and renamed neighbor rows current', () => {
  const source = readFileSync(
    'modules/analysis/src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-records.ts',
    'utf8'
  );
  expect(source.includes('// pointcloud-analysis/gpu-pointcloud-')).toBe(false);
  expect(source).toContain('// pointcloud-analysis/operations/gpu-pointcloud-height.ts');
  expect(source).toContain('gpu-nearest-feature-weights.ts: neighborRows');
  for (const match of source.matchAll(
    /\/\/ (pointcloud-analysis\/operations\/gpu-pointcloud-[a-z-]+\.ts)/g
  ))
    expect(existsSync(`modules/analysis/src/${match[1]}`), match[1]).toBe(true);
});

it('distinguishes neighbor identities from gather and weights row inputs', () => {
  expect(getGPUAnalysisOperator('GPUNearestFeatureJoin')!.limitation).toContain('neighborRows');
  expect(getGPUAnalysisOperator('GPUPairGather')!.limitation).toContain('neighbors.rows');
  expect(getGPUAnalysisOperator('GPUNearestFeatureWeights')!.limitation).toContain('neighborRows');
  expect(getGPUAnalysisOperator('GPUBufferSelection')!.limitation).toContain('nearestFeatureRows');
});
