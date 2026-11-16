// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_TRANCHE_ONE_CASES} from './kernel-audit/cases-coverage-tranche-one';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('coverage-tranche-one', COVERAGE_TRANCHE_ONE_CASES);

import {expect, it} from 'vitest';
import {createKernelAuditContext} from './kernel-audit/kernel-audit-context';
import {GPURasterFractionalZonalDistribution} from '../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {
  GPURasterFractionalCategoricalStatistics,
  type GPURasterFractionalCategoricalStatisticsProps
} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterFocalPairs} from '../../src/raster-analysis/focal-pairs/index';
import {GPURasterSAVI} from '../../src/raster-analysis/gpu-raster-spectral-indices';
import {addDasymetricRedistributionRecipe} from '../../src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe';

it('coverage tranche one refuses documented scalar and capacity bounds one step above maximum', () => {
  const context = createKernelAuditContext('coverage-boundaries');
  const view = context.view;
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const band = {
    id: 'input',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('float32', 2)}
  };
  const coverage = (count: number, zones = 1) => ({
    zoneIds: view('uint32', count),
    cellIds: view('uint32', count),
    fractions: view('float32', count),
    zoneValidity: view('uint32', zones),
    status: {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1)
    }
  });
  try {
    for (const [tupleCount, probabilityCount, zoneCount] of [
      [65537, 1, 1],
      [1, 33, 1],
      [1, 1, 65537]
    ]) {
      expect(
        () =>
          new GPURasterFractionalZonalDistribution({
            grid,
            input: band,
            zoneCount,
            coverage: coverage(tupleCount, zoneCount),
            probabilities: view('float32', probabilityCount),
            output: {
              means: view('float32', zoneCount),
              populationVariances: view('float32', zoneCount),
              sampleVariances: view('float32', zoneCount),
              effectiveSampleSizes: view('float32', zoneCount),
              weightSums: view('float32', zoneCount),
              squaredWeightSums: view('float32', zoneCount),
              coveredArea: view('float32', zoneCount),
              validArea: view('float32', zoneCount),
              quantiles: view('float32', zoneCount * probabilityCount),
              quantileValidity: view('uint32', zoneCount * probabilityCount),
              validity: view('uint32', zoneCount),
              status: view('uint32', zoneCount)
            }
          }),
        `distribution tuples=${tupleCount} quantiles=${probabilityCount} zones=${zoneCount}`
      ).toThrow();
    }
    for (const [categoryCapacity, zoneCount, tupleCount] of [
      [65535, 1, 1],
      [2, 21846, 1],
      [1, 1, 1048577]
    ]) {
      const props: GPURasterFractionalCategoricalStatisticsProps = {
        id: `categorical-boundary-${categoryCapacity}-${zoneCount}-${tupleCount}`,
        grid,
        input: {
          id: 'categories',
          format: 'uint32',
          storage: {kind: 'buffer', values: view('uint32', 2)}
        },
        zoneCount,
        categoryCapacity,
        coverage: coverage(tupleCount, zoneCount),
        output: {
          coverageSums: view('float32', zoneCount * categoryCapacity),
          classAreas: view('float32', zoneCount * categoryCapacity),
          proportions: view('float32', zoneCount * categoryCapacity),
          coveredArea: view('float32', zoneCount),
          validArea: view('float32', zoneCount),
          missingArea: view('float32', zoneCount),
          majority: view('uint32', zoneCount),
          minority: view('uint32', zoneCount),
          validity: view('uint32', zoneCount)
        }
      };
      const label = `categorical categories=${categoryCapacity} zones=${zoneCount} tuples=${tupleCount}`;
      // The opt-in reference route retains every historical scalar and capacity bound.
      expect(
        () => new GPURasterFractionalCategoricalStatistics({...props, mode: 'exact'}),
        `${label} exact`
      ).toThrow(
        tupleCount > 1048576
          ? /coverage.zoneIds length in exact mode.*1048576/
          : /zoneCount times.*65535/
      );
      if (categoryCapacity === 65535) {
        expect(() => new GPURasterFractionalCategoricalStatistics(props), label).toThrow(
          /categoryCapacity.*65534/
        );
      } else {
        // Tiled accumulation deliberately lifts the old group and tuple caps. Build the graph
        // too, so acceptance includes derived scratch and the declared command budget.
        const contributor = new GPURasterFractionalCategoricalStatistics(props);
        const nodes = contributor.getCommandNodes(context.graph);
        expect(nodes.length, label).toBeGreaterThan(0);
        expect(nodes.length, label).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        expect(
          nodes.every(node => node.id.startsWith(props.id!)),
          label
        ).toBe(true);
      }
    }

    for (const options of [
      {radius: 9, minimumCount: 2},
      {radius: 8, minimumCount: 290}
    ])
      expect(
        () =>
          new GPURasterFocalPairs({
            grid,
            predictor: band,
            response: band,
            ...options,
            output: {
              pairedCount: view('uint32', 2),
              status: view('uint32', 2),
              slope: view('float32', 2)
            }
          })
      ).toThrow();
    expect(
      () =>
        new GPURasterSAVI({
          width: 2,
          height: 1,
          nearInfrared: band,
          red: band,
          soilAdjustment: 1.0001,
          output: view('float32', 2),
          outputValidity: view('uint32', 2)
        })
    ).toThrow();
    expect(() =>
      addDasymetricRedistributionRecipe(context.graph, {
        grid,
        source: {kind: 'coverage', coverage: coverage(65537)},
        sourceTotals: view('float32', 1),
        targetZones: view('uint32', 2),
        targetZoneCount: 64,
        ancillary: band,
        pairCapacity: 65537
      })
    ).toThrow();
  } finally {
    context.device.destroy();
  }
});
