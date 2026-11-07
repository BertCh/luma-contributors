// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addDasymetricRedistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe';
import {GPURasterFractionalZonalDistribution} from '../../../src/gpu-raster/gpu-raster-fractional-zonal-distribution';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/gpu-raster/gpu-raster-fractional-categorical-statistics';
import {GPURasterFocalPairs} from '../../../src/gpu-raster/focal-pairs/index';
import {
  GPURasterNDWI,
  GPURasterNBR,
  GPURasterEVI,
  GPURasterSAVI
} from '../../../src/gpu-raster/gpu-raster-spectral-indices';
import type {GPURasterBufferBand, GPURasterMetadata} from '../../../src/gpu-raster/types';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const GRID: GPURasterMetadata = {
  width: 128,
  height: 128,
  affine: [2, 0.25, 0, 0.5, -2, 0],
  pixelInterpretation: 'area'
};
const cases: KernelAuditCase[] = [];
function add(exportName: string, name: string, build: KernelAuditCase['build']) {
  cases.push({exportName, name, build});
}
function signedBand(context: KernelAuditContext, length: number): GPURasterBufferBand<'sint32'> {
  return {
    id: 'calibrated-signed',
    format: 'sint32',
    storage: {kind: 'buffer', values: context.view('sint32', length)},
    validity: context.view('uint32', length),
    noDataValue: -9999,
    scale: 0.0001,
    offset: 0.01
  };
}
function coverage(context: KernelAuditContext, length: number, zoneCount: number) {
  return {
    zoneIds: context.view('uint32', length),
    cellIds: context.view('uint32', length),
    fractions: context.view('float32', length),
    zoneValidity: context.view('uint32', zoneCount),
    status: {
      count: context.view('uint32', 1),
      requiredCount: context.view('uint32', 1),
      overflow: context.view('uint32', 1)
    }
  };
}
add(
  'GPURasterFractionalZonalDistribution',
  '65536 tuples and zones,32 quantiles,calibrated masked input,every output',
  context => {
    const zoneCount = 65536;
    context.graph.add(
      new GPURasterFractionalZonalDistribution({
        grid: GRID,
        input: signedBand(context, 16384),
        zoneCount,
        coverage: coverage(context, 65536, zoneCount),
        probabilities: context.view('float32', 32),
        output: {
          means: context.view('float32', zoneCount),
          populationVariances: context.view('float32', zoneCount),
          sampleVariances: context.view('float32', zoneCount),
          effectiveSampleSizes: context.view('float32', zoneCount),
          weightSums: context.view('float32', zoneCount),
          squaredWeightSums: context.view('float32', zoneCount),
          coveredArea: context.view('float32', zoneCount),
          validArea: context.view('float32', zoneCount),
          quantiles: context.view('float32', zoneCount * 32),
          quantileValidity: context.view('uint32', zoneCount * 32),
          validity: context.view('uint32', zoneCount),
          status: context.view('uint32', zoneCount)
        }
      })
    );
  }
);
for (const [zoneCount, categoryCapacity] of [
  [63, 1024],
  [21845, 2]
]) {
  add(
    'GPURasterFractionalCategoricalStatistics',
    `${zoneCount} zones,${categoryCapacity} categories,1048576 tuples,masked nodata,every output`,
    context => {
      const classCount = zoneCount * categoryCapacity;
      context.graph.add(
        new GPURasterFractionalCategoricalStatistics({
          grid: GRID,
          input: {
            id: 'categories',
            format: 'uint32',
            storage: {kind: 'buffer', values: context.view('uint32', 16384)},
            validity: context.view('uint32', 16384),
            noDataValue: 0xffffffff
          },
          zoneCount,
          categoryCapacity,
          coverage: coverage(context, 1048576, zoneCount),
          output: {
            coverageSums: context.view('float32', classCount),
            classAreas: context.view('float32', classCount),
            proportions: context.view('float32', classCount),
            coveredArea: context.view('float32', zoneCount),
            validArea: context.view('float32', zoneCount),
            missingArea: context.view('float32', zoneCount),
            majority: context.view('uint32', zoneCount),
            minority: context.view('uint32', zoneCount),
            validity: context.view('uint32', zoneCount)
          }
        })
      );
    }
  );
}
for (const borderMode of ['nodata', 'clamp', 'reflect'] as const) {
  for (const noDataPolicy of ['ignore', 'propagate'] as const) {
    add(
      'GPURasterFocalPairs',
      `radius8 ${borderMode} ${noDataPolicy},population minimum289,calibrated masks,every output`,
      context => {
        context.graph.add(
          new GPURasterFocalPairs({
            grid: GRID,
            predictor: signedBand(context, 16384),
            response: signedBand(context, 16384),
            radius: 8,
            borderMode,
            noDataPolicy,
            minimumCount: 289,
            propagateCenterNoData: true,
            covarianceNormalization: 'population',
            output: {
              pairedCount: context.view('uint32', 16384),
              status: context.view('uint32', 16384),
              covariance: context.view('float32', 16384),
              correlation: context.view('float32', 16384),
              slope: context.view('float32', 16384),
              intercept: context.view('float32', 16384),
              rSquared: context.view('float32', 16384),
              residualSumSquares: context.view('float32', 16384)
            }
          })
        );
      }
    );
  }
}
add(
  'GPURasterFocalPairs',
  'largest4096-wide scratch binding grid,radius8,sample,every output',
  context => {
    const height = Math.floor(context.maximumRows(24) / 4096);
    const count = 4096 * height;
    context.graph.add(
      new GPURasterFocalPairs({
        grid: {...GRID, width: 4096, height},
        predictor: signedBand(context, count),
        response: signedBand(context, count),
        radius: 8,
        covarianceNormalization: 'sample',
        output: {
          pairedCount: context.view('uint32', count),
          status: context.view('uint32', count),
          covariance: context.view('float32', count),
          correlation: context.view('float32', count),
          slope: context.view('float32', count),
          intercept: context.view('float32', count),
          rSquared: context.view('float32', count),
          residualSumSquares: context.view('float32', count)
        }
      })
    );
  }
);
add(
  'GPURasterFocalPairs',
  'two texture channels,raw nodata,calibration and validity,radius8,every output',
  context => {
    const textureBand = () => ({
      id: 'texture-source',
      format: 'float32' as const,
      storage: {
        kind: 'texture' as const,
        view: context.graph.createTextureView(context.texture('rgba32float', 128, 128)),
        channel: 3 as const
      },
      validity: context.view('uint32', 16384),
      noDataValue: -9999,
      scale: 2,
      offset: 3
    });
    context.graph.add(
      new GPURasterFocalPairs({
        grid: GRID,
        predictor: textureBand(),
        response: textureBand(),
        radius: 8,
        borderMode: 'reflect',
        noDataPolicy: 'propagate',
        propagateCenterNoData: true,
        output: {
          pairedCount: context.view('uint32', 16384),
          status: context.view('uint32', 16384),
          covariance: context.view('float32', 16384),
          correlation: context.view('float32', 16384),
          slope: context.view('float32', 16384),
          intercept: context.view('float32', 16384),
          rSquared: context.view('float32', 16384),
          residualSumSquares: context.view('float32', 16384)
        }
      })
    );
  }
);
for (const maximum of [false, true]) {
  for (const method of ['gao', 'mcfeeters'] as const) {
    add(
      'GPURasterNDWI',
      `${method},${maximum ? 'maximum binding rows' : 'calibrated unsigned/signed'},masked epsilon clamp`,
      context => {
        const count = maximum ? context.maximumRows(4) : 16384;
        const common = {
          width: maximum ? 8192 : 128,
          height: maximum ? 4096 : 128,
          nearInfrared: signedBand(context, count),
          output: context.view('float32', count),
          outputValidity: context.view('uint32', count),
          epsilon: 1e-6,
          clamp: [-1, 1] as const
        };
        const other: GPURasterBufferBand<'uint32'> = {
          id: 'unsigned',
          format: 'uint32',
          storage: {kind: 'buffer', values: context.view('uint32', count)},
          validity: context.view('uint32', count),
          noDataValue: 0xffffffff,
          scale: 0.0001,
          offset: 0.02
        };
        new GPURasterNDWI(
          method === 'gao'
            ? {...common, method, shortwaveInfrared: other}
            : {...common, method, green: other}
        ).addToGraph(context.graph);
      }
    );
  }
  for (const element of ['GPURasterNBR', 'GPURasterEVI', 'GPURasterSAVI'] as const) {
    add(
      element,
      `${maximum ? 'maximum binding rows' : 'calibrated signed'},masked epsilon clamp`,
      context => {
        const count = maximum ? context.maximumRows(4) : 16384;
        const common = {
          width: maximum ? 8192 : 128,
          height: maximum ? 4096 : 128,
          nearInfrared: signedBand(context, count),
          output: context.view('float32', count),
          outputValidity: context.view('uint32', count),
          epsilon: 1e-6,
          clamp: [-1, 1] as const
        };
        if (element === 'GPURasterNBR')
          new GPURasterNBR({...common, shortwaveInfrared2: signedBand(context, count)}).addToGraph(
            context.graph
          );
        if (element === 'GPURasterEVI')
          new GPURasterEVI({
            ...common,
            red: signedBand(context, count),
            blue: signedBand(context, count)
          }).addToGraph(context.graph);
        if (element === 'GPURasterSAVI')
          new GPURasterSAVI({
            ...common,
            red: signedBand(context, count),
            soilAdjustment: 1
          }).addToGraph(context.graph);
      }
    );
  }
}

for (const denominator of ['zone', 'overlap'] as const) {
  for (const sourceKind of ['coverage', 'polygons'] as const) {
    add(
      'addDasymetricRedistributionRecipe',
      `${sourceKind},${denominator},max target-tuple product,masked calibration,caller outputs`,
      context => {
        const sourceCount = sourceKind === 'coverage' ? 1024 : 16;
        const targetCount = 64;
        const tupleCapacity = 65536;
        addDasymetricRedistributionRecipe(context.graph, {
          grid: GRID,
          source:
            sourceKind === 'coverage'
              ? {kind: 'coverage', coverage: coverage(context, tupleCapacity, sourceCount)}
              : {
                  kind: 'polygons',
                  tupleCapacity,
                  zones: {
                    positions: context.view('float32x2', 128),
                    featureOffsets: context.view('uint32', sourceCount + 1),
                    polygonOffsets: context.view('uint32', 33),
                    ringOffsets: context.view('uint32', 33)
                  }
                },
          sourceTotals: context.view('float32', sourceCount),
          targetZones: context.view('uint32', 16384),
          targetZoneCount: targetCount,
          ancillary: signedBand(context, 16384),
          mask: context.view('uint32', 16384),
          denominator,
          pairCapacity: tupleCapacity,
          outputs: {
            targetTotals: context.view('float32', targetCount),
            targetSupportWeights: context.view('float32', targetCount),
            sourceEligibleWeights: context.view('float32', sourceCount),
            sourceTransferredFractions: context.view('float32', sourceCount),
            sourceAllocatedTotals: context.view('float32', sourceCount),
            sourceUnallocatedTotals: context.view('float32', sourceCount),
            sourceValidity: context.view('uint32', sourceCount)
          }
        });
      }
    );
  }
}

export const COVERAGE_TRANCHE_ONE_CASES: readonly KernelAuditCase[] = cases;
