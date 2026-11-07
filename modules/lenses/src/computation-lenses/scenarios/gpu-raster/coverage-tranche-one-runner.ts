// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getRasterOperatorEntries} from '../../../gpu-raster/computation-pack/index';
import {getSpatialOperatorEntry} from '../../../gpu-spatial-analysis/computation-pack/index';
import {runRasterLens, type RasterLensSession} from './raster-lens-harness';
import {runExample as runDistribution} from './coverage-tranche-one-fractional-zonal-distribution.example';
import {runExample as runCategorical} from './coverage-tranche-one-fractional-categorical-statistics.example';
import {runExample as runFocalPairs} from './coverage-tranche-one-focal-pairs.example';
import {runExample as runNDWI} from './coverage-tranche-one-ndwi.example';
import {runExample as runNBR} from './coverage-tranche-one-nbr.example';
import {runExample as runEVI} from './coverage-tranche-one-evi.example';
import {runExample as runSAVI} from './coverage-tranche-one-savi.example';
import {runExample as runDasymetric} from './coverage-tranche-one-dasymetric.example';

export const COVERAGE_TRANCHE_ONE_GRID: GPURasterMetadata = {
  width: 3,
  height: 3,
  affine: [10, 0, 0, 0, 10, 0],
  pixelInterpretation: 'area'
};

/** Executes real public contributors and returns validated captures, never illustrative output data. */
export async function runCoverageTrancheOne(device: Device, element: string, scenarioId: string) {
  const domain =
    element === 'addDasymetricRedistributionRecipe' ? 'gpu-spatial-analysis' : 'gpu-raster';
  const entry =
    domain === 'gpu-raster'
      ? getRasterOperatorEntries()[element]
      : getSpatialOperatorEntry(element);
  const formula = entry?.formula;
  return runRasterLens({
    device,
    scenarioId,
    domain,
    label: element,
    grid: COVERAGE_TRANCHE_ONE_GRID,
    coordinateModel: 'projected',
    unit: 'm',
    dataset: {id: 'coverage-tranche-one-synthetic', version: '1'},
    formulas: formula && 'tex' in formula ? [formula] : [],
    keySpaces: [
      {id: 'zones', label: 'source zones', size: 2},
      {id: 'targets', label: 'target zones', size: 2},
      {id: 'classRows', label: 'zone-major class rows (three classes per zone)', size: 6},
      {id: 'quantileRows', label: 'zone-major quantiles (0.25,0.75 per zone)', size: 4},
      {id: 'probabilities', label: 'quantile probabilities', size: 2},
      {id: 'status', label: 'whole-domain completion', size: 1}
    ],
    build: session => buildCoverageTrancheOneScenario(session, element)
  });
}

/** Shared executable construction used by the viewer and CPU-only graph/wiring checks. */
export async function buildCoverageTrancheOneScenario(
  session: RasterLensSession,
  element: string
): Promise<void> {
  const grid = COVERAGE_TRANCHE_ONE_GRID;
  const rasterOutput = (name: string, meaning: string) =>
    session.output(name, 'float32', {role: 'output', geometry: 'raster', meaning});
  const zoneOutput = (name: string, meaning: string) =>
    session.output(name, 'float32', {role: 'output', keySpace: 'zones', meaning}, 2);
  const scalarInput = (name: string, value: number) =>
    session.input(name, 'uint32', [value], false);
  const coverage = () => ({
    zoneIds: session.input('coverageSourceZones', 'uint32', [0, 0, 0, 0, 0, 0, 1, 1, 1], {
      role: 'context',
      meaning: 'Source zone assigned to each prepared coverage tuple'
    }),
    cellIds: session.input('coverageCellIds', 'uint32', [0, 1, 2, 3, 4, 5, 6, 7, 8], false),
    fractions: session.input('coverageFractions', 'float32', [1, 0.5, 1, 0.25, 1, 0.5, 1, 0.5, 1], {
      role: 'input',
      geometry: 'raster',
      meaning: 'Exact supplied source-zone fraction of each 100 m² cell'
    }),
    zoneValidity: session.input('coverageZoneValidity', 'uint32', [1, 1], false),
    status: {
      count: scalarInput('coverageCount', 9),
      requiredCount: scalarInput('coverageRequiredCount', 9),
      overflow: scalarInput('coverageOverflow', 0)
    }
  });
  const floatBand = (
    name: string,
    values: number[],
    meaning: string
  ): GPURasterBufferBand<'float32'> => ({
    id: name,
    format: 'float32',
    storage: {
      kind: 'buffer',
      values: session.input(name, 'float32', values, {role: 'input', geometry: 'raster', meaning})
    }
  });
  if (element === 'GPURasterFractionalZonalDistribution') {
    await runDistribution(session.graph, {
      grid,
      input: floatBand(
        'observations',
        [1, 2, 4, 8, 16, 32, 3, 6, 12],
        'Measured values with fractional boundary support'
      ),
      zoneCount: 2,
      coverage: coverage(),
      probabilities: session.input('probabilities', 'float32', [0.25, 0.75], {
        role: 'context',
        keySpace: 'probabilities',
        meaning: 'Requested quantile probabilities'
      }),
      output: {
        means: zoneOutput('means', 'Fraction-weighted mean'),
        populationVariances: zoneOutput(
          'populationVariances',
          'Fraction-weighted population variance'
        ),
        sampleVariances: zoneOutput('sampleVariances', 'Reliability-weighted sample variance'),
        effectiveSampleSizes: zoneOutput(
          'effectiveSampleSizes',
          'Kish effective source-cell support'
        ),
        weightSums: zoneOutput('weightSums', 'Sum of valid coverage fractions'),
        squaredWeightSums: zoneOutput(
          'squaredWeightSums',
          'Sum of squared valid coverage fractions'
        ),
        coveredArea: zoneOutput('coveredArea', 'Covered footprint area (m²)'),
        validArea: zoneOutput('validArea', 'Valid observation support area (m²)'),
        quantiles: session.output(
          'quantiles',
          'float32',
          {
            role: 'output',
            keySpace: 'quantileRows',
            meaning: 'Zone-major lower and upper quartiles'
          },
          4
        ),
        quantileValidity: session.output(
          'quantileValidity',
          'uint32',
          {role: 'intermediate', keySpace: 'quantileRows', meaning: 'Defined quantile flags'},
          4
        ),
        validity: session.output(
          'validity',
          'uint32',
          {role: 'intermediate', keySpace: 'zones', meaning: 'Valid mean and population variance'},
          2
        ),
        status: session.output(
          'status',
          'uint32',
          {
            role: 'intermediate',
            keySpace: 'zones',
            meaning: 'Per-zone distribution diagnostic bits'
          },
          2
        )
      }
    });
    return;
  }
  if (element === 'GPURasterFractionalCategoricalStatistics') {
    await runCategorical(session.graph, {
      grid,
      input: {
        id: 'land-cover',
        format: 'uint32',
        storage: {
          kind: 'buffer',
          values: session.input('landCover', 'uint32', [0, 1, 1, 2, 2, 0, 1, 0, 2], {
            role: 'input',
            geometry: 'raster',
            meaning: 'Land-cover classes: 0 open, 1 vegetation, 2 built'
          })
        },
        validity: session.input('landCoverValidity', 'uint32', [1, 1, 1, 1, 0, 1, 1, 1, 1], {
          role: 'context',
          geometry: 'raster',
          meaning: 'Observation mask; masked class contributes to missing area'
        })
      },
      zoneCount: 2,
      categoryCapacity: 3,
      coverage: coverage(),
      output: {
        coverageSums: session.output(
          'coverageSums',
          'float32',
          {
            role: 'output',
            keySpace: 'classRows',
            meaning: 'Fractional support by source zone and class'
          },
          6
        ),
        classAreas: session.output(
          'classAreas',
          'float32',
          {role: 'output', keySpace: 'classRows', meaning: 'Planar class area (m²)'},
          6
        ),
        proportions: session.output(
          'proportions',
          'float32',
          {role: 'output', keySpace: 'classRows', meaning: 'Class share of valid area'},
          6
        ),
        coveredArea: zoneOutput('coveredArea', 'Total covered footprint area (m²)'),
        validArea: zoneOutput('validArea', 'Observed class area (m²)'),
        missingArea: zoneOutput('missingArea', 'Masked observation area (m²)'),
        majority: session.output(
          'majority',
          'uint32',
          {
            role: 'output',
            keySpace: 'zones',
            meaning: 'Largest support class; highest category breaks ties'
          },
          2
        ),
        minority: session.output(
          'minority',
          'uint32',
          {
            role: 'output',
            keySpace: 'zones',
            meaning: 'Smallest present class; lowest category breaks ties'
          },
          2
        ),
        validity: session.output(
          'validity',
          'uint32',
          {role: 'intermediate', keySpace: 'zones', meaning: 'Complete positive category support'},
          2
        )
      }
    });
    return;
  }
  if (element === 'GPURasterFocalPairs') {
    await runFocalPairs(session.graph, {
      grid,
      predictor: floatBand('predictor', [0, 1, 2, 3, 4, 5, 6, 7, 8], 'Synthetic predictor'),
      response: floatBand(
        'response',
        [3, 5, 7, 9, 11, 13, 15, 17, 19],
        'Synthetic response = 3 + 2 × predictor'
      ),
      radius: 1,
      borderMode: 'nodata',
      noDataPolicy: 'ignore',
      covarianceNormalization: 'sample',
      output: {
        pairedCount: session.output('pairedCount', 'uint32', {
          role: 'output',
          geometry: 'raster',
          meaning: 'Valid pairs inside each clipped 3 × 3 window'
        }),
        status: session.output('focalStatus', 'uint32', {
          role: 'intermediate',
          geometry: 'raster',
          meaning: 'Per-cell fit diagnostic bits'
        }),
        covariance: rasterOutput('covariance', 'Sample covariance'),
        correlation: rasterOutput('correlation', 'Pearson correlation'),
        slope: rasterOutput('slope', 'Local regression slope (expected 2)'),
        intercept: rasterOutput('intercept', 'Local regression intercept (expected 3)'),
        rSquared: rasterOutput('rSquared', 'Descriptive coefficient of determination'),
        residualSumSquares: rasterOutput('residualSumSquares', 'Sum of squared fitted residuals')
      }
    });
    return;
  }
  if (element === 'addDasymetricRedistributionRecipe') {
    const result = await runDasymetric(session.graph, {
      grid,
      source: {kind: 'coverage', coverage: coverage()},
      sourceTotals: session.input('sourceTotals', 'float32', [120, 90], {
        role: 'input',
        keySpace: 'zones',
        meaning: 'Extensive source population totals'
      }),
      targetZones: session.input('targetZones', 'uint32', [0, 0, 1, 0, 1, 1, 0, 1, 1], {
        role: 'context',
        geometry: 'raster',
        meaning: 'Whole-cell destination ownership'
      }),
      targetZoneCount: 2,
      ancillary: floatBand(
        'ancillary',
        [1, 2, 1, 3, 2, 1, 1, 2, 1],
        'Nonnegative ancillary density (weight per m²)'
      ),
      mask: session.input('eligibility', 'uint32', [1, 1, 1, 1, 0, 1, 1, 1, 1], {
        role: 'context',
        geometry: 'raster',
        meaning: 'Eligible redistribution support'
      }),
      denominator: 'overlap',
      pairCapacity: 9
    });
    const outputMeanings: Record<string, string> = {
      targetTotals: 'Redistributed target population totals',
      targetSupportWeights: 'Ancillary-weighted target support',
      sourceEligibleWeights: 'Eligible source support weight',
      sourceTransferredFractions: 'Fraction of each source population assigned',
      sourceAllocatedTotals: 'Allocated source population',
      sourceUnallocatedTotals: 'Retained source population',
      sourceValidity: 'Complete positive eligible support'
    };
    for (const [name, view] of Object.entries(result.outputs)) {
      session.capture(name, view, {
        role: 'output',
        keySpace: name.startsWith('target') ? 'targets' : 'zones',
        meaning: outputMeanings[name]
      });
    }
    session.capture('eligibleCellWeights', result.intermediates.cellWeights, {
      role: 'intermediate',
      geometry: 'raster',
      meaning: 'Eligible ancillary density × cell area'
    });
    for (const name of [
      'complete',
      'invalidTupleCount',
      'invalidSourceCount',
      'zeroEligibleCount'
    ] as const)
      session.capture(name, result.status[name], {
        role: 'intermediate',
        keySpace: 'status',
        meaning: name
      });
    return;
  }
  const reflectance = (name: string, values: number[]): GPURasterBufferBand<'sint32'> => ({
    id: name,
    format: 'sint32',
    storage: {
      kind: 'buffer',
      values: session.input(name, 'sint32', values, {
        role: 'input',
        geometry: 'raster',
        meaning: `${name}: scaled integer reflectance; multiply by 0.0001`
      })
    },
    scale: 0.0001,
    noDataValue: -9999
  });
  const nearInfrared = reflectance(
    'nearInfrared',
    [6000, 5000, 4000, 3000, 2000, 1000, 7000, 8000, -9999]
  );
  const common = {
    width: 3,
    height: 3,
    nearInfrared,
    output: rasterOutput('index', 'Dimensionless spectral index; nodata remains invalid'),
    outputValidity: session.output('indexValidity', 'uint32', {
      role: 'intermediate',
      geometry: 'raster',
      meaning: 'Intersection of calibrated source validity'
    }),
    epsilon: 1e-6
  };
  switch (element) {
    case 'GPURasterNDWI':
      await runNDWI(session.graph, {
        ...common,
        method: 'mcfeeters',
        green: reflectance('green', [1000, 2000, 3000, 4000, 5000, 6000, 1000, 2000, 3000])
      });
      return;
    case 'GPURasterNBR':
      await runNBR(session.graph, {
        ...common,
        shortwaveInfrared2: reflectance(
          'shortwaveInfrared2',
          [1000, 2000, 3000, 4000, 5000, 6000, 1000, 2000, 3000]
        )
      });
      return;
    case 'GPURasterEVI':
      await runEVI(session.graph, {
        ...common,
        red: reflectance('red', [1000, 1500, 2000, 2500, 3000, 3500, 1000, 1500, 2000]),
        blue: reflectance('blue', [500, 500, 500, 500, 500, 500, 500, 500, 500])
      });
      return;
    case 'GPURasterSAVI':
      await runSAVI(session.graph, {
        ...common,
        red: reflectance('red', [1000, 1500, 2000, 2500, 3000, 3500, 1000, 1500, 2000]),
        soilAdjustment: 0.5
      });
      return;
    default:
      throw new Error(`Unknown coverage scenario: ${element}`);
  }
}
