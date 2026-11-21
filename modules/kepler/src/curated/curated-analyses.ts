// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS} from './bindings/curated-cell-flow-topology-binding';
import {REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE} from './samples/regional-raster-change';
import {CURATED_ZONAL_EXPOSURE_DESCRIPTOR} from '../descriptors/descriptors-curated-zonal-exposure';
import {CURATED_DRIVE_TIME_BANDS_ANALYSIS} from './samples/drive-time-bands-data';
import {CURATED_NETWORK_DISRUPTION_ANALYSIS} from './samples/network-disruption';
import {CURATED_DRAINAGE_NETWORK_ANALYSIS} from './samples/drainage-network';
import type {
  GPUAnalysisFieldRoleSpec,
  GPUAnalysisQuestion
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {KeplerCuratedAnalysis, KeplerCuratedAnalysisSlot} from './curated-analysis-types';
import {CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS} from './bindings/curated-dasymetric-redistribution-binding';

// The 27 starting analyses plus drive-time bands of the analysis-experience design (section 4.6), in table order. Each is
// one step: a question, typed dataset slots with field roles, preset parameters and a few exposed
// ones. `available` means one registered catalog binding delivers what the title promises, with the
// presets given here. `needs-binding` names the analysis-package recipe that implements it and what
// the Kepler package lacks; those entries carry no execution inputs or exposed parameters.
//
// Where the nearest binding does only part of the README's analysis, the entry is `available` only
// when its title, name and method describe exactly what that binding computes (facility reach
// without bands, period comparison from two uploaded point sets). Curated analyses run on the fused
// bindings of `modules/kepler/src/curated/bindings`, which chain operations the catalog already
// computes on feature collections.

const NUMBER_COLUMN = ['number', 'integer'] as const;

function makeNumberRole(
  role: string,
  label: string,
  required: GPUAnalysisFieldRoleSpec['required'],
  measurement: GPUAnalysisFieldRoleSpec['measurement']
): GPUAnalysisFieldRoleSpec {
  return {role, label, accepts: NUMBER_COLUMN, measurement, required};
}

function makePointsSlot(
  name: string,
  label: string,
  fields: readonly GPUAnalysisFieldRoleSpec[],
  rowSemantics?: KeplerCuratedAnalysisSlot['rowSemantics']
): KeplerCuratedAnalysisSlot {
  return {name, label, kind: 'points', rowSemantics, fields};
}

function makeAreasSlot(
  name: string,
  label: string,
  fields: readonly GPUAnalysisFieldRoleSpec[],
  rowSemantics?: KeplerCuratedAnalysisSlot['rowSemantics']
): KeplerCuratedAnalysisSlot {
  return {name, label, kind: 'polygons', rowSemantics, fields};
}

function makeSurfaceSlot(
  name: string,
  label: string,
  fields: readonly GPUAnalysisFieldRoleSpec[]
): KeplerCuratedAnalysisSlot {
  return {name, label, kind: 'surface', fields};
}

export const KEPLER_CURATED_ANALYSES: readonly KeplerCuratedAnalysis[] = [
  {
    id: 'density-surface',
    question: 'where-concentrated',
    title: 'Where are the points concentrated?',
    name: 'Density surface',
    method:
      'Each point spreads its weight over the surrounding area with a Gaussian kernel whose width is the bandwidth. The spread weights are added on a regular grid, and each cell holds intensity per square metre for geographic coordinates, or per squared coordinate unit for planar coordinates. The map shows a smooth surface whose peaks identify the greatest local summed weight; with unit weights this represents point concentration, and it does not test whether a peak is more than chance. By default the kernel is Gaussian and the grid covers only the extent of the points. Padding extends the evaluated grid; it does not correct boundary bias.',
    wave: 1,
    slots: [
      makePointsSlot(
        'points',
        'Points',
        [makeNumberRole('weight', 'Weight (optional)', false, ['count', 'amount'])],
        ['events']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'kernel-density-grid',
      inputs: {points: 'points'},
      fieldParameters: {'points.weight': 'weightProperty'},
      parameters: {kernel: 'gaussian', includeEmptyCells: false}
    },
    exposedParameters: ['bandwidth', 'cellSize', 'kernel'],
    sampleId: 'city-incidents'
  },
  {
    id: 'hot-spot-map',
    question: 'where-significant',
    title: 'Where are the significant hot and cold spots?',
    name: 'Hot spot map',
    method:
      'Points are counted (or a chosen value is summed) in the cells of a square lattice. For every cell the Getis-Ord Gi* statistic compares the total in the cell and its surrounding cells (by default the eight that touch it) with what the whole lattice would give by chance, which gives a z-score and a p-value. By default the p-values are adjusted for the number of cells tested with the Benjamini-Hochberg procedure. The map shows each cell as a hot spot or cold spot at 90, 95 or 99 percent, or as not significant.',
    wave: 1,
    slots: [
      makePointsSlot(
        'points',
        'Points',
        [makeNumberRole('value', 'Value to sum (optional)', false, ['count', 'amount'])],
        ['events']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'recipe-hot-spot-lattice',
      inputs: {points: 'points'},
      fieldParameters: {'points.value': 'valueProperty'},
      parameters: {
        criterion: 'queen',
        radius: 1,
        multipleTesting: 'benjamini-hochberg'
      }
    },
    exposedParameters: ['cellSize', 'radius', 'multipleTesting'],
    sampleId: 'city-incidents'
  },
  {
    id: 'clusters-and-footprints',
    question: 'which-groups-or-regions',
    title: 'Which points form groups, and what area does each cover?',
    name: 'Clusters and footprints',
    method:
      'A point is a core point when at least the minimum number of points, itself included, lie within the search distance of it. Core points within reach of each other form one group, and points within reach of a core point but without enough neighbours of their own join a reachable group as border points. A border point reachable from multiple groups joins the group whose first core point occurs first in input order. Every other point is noise. The footprint of a group is the convex hull of its points, the smallest convex polygon that contains them all, with the number of points, the area and the perimeter. A group of fewer than three points, or of points on one line, has no footprint and is listed without one. The map colours each point by its group and marks the noise, and a second table lists the footprints.',
    wave: 1,
    slots: [makePointsSlot('points', 'Points', [], ['events', 'objects'])],
    execution: {
      status: 'available',
      bindingId: 'curated-clusters-and-footprints',
      inputs: {points: 'points'},
      fieldParameters: {},
      parameters: {}
    },
    exposedParameters: ['epsilon', 'minimumSamples'],
    sampleId: 'city-incidents'
  },
  {
    id: 'emerging-hot-spots',
    question: 'what-changed',
    title: 'How do hot spots change over time?',
    name: 'Emerging hot spots',
    method:
      'Points are counted in the cells of a lattice for each time period, which gives a cube of cells by periods. Each cell is tested for a hot or cold spot in every period and the series of results is classified as new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating or historical. The map shows each cell with its class.',
    wave: 1,
    slots: [
      makePointsSlot(
        'points',
        'Points with a time',
        [
          {
            role: 'time',
            label: 'Time',
            accepts: ['time', 'number', 'integer'],
            required: true
          }
        ],
        ['events']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-emerging-hot-spots',
      inputs: {points: 'points'},
      fieldParameters: {'points.time': 'timeProperty'},
      parameters: {}
    },
    exposedParameters: ['cellSize', 'timeStep', 'sliceCount', 'radius', 'temporalWindow'],
    sampleId: 'emerging-hot-spots'
  },
  {
    id: 'period-comparison',
    question: 'what-changed',
    title: 'What changed between the earlier and the later points?',
    name: 'Change between two point sets',
    method:
      'The points of both sets are counted (or a chosen value is summed or averaged) in the cells of one square lattice laid over the coordinates, whose cell size is in the units of the coordinates (degrees for longitude and latitude). For each cell the later value is compared with the earlier one, giving the difference and the relative change; a cell with points in only one set is marked as appeared or disappeared. The map shows each cell coloured by its change. The two totals are not normalised: a later period that is longer, or reported more completely, shows an increase that is not a change in the process. It is a comparison of cell totals, not a significance test, and the two periods must already be two datasets.',
    wave: 2,
    slots: [
      makePointsSlot('before', 'Earlier points', [
        makeNumberRole(
          'value',
          'Value (needed to sum or average)',
          {whenParameter: 'statistic', equals: ['sum', 'mean']},
          undefined
        )
      ]),
      makePointsSlot('after', 'Later points', [
        makeNumberRole(
          'value',
          'Value (the same column as in the earlier points)',
          {whenParameter: 'statistic', equals: ['sum', 'mean']},
          undefined
        )
      ])
    ],
    execution: {
      status: 'available',
      bindingId: 'cell-compare',
      inputs: {before: 'before', after: 'after'},
      fieldParameters: {'before.value': 'valueProperty', 'after.value': 'valueProperty'},
      parameters: {statistic: 'count'}
    },
    exposedParameters: ['cellSize', 'statistic'],
    sampleId: 'before-after-incidents'
  },

  {
    id: 'idw-surface',
    question: 'value-where-unmeasured',
    title: 'What is the value where nothing was measured?',
    name: 'Inverse distance weighted surface',
    method:
      'For the centre of each cell of a regular grid, the measured values of the nearest points within the search radius are averaged, with each value weighted by one over its distance raised to the power. A cell with fewer than the minimum number of measurements in reach is left empty. The map shows the interpolated surface; it gives no estimate of how sure each value is, and the surface cannot rise above the highest measurement or fall below the lowest.',
    wave: 2,
    slots: [
      makePointsSlot(
        'points',
        'Measurements',
        [makeNumberRole('value', 'Measured value', true, ['amount', 'rate'])],
        ['surface-samples']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'idw-grid',
      inputs: {points: 'points'},
      fieldParameters: {'points.value': 'valueProperty'},
      parameters: {
        power: 2,
        neighborCount: 12,
        minimumNeighborCount: 3,
        includeEmptyCells: false
      }
    },
    exposedParameters: ['cellSize', 'searchRadius', 'power'],
    sampleId: 'rain-gauges'
  },
  {
    id: 'kriging-with-diagnostics',
    question: 'value-where-unmeasured',
    title: 'What is the value where nothing was measured, and how sure is it?',
    name: 'Kriging prediction, variance and diagnostics',
    method:
      'The variation of the measurements with the distance between them is summarised as an empirical variogram, and a spherical, exponential or Gaussian model is fitted to it, the best of the three being kept. The surface is then predicted by ordinary kriging with the fitted model, and each cell gets a prediction and a kriging variance. Each measurement is also predicted from the nearest others within the search radius (the same neighbourhood as the grid, not all the others), without refitting the model, to give a prediction error and a standardised error for it. The table gives the root mean square error and the mean error of those predictions, and two checks of the kriging variance: the mean standardised error, which should be near zero, and the root mean square standardised error, which should be near one. The result is flagged when the variogram is not well identified (range beyond the largest lag, almost no spatial structure) or the standardised errors are off. The map shows the surface and its uncertainty.',
    wave: 1,
    slots: [
      makePointsSlot(
        'points',
        'Measurements',
        [makeNumberRole('value', 'Measured value', true, ['amount', 'rate'])],
        ['surface-samples']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-kriging-with-diagnostics',
      inputs: {points: 'points'},
      fieldParameters: {'points.value': 'valueProperty'},
      parameters: {
        variogram: 'fit',
        model: 'auto',
        binCount: 15,
        neighborCount: 16
      }
    },
    exposedParameters: ['cellSize', 'variogram', 'model'],
    sampleId: 'rain-gauges'
  },
  {
    id: 'count-by-area',
    question: 'how-much-per-area',
    title: 'How many, or how much, is in each area?',
    name: 'Count and summarise by area',
    method:
      'Each point is assigned to the polygon that contains it; points outside every polygon are counted and set aside. For each polygon the points are counted, or the chosen value is summed, averaged or reduced to its minimum or maximum. The map shows each area with its count or statistic, not a rate: areas of unequal size or population mostly show where the areas are large or the people live, so divide by area or population before comparing areas. Class breaks and colours come from the map styling.',
    wave: 1,
    slots: [
      makePointsSlot('points', 'Points', [
        makeNumberRole(
          'value',
          'Value (needed unless counting)',
          {whenParameter: 'statistic', equals: ['sum', 'mean', 'minimum', 'maximum']},
          undefined
        )
      ]),
      makeAreasSlot('areas', 'Areas', [])
    ],
    execution: {
      status: 'available',
      bindingId: 'recipe-points-in-polygons-choropleth',
      inputs: {points: 'points', areas: 'polygons'},
      fieldParameters: {'points.value': 'valueProperty'},
      parameters: {statistic: 'count'}
    },
    exposedParameters: ['statistic'],
    sampleId: 'city-district-incidents'
  },

  {
    id: 'distribution-by-zone',
    question: 'how-much-per-area',
    title: 'How do areas differ in what they contain?',
    name: 'Distribution by zone',
    method:
      'Each point is assigned to the polygon that contains it. For each polygon the values of its points are sorted and summarised by the median and the chosen quantiles (the quartiles by default), with the number of values. The map shows each area with those figures, so areas with the same average but a different spread can be told apart.',
    wave: 1,
    slots: [
      makePointsSlot('points', 'Points with a value', [
        makeNumberRole('value', 'Value', true, ['count', 'amount', 'rate', 'ordinal'])
      ]),
      makeAreasSlot('areas', 'Areas', [])
    ],
    execution: {
      status: 'available',
      bindingId: 'recipe-zonal-distribution',
      inputs: {points: 'points', areas: 'polygons'},
      fieldParameters: {'points.value': 'valueProperty'},
      parameters: {probabilities: [0.25, 0.75]}
    },
    exposedParameters: ['probabilities'],
    sampleId: 'city-district-incidents'
  },

  {
    id: 'straight-line-catchments',
    question: 'what-is-near-or-reachable',
    title: 'What does each facility serve, in a straight line?',
    name: 'Straight-line catchments',
    method:
      'A grid is laid over the facilities and the observations. Facilities snap to grid seed cells, and every cell is given to its nearest facility seed cell by straight-line distance. This approximates facility distance at the chosen resolution. Ties go to the first facility in input order; facilities sharing a seed cell leave later facilities with empty catchments. The observations are averaged per cell, and each catchment reports allocated cells, occupied cells that hold observations, and the mean of the occupied cell means. Without a value column each observation contributes one, so occupied cell means are one. These are cell figures, not observation figures: a catchment with 100 observations of 0 in one cell and one observation of 10 in each of ten other cells has a mean of 9.09, not 0.91. The default map shades facilities by allocated cell count; occupied cell count is separate, and meaningful value means require a value column. It does not follow roads.',
    wave: 2,
    slots: [
      makePointsSlot('facilities', 'Facilities', []),
      makePointsSlot('observations', 'Observations', [
        makeNumberRole('value', 'Value (averaged per cell)', false, undefined)
      ])
    ],
    execution: {
      status: 'available',
      bindingId: 'recipe-straight-line-catchments',
      inputs: {facilities: 'facilities', observations: 'observations'},
      fieldParameters: {'observations.value': 'valueProperty'},
      parameters: {}
    },
    exposedParameters: ['cellSize'],
    sampleId: 'facility-observations'
  },

  {
    id: 'cluster-and-outlier-map',
    question: 'where-significant',
    title: 'Where do similar values cluster, and where are the outliers?',
    name: 'Cluster and outlier map',
    method:
      'Each polygon is linked to the polygons that touch it at a corner or along an edge (queen contiguity, by default), and the links are given equal weight. Each polygon value is compared with the average of its neighbours as a local Moran statistic. The values are then shuffled among the polygons (999 times by default) to see how often chance gives a statistic as extreme, which gives a pseudo p-value. The map colours each polygon as high among highs, low among lows, high among lows or low among highs when its adjusted p-value is below the significance level, and leaves the rest unmarked. Benjamini-Hochberg correction is applied by default for testing many polygons; uncorrected and Bonferroni options are also available. Areas with no neighbour are not tested. For cases and a population, the rate cluster map also smooths small populations.',
    wave: 1,
    slots: [
      makeAreasSlot(
        'areas',
        'Areas with a value',
        [makeNumberRole('value', 'Value', true, ['rate'])],
        ['objects']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'local-moran-statistics',
      inputs: {areas: 'features'},
      fieldParameters: {'areas.value': 'valueProperty'},
      parameters: {neighborhood: 'queen', permutations: 999, seed: 1, significance: 0.05}
    },
    exposedParameters: ['neighborhood', 'significance', 'multipleTesting'],
    sampleId: 'city-district-rates'
  },

  {
    id: 'rate-cluster-map',
    question: 'where-significant',
    title: 'Where do rates cluster?',
    name: 'Rate cluster map',
    method:
      'The rate of each area is cases divided by population. Rates from small populations are shrunk toward the overall rate with empirical Bayes smoothing, areas that touch are linked with equal weight (queen contiguity, by default), and a local Moran statistic of the smoothed rate is computed for each area. The smoothed rates are then shuffled among the areas (999 times by default) to see how often chance gives a statistic as extreme, which gives a pseudo p-value, and by default the p-values are corrected for the number of areas tested with the Benjamini-Hochberg procedure. The map shows each area as a high or low cluster, an outlier or not significant; areas with no neighbour are not tested. In the bundled sample at engine-resolved defaults, the elevated block is recovered in 20 of 20 seeds, but the lone high-low outlier in only 12 of 20 (seed offsets 101 through 424 in steps of 17); smoothing and multiple-testing correction can leave an isolated contrast unmarked.',
    wave: 1,
    slots: [
      makeAreasSlot(
        'areas',
        'Areas with cases and population',
        [
          makeNumberRole('cases', 'Cases', true, ['count']),
          makeNumberRole('population', 'Population', true, ['count'])
        ],
        ['objects']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-rate-cluster-map',
      inputs: {areas: 'areas'},
      fieldParameters: {'areas.cases': 'casesProperty', 'areas.population': 'populationProperty'},
      parameters: {
        criterion: 'queen',
        permutations: 999,
        seed: 1,
        significance: 0.05,
        multipleTesting: 'benjamini-hochberg'
      }
    },
    exposedParameters: ['criterion', 'significance', 'multipleTesting'],
    sampleId: 'city-district-rates'
  },
  {
    id: 'excess-risk',
    question: 'where-significant',
    title: 'Where is risk above expectation?',
    name: 'Excess risk and smoothed rates',
    method:
      'The expected cases of each area are its population times the overall rate, so that the expected cases add up to the observed ones. The raw excess risk is the observed cases over the expected ones, above one where there are more cases than the population explains; it is unstable where the expected cases are few, because one case moves it by a large share. The smoothed excess risk is the empirical Bayes smoothed rate over the overall rate, which shrinks small populations toward one, and the map shows it by default. The rate is also pooled with the areas that touch each area (queen contiguity, by default), adding their cases and their populations before dividing. It does not test whether an excess is more than chance.',
    wave: 2,
    slots: [
      makeAreasSlot(
        'areas',
        'Areas with cases and population',
        [
          makeNumberRole('cases', 'Cases', true, ['count']),
          makeNumberRole('population', 'Population', true, ['count'])
        ],
        ['objects']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-excess-risk',
      inputs: {areas: 'areas'},
      fieldParameters: {'areas.cases': 'casesProperty', 'areas.population': 'populationProperty'},
      parameters: {criterion: 'queen', islandPolicy: 'undefined'}
    },
    exposedParameters: ['criterion', 'rateScale'],
    sampleId: 'city-district-rates'
  },
  {
    id: 'regression-with-diagnostics',
    question: 'how-related',
    title: 'How does one variable relate to others, and is anything spatial left over?',
    name: 'Regression with spatial diagnostics',
    method:
      'A linear model of the response on the chosen predictors is fitted by ordinary least squares. The table gives each coefficient with its standard error, t statistic and p-value, the R squared and adjusted R squared, the corrected Akaike criterion, the condition number of the predictors, and tests of the normality of the residuals (Jarque-Bera) and of constant variance (Breusch-Pagan and Koenker-Bassett). The residual of each area is mapped, and the Moran statistic of the residuals, with a permutation p-value, says whether the residuals cluster in space, which means the model misses a spatial pattern. It does not show whether the relationship itself varies over space (no geographically weighted fit), and it does not fit a spatial lag or spatial error model.',
    wave: 1,
    slots: [
      makeAreasSlot(
        'areas',
        'Areas with several columns',
        [
          makeNumberRole('response', 'Response', true, ['count', 'amount', 'rate', 'ordinal']),
          {
            role: 'predictors',
            label: 'Predictors',
            accepts: NUMBER_COLUMN,
            measurement: ['count', 'amount', 'rate', 'ordinal'],
            required: true,
            multiple: {minimum: 1, maximum: 15}
          }
        ],
        ['objects']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-regression-with-diagnostics',
      inputs: {areas: 'areas'},
      fieldParameters: {
        'areas.response': 'responseProperty',
        'areas.predictors': 'predictorProperties'
      },
      parameters: {neighborhood: 'auto', permutations: 999}
    },
    exposedParameters: ['neighborhood'],
    sampleId: 'district-regression'
  },
  {
    id: 'contiguity-correlogram',
    question: 'how-related',
    title: 'At what neighbourhood scale is there association?',
    name: 'Contiguity correlogram',
    method:
      'For each order of adjacency (neighbours, neighbours of neighbours, and so on) the areas at exactly that order are linked, and the global Moran statistic of the chosen value is computed with a normal-approximation p-value and a permutation p-value. The result is a table of Moran statistics by order, from which a chart of how association fades with the order can be drawn. Each order is tested on its own: no correction is made for testing several orders, so with many orders one significant order can be chance.',
    wave: 2,
    slots: [
      makeAreasSlot(
        'areas',
        'Areas with a value',
        [makeNumberRole('value', 'Value', true, ['rate', 'amount'])],
        ['objects']
      )
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-contiguity-correlogram',
      inputs: {areas: 'areas'},
      fieldParameters: {'areas.value': 'valueProperty'},
      parameters: {
        criterion: 'queen',
        maximumOrder: 6,
        permutations: 999,
        seed: 1,
        significance: 0.05
      }
    },
    exposedParameters: ['maximumOrder', 'criterion', 'significance'],
    sampleId: 'district-smooth-surface'
  },
  {
    id: 'change-of-support',
    question: 'prepare-data',
    title: 'How do I move values from one set of boundaries to another?',
    name: 'Areal interpolation',
    method:
      'Each source polygon is sampled on a regular grid of points. Every point falls in a target polygon or outside all of them, which gives the share of the source polygon in each target. A total (a count or an amount, such as residents) has its covered share transferred to the targets; total preservation requires complete target coverage of the sources; a rate is instead averaged with the shares as weights, never summed. Which of the two applies is chosen from what the value column measures (counts and amounts are totals, rates are averaged), and can be changed. The map shows the target polygons with the transferred value and the fraction of each target covered by sources. The shares come from sampling, so small slivers are approximate, and the transfer assumes the value is spread evenly inside each source polygon.',
    wave: 2,
    slots: [
      makeAreasSlot(
        'sources',
        'Source areas with a value',
        [
          makeNumberRole(
            'value',
            'Value to transfer (a total such as residents is split between targets; a rate is averaged)',
            true,
            ['count', 'amount', 'rate']
          )
        ],
        ['objects']
      ),
      makeAreasSlot('targets', 'Target areas', [])
    ],
    execution: {
      status: 'available',
      bindingId: 'areal-interpolation',
      inputs: {sources: 'sources', targets: 'targets'},
      fieldParameters: {'sources.value': 'valueProperty'},
      parameters: {samplesPerSide: 24}
    },
    exposedParameters: ['variableType'],
    sampleId: 'boundary-pair'
  },
  CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS,
  CURATED_DRAINAGE_NETWORK_ANALYSIS,
  {
    id: 'surface-preparation',
    question: 'prepare-data',
    title: 'Which gaps can be filled, and which remain?',
    name: 'Surface preparation',
    method:
      'Cells without data are filled from the cells around them, the filled cells are recorded so they can be told from measured ones, an optional rank filter removes isolated spikes, and the cells that could not be filled are counted. The map shows the prepared surface and the cells that remain empty.',
    wave: 2,
    slots: [
      {name: 'surface', label: 'Selected raster band with gaps', kind: 'surface', fields: []}
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-surface-preparation',
      inputs: {surface: 'surface'},
      fieldParameters: {},
      parameters: {}
    },
    exposedParameters: ['maximumDistance', 'method', 'filter'],
    sampleId: 'surface-preparation-gaps'
  },
  {
    id: 'zonal-exposure',
    question: 'how-much-per-area',
    title: 'What is in each zone, and who is exposed?',
    name: 'Zonal distribution and exposure',
    method:
      'Summarise a selected hazard band using explicit whole-cell zone ownership. Optionally sum population counts or densities above a threshold, using declared cell areas for densities. Report observed and total population, exposed share and coverage; missing hazard remains unknown.',
    wave: 2,
    slots: CURATED_ZONAL_EXPOSURE_DESCRIPTOR.inputs.map(input => ({
      name: input.name,
      label: input.name,
      kind: input.kind,
      rowSemantics: input.rowSemantics,
      fields: input.fields.map(({parameter, autoAssign, ...field}) => field)
    })),
    execution: {
      status: 'available',
      bindingId: 'curated-zonal-exposure',
      inputs: {surface: 'surface', zones: 'zones', cells: 'cells'},
      fieldParameters: Object.fromEntries(
        CURATED_ZONAL_EXPOSURE_DESCRIPTOR.inputs.flatMap(input =>
          input.fields.map(field => [`${input.name}.${field.role}`, field.parameter])
        )
      ),
      parameters: {}
    },
    exposedParameters: ['probabilities', 'weightMeaning'],
    sampleId: 'zonal-exposure-counts'
  },
  {
    id: 'regional-raster-change',
    question: 'what-changed',
    title: 'Where did it change?',
    name: 'Regional raster change',
    wave: 2,
    ...REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE
  },
  {
    id: 'landscape-metrics',
    question: 'shape-of-the-surface',
    title: 'How fragmented is it, and how isolated are the patches?',
    name: 'Landscape metrics',
    method:
      'Neighbouring cells of the same class are joined into patches, and the patch areas, edges and adjacency between classes are measured. These are summarised for each class and for the whole landscape, and the distance of each patch to the nearest patch of its class gives its isolation. The tables show the metrics by class and by patch.',
    wave: 2,
    slots: [makeSurfaceSlot('landcover', 'Selected land-cover raster band', [])],
    execution: {
      status: 'available',
      bindingId: 'curated-landscape-metrics',
      inputs: {landcover: 'landcover'},
      fieldParameters: {},
      parameters: {}
    },
    exposedParameters: ['connectivity', 'edgeDepth', 'searchRadius', 'excludeClass'],
    sampleId: 'landscape-metrics'
  },
  {
    // The id is kept for stability. It was named for the design's drive-time catchments, but the
    // binding is `network-accessibility`: one demand total per facility within a maximum cost, in
    // metres of street length unless a cost property is mapped, with no time bands and no polygons.
    id: 'drive-time-catchments',
    question: 'what-is-near-or-reachable',
    title: 'How much demand can each facility reach within a travel cost?',
    name: 'Demand within reach of each facility',
    method:
      'The street lines are joined into a network at shared vertices and each facility and each demand point is snapped to the nearest street within the snap distance. For each facility the least network cost to every demand point is found by Dijkstra search, in metres of street length by default or in the cost taken from a street property when the cost mode is changed and the cost columns are mapped. Each demand point inside the maximum cost (1,000 cost units by default: metres for street length) adds its weight to the facility total, or a decayed share of it if a decay is chosen. The map shows each facility with its reachable demand. It gives one total per facility, not bands of travel time or catchment polygons.',
    wave: 1,
    slots: [
      {
        name: 'network',
        label: 'Street network',
        kind: 'network',
        fields: [
          {
            role: 'forward-cost',
            label: 'Cost along the street (needed for a cost property)',
            accepts: NUMBER_COLUMN,
            measurement: ['amount'],
            required: {whenParameter: 'costMode', equals: ['property']}
          },
          {
            role: 'reverse-cost',
            label: 'Cost against the street (optional)',
            accepts: NUMBER_COLUMN,
            measurement: ['amount'],
            required: false
          },
          {
            role: 'one-way',
            label: 'One-way restriction (optional)',
            accepts: ['category', 'text', 'boolean', 'integer'],
            measurement: ['nominal'],
            required: false
          }
        ]
      },
      makePointsSlot('facilities', 'Facilities', []),
      makePointsSlot('demand', 'Demand points', [
        makeNumberRole('weight', 'Demand (optional)', false, ['count', 'amount'])
      ])
    ],
    execution: {
      status: 'available',
      bindingId: 'network-accessibility',
      inputs: {network: 'lines', facilities: 'origins', demand: 'opportunities'},
      fieldParameters: {
        'network.forward-cost': 'forwardCostProperty',
        'network.reverse-cost': 'reverseCostProperty',
        'network.one-way': 'onewayProperty',
        'demand.weight': 'weightProperty'
      },
      parameters: {costMode: 'length', decay: 'cumulative'}
    },
    exposedParameters: ['costMode', 'maximumCost', 'decay'],
    sampleId: 'street-grid-facilities'
  },

  CURATED_NETWORK_DISRUPTION_ANALYSIS,
  {
    id: 'fleet-dwell',
    question: 'how-it-moves',
    title: 'Where do vehicles stop, and for how long?',
    name: 'Fleet dwell',
    method:
      'Stops are found in each track from its speed and time, each stop is assigned to the zone that contains it, and the stops are grouped by zone. The map shows each zone with its number of stops and the time spent.',
    wave: 2,
    slots: [
      {
        name: 'tracks',
        label: 'Tracks',
        kind: 'trajectories',
        fields: [
          {
            role: 'track',
            label: 'Track identifier (or feature ID)',
            accepts: ['number', 'category', 'text'],
            measurement: ['identifier', 'nominal'],
            required: false
          },
          {
            role: 'time',
            label: 'Timestamp (or embedded vertex time)',
            accepts: ['number', 'time', 'text'],
            required: false
          },
          {
            role: 'longitude',
            label: 'Native point horizontal coordinate',
            accepts: ['number'],
            required: false
          },
          {
            role: 'latitude',
            label: 'Native point vertical coordinate',
            accepts: ['number'],
            required: false
          }
        ]
      },
      makeAreasSlot('zones', 'Zones', [])
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-fleet-dwell',
      inputs: {tracks: 'tracks', zones: 'zones'},
      fieldParameters: {
        'tracks.track': 'trackProperty',
        'tracks.time': 'timeProperty',
        'tracks.longitude': 'longitudeProperty',
        'tracks.latitude': 'latitudeProperty'
      },
      parameters: {}
    },
    exposedParameters: ['stopSpeedThreshold', 'stopMinimumDuration', 'timestampEncoding'],
    sampleId: 'fleet-dwell'
  },
  {
    id: 'track-quality',
    question: 'how-it-moves',
    title: 'Which observations need checking?',
    name: 'Track quality',
    method:
      'For each observation, the speed of the arriving step and the elapsed time gap are checked against explicit limits. All raw observations remain on the map, colored by whether they need checking, with independent flag bits and per-track counts, lengths, durations and maximum speeds. Turning angles and acceleration are not computed.',
    wave: 2,
    slots: [
      {
        name: 'tracks',
        label: 'Tracks',
        kind: 'trajectories',
        fields: [
          {
            role: 'track',
            label: 'Track identifier',
            accepts: ['number', 'category', 'text'],
            measurement: ['identifier', 'nominal'],
            required: true
          },
          {role: 'time', label: 'Timestamp', accepts: ['number', 'time', 'text'], required: true},
          {
            role: 'longitude',
            label: 'Native point horizontal coordinate',
            accepts: ['number'],
            required: false
          },
          {
            role: 'latitude',
            label: 'Native point vertical coordinate',
            accepts: ['number'],
            required: false
          }
        ]
      }
    ],
    execution: {
      status: 'available',
      bindingId: 'curated-track-quality',
      inputs: {tracks: 'tracks'},
      fieldParameters: {
        'tracks.track': 'trackProperty',
        'tracks.time': 'timeProperty',
        'tracks.longitude': 'longitudeProperty',
        'tracks.latitude': 'latitudeProperty'
      },
      parameters: {}
    },
    exposedParameters: ['maximumSpeed', 'maximumGap', 'speedOutlierMode', 'timestampEncoding'],
    sampleId: 'track-quality'
  },
  CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
  CURATED_DRIVE_TIME_BANDS_ANALYSIS
];

/** Returns one curated analysis by id, or undefined when no analysis has it. */
export function getKeplerCuratedAnalysis(id: string): KeplerCuratedAnalysis | undefined {
  return KEPLER_CURATED_ANALYSES.find(analysis => analysis.id === id);
}

/** Filters for `listKeplerCuratedAnalyses`; omitted members match everything. */
export type KeplerCuratedAnalysisFilter = {
  question?: GPUAnalysisQuestion;
  wave?: 1 | 2;
  status?: 'available' | 'needs-binding';
};

/** Lists curated analyses, in table order, that match every given filter member. */
export function listKeplerCuratedAnalyses(
  filter: KeplerCuratedAnalysisFilter = {}
): readonly KeplerCuratedAnalysis[] {
  return KEPLER_CURATED_ANALYSES.filter(
    analysis =>
      (filter.question === undefined || analysis.question === filter.question) &&
      (filter.wave === undefined || analysis.wave === filter.wave) &&
      (filter.status === undefined || analysis.execution.status === filter.status)
  );
}
