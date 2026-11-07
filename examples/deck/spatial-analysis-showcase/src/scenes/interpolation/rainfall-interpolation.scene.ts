// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {getRainfallSurfaceStyle, RAINFALL_VARIABLES} from './b7-rainfall-style';
import type {RainfallOptions} from './rainfall-interpolation.compute';

const RAMP_OPTIONS = [
  {value: 'viridis', label: 'Viridis'},
  {value: 'cividis', label: 'Cividis (color-blind optimised)'},
  {value: 'magma', label: 'Magma'},
  {value: 'inferno', label: 'Inferno'}
] as const;

/** Chapter `interpolation`, scene 1: Hurricane Helene rain gauges turned into a surface. */
export default defineScene<RainfallOptions>({
  id: 'rainfall-interpolation',
  title: 'Hurricane Helene rainfall: interpolated depth between gauges',
  chapter: 'interpolation',
  order: 1,
  summary:
    'GPU inverse-distance weighting and ordinary kriging interpolate 6,135 NOAA gauges into rainfall and uncertainty rasters. Held-out errors assess prediction, but sparse coverage and omitted elevation limit the surface.',
  contributors: [
    'GPUInverseDistanceWeighting',
    'GPUKriging',
    'GPUVariogram',
    'GPUVariogramModelFit',
    'addInterpolationDiagnosticsRecipe',
    'GPUFocalStatistics',
    'GPUSpatialCrossValidation'
  ],
  datasets: [{id: 'ghcn-stations', role: 'daily gauge readings'}],
  initialView: {longitude: -83, latitude: 32.3, zoom: 5.2},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Helene rainfall interpolation',
      subtitle: 'Gauge-derived depth and uncertainty'
    },
    scaleBar: {units: 'metric'},
    credit: 'NOAA Global Historical Climatology Network Daily',
    caveat: 'Interpolation omits elevation and does not measure rainfall between gauges.'
  },

  options: [
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable',
      group: 'Data',
      apply: 'param',
      default: 'prcp',
      help: 'Which GHCN-Daily reading is interpolated. Switching rewrites the value buffer; nothing is recompiled. Gauges without a reading (NaN) are skipped by every contributor.',
      options: [
        {
          value: 'prcp',
          label: RAINFALL_VARIABLES.prcp.label,
          help: 'Landfall day: 6,135 gauges.'
        },
        {
          value: 'prcpPrevDay',
          label: RAINFALL_VARIABLES.prcpPrevDay.label,
          help: 'The day before landfall: 5,759 gauges. Rain-band arrival ahead of the eye.'
        },
        {
          value: 'tmax',
          label: RAINFALL_VARIABLES.tmax.label,
          help: 'Only 1,100 stations report temperature: a sparse network, so the uncertainty map matters more.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'validationSplit',
      label: 'Validation design',
      group: 'Data',
      apply: 'param',
      default: 'random',
      help: 'Random holds out individual gauges. Spatial blocks keep every gauge in a 250 km tile together, then score one of five GPU-assigned folds. The block assignment is deterministic for the selected seed.',
      options: [
        {value: 'random', label: 'Random gauge holdout'},
        {value: 'spatial', label: 'Spatially blocked (one of five folds)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'holdOut',
      label: 'Hold out gauges for validation',
      group: 'Data',
      apply: 'param',
      default: false,
      help: 'Removes gauges from the interpolation and variogram, draws them with a ring, and scores both methods there. Choose the validation design below.'
    },
    {
      kind: 'button',
      id: 'reseed',
      label: 'Draw a new hold-out split',
      group: 'Data',
      help: 'Picks a new deterministic assignment. Random mode changes gauge membership; spatial mode changes which 250 km blocks belong to the scored fold.'
    },
    {
      kind: 'select',
      id: 'surface',
      label: 'Surface',
      group: 'Surface',
      apply: 'param',
      default: 'idw',
      help: 'Both interpolators run in one compiled graph; a parameter word picks which result is drawn (and fed to the focal statistics).',
      options: [
        {
          value: 'idw',
          label: 'Inverse distance weighting',
          help: 'Weighted average of nearby gauges, weight 1 / d^p.'
        },
        {
          value: 'kriging',
          label: 'Ordinary kriging',
          help: 'Best linear unbiased prediction under the fitted variogram.'
        },
        {
          value: 'difference',
          label: 'Kriging minus IDW',
          help: 'Where the two methods disagree (diverging ramp, zero is white).'
        },
        {
          value: 'error',
          label: 'Kriging standard error',
          help: 'Square root of the kriging variance: how uncertain the prediction is. It depends on gauge geometry and the variogram, not on the observed values.'
        },
        {
          value: 'support',
          label: 'Gauges used per cell (IDW)',
          help: 'How many gauges contribute to each cell after the radius and nearest-k limits. Cells with fewer than the minimum are no data.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'searchRadiusKm',
      label: 'Search radius',
      group: 'Surface',
      apply: 'param',
      min: 20,
      max: 500,
      step: 10,
      default: 120,
      unit: 'km',
      help: 'Gauges farther than this are ignored by both methods. Cells with too few gauges inside the radius are no data and stay transparent, which is how the map avoids inventing rain over the ocean. Distances are measured in an azimuthal equidistant frame about 83 W, 32 N: exact from that centre and within 0.65% anywhere in the gauge box.'
    },
    {
      kind: 'slider',
      id: 'minimumStations',
      label: 'Minimum gauges per cell',
      group: 'Surface',
      apply: 'param',
      min: 1,
      max: 8,
      step: 1,
      default: 3,
      help: 'A cell with fewer contributing gauges is no data. Kriging always needs at least 3 (a smaller value is raised to 3 for kriging only).'
    },
    {
      kind: 'slider',
      id: 'power',
      label: 'Distance power p',
      group: 'Inverse distance weighting',
      apply: 'param',
      min: 0,
      max: 6,
      step: 0.5,
      default: 2,
      help: 'Weight is 1 / d^p. p = 0 is a plain average of the neighbours (flat plateaus), p = 2 is the common default, large p snaps each cell to its nearest gauge and leaves bullseyes around every one.',
      disabledWhen: state => state.surface === 'kriging' || state.surface === 'error'
    },
    {
      kind: 'slider',
      id: 'neighborCount',
      label: 'Nearest gauges k',
      group: 'Inverse distance weighting',
      apply: 'param',
      min: 0,
      max: 32,
      step: 1,
      default: 12,
      format: value => (value === 0 ? 'all in radius' : String(value)),
      help: 'Keeps only the k nearest gauges inside the radius (compile-time capacity 32). 0 uses every gauge in the radius, which smooths away small features in dense networks.'
    },
    {
      kind: 'select',
      id: 'variogramModel',
      label: 'Variogram model',
      group: 'Variogram',
      apply: 'param',
      default: 'spherical',
      help: 'CPU and GPU fit the same Float32-rounded omnidirectional bins; the CPU fit continues to drive the kriging parameters. Changing the model never recompiles.',
      options: [
        {value: 'spherical', label: 'Spherical', help: 'Reaches the sill at a finite range.'},
        {
          value: 'exponential',
          label: 'Exponential',
          help: 'Rises faster, approaches the sill asymptotically (95% at the range).'
        },
        {
          value: 'gaussian',
          label: 'Gaussian',
          help: 'Very smooth near zero; assumes an extremely continuous field. Can be unstable with a small nugget.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'weighting',
      label: 'Fit weighting',
      group: 'Variogram',
      apply: 'param',
      default: 'cressie',
      help: 'Least-squares weights of the model fit. Cressie (1985) weights N / γ² and is the gstat default; it trusts short lags with many pairs.',
      options: [
        {value: 'cressie', label: 'Cressie (N / γ²)'},
        {value: 'pairs', label: 'Pair counts (N)'},
        {value: 'none', label: 'Ordinary least squares'}
      ]
    },
    {
      kind: 'toggle',
      id: 'robust',
      label: 'Robust (Cressie–Hawkins) estimator',
      group: 'Variogram',
      apply: 'param',
      default: false,
      help: 'Fits the Cressie–Hawkins robust semivariances instead of the Matheron mean of squared differences. Robust bins are less pulled by a few extreme storm totals.'
    },
    {
      kind: 'slider',
      id: 'maximumLagKm',
      label: 'Maximum lag',
      group: 'Variogram',
      apply: 'param',
      min: 100,
      max: 1000,
      step: 50,
      default: 300,
      unit: 'km',
      help: 'Pairs farther apart than this are ignored; the 16 lag bins split [0, max]. A common rule is about half the extent of the study area, but here the rain has a regional trend, so very long lags overshoot the sample variance.'
    },
    {
      kind: 'select',
      id: 'direction',
      label: 'Direction sector (diagnostic)',
      group: 'Variogram',
      apply: 'param',
      default: 'all',
      help: 'GPUVariogram bins pairs into 4 direction sectors of 45°. Pick one to fit its range separately; a different range means the field is anisotropic. GPUKriging itself is isotropic and always uses the omnidirectional fit.',
      options: [
        {value: 'all', label: 'All directions'},
        {value: '0', label: 'Sector 1'},
        {value: '1', label: 'Sector 2'},
        {value: '2', label: 'Sector 3'},
        {value: '3', label: 'Sector 4'}
      ]
    },
    {
      kind: 'slider',
      id: 'azimuth',
      label: 'Sector start angle',
      group: 'Variogram',
      apply: 'param',
      min: 0,
      max: 135,
      step: 15,
      default: 0,
      unit: '°',
      help: 'Rotates the 4 sectors (counter-clockwise from east). Sector 1 covers this angle to +45°.',
      disabledWhen: state => state.direction === 'all'
    },
    {
      kind: 'slider',
      id: 'krigingNeighborCount',
      label: 'Neighbourhood size k',
      group: 'Kriging',
      apply: 'param',
      min: 3,
      max: 16,
      step: 1,
      default: 12,
      help: 'Each cell solves a (k + 1) × (k + 1) kriging system over its k nearest gauges (compile-time capacity 16). Larger k adds distant information and cost; local kriging assumes the variogram holds in the neighbourhood.'
    },
    {
      kind: 'select',
      id: 'focalStatistic',
      label: 'Window statistic',
      group: 'Focal statistics',
      apply: 'param',
      default: 'mean',
      help: 'GPUFocalStatistics computes every statistic over a moving window of cells; this chooses which output buffer is drawn. It only applies when the window radius is above 0.',
      options: [
        {value: 'mean', label: 'Mean', help: 'Smooths the surface.'},
        {
          value: 'max',
          label: 'Maximum',
          help: 'Highest value in the window: the peak-rain footprint.'
        },
        {value: 'min', label: 'Minimum'},
        {
          value: 'range',
          label: 'Range (max − min)',
          help: 'Local contrast: where the surface changes quickly.'
        },
        {
          value: 'standardDeviation',
          label: 'Standard deviation',
          help: 'Local roughness of the surface.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'focalRadius',
      label: 'Window radius',
      group: 'Focal statistics',
      apply: 'param',
      min: 0,
      max: 8,
      step: 1,
      default: 0,
      format: value => (value === 0 ? 'off' : `${value} cells`),
      help: 'Radius of the moving window in raster cells (compile-time cap 8). The raster follows the camera, so a cell is about 5 km at full view and 1 km zoomed in.'
    },
    {
      kind: 'select',
      id: 'focalShape',
      label: 'Window shape',
      group: 'Focal statistics',
      apply: 'param',
      default: 'circle',
      disabledWhen: state => state.focalRadius === 0,
      help: 'A circle treats all directions alike; a square reaches farther along the diagonals.',
      options: [
        {value: 'circle', label: 'Circle'},
        {value: 'square', label: 'Square'}
      ]
    },
    {
      kind: 'slider',
      id: 'focalMinimumCount',
      label: 'Minimum valid cells',
      group: 'Focal statistics',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 1,
      disabledWhen: state => state.focalRadius === 0,
      help: 'A cell whose window holds fewer valid (non no-data) cells becomes no data. Raise it to trim ragged edges.'
    },
    {
      kind: 'toggle',
      id: 'keepHoles',
      label: 'Keep no-data holes',
      group: 'Focal statistics',
      apply: 'param',
      default: true,
      disabledWhen: state => state.focalRadius === 0,
      help: 'On: a no-data centre cell stays no data. Off: windows fill holes and grow the surface outward by the window radius (ArcGIS default).'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the interpolation and variogram graphs',
      group: 'Compare',
      help: 'Runs both compiled graphs outside the frame and reports GPU time per run.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      help: 'Used for the interpolated value and for the gauge dots, so they can be compared. Spread and error maps use fixed ramps.',
      options: RAMP_OPTIONS
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Surface opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Lower it to read the basemap under the surface.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Show gauges',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Dots use the same ramp as the surface. A ringed dot is a held-out gauge.'
    }
  ],

  readouts: [
    {id: 'stations', label: 'Gauges', help: 'Gauges in the dataset (NaN readings are skipped).'},
    {id: 'training', label: 'Gauges used', help: 'Gauges with a reading that are not held out.'},
    {id: 'heldOut', label: 'Held out', help: 'Finite gauges removed for validation.'},
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Width × height of one raster cell, in meters of the azimuthal equidistant analysis frame about 83 W, 32 N (within 0.65% of ground distance across the gauge box). Follows the camera: the extent buffer is rewritten each frame.'
    },
    {
      id: 'variogramModel',
      label: 'Fitted variogram',
      help: 'Nugget (noise at zero distance), partial sill (variance that is spatially structured) and effective range.'
    },
    {
      id: 'empirical',
      label: 'Empirical γ(h)',
      help: 'GPU semivariance per lag, omnidirectional (bars rise with distance). Read it left to right, from short to long lag.'
    },
    {
      id: 'fitted',
      label: 'Fitted model γ(h)',
      help: 'The fitted curve at the same lags and the same vertical scale.'
    },
    {id: 'lagAxis', label: 'Lags'},
    {
      id: 'variogramChart',
      label: 'Spatial correlation evidence',
      kind: 'chart',
      help: 'Actual omnidirectional bins and CPU/GPU fits from the same Float32 inputs, with the sample variance reference.'
    },
    {
      id: 'fitComparison',
      label: 'CPU / GPU fit check',
      help: 'Parameter differences compare both backends on the same selected variable, robust estimator, bins, weights and model. GPU status 3 is only a boundary or budget candidate; statuses 1 and 2 withhold the fit.'
    },
    {
      id: 'variogramInputBins',
      label: 'Exact empirical fit bins',
      help: 'One row per lag: Float32 distance in meters, selected Float32 semivariance, and pair count. These values are the inputs shared by CPU and GPU fitting.'
    },
    {
      id: 'sampleVariance',
      label: 'Sample variance',
      help: 'Variance of the gauge values: the sill a stationary field should approach.'
    },
    {
      id: 'krigingParameters',
      label: 'Kriging buffer parameters',
      help: 'Exact float32 nugget, partial sill and range passed to GPUKriging, plus the sample-variance scale and the numeric floors applied during parameter packing.'
    },
    {id: 'sectorRange', label: 'Direction sector'},
    {
      id: 'validationIdw',
      label: 'IDW at held-out gauges',
      help: 'Root-mean-square error, mean absolute error and bias of the GPU-generated IDW surface, sampled at held-out gauges in the current map extent.'
    },
    {id: 'validationKriging', label: 'Kriging at held-out gauges'},
    {id: 'validationVerdict', label: 'Verdict'},
    {id: 'validationDesign', label: 'Validation design in use'},
    {
      id: 'validationSupport',
      label: 'Scored / unsupported',
      help: 'Counts held-out gauges inside the current raster extent, then shows how many each method could score. Unsupported means a no-data prediction at a visible held-out gauge; gauges outside the current view are not counted as support failures.'
    },
    {
      id: 'validationChart',
      label: 'Validation comparison',
      kind: 'chart',
      help: 'Paired error metrics from the same held-out gauges in the current map extent.'
    },
    {
      id: 'leaveOneOutStatus',
      label: 'Conditional LOO status',
      help: 'The leading counts are valid gauge predictions and standardized gauge residuals with positive prediction variance. Kriging invalid/uncertain counts refer to output raster cells; the summary invalid count refers to the LOO metric summary stage. Leave-one-out diagnostics fit the variogram on the full selected gauge set first, so they are conditional diagnostics, not spatial cross-validation or calibrated deployment error.'
    },
    {id: 'leaveOneOutMetrics', label: 'Conditional LOO errors'},
    {
      id: 'leaveOneOutChart',
      label: 'Observed versus LOO prediction',
      kind: 'chart',
      help: 'The chart compares observed and predicted rainfall in the same units. Standardized residuals are available only when prediction variance is positive.'
    },
    {
      id: 'leaveOneOutValues',
      label: 'All-gauge LOO numeric table',
      help: 'One row per source station ID with observed value, LOO prediction, prediction variance, residual and standardized residual. Standardized values are NaN unless prediction variance is positive.'
    },
    {id: 'graphTime', label: 'Interpolation graph'},
    {id: 'variogramTime', label: 'Variogram graph'}
  ],

  legends: state => {
    const style = getRainfallSurfaceStyle(state);
    const meta = RAINFALL_VARIABLES[state.variable];
    const entries = [
      {
        kind: 'ramp' as const,
        id: 'surface',
        title: style.title,
        ramp: style.ramp,
        unit: style.unit,
        extent: style.range ?? ('gpu' as const),
        format: (value: number) => (Math.abs(value) < 10 ? value.toFixed(1) : value.toFixed(0))
      }
    ];
    if (state.showStations && (style.ramp !== state.ramp || state.surface === 'error')) {
      entries.push({
        kind: 'ramp' as const,
        id: 'stations',
        title: `Gauges: ${meta.short.toLowerCase()}`,
        ramp: state.ramp,
        unit: meta.unit,
        extent: meta.range as unknown as 'gpu',
        format: (value: number) => value.toFixed(0)
      });
    }
    return state.holdOut && state.showStations
      ? [
          ...entries,
          {
            kind: 'categories' as const,
            title: 'Gauges',
            entries: [
              {color: [160, 160, 160, 255] as const, label: 'Used by the surface'},
              {color: [255, 255, 255, 255] as const, label: 'Ringed: held out for validation'}
            ],
            note: 'Dots are colored by the gauge reading with the ramp above.'
          }
        ]
      : entries;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {makeGPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUInverseDistanceWeighting, GPUKriging, GPUFocalStatistics, GPUSpatialCrossValidation,
  addInterpolationDiagnosticsRecipe,
  getGPUInverseDistanceWeightingParameterValues, getGPUKrigingParameterValues,
  getGPUFocalStatisticsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUVariogram, GPUVariogramModelFit, fitVariogramModel, getGPUVariogramParameterValues}
  from '@luma.gl/graph/gpu-dataframe';

// 1. Empirical variogram on the GPU; CPU and GPU fit the same Float32-rounded omni bins
variogramGraph.add(new GPUVariogram({
  spatialContext: projection.spatialContext,
  positions, values, mask,                  // mask: 0 = held-out gauge
  parameters: variogramParameters.importToGraph(variogramGraph),
  gridSize: [32, 32], lagCount: 16, directionCount: 4,
  output: {semivariances, pairCounts, meanDistances, robustSemivariances, statistics, status: {overflow}}
}));
variogramParameters.write(getGPUVariogramParameterValues({
  bounds: indexBounds, maximumDistance: ${state.maximumLagKm * 1000}
}));
const model = fitVariogramModel(
  {distances, semivariances, pairCounts},
  {model: '${state.variogramModel}', weighting: '${state.weighting}'}
);
fitGraph.add(new GPUVariogramModelFit({
  distances: fitDistances, semivariances: fitSemivariances, pairCounts: fitPairCounts,
  models: ['spherical', 'exponential', 'gaussian'], weighting: '${state.weighting}',
  rangeSteps: 200, maximumIterations: 60, output: {parameters, diagnostics, status}
}));

// 2. One graph: IDW + kriging + focal statistics over a raster that follows the camera
const grid = makeGPURasterMetadata({width: 384, height: 240});  // placement is per frame
const common = {
  spatialContext: projection.spatialContext, // AEQD about 83 W, 32 N
  positions, values, mask, grid,
  indexGridSize: [64, 64], indexBounds};
graph.add(new GPUInverseDistanceWeighting({...common, maximumNeighborCount: 32,
  parameters: idwParameters.importToGraph(graph), output: {values: idw, counts}}));
graph.add(new GPUKriging({...common, maximumNeighborCount: 16,
  parameters: krigingParameters.importToGraph(graph), output: {values: kriging, variance}}));
graph.add(new GPUFocalStatistics({values: selected, grid,
  maximumRadius: 8, parameters: focalParameters.importToGraph(graph),
  output: {mean, max, range}}));
// Conditional leave-one-out diagnostics; this does not change the mapped surface.
addInterpolationDiagnosticsRecipe(validationGraph, {
  id: 'rainfall-leave-one-out', kriging: {...common,
    parameters: krigingParameters.importToGraph(validationGraph), maximumNeighborCount: 16}
});
const compiled = await graph.compileAsync();  // once

// 3. Every frame: parameter writes, then encode
// The viewport's grid: rows advance north from the bounds' minimum corner
const [minX, minY, maxX, maxY] = viewBounds;
const viewGrid = makeGPURasterMetadata({
  width: 384, height: 240, cellSize: [(maxX - minX) / 384, (maxY - minY) / 240],
  origin: [minX, minY], rowDirection: 'north'
});
idwParameters.write(getGPUInverseDistanceWeightingParameterValues({
  searchRadius: ${state.searchRadiusKm * 1000}, power: ${state.power},
  neighborCount: ${state.neighborCount}, minimumNeighborCount: ${state.minimumStations}
}, viewGrid));
krigingParameters.write(getGPUKrigingParameterValues({
  searchRadius: ${state.searchRadiusKm * 1000},
  neighborCount: ${state.krigingNeighborCount}, variogram: model   // {model, nugget, sill, range}
}, viewGrid));
focalParameters.write(getGPUFocalStatisticsParameterValues({
  radius: ${state.focalRadius}, shape: '${state.focalShape}', minimumCount: ${state.focalMinimumCount}
}));
// A separate graph makes block masks; fit each model using only its training mask.
validationGraph.add(new GPUSpatialCrossValidation({
  positions, blocks, mask, foldCount: 5,
  output: {folds, training, testing, counts, status}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUInverseDistanceWeighting` and `GPUKriging` turn scattered samples into a raster, one invocation per cell over a rebuilt grid index of the samples. IDW averages the nearest gauges with weights 1/d^p. Ordinary kriging solves a small linear system over the k nearest gauges using a variogram `γ(h)` fitted by `GPUVariogram` bins, and also returns the kriging variance. `GPUVariogramModelFit` compares GPU fitting against the CPU fitter on the same Float32-rounded omnidirectional bins. `addInterpolationDiagnosticsRecipe` adds conditional leave-one-out predictions and residual summaries without replacing the mapped surface. `GPUFocalStatistics` then summarises the raster over a moving window. `GPUSpatialCrossValidation` assigns spatial blocks to folds and produces training and test masks; this scene uses those masks for a separate interpolation fit and score.',
    why: 'Rainfall is measured at points but decisions (flood warnings, crop losses, insurance) are about places. How you interpolate decides the answer between gauges, and kriging adds the part IDW cannot: an honest uncertainty map and a variogram that says how far rain correlates.',
    howToRead:
      'Colors show the interpolated value in the legend units; transparent cells have too few gauges inside the radius. Dots are the gauges on the same scale: where a dot disagrees with the surface under it, the surface is smoothing. The kriging standard-error map shows model variance from gauge geometry and the fitted variogram. In validation, rings mark withheld gauges and unsupported counts disclose where no prediction could be scored.'
  },

  create: async ctx =>
    (await import('./rainfall-interpolation.compute')).createRainfallInterpolation(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Gauges leave rainfall unmeasured between stations',
      textAlternative:
        'Rain gauges appear over a continuous GPU-interpolated precipitation surface across the Southeast.',
      title: 'Where did Helene’s rain fall between the gauges?',
      body: 'On 26–27 September 2024 Hurricane Helene crossed Florida’s Big Bend and drove record rain into the southern Appalachians. NOAA’s GHCN-Daily network recorded **6,135 gauges** across the Southeast that day, but a gauge only knows its own backyard. A flood manager needs the rain *between* the gauges.\n\nThe colored surface is built on the GPU from those points by **`GPUInverseDistanceWeighting`**, over a raster that follows the camera. Dots are the gauges on the same scale: hover one for its name and reading, hover anywhere else for the interpolated value. Transparent cells are where no gauge is close enough to say anything. **Variable** and **Surface**, below, choose the reading and the method.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2},
      options: {surface: 'idw'},
      callout: {coordinate: [-83.58, 30.12], text: 'Landfall near Perry, FL'},
      controls: ['variable', 'surface'],
      readouts: ['stations'],
      evidence:
        'The map starts from gauge observations; transparent cells explicitly mark places where the selected neighbourhood rule has insufficient support.',
      caveat:
        'A continuous surface is an estimate between instruments, not a direct measurement of rainfall at every pixel.'
    },
    {
      id: 'idw',
      headline: 'Distance power controls smoothing around gauges',
      textAlternative:
        'Inverse-distance rainfall forms broader or narrower surfaces around gauge readings as power changes.',
      title: 'Nearby gauges vote, nearer ones louder',
      body: 'Inverse distance weighting estimates a cell as a weighted average of the gauges around it, `z = Σ wᵢ zᵢ / Σ wᵢ` with `wᵢ = 1 / dᵢᵖ`. Doubling the distance quarters a gauge’s say when **p = 2**.\n\nThe Blue Ridge near Asheville is the story: Connestee Falls, NC recorded 343 mm. Slide **Distance power p** to 0 (a plain average, flat plateaus) and to 6 (each cell snaps to its nearest gauge, bullseyes around every dot), and watch **Search radius** and **Nearest gauges k** carve no-data holes and limit the vote. IDW never predicts above the highest or below the lowest gauge, so it can only spread a peak, never place one between gauges.',
      camera: {longitude: -82.8, latitude: 35.4, zoom: 7.6, transitionMs: 2200},
      options: {surface: 'idw', power: 2, searchRadiusKm: 120, neighborCount: 12},
      callout: {coordinate: [-82.7364, 35.141], text: 'Connestee Falls: 343 mm'},
      controls: ['power', 'searchRadiusKm', 'neighborCount']
    },
    {
      id: 'variogram',
      headline: 'Rainfall similarity declines with gauge separation',
      textAlternative:
        'Empirical and fitted semivariogram bars compare rainfall difference against station-pair distance.',
      title: 'How fast does rain stop resembling its neighbour?',
      body: 'IDW assumes one fixed distance decay everywhere. The **semivariogram** measures how gauge differences change with separation, `γ(h) = ½ · mean (zᵢ − zⱼ)²`. **`GPUVariogram`** computes the bins in one deterministic GPU pass. The omnidirectional bins are reduced from its four directional sectors and read back for fitting.\n\n**Which backend fits the same bins?** The CPU fitter and **`GPUVariogramModelFit`** both receive the same selected variable, robust or Matheron estimator, Float32-rounded lag distances and semivariances, pair counts, weighting and model. The chart’s **Show values** table exposes those actual bins. Compare nugget, sill and range, then switch **Variogram model**, **Fit weighting**, or the robust estimator. The GPU search uses 200 range steps and 60 refinement iterations over the CPU fitter’s range domain. A boundary/budget result is only a candidate range; it is not identified. The AIC shown is descriptive for this exact bin residual, not a spatial likelihood and not comparable across different bins. Stretch **Maximum lag** past about 400 km and the points overshoot sample variance: Helene’s regional trend violates stationarity.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2, transitionMs: 2000},
      options: {surface: 'idw', variogramModel: 'spherical', maximumLagKm: 300},
      highlight: {readout: 'variogramModel'},
      controls: ['variable', 'robust', 'variogramModel', 'weighting', 'maximumLagKm'],
      readouts: ['variogramModel', 'fitComparison', 'variogramInputBins', 'variogramChart'],
      evidence:
        'The exact shared GPU-binned inputs and seven-significant-digit CPU/GPU parameters are exposed for independent checking; the chart overlays both model curves with the sample-variance reference.',
      caveat:
        'The omnidirectional fit assumes stationarity and isotropy; Helene’s regional gradient and banding violate both assumptions.'
    },
    {
      id: 'kriging',
      headline: 'Kriging redistributes weights among clustered gauges',
      textAlternative:
        'Ordinary kriging maps rainfall using fitted spatial covariance and nearby gauge geometry.',
      title: 'Kriging lets the variogram choose the weights',
      body: '**`GPUKriging`** is ordinary kriging: each cell uses its nearest gauges and the fitted variogram to choose weights. Its map variance is model-based uncertainty from gauge geometry.\n\n**How close are leave-one-out predictions to observed gauge readings?** `addInterpolationDiagnosticsRecipe` predicts each gauge from its neighbors and returns predictions, prediction variance, residuals and summary errors. The observed-versus-predicted chart compares values in the same units; its table lists both actual values. The all-gauge numeric table also exposes variance, residual and standardized residual, and the first three source IDs are repeated as fixed probes. Standardized residuals are counted only where prediction variance is positive; no Gaussian 95% coverage claim is made. The variogram is fitted on all gauges before these predictions, so this is conditional diagnostics, not leakage-resistant spatial cross-validation or calibrated deployment error. Use the separate held-out and spatial-block controls for those distinct validation designs. The ordinary kriging map and its variance remain the displayed surface.',
      camera: {longitude: -82.8, latitude: 35.4, zoom: 7.3, transitionMs: 2000},
      options: {
        surface: 'kriging',
        holdOut: false,
        krigingNeighborCount: 12,
        searchRadiusKm: 120
      },
      controls: ['surface', 'krigingNeighborCount'],
      readouts: [
        'leaveOneOutStatus',
        'leaveOneOutMetrics',
        'leaveOneOutChart',
        'leaveOneOutValues'
      ],
      evidence:
        'The GPU recipe returns leave-one-out predictions, positive-variance-gated standardized residuals, valid counts, bias, MAE and RMSE; the chart compares actual observations and predictions, with full numeric station rows available.',
      caveat:
        'The variogram is fitted on all gauges before leave-one-out prediction, so these conditional diagnostics are not spatial cross-validation or deployment error calibration. Parameter packing floors nugget and sill at 1e−4 times sample variance and range at 1,000 m; the Kriging buffer parameters readout exposes the exact float32 values used by both the surface and LOO predictions.'
    },
    {
      id: 'blocked-validation',
      headline: 'Blocked holdout asks whether a surface travels into a region',
      textAlternative:
        'A map shows held-out gauges grouped into spatial blocks, with IDW and kriging error metrics beneath it.',
      title: 'Withhold whole regions, then test the prediction',
      body: 'Start with **Random gauge holdout** to score a reproducible tenth of gauges, then switch **Validation design** below to **Spatially blocked**. **`GPUSpatialCrossValidation`** assigns each coarse tile as a unit to a fold; **{{validationDesign}}** states the current block and fold design. The GPU returns training and testing masks; only training gauges feed the variogram and interpolators. Ringed test gauges are scored against the resulting GPU surfaces. **Scored / unsupported** counts held-out gauges with predictions and those with no supported prediction.\n\nThis is a spatial separation stress test, not an unbiased estimate of future error: it scores one fold at a time, and points near tile edges can still be close to training gauges in adjacent tiles. Press **Draw a new hold-out split** to reassign blocks. The random split can look better because nearby gauges remain in training. The station file has point elevation readings but no prediction DEM; neither method corrects the Blue Ridge rainfall maximum for orography.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2, transitionMs: 1800},
      options: {
        surface: 'kriging',
        holdOut: true,
        validationSplit: 'spatial',
        searchRadiusKm: 180
      },
      highlight: {readout: 'validationSupport'},
      controls: ['holdOut', 'validationSplit', 'reseed'],
      readouts: ['validationDesign', 'validationSupport', 'validationVerdict', 'validationChart'],
      evidence:
        'The GPU spatial fold supplies the training and test masks. Both methods are scored only at withheld gauges with supported raster predictions, and missing predictions are counted separately.',
      caveat:
        'This single-fold stress test is not a five-fold average; adjacent tiles may still contain nearby gauges. Elevation is not used because no prediction DEM is supplied.'
    },
    {
      id: 'focal',
      headline: 'Focal windows summarize local rainfall variation',
      textAlternative:
        'A moving-window raster reports local rainfall maxima, ranges or standard deviations.',
      title: 'Summarise the surface with focal statistics, then try your own',
      body: '**`GPUFocalStatistics`** slides a window over the raster and reports its mean, minimum, maximum, range or standard deviation (like ArcGIS `FocalStatistics` and GRASS `r.neighbors`). Here a 3-cell circle (**Window radius**) with the *Maximum* **Window statistic** turns the surface into a peak-rain footprint; *Range* or *Standard deviation* highlight where rain changes abruptly. Switching **Keep no-data holes** off lets the window grow the surface outward.\n\nLimits: the raster is 384 × 240 cells whatever the zoom, so zooming in refines it; interpolation is isotropic here although Helene’s rain was banded along its track (a **Direction sector** in the variogram would show it, but it does not change the map). Try **Variable** *Daily high temperature* (only 1,100 stations), or set **Surface** to the difference map.',
      camera: {longitude: -82.8, latitude: 35.4, zoom: 7.0, transitionMs: 1800},
      options: {
        surface: 'kriging',
        holdOut: false,
        focalStatistic: 'max',
        focalRadius: 3,
        focalShape: 'circle'
      },
      controls: ['focalStatistic', 'focalRadius', 'keepHoles', 'variable', 'surface']
    }
  ]
});
