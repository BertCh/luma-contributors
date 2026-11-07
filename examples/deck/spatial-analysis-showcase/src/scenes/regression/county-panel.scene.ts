// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {ground} from '../../cartography/grounds';
import type {CountyPanelOptions} from './county-panel.compute';

export default defineScene<CountyPanelOptions>({
  id: 'county-panel',
  title: 'Does the model change when counties are followed over time?',
  chapter: 'regression',
  order: 90,
  summary:
    'Compare pooled and unit fixed-effects spatial 2SLS with random- and unit-effects spatial-lag ML and unit-effects spatial-error ML on 21 historical New Jersey county units.',
  contributors: [
    'GPUSpatialPanelRegression',
    'GPUSpatialPanelLagMaximumLikelihood',
    'GPUSpatialPanelErrorMaximumLikelihood'
  ],
  datasets: [
    {id: 'ncovr-new-jersey', role: 'historical county panel and matching common-support polygons'}
  ],
  initialView: {longitude: -74.55, latitude: 40.05, zoom: 7.1},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'New Jersey unemployment panel',
      subtitle: 'Spatial models on historical county units',
      sample: '21 common-support units · 1970 / 1980 / 1990',
      chips: ['Historical']
    },
    scaleBar: {units: 'metric'},
    credit: 'NCOVR; Messner et al. (2000); GeoDa archive',
    caveat: 'Three decennial observations; ecological associations are not causal.'
  },
  options: [
    {
      kind: 'select',
      id: 'period',
      label: 'Decennial observation',
      apply: 'param',
      default: '1970',
      disabledWhen: state => state.estimator.startsWith('cross-'),
      options: [
        {value: '1970', label: '1970'},
        {value: '1980', label: '1980'},
        {value: '1990', label: '1990'}
      ],
      help: 'Choose one of the three observed census-era slices. These are decennial snapshots, not annual observations.'
    },
    {
      kind: 'select',
      id: 'estimator',
      label: 'Estimator',
      apply: 'param',
      default: 'panel-2sls-pooled',
      group: 'Spatial model',
      options: [
        {value: 'panel-2sls-pooled', label: 'Spatial 2SLS · pooled'},
        {value: 'panel-2sls-unit', label: 'Spatial 2SLS · unit effects'},
        {value: 'lag-ml-random', label: 'Spatial lag ML · random effects'},
        {value: 'lag-ml-unit', label: 'Spatial lag ML · unit effects'},
        {value: 'error-ml-unit', label: 'Spatial error ML · unit effects'},
        {value: 'cross-lag-ml-1990', label: '1990 spatial lag ML · cross-section'},
        {value: 'cross-error-ml-1990', label: '1990 spatial error ML · cross-section'}
      ],
      help: 'Switch among estimators with different assumptions. 2SLS instruments the spatially lagged response; lag ML estimates a spatial outcome parameter by Gaussian likelihood; error ML models spatially correlated disturbances. The last two fit 1990 alone and do not estimate panel effects.'
    },
    {
      kind: 'select',
      id: 'map',
      label: 'Mapped evidence',
      apply: 'param',
      default: 'observed',
      group: 'Display',
      options: [
        {value: 'observed', label: 'Observed unemployment'},
        {value: 'residual', label: 'Model residual'}
      ],
      help: 'Cross-sectional estimators always map the 1990 source observation, even if the control previously selected another year. Residuals use that estimator’s own definition and are masked when its fit status is invalid or incomplete.'
    },
    {
      kind: 'select',
      id: 'covariance',
      label: 'Pooled 2SLS covariance',
      apply: 'param',
      default: 'HC0',
      group: 'Inference sensitivity',
      disabledWhen: state => state.estimator !== 'panel-2sls-pooled',
      options: [
        {value: 'HC0', label: 'HC0'},
        {value: 'HC1', label: 'HC1 · n/(n−p)'},
        {value: 'HAC-50', label: 'Gaussian space-time · 50 km'},
        {value: 'HAC-100', label: 'Gaussian space-time · 100 km'}
      ],
      help: 'Sensitivity for the pooled 63-row spatial 2SLS fit only. HAC uses a full dense Gaussian space-time kernel with authored 50/100 km spatial scales and 10-year temporal scale; these are not tuned bandwidths. It changes covariance estimates, never coefficients.'
    }
  ],
  story: [
    {
      id: 'three-snapshots',
      title: 'Begin with three observed snapshots',
      headline: 'The panel contains three decennial rates',
      body: 'What changed in unemployment across these 21 historical county units? The source records decennial census-era values for 1970, 1980 and 1990. The **Decennial observation** control below changes the mapped year.',
      textAlternative:
        'A county map of New Jersey colored by the observed unemployment percentage in one selected decennial year; the 21 historical common-support county units are outlined.',
      evidence:
        'Every county unit remains on the source common-support geography; the fill shows the selected observed unemployment percentage.',
      caveat:
        'These are county units in the archive historical common-support template; the template alone does not imply changed present county identities.',
      controls: ['period'],
      readouts: ['mappedUnits'],
      camera: {longitude: -74.55, latitude: 40.05, zoom: 7.1}
    },
    {
      id: 'pooled-and-within',
      title: 'Pooled and within-unit fits answer different questions',
      headline: 'Unit effects remove stable county differences',
      body: 'The panel 2SLS estimator uses the spatially lagged unemployment outcome as an endogenous term. Pooled mode keeps a common intercept; unit effects remove each county unit’s time-constant level. Use **Estimator** below to compare those specifications.',
      textAlternative:
        'The map retains the selected observed rate or shows model residuals for each county unit, while a status and coefficient readout describes the selected pooled or unit-effects 2SLS fit.',
      evidence:
        'The fitted coefficient table and status come from the selected GPU run; the panel is only 21 units by 3 periods.',
      caveat:
        'These Gaussian working estimates use homoskedastic inference, without entity clustering or serial-correlation correction; they do not identify causal effects.',
      options: {estimator: 'panel-2sls-pooled'},
      controls: ['estimator'],
      readouts: ['fitStatus', 'povertyCoefficient', 'densityCoefficient']
    },
    {
      id: 'lag-versus-error',
      title: 'Spatial dependence can enter different parts of a model',
      headline: 'Lag and error models encode distinct dependence',
      body: 'Spatial-lag ML places dependence in the outcome equation; spatial-error ML places it in the disturbance process. The random-effects lag fit also reports an explicit Hausman comparison status. Choose **Estimator** below to inspect each method.',
      textAlternative:
        'A diverging map of the selected estimator residuals, with invalid or nonconverged county values hidden and the solver status shown beside the map.',
      evidence:
        'Residual fills are read from the selected estimator’s GPU output, and invalid fits are rendered as no data.',
      caveat:
        'The lag random-effects likelihood is a small-panel Gaussian fit. Hausman is informative only when both covariance estimates support a valid comparison.',
      options: {estimator: 'lag-ml-random', map: 'residual'},
      controls: ['estimator', 'map'],
      readouts: ['fitStatus', 'lagOrErrorParameter', 'hausmanStatus', 'panelDiagnostics']
    },
    {
      id: 'pooled-covariance-sensitivity',
      title: 'How much do uncertainty estimates depend on the covariance rule?',
      headline: 'The pooled 2SLS coefficients stay fixed',
      body: 'For the pooled 63-row fit, compare HC0, HC1, and two authored Gaussian space-time HAC kernels. The map and point estimates do not change. HC1 multiplies the finite-sample HC0 covariance by **63/59**; the HAC kernels pair actual county interior anchors across the three decades, using a 10-year temporal scale and either 50 or 100 km spatial scale.',
      textAlternative:
        'A bar chart shows the square roots of the selected GPU covariance diagonal for the intercept, standardized poverty, standardized log density, and spatially lagged outcome. A text readout exposes all 16 covariance entries.',
      evidence:
        'The covariance contributor reads the actual residual buffer from the pooled spatial 2SLS fit. Spatial-lag and spatial-error models, including unit-effects 2SLS, are not passed to this covariance calculation.',
      caveat:
        'HC0 and HC1 do not correct serial or spatial dependence. The HAC scales are transparent sensitivity choices, not selected or validated bandwidths. These standard errors are not automatic significance decisions and none of the coefficients is a causal effect.',
      options: {estimator: 'panel-2sls-pooled', covariance: 'HC0'},
      controls: ['covariance'],
      readouts: [
        'ivCovarianceStatus',
        'ivCovarianceStandardErrors',
        'ivCovarianceMatrix',
        'ivCovarianceChart'
      ]
    },
    {
      id: 'compare-decades',
      title: 'Compare periods without inventing an annual trajectory',
      headline: 'Decennial contrasts do not imply annual change',
      body: 'Move **Decennial observation** below and compare observed rates or a chosen model’s period-specific residuals. Poverty and source log density enter as standardized columns, so the reported slopes are per one standard deviation of each predictor.',
      textAlternative:
        'The same 21 historical county polygons show either observed unemployment or residuals for one of three decades; no interpolated years are drawn.',
      evidence:
        'The map is limited to the three source years; **Mapped units** reports finite mapped values for the selected view.',
      caveat:
        'Poverty fields follow the source census-vintage definitions. Neither a residual map nor a coefficient establishes that poverty or density caused unemployment.',
      options: {period: '1990', map: 'residual'},
      controls: ['period', 'map'],
      readouts: ['mappedUnits', 'fitStatus']
    },
    {
      id: 'cross-sectional-check',
      title: 'A single-year model is a different analysis',
      headline: 'The 1990 cross-section drops all panel effects',
      body: 'Choose **Estimator** below to run the fixed 1990 spatial-lag or spatial-error ML model. Its map and residuals always use 1990, regardless of the last selected panel year. The lag model also computes average direct, indirect and total impacts from its fitted coefficient table and exact weight traces.',
      textAlternative:
        'The map shows 1990 unemployment residuals from a 21-unit cross-sectional model; status guards both the residual map and the impact summary.',
      evidence:
        'The 1990 spatial-lag impacts readout is computed on the GPU and displayed only when solver and convergence statuses are valid.',
      caveat:
        'A 21-unit cross-section is small, has no unit or time effects, and does not support a causal interpretation of the impacts.',
      options: {estimator: 'cross-lag-ml-1990', map: 'residual'},
      controls: ['estimator', 'map'],
      readouts: ['fitStatus', 'crossImpacts', 'povertyCoefficient', 'densityCoefficient']
    }
  ],
  legends: state => [
    state.map === 'observed'
      ? {
          kind: 'ramp',
          title: 'Unemployment rate',
          ramp: 'cividis',
          extent: [0, 15],
          unit: 'percent',
          note: `NCOVR ${state.estimator.startsWith('cross-') ? '1990' : state.period} census-era observation`
        }
      : {
          kind: 'ramp',
          title: 'Estimator residual output',
          ramp: 'diverging',
          extent: [-5, 5],
          unit: 'percentage points',
          midpoint: 0,
          midpointLabel: 'zero',
          note: 'Residual definitions differ by estimator. Symmetric fixed range; values outside are clipped.'
        }
  ],
  readouts: [
    {
      id: 'mappedUnits',
      label: 'Finite mapped units',
      format: 'integer',
      help: 'Count of county units whose GPU map value is finite.'
    },
    {
      id: 'fitStatus',
      label: 'Solver status',
      format: 'text',
      help: 'Status is checked before residual interpretation.'
    },
    {
      id: 'povertyCoefficient',
      label: 'Poverty slope',
      format: 'decimal',
      help: 'First predictor coefficient; standardized poverty percentage points.'
    },
    {
      id: 'densityCoefficient',
      label: 'Log-density slope',
      format: 'decimal',
      help: 'Second predictor coefficient; standardized source log density.'
    },
    {
      id: 'lagOrErrorParameter',
      label: 'Spatial parameter',
      format: 'decimal',
      help: 'Spatial lag or error parameter when reported in the table.'
    },
    {
      id: 'hausmanStatus',
      label: 'Hausman comparison',
      format: 'text',
      help: 'Random-versus-unit lag comparison; invalid covariance differences are reported explicitly.'
    },
    {
      id: 'panelDiagnostics',
      label: 'Pooled LM screen',
      format: 'text',
      help: 'Pooled untransformed LM screen for error and robust lag/error pairs; these are not diagnostics of the fitted unit-error residuals.'
    },
    {
      id: 'crossImpacts',
      label: '1990 spatial lag impacts',
      format: 'text',
      help: 'GPU-computed direct, indirect and total average impacts for the 1990 cross-sectional spatial-lag model; read only when fit and convergence statuses are valid.'
    },
    {
      id: 'ivCovarianceStatus',
      label: 'Pooled 2SLS covariance status',
      format: 'text',
      help: 'A covariance sensitivity for pooled 2SLS only. It does not refit the model or choose a significance threshold.'
    },
    {
      id: 'ivCovarianceStandardErrors',
      label: 'Selected covariance diagonal',
      format: 'text',
      help: 'Square roots of the four diagonal entries in coefficient order: intercept, poverty, log density, and spatially lagged outcome.'
    },
    {
      id: 'ivCovarianceMatrix',
      label: 'Selected 4 × 4 covariance matrix',
      format: 'text',
      help: 'All matrix entries from GPUInstrumentalVariablesCovariance, in intercept, poverty, log density, spatial-lag order. The method selector changes covariance only.'
    },
    {
      id: 'ivCovarianceChart',
      label: 'Selected covariance standard errors',
      kind: 'chart',
      help: 'The bars are square roots of the selected covariance diagonal; they are descriptive uncertainty summaries, not a model-selection rule.'
    }
  ],
  snippet: state =>
    state.estimator.startsWith('cross-')
      ? `// 1990 cross-section only; no panel effects
new ${state.estimator === 'cross-lag-ml-1990' ? 'GPUSpatialLagMaximumLikelihood' : 'GPUSpatialErrorMaximumLikelihood'}({
  maximumPower: 64, traceMode: 'exact', predictorCount: 2,
  response: unemployment1990, predictors: standardizedPovertyAndLogDensity1990,
  weights: queenContiguityRowStandardized
});`
      : `// ${state.estimator}; balanced 21-unit x 3-period NCOVR panel
new ${state.estimator === 'error-ml-unit' ? 'GPUSpatialPanelErrorMaximumLikelihood' : state.estimator.startsWith('lag-ml') ? 'GPUSpatialPanelLagMaximumLikelihood' : 'GPUSpatialPanelRegression'}({
  periodCount: 3, predictorCount: 2, effects: '${state.estimator.includes('unit') ? 'unit' : state.estimator === 'lag-ml-random' ? 'random' : 'pooled'}',
  response: unemployment[1970, 1980, 1990], predictors: standardizedPovertyAndLogDensity,
  weights: queenContiguityRowStandardized
});`,
  about: {
    what: 'A compact historical panel model comparison using unemployment as the outcome and poverty plus log population density as time-varying predictors.',
    why: 'Pooled, unit-effects, random-effects, spatial-lag and spatial-error specifications encode different ways to handle repeated county observations and spatial dependence.',
    howToRead:
      'Observed rates are percentages. The residual map uses the selected contributor output and a diverging zero-centered scale. Read solver status before interpreting coefficients or residual geography. The source archive has only 21 New Jersey common-support units and three decennial periods; no time effects, annual dynamics or cluster-robust inference are claimed. Cross-sectional spatial impacts are model summaries, not causal effects.'
  },
  create: async ctx => (await import('./county-panel.compute')).createCountyPanel(ctx)
});
