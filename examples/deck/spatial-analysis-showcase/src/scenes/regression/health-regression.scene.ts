// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {HealthRegressionOptions} from './health-regression.compute';
import {VALIDATION_FOLD_COLORS} from './health-regression.constants';

const CLUSTER_RGBA = {
  hh: [214, 69, 65, 235],
  lh: [138, 176, 232, 235],
  ll: [49, 104, 190, 235],
  hl: [240, 160, 60, 235],
  none: [150, 150, 158, 120]
} as const;

const OUTCOME_UNITS: Record<HealthRegressionOptions['outcome'], string> = {
  diabetes: '% of adults with diabetes',
  obesity: '% of adults with obesity',
  asthma: '% of adults with asthma',
  depression: '% of adults with depression',
  heartDisease: '% of adults with heart disease',
  smoking: '% of adults who smoke'
};

/** Health regression on Chicago tracts: OLS, diagnostics, spatial lag and spatial error models. */
export default defineScene<HealthRegressionOptions>({
  id: 'health-regression',
  title: 'Chicago diabetes residuals across tract regression models',
  chapter: 'regression',
  order: 1,
  summary:
    'OLS, spatial diagnostics, lag and error models use CDC PLACES and ACS data for Chicago tracts to map prevalence residuals and local clusters; the ecological, model-based inputs do not support causal or individual inference.',
  contributors: [
    'GPUOrdinaryLeastSquares',
    'GPUSpatialRegressionDiagnostics',
    'GPUSpatialTwoStageLeastSquares',
    'GPUSpatialErrorGM',
    'GPUSpatialCrossValidation',
    'GPUSpatialCrossValidatedOLS',
    'addSpatialRegressionRecipe'
  ],
  datasets: [
    {id: 'chicago-tracts', role: 'tract polygons with CDC PLACES outcomes, ACS and SVI covariates'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.9},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Chicago diabetes residuals',
      subtitle: 'Tract models using CDC PLACES and ACS covariates'
    },
    scaleBar: {units: 'metric'},
    credit: 'CDC PLACES; US Census Bureau ACS',
    caveat: 'Ecological associations from model-based tract estimates; not causal effects.'
  },

  options: [
    {
      kind: 'select',
      id: 'outcome',
      label: 'Health outcome (response)',
      group: 'Model',
      apply: 'param',
      default: 'diabetes',
      help: 'CDC PLACES model-based prevalence per tract (percent of adults). Changing it rewrites the response buffer; nothing is recompiled.',
      options: [
        {value: 'diabetes', label: 'Diabetes'},
        {value: 'obesity', label: 'Obesity'},
        {value: 'asthma', label: 'Current asthma'},
        {value: 'depression', label: 'Depression'},
        {value: 'heartDisease', label: 'Coronary heart disease'},
        {value: 'smoking', label: 'Current smoking'}
      ]
    },
    {
      kind: 'select',
      id: 'model',
      label: 'Predictors',
      group: 'Model',
      apply: 'compile',
      default: 'economic',
      help: 'Which covariates enter the model (all standardized to z-scores, so a coefficient is points of prevalence per one standard deviation). The predictor count is a compile-time property of every contributor, so each set compiles its graphs the first time you choose it.',
      options: [
        {value: 'income', label: 'Income only (1)', help: 'ln income per capita.'},
        {
          value: 'economic',
          label: 'Economic (4)',
          help: 'Income, poverty, uninsured and unemployment rates.'
        },
        {
          value: 'full',
          label: 'Full (8)',
          help: 'Adds age 65+, disability, no-vehicle households, Black and Hispanic shares. Predictors overlap heavily, so coefficients become unstable: try the ridge penalty.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'ridge',
      label: 'Ridge penalty λ (log₁₀)',
      group: 'Model',
      apply: 'param',
      min: -2,
      max: 3,
      step: 0.1,
      default: -2,
      format: value => (value <= -2 ? 'off' : (10 ** value).toPrecision(2)),
      help: 'Adds λ to the diagonal of XᵀX (intercept unpenalized). Coefficients shrink toward zero and standard errors tighten; watch the full model stabilize. A per-frame parameter write.'
    },
    {
      kind: 'select',
      id: 'weights',
      label: 'Spatial weights',
      group: 'Neighbours',
      apply: 'compile',
      default: 'queen',
      help: 'Who counts as a neighbour in the diagnostics, the residual map and the lag and error models. Weights are always row-standardized. Contiguity is symmetric; k nearest neighbours is directed.',
      options: [
        {value: 'queen', label: 'Queen contiguity', help: 'Tracts sharing a boundary point.'},
        {value: 'rook', label: 'Rook contiguity', help: 'Tracts sharing a boundary edge.'},
        {value: 'knn4', label: '4 nearest centroids'},
        {value: 'knn8', label: '8 nearest centroids'},
        {value: 'knn12', label: '12 nearest centroids'}
      ]
    },
    {
      kind: 'select',
      id: 'instruments',
      label: 'Lag-model instruments',
      group: 'Neighbours',
      apply: 'compile',
      default: '1',
      help: 'Instruments of the spatially lagged response W·y in two-stage least squares (spreg w_lags): [1, X, WX], or also W²X.',
      options: [
        {value: '1', label: 'X and WX (w_lags = 1)'},
        {value: '2', label: 'X, WX and W²X (w_lags = 2)'}
      ]
    },
    {
      kind: 'slider',
      id: 'significance',
      label: 'Residual cluster significance',
      group: 'Diagnostics',
      apply: 'param',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      default: 0.05,
      format: value => `p ≤ ${value.toFixed(3)}`,
      help: 'Level of the analytic local Moran test on the OLS residuals (used by the residual clusters map).'
    },
    {
      kind: 'select',
      id: 'validationDesign',
      label: 'Validation split',
      group: 'Blocked validation',
      apply: 'compile',
      default: 'spatial',
      help: 'Compare seeded 3 km spatial block holdouts with seeded random row holdouts. Random rows are a contrast, not a substitute for spatial transfer assessment.',
      options: [
        {value: 'spatial', label: '3 km spatial blocks'},
        {value: 'random', label: 'Random rows'}
      ]
    },
    {
      kind: 'select',
      id: 'validationBuffer',
      label: 'Training exclusion',
      group: 'Blocked validation',
      apply: 'compile',
      default: '500',
      help: 'Exclude training tracts within this projected distance of test tracts. The split seed remains 31.',
      options: [
        {value: '0', label: '0 m'},
        {value: '500', label: '500 m'},
        {value: '1000', label: '1,000 m'}
      ]
    },
    {
      kind: 'select',
      id: 'map',
      label: 'Map',
      group: 'Display',
      apply: 'param',
      default: 'residuals',
      help: 'What the tract fill shows. Switching only changes which result buffer the fill layer binds.',
      options: [
        {value: 'residuals', label: 'OLS residuals'},
        {value: 'fitted', label: 'OLS fitted value'},
        {value: 'observed', label: 'Observed value'},
        {value: 'clusters', label: 'OLS residual clusters (local Moran)'},
        {value: 'validationResiduals', label: 'Blocked out-of-fold residuals'},
        {value: 'validationFolds', label: 'Spatial validation fold'},
        {value: 'lag', label: 'Spatial lag model residuals'},
        {value: 'error', label: 'Spatial error model residuals'}
      ]
    }
  ],

  readouts: [
    {id: 'localInference', label: 'Local test threshold'},
    {
      id: 'rows',
      label: 'Tracts in the model',
      help: 'Tracts with every variable present, population of at least 300, in the largest contiguous group (the O’Hare tract is an island).'
    },
    {id: 'links', label: 'Weight links'},
    {
      id: 'fit',
      label: 'OLS R² and adjusted R²',
      help: 'R² and adjusted R² of the ordinary least squares fit.'
    },
    {
      id: 'criteria',
      label: 'Information criteria',
      help: 'AIC and BIC, GeoDa convention. Lower is better; compare between models of the same outcome.'
    },
    {
      id: 'normality',
      label: 'Jarque-Bera (normal residuals)',
      help: 'Small p: residuals are not normally distributed.'
    },
    {
      id: 'heteroskedasticity',
      label: 'Koenker-Bassett (constant variance)',
      help: 'The studentized Breusch-Pagan test, which does not assume normal residuals. Small p: residual variance changes with the predictors.'
    },
    {
      id: 'coefficient0',
      label: 'β₀',
      help: 'Estimate, standard error and p-value of the coefficient in OLS (t), the lag model and the error model (z).'
    },
    {id: 'coefficient1', label: 'β₁'},
    {id: 'coefficient2', label: 'β₂'},
    {id: 'coefficient3', label: 'β₃'},
    {id: 'coefficient4', label: 'β₄'},
    {id: 'coefficient5', label: 'β₅'},
    {id: 'coefficient6', label: 'β₆'},
    {id: 'coefficient7', label: 'β₇'},
    {id: 'coefficient8', label: 'β₈'},
    {
      id: 'moran',
      label: 'Moran’s I of the residuals',
      help: 'Spatial autocorrelation of OLS residuals with the chosen weights; positive means similar residuals sit next to each other.'
    },
    {
      id: 'lmLag',
      label: 'LM-lag',
      help: 'Lagrange multiplier test for a spatially lagged response (statistic, p).'
    },
    {
      id: 'lmError',
      label: 'LM-error',
      help: 'Lagrange multiplier test for spatially correlated errors.'
    },
    {
      id: 'robustLag',
      label: 'Robust LM-lag',
      help: 'LM-lag test robust to a possible spatial error process.'
    },
    {
      id: 'robustError',
      label: 'Robust LM-error',
      help: 'LM-error test robust to a possible spatial lag.'
    },
    {id: 'sarma', label: 'LM-SARMA', help: 'Joint test for lag and error.'},
    {
      id: 'verdict',
      label: 'Spatial diagnostic screen',
      help: 'Anselin’s LM decision rule at 5 percent is a diagnostic screen. Failure to detect spatial dependence does not establish that OLS is adequate; the screen does not prove a spatial model is correct.'
    },
    {
      id: 'validation',
      label: 'Blocked OLS validation',
      help: 'Five spatial block holdouts with a 500 m training exclusion. Each unpenalized OLS fit and its centering use training rows only. Scores measure this split and are not a guarantee of future predictive performance.'
    },
    {id: 'validationFolds', label: 'Held-out RMSE by fold', kind: 'chart'},
    {
      id: 'rho',
      label: 'Lag model ρ',
      help: 'Spatial lag coefficient from two-stage least squares: how much neighbours’ prevalence spills into a tract after controlling for X.'
    },
    {id: 'lagFit', label: 'Lag model fit'},
    {
      id: 'conditioning',
      label: 'Collinearity (condition numbers)',
      help: 'Belsley condition number of each design (above 30 is harmful collinearity) and the largest variance inflation factor. A float32 precision warning means the normal equations may have lost digits; the fit is still shown.'
    },
    {
      id: 'lambda',
      label: 'Error model λ',
      help: 'Spatial error coefficient from generalized moments (homoskedastic, spreg GM_Error_Hom), with its standard error, z and p-value.'
    },
    {
      id: 'clusters',
      label: 'Residual clusters',
      help: 'Significant local Moran quadrants of the OLS residuals.'
    },
    {
      id: 'status',
      label: 'Status (OLS / diagnostics / lag / error)',
      help: '0 means every fit succeeded.'
    }
  ],

  legends: state => {
    const unit = OUTCOME_UNITS[state.outcome];
    const entries: LegendSpec[] = [];
    if (state.map === 'fitted' || state.map === 'observed') {
      entries.push({
        kind: 'ramp',
        id: 'value',
        title: state.map === 'fitted' ? 'Fitted by OLS' : 'Observed',
        ramp: 'ylorrd',
        extent: 'gpu',
        unit,
        format: value => value.toFixed(1)
      });
    } else if (state.map === 'clusters') {
      entries.push({
        kind: 'categories',
        title: 'Cluster of OLS residuals',
        entries: [
          {color: CLUSTER_RGBA.hh, label: 'High-High: model under-predicts'},
          {color: CLUSTER_RGBA.ll, label: 'Low-Low: model over-predicts'},
          {color: CLUSTER_RGBA.lh, label: 'Low-High outlier'},
          {color: CLUSTER_RGBA.hl, label: 'High-Low outlier'},
          {color: CLUSTER_RGBA.none, label: 'Not significant'}
        ],
        note: 'Analytic local Moran of the residuals at the chosen level. Clusters are what a lag or error term should absorb.'
      });
    } else if (state.map === 'validationFolds') {
      entries.push({
        kind: 'categories',
        title: 'Spatial validation fold',
        entries: [1, 2, 3, 4, 5].map(fold => ({
          color: VALIDATION_FOLD_COLORS[fold],
          label: `Held-out fold ${fold}`
        })),
        note: 'Fold assignment used for the blocked out-of-fold predictions; buffered tracts are omitted from that fold’s training fit.'
      });
    } else {
      const names = {
        residuals: 'OLS residual',
        lag: 'Lag model residual',
        error: 'Error model residual',
        validationResiduals: 'Blocked out-of-fold residual'
      };
      entries.push({
        kind: 'ramp',
        id: 'residual',
        title: `${names[state.map]} (observed minus fitted)`,
        ramp: 'diverging',
        midpoint: 0,
        extent: 'gpu',
        unit: 'percentage points',
        labels: ['over-predicted', 'under-predicted'],
        format: value => value.toFixed(1)
      });
    }
    entries.push({
      kind: 'categories',
      title: 'Tracts',
      entries: [
        {color: [170, 172, 178, 150], label: 'Left out of the model (missing data or island)'}
      ]
    });
    return entries;
  },

  snippet: state => {
    const knn = state.weights.startsWith('knn');
    return `import {
  addSpatialRegressionRecipe, GPUContiguityWeights, GPUSpatialWeightsTransform,
  GPUSpatialTwoStageLeastSquares, GPUSpatialErrorGM
} from '@luma-contributors/graph/gpu-spatial-analysis';

// 1. Weights ${knn ? `(directed k nearest neighbours, row-standardized)` : `(${state.weights} contiguity from shared vertices, then row-standardized)`}
${
  knn
    ? `graph.add(new GPUNeighborSearch({spatialContext: projection.spatialContext, mode: 'knn', k: ${state.weights.slice(3)}, gridSize: [64, 64], positions, parameters, output: {...weights, status: {overflow}}}));`
    : `graph.add(new GPUContiguityWeights({criterion: '${state.weights}', polygons: {kind: 'polygons', positions: vertices, featureOffsets, polygonOffsets, ringOffsets}, output: {...weights, status: {overflow, requiredCount, candidateOverflow}}}));
graph.add(new GPUSpatialWeightsTransform({spatialContext: projection.spatialContext, operation: 'row', weights}));`
}

// 2. OLS + LM diagnostics + residual local Moran, one recipe
const regression = addSpatialRegressionRecipe(graph, {
  predictors, response, predictorCount: ${state.model === 'income' ? 1 : state.model === 'economic' ? 4 : 8},
  weights, settings: {significanceLevel: ${state.significance}},   // result.setSettings repacks it per frame
  parameters: moranParameters,         // GPUParameterBuffer the recipe fills from settings
  olsParameters: ridgeParameters,      // [lambda]: a per-frame buffer write
  ols: {coefficients, standardErrors, tStatistics, summary, status: {code: olsStatus}, residuals, fitted},
  diagnostics: {tests, summary, status: {code: diagnosticsStatus}},
  residualMoran: {zScores, localI, quadrants, pValues}
});

// 3. The two alternatives the diagnostics choose between
lagGraph.add(new GPUSpatialTwoStageLeastSquares({
  weights, predictors, response, predictorCount, instrumentOrder: ${state.instruments}, output: lag
}));
errorGraph.add(new GPUSpatialErrorGM({weights, predictors, response, predictorCount, output: error}));

// 4. Per frame only on change: ridgeParameters.write(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: ${state.ridge <= -2 ? 0 : (10 ** state.ridge).toPrecision(2)}}))`;
  },

  about: {
    what: '`GPUOrdinaryLeastSquares` fits the regression and reports R², AIC/BIC, Jarque-Bera and Koenker-Bassett, with a standard error and p-value for each coefficient and collinearity evidence. `GPUSpatialRegressionDiagnostics` screens residuals for spatial dependence (LM-lag, LM-error, robust variants, SARMA, Moran’s I). `GPUSpatialTwoStageLeastSquares` fits a spatial lag model and `GPUSpatialErrorGM` a spatial error model. `GPUSpatialCrossValidatedOLS` fits five separate OLS models, each using only its training rows, and scores held-out tracts.',
    why: 'Neighbouring tracts share air, food environments, clinics and history. Spatially clustered residuals can make ordinary standard errors misleading. The LM tests screen for remaining dependence but cannot establish that a model is adequate; blocked validation measures how this unpenalized OLS design transfers across separated parts of the city.',
    howToRead:
      'On the residual maps blue tracts have lower diabetes estimates than their fitted values and red tracts higher. A specification that captures the spatial pattern may leave residuals with less clustering, though a global Moran statistic or local cluster map cannot verify every model assumption. Compare coefficients across the three specifications: a large change after adding a spatial term may indicate sensitivity to spatial structure or other specification choices.'
  },

  create: async ctx => (await import('./health-regression.compute')).createHealthRegression(ctx),

  story: [
    {
      id: 'question',
      title: 'What do incomes explain about diabetes in Chicago?',
      headline: 'Observed diabetes prevalence varies across Chicago tracts',
      textAlternative:
        'Chicago tracts are shaded by estimated diabetes prevalence, with higher values concentrated on the South and West sides.',
      body: 'Diabetes is not spread evenly across Chicago: **CDC PLACES** estimates its prevalence for every census tract, and the highest rates sit on the South and West sides. Part of that is income. The question for a planner is sharper: **how much does a tract’s income and poverty explain, and where does the model get it wrong?**\n\nThe map shows the observed prevalence (**Map** below) for the tracts in the model; pick another **Health outcome (response)** to see its geography. Tracts left out (missing data, or O’Hare, which touches no other tract) are gray.',
      options: {model: 'income', map: 'observed'},
      controls: ['outcome', 'map'],
      readouts: ['rows']
    },
    {
      id: 'ols',
      title: 'One predictor, one line: ordinary least squares',
      headline: 'Income-only residuals remain spatially patterned',
      textAlternative:
        'A diverging tract map shows adjacent areas with similarly positive or negative income-only OLS residuals.',
      body: '**`GPUOrdinaryLeastSquares`** first fits `prevalence = β₀ + β₁·ln(income per capita) + ε` by least squares on the GPU. Predictors are standardized, so β₁ is points of prevalence per standard deviation of ln income. The map shows the **residual**: observed minus fitted. Blue tracts have lower prevalence than the fitted model predicts, red tracts higher. Hover a tract for its numbers.\n\nSwitch **Predictors** from *Income only* to *Economic (4)* to add poverty, uninsured and unemployment rates. Each coefficient is an adjusted ecological association conditional on the included predictors, not a causal effect. Compare fit and information criteria across these designs; the predictor count is compile-time, so the design switch recompiles its graphs.',
      options: {model: 'income', map: 'residuals'},
      controls: ['model', 'map'],
      readouts: ['fit', 'criteria', 'coefficient1']
    },
    {
      id: 'ridge',
      title: 'When predictors overlap: ridge',
      headline: 'Ridge shrinkage reduces coefficient instability',
      textAlternative:
        'The residual map accompanies coefficient readouts that contract as the ridge penalty increases for correlated tract predictors.',
      body: '**Predictors** is now *Full (8)*, which adds age, disability, vehicle access and race/ethnicity shares. Tract covariates are highly correlated, so individual coefficients become unstable and their standard errors inflate. Slide the **Ridge penalty λ (log₁₀)**: `GPUOrdinaryLeastSquares` adds λ to the diagonal of XᵀX, which shrinks the coefficients and tightens the standard errors at the price of a little bias. The penalty is a one-float parameter buffer, so the slider is a buffer write and no graph is rebuilt.',
      options: {model: 'full', ridge: 0.5},
      controls: ['model', 'ridge'],
      readouts: ['coefficient1', 'coefficient2', 'coefficient3', 'conditioning']
    },
    {
      id: 'diagnostics',
      title: 'Is the leftover spatial?',
      headline: 'Local residual patterns require a multiple-testing caveat',
      textAlternative:
        'Red and blue tract groups mark high-high and low-low economic-model residual patterns passing uncorrected analytic local tests.',
      body: 'Ordinary least squares assumes independent errors. **`GPUSpatialRegressionDiagnostics`** tests that against the spatial weights: **LM-lag** (does a neighbour’s outcome matter?), **LM-error** (are the errors correlated?), their **robust** versions and **Moran’s I** of the residuals. The matching spreg tests are `LMtests` and `MoranRes`.\n\n**Map** now shows *OLS residual clusters (local Moran)*: exploratory **clusters** of OLS residuals from a local Moran run inside the same graph by **`addSpatialRegressionRecipe`**: red clusters are places the model under-predicts, blue clusters where it over-predicts. These are analytic local Moran tests at **{{localInference}}**, without multiple-testing correction across tracts; false positives are expected. The **Moran’s I** readout is a separate global residual test and does not establish significance of individual mapped clusters.',
      options: {model: 'economic', ridge: -2, map: 'clusters'},
      controls: ['weights', 'significance', 'map'],
      readouts: ['localInference', 'moran', 'lmLag', 'lmError']
    },
    {
      id: 'error',
      title: 'Compare spatial lag and spatial error models',
      headline: 'Alternative specifications represent different dependence',
      textAlternative:
        'Spatial-lag and spatial-error residuals are mapped by tract beside their coefficients and held-out validation results.',
      body: 'If neighboring outcomes may be associated with a tract’s outcome, a spatial-lag specification is one candidate: `y = ρ·Wy + Xβ + ε`. **`GPUSpatialTwoStageLeastSquares`** estimates ρ using `WX` (and optionally `W²X`) as instruments, following spreg `GM_Lag`. If neighboring tracts resemble one another because of shared or omitted conditions, a spatial-error specification is another candidate: `u = λ·Wu + ε`. **`GPUSpatialErrorGM`** estimates λ by a generalized-moments scan, then re-estimates β on spatially filtered data. These fitted coefficients describe adjusted ecological associations.\n\nThe **Spatial diagnostic screen** applies Anselin’s LM rule; it can indicate candidate specifications, but a non-significant test does not establish OLS adequacy and the smaller p-value does not select a winner. **Blocked OLS validation** fits each fold on training tracts only. Compare 3 km spatial blocks with seeded random rows, and vary the 0, 500 or 1,000 m training exclusion. The chart reports fold RMSE; the map can show out-of-fold residuals or fold assignments. These unpenalized fits are a predictive stress test, separate from the ridge-enabled full-sample models above.\n\nTry other **Spatial weights** (rook, k nearest) and another **Health outcome (response)**: diagnostics, ρ and λ depend on how neighbors are defined. **Limits:** PLACES values are model-based estimates, not tract surveys; these are ecological associations, not individual risk; and neither diagnostic screens nor one blocked split proves the model is correct or will generalize elsewhere.',
      options: {map: 'error'},
      controls: ['instruments', 'validationDesign', 'validationBuffer', 'map'],
      readouts: ['validation', 'validationFolds', 'lambda', 'rho']
    },
    {
      id: 'blocked-validation',
      title: 'Test prediction on held-out tracts',
      headline: 'Training-only fits leave a mapped out-of-fold residual for every tract',
      textAlternative:
        'A diverging Chicago tract map shows observed minus held-out predicted diabetes prevalence; a chart reports RMSE across five spatial or random folds.',
      body: '**Blocked OLS validation** fits and centers a separate unpenalized model inside each training fold; it predicts only the held-out tracts and records one out-of-fold residual per tract. The map shows observed minus that held-out prediction, and hover reports the fold and residual. **Validation split** compares seeded 3 km spatial blocks with random rows; random rows are a contrast, while spatial blocks ask how the fit transfers to separated neighborhoods. **Training exclusion** removes nearby tracts around each test block. The fold chart and pooled score are predictive checks for this chosen design, not evidence of causal effects or a guarantee about other cities.',
      options: {
        model: 'economic',
        ridge: -2,
        map: 'validationResiduals',
        validationDesign: 'spatial',
        validationBuffer: '500'
      },
      controls: ['model', 'validationDesign', 'validationBuffer', 'map'],
      readouts: ['validation', 'validationFolds'],
      evidence:
        'The GPU cross-validation operator receives the raw predictor table and applies fold-specific masks; no full-data centering or response-derived transformation enters the held-out fits.',
      caveat:
        'The random-row split keeps nearby tracts on opposite sides of the split and therefore tends to test interpolation among familiar neighbors rather than transfer to a separated region.'
    }
  ]
});
