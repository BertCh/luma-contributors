// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseSpatialRegimesOptions} from './meuse-spatial-regimes.compute';

export default defineScene<MeuseSpatialRegimesOptions>({
  id: 'meuse-spatial-regimes',
  title: 'Meuse lime regimes: pooled and varying associations',
  chapter: 'regression',
  order: 10,
  summary:
    'Nested OLS restrictions ask whether log-zinc associations differ between the source measured-lime groups, with actual covariance, fitted values and residuals.',
  contributors: ['GPURestrictedRegimeRegression', 'GPUSpatialRegimeRegression'],
  datasets: [
    {id: 'meuse-validation', role: '155 measured soil-metal samples and observed lime indicator'}
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.8},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Do log-zinc associations vary by measured lime group?',
      subtitle: 'Pooled, partly varying and fully varying OLS restrictions'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'Lime 0/1 is a measured field grouping, not a set of contiguous spatial regions. Classical covariance, Chow and Wald quantities assume independent homoskedastic errors; they do not provide spatially corrected inference or a causal decision.'
  },
  options: [
    {
      kind: 'select',
      id: 'profile',
      label: 'Coefficient restrictions',
      group: 'Model form',
      apply: 'param',
      default: 'pooled',
      help: 'Compare the same complete-case rows under four authored restriction sets. Common terms share a coefficient across measured lime 0/1 groups; the fully varying fit estimates separate intercepts and slopes.',
      options: [
        {value: 'pooled', label: 'All coefficients common'},
        {value: 'varying-intercept', label: 'Varying intercept'},
        {value: 'intercept-distance', label: 'Varying intercept + river distance'},
        {value: 'fully-varying', label: 'All terms vary by lime'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Mapped values',
      group: 'Map',
      apply: 'param',
      default: 'observed',
      options: [
        {value: 'lime-group', label: 'Measured lime group'},
        {value: 'observed', label: 'Observed log zinc'},
        {value: 'fitted', label: 'Profile fitted values'},
        {value: 'residual', label: 'Profile residuals'}
      ]
    }
  ],
  story: [
    {
      id: 'measured-groups',
      title: 'Start from the measured lime field',
      headline: 'Two observed categories split the complete cases 109 to 44',
      textAlternative:
        'Meuse sample locations show the two measured zero-one lime groups with distinct categorical colors; these points are not contiguous spatial regions.',
      body: 'This lab uses **153 common complete cases** from the source’s 155 measured topsoil composites: **{{sampleCount}}**. The response is log zinc. Predictors are globally standardized measured elevation, organic matter and distance to the river. The source `lime` indicator defines two observed groups; those groups are not contiguous spatial regions and are not treated as a treatment assignment.',
      options: {display: 'lime-group', profile: 'pooled'},
      optionsMode: 'fresh',
      controls: ['display'],
      readouts: ['sampleCount', 'mapStatus'],
      evidence:
        'The complete-case filter excludes the two rows with missing organic matter; all profiles use the same 153 sample IDs.'
    },
    {
      id: 'pooled-association',
      title: 'Fit one common coefficient vector',
      headline: 'The pooled model makes every coefficient common',
      textAlternative:
        'Observed log-zinc values or fitted values from an OLS model with one shared intercept and three shared slopes.',
      body: '**`GPURestrictedRegimeRegression`** fits one log-zinc model with the intercept and all three slopes shared between measured lime groups. Coefficients and standard errors come from the fitted covariance matrix; the RMSE chart is an in-sample description, not held-out prediction.',
      options: {display: 'fitted', profile: 'pooled'},
      controls: ['display', 'profile'],
      readouts: ['profileStatus', 'profileDiagnostics', 'profileRmse'],
      caveat:
        'The classical standard errors and nested-model statistics assume independent homoskedastic errors; sample locations may have spatially dependent residuals.'
    },
    {
      id: 'vary-intercept',
      title: 'Let the baseline differ between groups',
      headline: 'A varying intercept changes baselines while keeping slopes common',
      textAlternative:
        'The selected model allows each measured lime group its own intercept and shares elevation, organic matter and river-distance slopes.',
      body: 'Choose **Varying intercept**. The model estimates separate baselines but keeps all three predictor slopes common. Read the covariance-based standard errors and the Chow F statistic against a fully varying alternative; the same 153 observations appear in every fit.',
      options: {display: 'fitted', profile: 'varying-intercept'},
      controls: ['display', 'profile'],
      readouts: ['profileStatus', 'profileDiagnostics', 'profileRmse'],
      caveat:
        'A smaller in-sample error is expected as restrictions are relaxed. RMSE does not penalize added parameters; use the nested statistic only with its stated classical assumptions.'
    },
    {
      id: 'vary-distance',
      title: 'Allow the river-distance slope to vary too',
      headline: 'The third restriction set separates intercepts and distance slopes',
      textAlternative:
        'A profile fit map and coefficient diagnostics show separate intercepts and river-distance slopes, with elevation and organic-matter slopes shared.',
      body: 'Choose **Varying intercept + river distance**. This relaxes two source-column restrictions while retaining common elevation and organic-matter slopes. The coefficient readout reports regime-specific estimates where terms vary, covariance-based standard errors, per-term stability statistics and the joint test when available.',
      options: {display: 'fitted', profile: 'intercept-distance'},
      controls: ['display', 'profile'],
      readouts: ['profileStatus', 'profileDiagnostics', 'profileRmse'],
      evidence:
        'All profile predictions use source `logZn`, measured `elev`, `om` and `dist.m` after the same complete-case filter.'
    },
    {
      id: 'full-fit-residuals',
      title: 'Inspect the fully varying fit and its residuals',
      headline: 'The fully varying profile is checked against direct regime OLS',
      textAlternative:
        'The map can show fitted log zinc or residuals for the fully varying lime-group model; readouts compare its coefficients with a direct GPU spatial-regime OLS fit.',
      body: 'Choose **All terms vary by lime**, then switch **Mapped values** between fitted log zinc and residuals. The direct `GPUSpatialRegimeRegression` fit reports GPU group counts and a coefficient difference against the fully varying restricted model as an implementation cross-check, not independent statistical evidence. The residual map uses one fixed zero-centered scale across profiles.',
      options: {display: 'residual', profile: 'fully-varying'},
      controls: ['display', 'profile'],
      readouts: ['directCheck', 'profileStatus', 'profileDiagnostics', 'profileRmse'],
      caveat:
        'The unrestricted model has no coefficient restrictions, so its Chow F is unavailable. These point-composite groups and classical OLS diagnostics do not establish causal effects or spatially calibrated significance.'
    }
  ],
  legends: state => [
    state.display === 'lime-group'
      ? {
          kind: 'categories',
          title: 'Measured lime field',
          entries: [
            {color: [37, 99, 235, 255], label: 'lime = 0'},
            {color: [220, 38, 38, 255], label: 'lime = 1'}
          ],
          note: 'Measured sample groups; not contiguous regions or an assigned treatment.'
        }
      : {
          kind: 'ramp',
          title:
            state.display === 'residual'
              ? 'Log-zinc residual'
              : `${state.display === 'fitted' ? 'Fitted' : 'Observed'} log zinc`,
          unit: state.display === 'residual' ? 'log units' : 'ln(mg/kg)',
          ramp: state.display === 'residual' ? 'diverging' : 'cividis',
          extent: state.display === 'residual' ? [-1.5, 1.5] : [4.7, 7.6],
          ...(state.display === 'residual' ? {midpoint: 0} : {})
        }
  ],
  readouts: [
    {id: 'sampleCount', label: 'Complete cases and measured groups'},
    {id: 'profileStatus', label: 'Restricted fit statuses'},
    {id: 'profileDiagnostics', label: 'Coefficients, covariance and nested statistics'},
    {id: 'profileRmse', label: 'In-sample RMSE by restrictions', kind: 'chart'},
    {id: 'directCheck', label: 'Direct fully varying fit cross-check'},
    {id: 'mapStatus', label: 'Map fit status'}
  ],
  about: {
    what: 'Nested OLS restrictions compare log-zinc associations across the source measured-lime groups.',
    why: 'The same measured rows and predictors support transparent common-versus-varying coefficient fits without inventing a spatial partition.',
    howToRead:
      'Select a coefficient restriction set, then inspect its fitted values or residuals. RMSE is in-sample. Covariance-based standard errors, Wald stability tests and Chow F statistics use classical independent homoskedastic errors and are not spatially corrected.'
  },
  create: async ctx =>
    (await import('./meuse-spatial-regimes.compute')).createMeuseSpatialRegimes(ctx)
});
