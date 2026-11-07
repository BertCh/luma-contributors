// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {BixiDemandModelOptions} from './bixi-demand-model.compute';

const CREDIT = joinCredits(
  'BIXI Montréal open data, trip history 2024; monthly station-pair counts, August 2024',
  CREDITS.openStreetMap
);

export default defineScene<BixiDemandModelOptions>({
  id: 'bixi-demand-model',
  title: 'How does distance shape BIXI trips?',
  chapter: 'flows',
  order: 22,
  summary:
    'A conditional gravity experiment fits the same 16 high-activity BIXI stations under four different margin constraints, then compares grid search with bounded coefficient profiling.',
  contributors: ['GPUSpatialInteraction'],
  datasets: [
    {id: 'bixi-flows', role: 'BIXI station activity and monthly origin-destination counts'}
  ],
  initialView: {longitude: -73.57774, latitude: 45.51244, zoom: 12.4},
  options: [
    {
      kind: 'select',
      id: 'constraint',
      label: 'Marginal constraint',
      group: 'Gravity model',
      apply: 'param',
      default: 'doubly',
      display: 'segmented',
      help: 'Choose whether the model ignores both margins, preserves origin totals, preserves destination totals, or balances both. All four graph variants are compiled before interaction.',
      options: [
        {value: 'unconstrained', label: 'None'},
        {value: 'origin', label: 'Origins'},
        {value: 'destination', label: 'Destinations'},
        {value: 'doubly', label: 'Both'}
      ]
    },
    {
      kind: 'select',
      id: 'calibration',
      label: 'Coefficient calibration',
      group: 'Gravity model',
      apply: 'param',
      default: 'grid',
      display: 'segmented',
      help: 'Grid tests nine fixed distance-decay coefficients; bounded calibration profiles continuously over −1.5 to 0 per kilometre and reports the final bracket.',
      options: [
        {value: 'grid', label: 'Nine-point grid'},
        {value: 'bounded', label: 'Bounded profile'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map evidence',
      group: 'Display',
      apply: 'param',
      default: 'predicted',
      display: 'segmented',
      help: 'Show observed recorded pairs, fitted values for those same observed pairs, or origin marginal residuals. Missing pairs remain absent from the model and are never drawn as zero.',
      options: [
        {value: 'observed', label: 'Observed'},
        {value: 'predicted', label: 'Fitted'},
        {value: 'residual', label: 'Margin residuals'}
      ]
    }
  ],
  readouts: [
    {id: 'constraintLabel', label: 'Selected constraint'},
    {id: 'fitStatus', label: 'Fit status'},
    {
      id: 'coefficient',
      label: 'Distance coefficient',
      help: 'The fitted coefficient multiplies straight-line distance measured in kilometres. Negative values indicate decay with distance.'
    },
    {
      id: 'deviance',
      label: 'In-sample Poisson deviance',
      help: 'Descriptive fit on the observed induced pair support. It is not an uncertainty estimate or a causal travel effect.'
    },
    {id: 'marginalError', label: 'Maximum relative margin error'},
    {id: 'balancingIterations', label: 'Balancing iterations'},
    {id: 'boundedWidth', label: 'Final coefficient bracket'},
    {id: 'support', label: 'Model support'},
    {id: 'missingPairs', label: 'Missing off-diagonal pairs'},
    {id: 'pairCounts', label: 'Observed and fitted station pairs', kind: 'chart'},
    {id: 'marginalFit', label: 'Observed and fitted station margins', kind: 'chart'},
    {id: 'marginalResiduals', label: 'Origin and destination residuals', kind: 'chart'},
    {id: 'coefficientProfile', label: 'Grid coefficient profile deviance', kind: 'chart'}
  ],
  legends: state =>
    state.display === 'residual'
      ? [
          {
            kind: 'ramp',
            id: 'bixi-model-residuals',
            title:
              state.constraint === 'destination'
                ? 'Observed minus fitted destination rides'
                : 'Observed minus fitted origin rides',
            ramp: 'diverging',
            extent: 'gpu',
            unit: 'rides in August'
          }
        ]
      : [
          {
            kind: 'ramp',
            id: 'bixi-model-flows',
            title: state.display === 'observed' ? 'Observed monthly rides' : 'Fitted monthly rides',
            ramp: state.display === 'observed' ? 'cividis' : 'blues',
            extent: 'gpu',
            unit: 'rides in August'
          }
        ],
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Observed BIXI station pairs'},
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat:
      'Only observed monthly pairs are fitted; rare pairs omitted from the bundled aggregate remain missing.'
  },
  snippet: state => `graph.add(new GPUSpatialInteraction({
  origins, destinations, flows, covariate: distanceKm,
  pairState, // 0 structural zero, 1 observed, 2 missing
  originCount: 16, destinationCount: 16,
  constraint: '${state.constraint}',
  coefficients: ${state.calibration === 'grid' ? '[-1.5, -1.3125, -1.125, -0.9375, -0.75, -0.5625, -0.375, -0.1875, 0]' : '[-1.5, 0]'},
  calibration: '${state.calibration}',
  maximumIterations: 512, tolerance: 1e-5,
  output: {predictions, summary, status, candidates, originResiduals, destinationResiduals}
}));`,
  about: {
    what: 'Previously: BIXI riding groups (bixi-communities). Next: hypothetical BIXI service locations (bixi-service-locations). The model uses the public `GPUSpatialInteraction` contributor. It profiles a one-covariate Poisson gravity model over distance in kilometres, and the four constraint modes answer different conditional questions.',
    why: 'The origin-only form preserves departures at each sampled station; the destination-only form preserves arrivals; the doubly constrained form preserves both. Compare fitted values and residual margins to see what those constraints change in the same observed support.',
    howToRead:
      'The lines are the observed station pairs or their fitted counts. Lines have no arrowheads; the model retains origin-to-destination direction even where opposite links overlap. The residual view colors sampled stations by their selected origin or destination margin residual. A blank pair is omitted or otherwise unobserved in the bundled monthly table, not a zero. The coefficient is descriptive; the deviance is in-sample and no prediction uncertainty is estimated.'
  },
  create: async context =>
    (await import('./bixi-demand-model.compute')).createBixiDemandModel(context),
  story: [
    {
      id: 'observed-support',
      title: 'What can the monthly pair table support?',
      headline: 'Rare pairs stay missing in the fit',
      textAlternative:
        'A map of 16 active BIXI stations with observed directed monthly trips drawn between them; pairs omitted by the bundled aggregation are absent.',
      body: 'This experiment takes the 16 stations with the largest August departures plus arrivals. The showcase aggregation retains station pairs with at least three rides and removes same-station trips from this table. BIXI supplies the underlying trip records; these are bundling choices. The induced table has observed pairs, analytically excluded diagonal cells, and missing off-diagonal pairs.',
      evidence: '**{{support}}**; **{{missingPairs}}** missing off-diagonal pairs are in view.',
      caveat:
        'Results are conditional on this high-activity station sample and on the bundled aggregate’s pair threshold.',
      camera: {
        bounds: [-73.607, 45.486, -73.548, 45.539],
        padding: 56,
        zoom: 13.2,
        transitionMs: 1200
      },
      options: {display: 'observed'},
      controls: ['display'],
      readouts: ['support', 'missingPairs', 'pairCounts']
    },
    {
      id: 'distance-decay',
      title: 'Fitted trip counts decay with distance',
      headline: 'The fitted distance coefficient is negative',
      textAlternative:
        'Observed station links are replaced by fitted link counts from a doubly constrained gravity model, with the distance coefficient and deviance shown beside the map.',
      body: 'The fitted Poisson mean uses straight-line distance in kilometres as its only covariate. Change **Marginal constraint** below to compare models that preserve neither station margin, departures, arrivals, or both.',
      evidence:
        'The fitted distance coefficient is **{{coefficient}}**; the observed-support deviance is **{{deviance}}**.',
      caveat:
        'The distance term is an in-sample association and is not a causal effect or a route-time estimate.',
      options: {display: 'predicted', constraint: 'doubly'},
      controls: ['constraint'],
      readouts: ['fitStatus', 'coefficient', 'deviance', 'marginalError']
    },
    {
      id: 'profile-calibration',
      title: 'A bounded search refines the coefficient',
      headline: 'Bounded search reports its final bracket',
      textAlternative:
        'The same fitted station flows are shown with the selected coefficient and the width of the bounded calibration bracket.',
      body: 'The grid checks nine fixed coefficients between −1.5 and 0 per kilometre. **Coefficient calibration** below switches to a bounded profile search on the same interval; it reports whether the final bracket reached its width tolerance.',
      evidence: 'Fit status: **{{fitStatus}}**; final bracket width: **{{boundedWidth}}**.',
      caveat:
        'A bracket or balancing iteration limit is reported explicitly; failed fits do not supply predictions.',
      options: {calibration: 'bounded'},
      controls: ['calibration'],
      readouts: ['coefficient', 'boundedWidth', 'balancingIterations']
    },
    {
      id: 'residual-margins',
      title: 'Residual margins expose the constraint choice',
      headline: 'Residuals test the selected margin',
      textAlternative:
        'Stations are colored by signed observed-minus-fitted margin residuals, with one residual bar per station.',
      body: 'Read the selected margin residuals alongside the balancing diagnostics. Origin-only and doubly constrained fits target departure margins; destination-only fits target arrivals. The residuals answer whether that target was met on the induced support.',
      evidence:
        'The fitted margin error is **{{marginalError}}** after **{{balancingIterations}}** balancing passes.',
      caveat:
        'The support excludes missing pairs, so these are sample margins rather than complete BIXI station totals.',
      options: {display: 'residual'},
      controls: ['display', 'constraint'],
      readouts: ['marginalError', 'marginalFit', 'marginalResiduals']
    }
  ]
});
