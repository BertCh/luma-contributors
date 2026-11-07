// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseNeighborhoodOptions} from './meuse-neighborhood.compute';

const RESPONSE_LABELS: Record<MeuseNeighborhoodOptions['response'], string> = {
  zinc: 'Zinc',
  lead: 'Lead'
};

const DISPLAY_LABELS: Record<MeuseNeighborhoodOptions['display'], string> = {
  'local-geary': 'Local Geary statistic',
  'local-geary-p': 'Local Geary BY-adjusted pseudo-p',
  'multivariate-geary-p': 'Multivariate Local Geary equal-tail pseudo-p',
  losh: 'LOSH local dispersion',
  'losh-p': 'LOSH fixed-residual BY-adjusted pseudo-p',
  'lee-l': 'Local Lee L statistic',
  'lee-l-p': 'Local Lee L BY-adjusted pseudo-p',
  'quantile-membership': 'Selected type-5 quantile membership',
  'quantile-join-count': 'Selected-quantile local join count'
};

export default defineScene<MeuseNeighborhoodOptions>({
  id: 'meuse-neighborhood-statistics',
  title: 'Meuse topsoil: local association and variability',
  chapter: 'statistics',
  order: 7,
  summary:
    'A measured zinc, copper and lead field compares local Geary association, joint-vector dissimilarity, local variability and cross-metal edge statistics on one documented GPU kNN graph.',
  contributors: [
    'GPUNeighborSearch',
    'GPULocalGeary',
    'GPULocalPermutationTest',
    'GPUMultivariateLocalGearyPermutationTest',
    'GPUQuantileLISA',
    'addLocalVariabilityRecipe',
    'GPULOSHResidualPermutationTest',
    'GPUSpatialPearson',
    'GPUGammaStatistic'
  ],
  datasets: [
    {id: 'meuse-validation', role: '155 measured topsoil samples and the documented river outline'}
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.8},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Meuse local association and variability',
      subtitle: 'Measured metals · eight-neighbor row-standardized kNN'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'Local permutation maps are exploratory across 155 dependent sites. LOSH’s fixed-residual test does not refit local means; edge statistics have no analytic p-values.'
  },
  options: [
    {
      kind: 'select',
      id: 'response',
      label: 'Focal metal',
      group: 'Data',
      apply: 'param',
      default: 'zinc',
      display: 'segmented',
      help: 'Selects the response for univariate local Geary, LOSH, analytic local Moran, local Lee L’s focal variable, Spatial Pearson and Gamma. The multivariate test always uses log zinc, copper and lead together.',
      options: [
        {value: 'zinc', label: 'Zinc'},
        {value: 'lead', label: 'Lead'}
      ]
    },
    {
      kind: 'select',
      id: 'gammaOperation',
      label: 'Gamma operation',
      group: 'Edge statistics',
      apply: 'param',
      default: 'squared-difference',
      help: 'Selects one of three directed-edge Gamma summaries. Each is descriptive and has no analytic p-value in this scene.',
      options: [
        {value: 'cross-product', label: 'Centered cross-product'},
        {value: 'squared-difference', label: 'Squared difference'},
        {value: 'absolute-difference', label: 'Absolute difference'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Mapped statistic',
      group: 'Map',
      apply: 'param',
      default: 'local-geary',
      options: [
        {value: 'local-geary', label: 'Local Geary'},
        {value: 'local-geary-p', label: 'Local Geary BY-adjusted p'},
        {value: 'multivariate-geary-p', label: 'Joint-vector Geary equal-tail p'},
        {value: 'losh', label: 'LOSH dispersion'},
        {value: 'losh-p', label: 'LOSH fixed-residual BY-adjusted p'},
        {value: 'lee-l', label: 'Local Lee L'},
        {value: 'lee-l-p', label: 'Local Lee L BY-adjusted p'},
        {value: 'quantile-membership', label: 'Quantile membership'},
        {value: 'quantile-join-count', label: 'Quantile local join count'}
      ]
    },
    {
      kind: 'select',
      id: 'loshPower',
      label: 'LOSH deviation power',
      group: 'Local variability',
      apply: 'param',
      default: 'variance',
      help: 'Power 2 measures variance-like dispersion; power 1 measures absolute-deviation dispersion. The matching fixed-residual permutation test uses the same power.',
      options: [
        {value: 'variance', label: 'Power 2 · squared deviation'},
        {value: 'absolute-deviation', label: 'Power 1 · absolute deviation'}
      ]
    },
    {
      kind: 'select',
      id: 'quantileCount',
      label: 'Quantile count',
      group: 'Quantile joins',
      apply: 'param',
      default: '4',
      help: 'Partitions finite log-metal values into 4, 5 or 10 type-5 quantiles. The selected band is chosen below; boundaries are lower-inclusive and upper-exclusive.',
      options: [
        {value: '4', label: '4 quantiles'},
        {value: '5', label: '5 quantiles'},
        {value: '10', label: '10 quantiles'}
      ]
    },
    {
      kind: 'select',
      id: 'quantileBand',
      label: 'Selected band',
      group: 'Quantile joins',
      apply: 'param',
      default: 'highest',
      help: 'Select the lowest, middle or highest quantile. Local join count is xᵢ times the number of connected neighbors in the same selected band.',
      options: [
        {value: 'lowest', label: 'Lowest'},
        {value: 'middle', label: 'Middle'},
        {value: 'highest', label: 'Highest'}
      ]
    }
  ],
  story: [
    {
      id: 'measured-sites',
      title: 'Work with measured metal concentrations',
      headline: '155 field sites provide three correlated metal responses',
      textAlternative:
        'Measured Meuse field sites appear around the documented river with the selected local statistic encoded by point color.',
      body: 'The Meuse package provides 155 topsoil composite samples represented by points. Zinc, copper and lead concentrations are log-transformed from the source measurements. The computational coordinates stay in the Dutch RD plane (EPSG:28992), where distances are in metres; the map uses the dataset’s geographic sample positions and river outline. Cadmium is omitted because source zeros were replaced by 0.2 mg/kg, while these three responses are strictly positive measurements.',
      controls: ['response'],
      readouts: ['sampleCount', 'weightsStatus'],
      caveat:
        'The samples are spatially dependent and the metals are correlated. The map is a local exploratory analysis, not a field-wide causal model.'
    },
    {
      id: 'local-geary',
      title: 'Test local similarity with Local Geary',
      headline: 'Local Geary targets similar neighboring values with a lower-tail test',
      textAlternative:
        'A map of local Geary values and adjusted permutation p-values accompanies the one-sided local conditional-randomization summary.',
      body: '**`GPUNeighborSearch`** creates eight nearest neighbors per sample and row-standardizes binary weights. **`GPULocalGeary`** measures the variance-scaled squared difference between each focal metal value and its neighbors. **`GPULocalPermutationTest`** holds the focal value fixed and conditionally reassigns other site values to its neighbor slots; the one-sided lesser tail targets positive local similarity. P-values receive a Benjamini–Yekutieli adjustment for arbitrary dependence among the 155 local tests.',
      controls: ['response', 'display'],
      readouts: ['localGearyStatus'],
      caveat:
        'The local pseudo-p-values are finite Monte Carlo estimates from 999 draws. They do not establish a causal process or independent replications across sites. This scene does not use the legacy directed quantileLISA pseudo-p, which is anti-conservative.'
    },
    {
      id: 'joint-vector',
      title: 'Keep the metal vector intact',
      headline: 'Multivariate Local Geary permutes whole zinc–copper–lead vectors',
      textAlternative:
        'The map shows equal-tail conditional-randomization p-values for multivariate local Geary over all three measured log-metal variables.',
      body: '**`GPUMultivariateLocalGearyPermutationTest`** population-standardizes the three variables, then draws complete neighbor vectors together without replacement; it does not shuffle each metal independently. This scene uses the conservative equal-tail convention and no multiplicity adjustment. With 999 permutations, the smallest possible equal-tail p-value is 0.002, while Bonferroni α/155 is about 0.000323, so that correction could not reject any result at this resolution.',
      options: {display: 'multivariate-geary-p'},
      controls: ['display'],
      readouts: ['multivariateStatus', 'inferenceResolution'],
      caveat:
        'The multivariate equal-tail pseudo-p-values are unadjusted. The resolution chart makes the Bonferroni limit explicit; do not interpret the raw map as familywise-controlled significance.'
    },
    {
      id: 'variability',
      title: 'Separate dispersion from association',
      headline: 'LOSH describes local variability while Moran tests a different null',
      textAlternative:
        'The map switches between descriptive local dispersion and its explicit fixed-residual permutation comparison, while analytic local Moran results remain separate.',
      body: '**`addLocalVariabilityRecipe`** computes power-2 local spatial heteroscedasticity (LOSH) as a descriptive local-dispersion measure and analytic-normal local Moran statistics on the same graph. The power selector compares squared-deviation power 2 with absolute-deviation power 1; each has a matching fixed-residual permutation test. Those outputs remain separate; no combined significance class is implied. **`GPULOSHResidualPermutationTest`** computes normalized LOSH residuals once, keeps each focal residual fixed, and reassigns other residual labels to that site’s neighbor slots. It never recomputes local means inside the permutations.',
      options: {display: 'losh'},
      controls: ['response', 'display', 'loshPower'],
      readouts: ['variabilityStatus'],
      caveat:
        'The selected LOSH residual-label null is not esda LOSH inference or a raw-observation permutation test. It tests a fixed normalized residual field at the selected deviation power.'
    },
    {
      id: 'quantile-joins',
      title: 'Where do measured-metal quantiles meet?',
      headline: 'Selected-quantile join counts describe local co-membership',
      textAlternative:
        'A map shows which measured samples fall in the selected type-5 quantile and how many of their eight binary-connected neighbors share that membership.',
      body: '**`GPUQuantileLISA`** assigns the selected log-metal response to type-5 quantile bands. Choose 4, 5 or 10 quantiles and the lowest, middle or highest band. The interval includes its lower boundary and excludes its upper boundary. The Local Join Count is **xᵢ Σⱼ xⱼ** over the graph’s binary connectivity: it is a raw directed neighbor count from zero to eight, not a sum of the row-standardized weights used by other analyses. Non-members receive zero. The map and summary show selected-band membership and observed joins only; the recipe’s legacy directed pseudo-p-value is anti-conservative and is intentionally withheld.',
      options: {display: 'quantile-membership', quantileCount: '4', quantileBand: 'highest'},
      optionsMode: 'fresh',
      controls: ['response', 'display', 'quantileCount', 'quantileBand'],
      readouts: ['weightsStatus', 'quantileSummary', 'quantileJoinSummary'],
      caveat:
        'Membership and local join counts are descriptive. They do not test clustering or establish a spatial process; no quantile-LISA pseudo-p-value is presented.'
    },
    {
      id: 'cross-metal',
      title: 'Compare cross-metal edge statistics',
      headline: 'Spatial Pearson, local Lee L and Gamma summarize different quantities',
      textAlternative:
        'Local Lee L and its BY-adjusted pseudo-p map complement global Spatial Pearson edge correlation and the selected directed-edge Gamma summary.',
      body: '**`GPUSpatialPearson`** is the weighted Pearson correlation across directed edges from focal metal at site i to copper at neighboring site j. **`GPULocalPermutationTest`** estimates local Lee L for the same pair while drawing neighbor rows jointly; its two-sided conditional-randomization pseudo-p-values receive BY correction. **`GPUGammaStatistic`** offers centered cross-product, squared difference and absolute difference on directed focal-metal edges. Spatial Pearson and Gamma are descriptive edge statistics here; neither is shown with an analytic p-value.',
      options: {display: 'lee-l'},
      controls: ['response', 'display', 'gammaOperation'],
      readouts: ['leeLStatus', 'globalAssociation']
    }
  ],
  legends: state => [
    state.display === 'quantile-membership'
      ? {
          kind: 'categories' as const,
          title: 'Selected type-5 quantile membership',
          entries: [
            {
              color: [202, 211, 219, 255] as [number, number, number, number],
              label: 'Outside selected band'
            },
            {
              color: [226, 126, 44, 255] as [number, number, number, number],
              label: 'Inside selected band'
            }
          ]
        }
      : {
          kind: 'ramp',
          title:
            state.display === 'quantile-join-count'
              ? 'Raw binary local join count · 0–8'
              : `${DISPLAY_LABELS[state.display]} · ${RESPONSE_LABELS[state.response]}`,
          unit: state.display.endsWith('-p')
            ? 'pseudo-probability'
            : state.display === 'quantile-join-count'
              ? 'directed neighbors'
              : state.display === 'local-geary' || state.display === 'lee-l'
                ? 'dimensionless'
                : 'dimensionless ratio',
          ramp: state.display.endsWith('-p') ? 'magma' : 'cividis',
          extent: state.display.endsWith('-p')
            ? [0, 1]
            : state.display === 'quantile-join-count'
              ? [0, 8]
              : state.display === 'local-geary'
                ? [0, 3]
                : state.display === 'losh'
                  ? [0, 2]
                  : [-1, 1]
        },
    {
      kind: 'categories',
      title: 'Map features',
      entries: [
        {color: [49, 113, 151, 230], label: 'Meuse river outline'},
        {color: [19, 24, 28, 255], label: 'Measured sample outline'}
      ]
    }
  ],
  readouts: [
    {id: 'sampleCount', label: 'Measured field sites', format: 'integer'},
    {id: 'weightsStatus', label: 'Neighbor weights'},
    {id: 'localGearyStatus', label: 'Local Geary inference'},
    {id: 'multivariateStatus', label: 'Joint-vector Local Geary inference'},
    {id: 'variabilityStatus', label: 'LOSH and analytic Moran'},
    {id: 'leeLStatus', label: 'Local Lee L zinc/copper'},
    {id: 'quantileSummary', label: 'Selected type-5 quantile band'},
    {id: 'quantileJoinSummary', label: 'Raw directed local join counts'},
    {id: 'globalAssociation', label: 'Spatial Pearson / selected Gamma'},
    {id: 'inferenceResolution', label: 'Permutation resolution', kind: 'chart'}
  ],
  about: {
    what: 'The scene combines GPU kNN weights, local Geary and Lee L tests, multivariate whole-vector Geary, LOSH variability, descriptive quantile joins and directed edge association on measured Meuse metal samples.',
    why: 'It contrasts local similarity, multivariate dissimilarity, dispersion, quantile co-membership, bivariate association and edge-level statistics without treating them as interchangeable tests.',
    howToRead:
      'Local Geary, LOSH-residual and Lee L maps show BY-adjusted local pseudo-p-values. Multivariate Local Geary shows unadjusted equal-tail pseudo-p-values because 999 draws cannot resolve Bonferroni significance across 155 sites. The quantile chapter shows observed type-5 membership and raw binary-edge join counts; it withholds the legacy directed anti-conservative quantile-LISA pseudo-p. LOSH itself is descriptive; the residual-label test uses a fixed normalized residual field whose test statistic is checked against the recipe LOSH values.'
  },
  create: async ctx => (await import('./meuse-neighborhood.compute')).createMeuseNeighborhood(ctx)
});
