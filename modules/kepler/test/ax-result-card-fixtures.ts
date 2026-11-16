// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  applyKeplerDatasetProfileDeclarations,
  createKeplerCuratedAnalysisSample,
  getCatalogBinding,
  getCatalogFeatureInputs,
  getKeplerCuratedAnalysis,
  getKeplerOperationDescriptor,
  KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS,
  KEPLER_GRID_CELL_CAPS,
  profileKeplerFeatureCollection,
  resolveKeplerOperationParameters,
  runCatalogOperationAsync,
  type KeplerDatasetProfile,
  type KeplerResultReportContext
} from '../src/index';
import {createAnalysisResultCardModel} from '../../../examples/deck/kepler-compute/src/analysis-result-card-model';

// These twelve seeded, licence-clean CPU runs are the review cards. Their documented sentences
// are stored in __snapshots__/ax-result-card.node.spec.ts.snap; card-review.md names each method.
export const CARD_IDS = [
  'density-surface',
  'hot-spot-map',
  'clusters-and-footprints',
  'idw-surface',
  'kriging-with-diagnostics',
  'count-by-area',
  'distribution-by-zone',
  'cluster-and-outlier-map',
  'rate-cluster-map',
  'excess-risk',
  'regression-with-diagnostics',
  'contiguity-correlogram'
] as const;
export type CardId = (typeof CARD_IDS)[number];
export async function createReviewCard(id: CardId) {
  const curated = getKeplerCuratedAnalysis(id)!;
  if (curated.execution.status !== 'available' || !curated.sampleId)
    throw new Error(`Review card unavailable: ${id}`);
  const execution = curated.execution;
  const sample = getCatalogFeatureInputs(createKeplerCuratedAnalysisSample(curated.sampleId));
  const columns =
    KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS[
      curated.sampleId as keyof typeof KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS
    ];
  const inputs = Object.fromEntries(
    Object.entries(execution.inputs).map(([slot, input]) => [input, sample[slot]])
  );
  const profiles: Record<string, KeplerDatasetProfile> = Object.fromEntries(
    Object.entries(inputs).map(([name, collection]) => [
      name,
      applyKeplerDatasetProfileDeclarations(
        profileKeplerFeatureCollection(collection, {datasetId: name, label: name}),
        {coordinateModel: 'geographic'}
      )
    ])
  );
  const fieldBindings = Object.fromEntries(
    Object.entries(execution.fieldParameters).flatMap(([role, parameter]) =>
      columns[role] ? [[parameter, columns[role]]] : []
    )
  );
  const binding = getCatalogBinding(execution.bindingId)!;
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles, {
    fieldBindings,
    overrides: execution.parameters,
    executionEngine: 'cpu',
    maximumCellCount: KEPLER_GRID_CELL_CAPS[binding.id]?.maximumCellCount
  });
  const result = await runCatalogOperationAsync(binding, resolved.parameters, inputs, {
    engine: 'cpu'
  });
  const context: KeplerResultReportContext = {
    profiles,
    fieldBindings,
    resolutions: resolved.resolutions,
    presetParameters: Object.keys(execution.parameters)
  };
  return {model: createAnalysisResultCardModel(result, context), inputs};
}

export const CARD_SENTENCES: Readonly<Record<CardId, string>> = {
  'cluster-and-outlier-map':
    '12 of 144 features are significant at a pseudo p-value of 0.05 or less after false discovery rate correction: 12 high-high cores.',
  'clusters-and-footprints':
    'Found 3 clusters; 142 of 860 points (16.5%) are noise and the largest cluster holds 35.6% of all points.',
  'contiguity-correlogram':
    "Moran's I at neighbour order 1 is 0.898 (a positive value means similar values near each other), with pseudo p-value 0.001 from 999 permutations (the number of permutations with a value at least as extreme as the observed one in either direction, plus one, divided by 999 plus one); a result this extreme would be rare under a random pattern at the 0.05 level.",
  'count-by-area': '600 points across 144 areas: 4.17 per area on average, up to 216 in one area.',
  'density-surface': 'Density ranges from 0.00000102 to 0.0000801 across 62,985 values.',
  'distribution-by-zone': 'Median ranges from 1 to 8 across 97 values.',
  'excess-risk': 'Smoothed excess risk ranges from 0.268 to 2.21 across 144 values.',
  'hot-spot-map':
    '43 of 420 cells are significant hot spots and 0 are significant cold spots at 90% confidence after false discovery rate correction.',
  'idw-surface': 'Value ranges from 4.08 to 24.2 across 62,250 values.',
  'kriging-with-diagnostics': 'Value ranges from 3.88 to 25 across 16,002 values.',
  'rate-cluster-map':
    '13 of 144 features are significant at a pseudo p-value of 0.05 or less after false discovery rate correction: 12 high-high cores and 1 high-low outlier.',
  'regression-with-diagnostics':
    'R² is 0.90 on 144 observations (adjusted R² 0.90), a measure of how closely the model fits the data rather than of cause.'
};
