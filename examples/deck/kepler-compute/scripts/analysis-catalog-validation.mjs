// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared UI assertions, called by analysis-catalog.mjs with its URL argument and GPU device.
import {setAnalysisSheetParameter} from './analysis-sheet-controls.mjs';
import assert from 'node:assert/strict';

export const VALIDATION_OPERATIONS = [
  ['regression-gtwr', 'GPUGeographicallyTemporallyWeightedRegression'],
  ['regression-mgtwr', 'GPUMultiscaleGeographicallyTemporallyWeightedRegression'],
  ['interpolation-space-time-kriging', 'GPUSpaceTimeKriging'],
  ['validation-vecchia-likelihood', 'GPUVecchiaLogLikelihood'],
  ['validation-vecchia-fisher', 'GPUVecchiaFisherScoring'],
  ['validation-vecchia-space-time', 'GPUVecchiaSpaceTimeLogLikelihood'],
  ['validation-area-of-applicability', 'GPUAreaOfApplicability'],
  ['validation-conformal', 'GPUConformalPredictionIntervals'],
  ['validation-geo-conformal', 'GPUGeoConformalPredictionIntervals'],
  ['validation-ensemble-scores', 'GPUEnsembleScores'],
  ['validation-gaussian-scores', 'GPUGaussianPredictiveScores'],
  ['validation-fdr', 'GPURasterFalseDiscoveryRateMask']
];

// The authored fixture has several numeric columns. AX asks the analyst to choose the model
// and ambiguous roles; these explicit choices preserve the validation methods exercised here.
export const VALIDATION_SAMPLE_FIELDS = {
  responseProperty: 'observed',
  predictorProperties: 'predictor',
  timeProperty: 'time',
  valueProperty: 'observed',
  observationProperty: 'observed',
  predictionProperty: 'predicted',
  meanProperty: 'predicted',
  varianceProperty: 'variance',
  memberProperties: 'member1,member2,member3',
  pValueProperty: 'pvalue'
};

async function mapValidationSample(panel) {
  for (const [name, value] of Object.entries(VALIDATION_SAMPLE_FIELDS))
    await setAnalysisSheetParameter(panel, name, value, {optional: true});
}

export async function exerciseValidationCatalog(page, panel) {
  for (const [operation, contributor] of VALIDATION_OPERATIONS) {
    await panel.getByLabel('Operation', {exact: true}).selectOption(operation);
    await mapValidationSample(panel);
    await panel.getByTestId('catalog-run').click();
    await panel.getByTestId('catalog-result').waitFor({timeout: 60000});
    assert.match(await panel.getByTestId('catalog-engine').innerText(), new RegExp(`Engine: GPU \\(.*${contributor}`));
    const result = panel.getByTestId('catalog-result');
    assert.match(await result.innerText(), /diagnostics: [1-9]\d* rows/);
    const previews = result.getByRole('table', {name: 'Result preview', exact: true});
    assert((await previews.count()) >= 2, `${operation}: outputs and diagnostics are both shown`);
    assert((await previews.first().locator('tbody tr').count()) > 0);
    await panel.getByRole('button', {name: 'Serialize invocation', exact: true}).click();
    const invocation = JSON.parse(await panel.getByLabel('Serialized invocation', {exact: true}).inputValue());
    assert.equal(invocation.operationId, operation);
    assert(Object.keys(invocation.inputs).length > 0);
  }
  // A fitted likelihood is an explicit choice and its diagnostics must still publish.
  await panel.getByLabel('Operation', {exact: true}).selectOption('validation-vecchia-fisher');
  await mapValidationSample(panel);
  await setAnalysisSheetParameter(panel, 'fit', true);
  await panel.getByTestId('catalog-run').click();
  await panel.getByTestId('catalog-result').waitFor({timeout: 60000});
  assert.match(await panel.getByTestId('catalog-result').innerText(), /log_likelihood/);
  assert.match(await panel.getByTestId('catalog-result').innerText(), /converged/);
}
