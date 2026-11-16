// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_OLS_PREDICTORS,
  MAXIMUM_KEPLER_OLS_SOURCE_ROWS,
  parseKeplerOLSSettings,
  type KeplerOLSSettings,
  type CatalogParameterValue
} from '@luma-contributors/kepler';
import {useAnalysisFormInitialization} from './analysis-form-initialization';
import type {AnalysisRuntime} from './analysis-runtime';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  createResidentSheetFields,
  getResidentSheetChecks,
  readResidentSheetNumber
} from './analysis-sheet-model';
import {
  usePointAnalysisSheet,
  makePointSheetColumn,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';

export function makeOLSSheetDefaults(
  saved: KeplerOLSSettings | undefined,
  responseColumn: string
): Record<string, CatalogParameterValue> {
  return {
    responseColumn: saved?.responseColumn ?? responseColumn,
    predictorCount: saved?.predictorColumns.length ?? 1,
    ...Object.fromEntries(
      Array.from({length: MAXIMUM_KEPLER_OLS_PREDICTORS}, (_, index) => [
        `predictor${index + 1}`,
        saved?.predictorColumns[index] ?? ''
      ])
    ),
    ridgeLambda: saved?.ridgeLambda ?? 0,
    editedFields: '[]'
  };
}

/** Individual ordered roles preserve arbitrary column names, including commas, through replay. */
export function getOLSSheetSettings(values: PointSheetValues): KeplerOLSSettings {
  const count = readResidentSheetNumber(values, 'predictorCount');
  if (!Number.isInteger(count) || count < 1 || count > MAXIMUM_KEPLER_OLS_PREDICTORS)
    throw new Error('OLS requires 1 to 15 predictor columns');
  return parseKeplerOLSSettings({
    responseColumn: values['responseColumn'],
    predictorColumns: Array.from({length: count}, (_, index) => values[`predictor${index + 1}`]),
    intercept: 'included',
    ridgeLambda: readResidentSheetNumber(values, 'ridgeLambda'),
    variance: 'degrees-of-freedom',
    missing: 'complete-cases'
  });
}

/** Submit response and ordered predictors atomically while preserving the bespoke report. */
export function OLSPanel({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const saved = analysis.settings.ols;
  const sheet = usePointAnalysisSheet(
    makeOLSSheetDefaults(saved, analysis.settings.column || initialization.column),
    saved,
    analysis.coordinateRestorationKey ?? analysis.datasetId
  );
  const count = Number(sheet.values['predictorCount']);
  const specifications: PointSheetSpecification[] = [
    makePointSheetColumn('responseColumn', 'OLS response field', 'analysis-ols-response'),
    {
      name: 'predictorCount',
      label: 'OLS predictor count',
      controlId: 'analysis-ols-predictor-count',
      kind: 'integer',
      minimum: 1,
      maximum: MAXIMUM_KEPLER_OLS_PREDICTORS
    },
    ...Array.from(
      {
        length:
          Number.isInteger(count) && count >= 1 && count <= MAXIMUM_KEPLER_OLS_PREDICTORS
            ? count
            : 0
      },
      (_, index) =>
        makePointSheetColumn(
          `predictor${index + 1}`,
          `OLS predictor ${index + 1}`,
          `analysis-ols-predictor-${index}`
        )
    ),
    {
      name: 'ridgeLambda',
      label: 'OLS ridge penalty',
      controlId: 'analysis-ols-ridge',
      kind: 'number',
      minimum: 0,
      level: 'advanced'
    }
  ];
  const fields = createResidentSheetFields(
    sheet.values,
    specifications,
    name =>
      !saved && name === 'responseColumn' && sheet.source(name) !== 'User override'
        ? 'Numeric field above (publication role or user selection)'
        : sheet.source(name),
    name =>
      name === 'responseColumn'
        ? analysis.availableColumns
        : analysis.availableColumns.filter(column => column !== sheet.values['responseColumn'])
  );
  const checks = getResidentSheetChecks(fields, () => getOLSSheetSettings(sheet.values));
  if (analysis.rowCount < 1 || analysis.rowCount > MAXIMUM_KEPLER_OLS_SOURCE_ROWS)
    checks.push(`OLS supports 1 to ${MAXIMUM_KEPLER_OLS_SOURCE_ROWS} source rows.`);
  return (
    <details className="analysis-ols-settings">
      <summary>Configure ordinary least squares regression</summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([analysis.coordinateRestorationKey, saved])}
        fields={fields}
        checks={checks}
        method="Resident OLS fits a numeric response from 1 to 15 ordered numeric predictors with an intercept. Only complete finite rows enter the fit. Penalty 0 fits ordinary least squares; a positive penalty shrinks slopes in original predictor units without penalizing the intercept. No standardization or penalty selection is performed. Ridge reports are descriptive only. Float32 can fail or lose precision for poorly scaled or collinear predictors. Classical standard errors assume homoskedastic independent errors and use residual variance divided by n − p, counting the intercept. No robust or spatial covariance correction is applied. Selection refits complete selected rows; fitted values and residuals describe the training population, not held-out predictions. Excluded rows remain missing."
        submitLabel="Apply fields and fit regression"
        onDraftChange={sheet.markEdited}
        onChange={sheet.change}
        onApply={current =>
          analysis.configureOLS(getOLSSheetSettings({...sheet.values, ...current}))
        }
      />
    </details>
  );
}

/** A bounded coefficient and diagnostic report; no source-row arrays are read for this view. */
export function OLSResults({analysis}: {analysis: AnalysisRuntime}) {
  const report = analysis.summary?.ols;
  if (!report) return null;
  const penalized = report.inference === 'penalized-descriptive';
  const testNames: Record<string, string> = {
    jarqueBera: 'Jarque–Bera normality',
    breuschPagan: 'Breusch–Pagan heteroskedasticity',
    koenkerBassett: 'Koenker–Bassett heteroskedasticity'
  };
  return (
    <section className="analysis-ols-results" aria-label="Regression report">
      <p>
        {penalized ? 'Ridge descriptive fit' : 'Ordinary least squares fit'} · {report.rowCount}{' '}
        complete observations
        {!penalized && ` · ${report.degreesOfFreedom} residual degrees of freedom`}
      </p>
      {report.conditioning?.illConditioned && (
        <p role="status" className="error">
          Precision warning: the predictor correlation matrix is ill-conditioned. Float32 estimates
          may have lost several significant digits; inspect predictor scaling and collinearity.
        </p>
      )}
      <div style={{overflowX: 'auto'}}>
        <table aria-label="Regression coefficients">
          <thead>
            <tr>
              <th scope="col">Term</th>
              <th scope="col">Estimate</th>
              <th scope="col">Standard error</th>
              <th scope="col">t statistic</th>
              <th scope="col">Two-sided p</th>
            </tr>
          </thead>
          <tbody>
            {report.coefficients.map((coefficient, index) => (
              <tr key={index}>
                <th scope="row">{coefficient.name}</th>
                <td>{formatOLSNumber(coefficient.estimate)}</td>
                <td>{formatOLSNumber(coefficient.standardError)}</td>
                <td>{formatOLSNumber(coefficient.tStatistic)}</td>
                <td>{formatOLSNumber(coefficient.pValue)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        R²: {formatOLSNumber(report.rSquared)} · residual sum of squares:{' '}
        {formatOLSNumber(report.residualSumOfSquares)}
        {penalized
          ? ` · ridge penalty: ${formatOLSNumber(report.ridgeLambda)}`
          : ` · adjusted R²: ${formatOLSNumber(report.adjustedRSquared)}`}
      </p>
      {penalized ? (
        <p className="hint">
          Penalized coefficients are descriptive. Standard errors, t tests, p-values, residual
          variance, likelihood and information criteria are not reported. No effective degrees of
          freedom or predictive validation is claimed.
        </p>
      ) : (
        <p className="hint">
          Residual variance: {formatOLSNumber(report.sigmaSquared)} · Gaussian log-likelihood:{' '}
          {formatOLSNumber(report.logLikelihood)} · AIC: {formatOLSNumber(report.aic)} · BIC:{' '}
          {formatOLSNumber(report.bic)}. Coefficient p-values use two-sided Student t tests.
          Undefined values can occur for exact fits or degenerate diagnostics; they are not evidence
          that an assumption passed.
        </p>
      )}
      {report.conditioning && (
        <p className="hint">
          Design condition number: {formatOLSNumber(report.conditioning.conditionNumber)} ·
          {penalized
            ? 'penalized solve condition number'
            : 'predictor correlation condition number'}
          : {formatOLSNumber(report.conditioning.correlationConditionNumber)}
          {!penalized &&
            ` · maximum variance inflation factor: ${formatOLSNumber(report.conditioning.maximumVarianceInflationFactor)}`}
          .
        </p>
      )}
      {report.tests && (
        <div style={{overflowX: 'auto'}}>
          <table aria-label="Regression diagnostic tests">
            <thead>
              <tr>
                <th scope="col">Diagnostic</th>
                <th scope="col">Statistic</th>
                <th scope="col">Degrees of freedom</th>
                <th scope="col">p-value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(report.tests).map(([name, result]) => (
                <tr key={name}>
                  <th scope="row">{testNames[name] ?? name}</th>
                  <td>{formatOLSNumber(result.statistic)}</td>
                  <td>{result.degreesOfFreedom}</td>
                  <td>{formatOLSNumber(result.pValue)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function formatOLSNumber(value: number | null): string {
  return value !== null && Number.isFinite(value)
    ? String(Number(value.toPrecision(6)))
    : 'undefined';
}
