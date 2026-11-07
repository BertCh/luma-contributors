// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {
  MAXIMUM_KEPLER_OLS_PREDICTORS,
  MAXIMUM_KEPLER_OLS_SOURCE_ROWS,
  type KeplerOLSSettings
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {ResidentConsumerNotice} from './resident-consumer-notice';

/** Submit the response and ordered predictor set together, without partially updating a fit. */
export function OLSPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.ols;
  return (
    <OLSControls
      key={`${analysis.datasetId}:${JSON.stringify(settings)}`}
      analysis={analysis}
      settings={settings}
    />
  );
}

function OLSControls({
  analysis,
  settings
}: {
  analysis: AnalysisRuntime;
  settings?: KeplerOLSSettings;
}) {
  const [responseColumn, setResponseColumn] = useState(settings?.responseColumn ?? '');
  const [predictorColumns, setPredictorColumns] = useState<readonly string[]>(
    settings?.predictorColumns ?? []
  );
  const [ridgeLambda, setRidgeLambda] = useState(String(settings?.ridgeLambda ?? 0));
  const [error, setError] = useState('');
  return (
    <details className="analysis-ols-settings">
      <summary>Configure ordinary least squares regression</summary>
      <p className="hint">
        Fit a numeric response from one to 15 numeric predictors. An intercept is always included.
        Only complete rows with finite response and predictor values enter the fit.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            analysis.configureOLS({
              responseColumn,
              predictorColumns: analysis.availableColumns.filter(column =>
                predictorColumns.includes(column)
              ),
              intercept: 'included',
              ridgeLambda: Number(ridgeLambda),
              variance: 'degrees-of-freedom',
              missing: 'complete-cases'
            });
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <label htmlFor="analysis-ols-response">OLS response field</label>
        <select
          id="analysis-ols-response"
          aria-label="OLS response field"
          required
          value={responseColumn}
          onChange={event => {
            setResponseColumn(event.target.value);
            setPredictorColumns(predictorColumns.filter(column => column !== event.target.value));
          }}
        >
          <option value="">Choose a response</option>
          {analysis.availableColumns.map(column => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        <fieldset>
          <legend>OLS predictors · {predictorColumns.length} selected</legend>
          {analysis.availableColumns.map((column, index) => {
            const selected = predictorColumns.includes(column);
            return (
              <label key={column} htmlFor={`analysis-ols-predictor-${index}`}>
                <input
                  id={`analysis-ols-predictor-${index}`}
                  aria-label={`OLS predictor ${column}`}
                  type="checkbox"
                  checked={selected}
                  disabled={
                    column === responseColumn ||
                    (!selected && predictorColumns.length >= MAXIMUM_KEPLER_OLS_PREDICTORS)
                  }
                  onChange={event =>
                    setPredictorColumns(
                      event.target.checked
                        ? [...predictorColumns, column]
                        : predictorColumns.filter(value => value !== column)
                    )
                  }
                />
                {column}
                {column === responseColumn ? ' (response)' : ''}
              </label>
            );
          })}
        </fieldset>
        <label htmlFor="analysis-ols-ridge">OLS ridge penalty</label>
        <input
          id="analysis-ols-ridge"
          type="number"
          min="0"
          step="any"
          required
          value={ridgeLambda}
          onChange={event => setRidgeLambda(event.target.value)}
        />
        <p className="hint">
          Penalty 0 fits ordinary least squares. A positive penalty shrinks slopes in the original
          predictor units, without penalizing the intercept. It depends on predictor scale; no
          standardization or penalty selection is performed. Ridge reports are descriptive only.
        </p>
        <p className="hint">
          At most 65,536 source rows. Calculations use Float32; strongly collinear or poorly scaled
          predictors can fail or lose precision. Ordinary least squares standard errors assume
          homoskedastic independent errors and use residual variance divided by n − p, with the
          intercept counted in p. No robust or spatial covariance correction is applied.
        </p>
        <p className="hint">
          The selected population is refitted using only selected complete rows. Fitted values and
          residuals describe that training population, not held-out predictions. Excluded source
          rows remain missing in both outputs.
        </p>
        <button
          type="submit"
          disabled={!predictorColumns.length || analysis.rowCount > MAXIMUM_KEPLER_OLS_SOURCE_ROWS}
        >
          Apply fields and fit regression
        </button>
      </form>
      {error && (
        <ResidentConsumerNotice
          consumer="analysis"
          status="error"
          className="error"
          reason={error}
        />
      )}
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
