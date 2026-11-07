// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AnalysisRuntime} from './analysis-runtime';

/** Reuses bounded summary counters; no source rows or additional GPU diagnostics are read. */
export function RateResults({analysis}: {analysis: AnalysisRuntime}) {
  const mode = analysis.settings.mode;
  if (mode !== 'excess-risk' && mode !== 'spatial-rate-points' && mode !== 'spatial-rate-polygons')
    return null;
  const summary = analysis.summary;
  const rateExclusions = analysis.describeRateExclusions();
  if (!summary || summary.operation !== mode) return null;
  const consideredCount = summary.count + summary.nullCount;
  const excludedCount = analysis.rowCount - consideredCount;
  const hasInfiniteOutputs =
    summary.count > 0 &&
    (summary.minimum === Infinity ||
      summary.minimum === -Infinity ||
      summary.maximum === Infinity ||
      summary.maximum === -Infinity);
  return (
    <section aria-label="Rate diagnostics" className="analysis-rate-results">
      <h3>Rate output diagnostics</h3>
      <dl>
        <dt>Rows included in result summary</dt>
        <dd>{consideredCount.toLocaleString()}</dd>
        <dt>
          {hasInfiniteOutputs ? 'Defined outputs (including infinite values)' : 'Defined outputs'}
        </dt>
        <dd>{summary.count.toLocaleString()}</dd>
        <dt>Undefined outputs</dt>
        <dd>{summary.nullCount.toLocaleString()}</dd>
        <dt>Excluded source rows</dt>
        <dd>{excludedCount.toLocaleString()}</dd>
      </dl>
      {hasInfiniteOutputs && (
        <p role="status" className="error">
          Infinite rate outputs are present. They are included in the defined-output count, not the
          undefined-output count. The summary does not report how many outputs are infinite.
        </p>
      )}
      <p className="hint">
        Undefined outputs were included in the summary but did not produce a rate. Excluded source
        rows were omitted by input validity, coordinate or polygon requirements, or the selected
        population. These counts do not distinguish individual causes.
      </p>
      {rateExclusions && <p className="hint">Source columns: {rateExclusions}</p>}
      {summary.pointWeights && (
        <p>
          {summary.pointWeights.weightsRequiredCount.toLocaleString()} point neighbor links ·{' '}
          {summary.pointWeights.invalidCount.toLocaleString()} invalid coordinate rows across the
          whole source
        </p>
      )}
      {summary.polygonWeights && (
        <p>
          {summary.polygonWeights.weightsRequiredCount.toLocaleString()} polygon neighbor links ·{' '}
          {summary.polygonWeights.invalidCount.toLocaleString()} rows without eligible polygon parts
          across the whole source
        </p>
      )}
      {(summary.pointWeights || summary.polygonWeights) && (
        <p className="hint">
          Neighbor links and coordinate or geometry exclusions describe the whole source graph. They
          are separate from the current population's output counts and may overlap with excluded
          source rows above; do not add the counts together.
        </p>
      )}
    </section>
  );
}
