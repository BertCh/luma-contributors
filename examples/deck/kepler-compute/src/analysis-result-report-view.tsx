// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useSyncExternalStore} from 'react';
import {formatKeplerResultReport} from '@luma-contributors/kepler';
import {
  dismissAnalysisResultReport,
  getAnalysisResultReportView,
  subscribeAnalysisResultReports,
  type AnalysisResultPublication
} from './analysis-result-publication';

export function AnalysisResultReport({result}: {result: AnalysisResultPublication}) {
  const {encoding, report, table} = result;
  return (
    <section aria-label="Analysis result report" data-testid="analysis-result-report">
      {result.mapMessage && (
        <p role="status" data-testid="analysis-result-map-message">
          {result.mapMessage}
        </p>
      )}
      {report && <pre style={{whiteSpace: 'pre-wrap'}}>{formatKeplerResultReport(report)}</pre>}
      {encoding && (
        <>
          <h3>Colour: {encoding.presentation.column}</h3>
          <ul>
            {encoding.legend.map((entry, index) => (
              <li key={encoding.categories[index]}>
                <span
                  aria-hidden
                  style={{
                    display: 'inline-block',
                    width: 14,
                    height: 14,
                    marginRight: 8,
                    background: `rgb(${entry.color.join(',')})`
                  }}
                />
                {entry.label}
                {entry.count !== undefined ? ` (${entry.count})` : ''}
              </li>
            ))}
          </ul>
        </>
      )}
      <p>
        {table.rowCount} rows. Showing the first {Math.min(20, table.rowCount)}.
      </p>
      <div style={{overflow: 'auto'}}>
        <table>
          <thead>
            <tr>
              {Object.keys(table.columns).map(name => (
                <th key={name}>{name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({length: Math.min(20, table.rowCount)}, (_, row) => (
              <tr key={row}>
                {Object.entries(table.columns).map(([name, values]) => (
                  <td key={name}>{values[row] === null ? 'No value' : String(values[row])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function AnalysisResultReportView() {
  const report = useSyncExternalStore(subscribeAnalysisResultReports, getAnalysisResultReportView);
  if (!report) return null;
  return (
    <aside
      role="dialog"
      aria-label={report.label}
      style={{
        position: 'fixed',
        zIndex: 1000,
        right: 24,
        top: 64,
        maxHeight: '75vh',
        width: 480,
        maxWidth: '90vw',
        overflow: 'auto',
        background: '#182435',
        color: 'white',
        padding: 20
      }}
    >
      <button onClick={() => dismissAnalysisResultReport(report)}>Close report</button>
      <h2>{report.label}</h2>
      <AnalysisResultReport result={report.analysisResult} />
    </aside>
  );
}
