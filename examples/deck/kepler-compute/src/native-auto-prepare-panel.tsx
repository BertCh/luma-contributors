// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useSyncExternalStore} from 'react';
import type {AutoPrepareReport, NativeAutoPrepare} from './native-auto-prepare';

const STATE_LABELS: Record<AutoPrepareReport['state'], string> = {
  queued: 'Waiting to prepare',
  preparing: 'Preparing',
  ready: 'Prepared',
  failed: 'Not prepared'
};

/** Visible record of the roles and precision policy chosen for each automatically prepared dataset. */
export function NativeAutoPreparePanel({controller}: {controller: NativeAutoPrepare}) {
  useSyncExternalStore(
    listener => controller.subscribe(listener),
    () => controller.getVersion()
  );
  const entries = [...controller.reports.values()];
  if (!entries.length) return null;
  return (
    <section aria-label="Automatic preparation" data-testid="auto-prepare-report">
      <h2>Automatic preparation</h2>
      {entries.map(report => (
        <details key={report.datasetId} open>
          <summary>
            {report.label}: {STATE_LABELS[report.state]} ({report.rowCount.toLocaleString()} rows)
          </summary>
          {report.error && <p role="alert">{report.error}</p>}
          <ul>
            {report.roleSummary.map(line => (
              <li key={line}>{line}</li>
            ))}
            {report.policySummary.map(line => (
              <li key={line}>{line}</li>
            ))}
            {report.columnNotes.map(line => (
              <li key={line}>{line}</li>
            ))}
            {report.excluded.map(entry => (
              <li key={entry.name}>
                Excluded {entry.name}: {entry.reason}
              </li>
            ))}
            {report.warnings.map(line => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </details>
      ))}
    </section>
  );
}
