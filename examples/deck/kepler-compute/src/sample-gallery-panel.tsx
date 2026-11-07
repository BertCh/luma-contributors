// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useId, useMemo, useState} from 'react';
import * as arrow from 'apache-arrow';
import {
  SANDBOX_SAMPLES,
  createSandboxSample,
  type SandboxSample,
  type SandboxSampleId
} from './sandbox-samples';

/** Local reproducible sources; preparation still requires the user's schema review and publication. */
export function SampleGalleryPanel({onPrepare}: {onPrepare(sample: SandboxSample): void}) {
  const identifier = useId();
  const [sampleId, setSampleId] = useState<SandboxSampleId>('points-time');
  const [status, setStatus] = useState('');
  const sample = useMemo(() => createSandboxSample(sampleId), [sampleId]);
  return (
    <details className="sample-gallery">
      <summary>Try a sample dataset</summary>
      <p>
        Small, deterministic synthetic datasets. Use the same fields and publication workflow with
        your own data.
      </p>
      <label htmlFor={`${identifier}-sample`}>Sample dataset</label>
      <select
        id={`${identifier}-sample`}
        value={sampleId}
        onChange={event => {
          const selected = SANDBOX_SAMPLES.find(candidate => candidate.id === event.target.value);
          if (selected) {
            setSampleId(selected.id);
            setStatus('');
          }
        }}
      >
        {SANDBOX_SAMPLES.map(description => (
          <option key={description.id} value={description.id}>
            {description.title}
          </option>
        ))}
      </select>
      <p>{sample.description.description}</p>
      <p>{sample.description.workflow}</p>
      {sample.description.limitation && <p className="hint">{sample.description.limitation}</p>}
      <p>
        {sample.table.numRows} rows · {sample.table.batches.length} batches · generated locally
      </p>
      <div className="actions">
        <button
          onClick={() => {
            onPrepare(sample);
            setStatus('Sample ready below. Review its fields and check schema before publishing.');
          }}
        >
          Prepare sample
        </button>
        <button
          onClick={() => {
            // Copy into an owned ArrayBuffer so the browser Blob never borrows shared Arrow storage.
            const bytes = Uint8Array.from(arrow.tableToIPC(sample.table, 'stream'));
            const url = URL.createObjectURL(
              new Blob([bytes], {type: 'application/vnd.apache.arrow.stream'})
            );
            try {
              const anchor = document.createElement('a');
              anchor.href = url;
              anchor.download = `kepler-sample-${sampleId}.arrow`;
              anchor.click();
              setStatus('Sample Arrow downloaded with its original schema, metadata and batches.');
            } finally {
              setTimeout(() => URL.revokeObjectURL(url), 0);
            }
          }}
        >
          Download sample Arrow
        </button>
      </div>
      <details>
        <summary>Fields and units</summary>
        <dl>
          {sample.description.fields.map(field => (
            <div key={field.name}>
              <dt>{field.name}</dt>
              <dd>{field.meaning}</dd>
            </div>
          ))}
        </dl>
        <p className="hint">
          source_row_id is the original zero-based row identity in every sample. Grid shape and
          coordinate roles are included in Arrow schema metadata.
        </p>
      </details>
      {status && <p role="status">{status}</p>}
    </details>
  );
}
