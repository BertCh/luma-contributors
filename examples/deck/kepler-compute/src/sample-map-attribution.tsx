// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  describeSampleLicense,
  extractSampleAttribution,
  SAMPLE_LICENSE_URL,
  SAMPLE_REPOSITORY_LICENSE,
  type SampleMap
} from './sample-map-catalog';
import {listSampleMapAttributions} from './sample-map-provenance';
import {REMOTE_SAMPLE_RECOVERY_NOTICE} from './sample-map-lifecycle';

export function SampleAttribution({sample}: {sample: SampleMap}) {
  return (
    <div data-testid={`sample-attribution-${sample.id}`}>
      <p>
        {extractSampleAttribution(sample.detail).map((part, index) =>
          part.url ? (
            <a
              key={index}
              href={part.url}
              target="_blank"
              rel="noreferrer"
              style={{color: 'inherit'}}
            >
              {part.text}
            </a>
          ) : (
            <span key={index}>{part.text}</span>
          )
        )}
      </p>
      <p>
        {describeSampleLicense(sample)} Repository licence:{' '}
        <a href={SAMPLE_LICENSE_URL} target="_blank" rel="noreferrer" style={{color: 'inherit'}}>
          {SAMPLE_REPOSITORY_LICENSE}
        </a>
        .
      </p>
    </div>
  );
}

export function SampleDatasetAttributions({
  datasets
}: {
  datasets: readonly {id: string; metadata?: unknown}[];
}) {
  return (
    <>
      {listSampleMapAttributions(datasets).map(provenance => (
        // Collapsed to its summary line so the side panel keeps Kepler's tabs in view; a remote
        // sample stays open because its recovery notice must be read.
        <details
          key={provenance.datasetId}
          open={!provenance.sample.dataUrl}
          data-testid="retained-sample-attribution"
          data-dataset-id={provenance.datasetId}
        >
          <summary>Dataset attribution: {provenance.sample.label}</summary>
          <SampleAttribution sample={provenance.sample} />
          <p>{provenance.licenseNotice}</p>
          {!provenance.sample.dataUrl ? (
            <p role="status" data-testid="sample-recovery-notice">
              {REMOTE_SAMPLE_RECOVERY_NOTICE}
            </p>
          ) : null}
        </details>
      ))}
    </>
  );
}
