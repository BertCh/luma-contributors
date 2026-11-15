// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {getSampleUnsupportedReason, type SampleMap} from './sample-map-catalog';
import {SampleAttribution} from './sample-map-attribution';
import {fetchSampleMaps} from './sample-map-loader';

export function SampleMapsPanel({
  loadingId,
  error,
  onLoad
}: {
  loadingId?: string;
  error: string;
  onLoad(sample: SampleMap): Promise<void>;
}) {
  const [samples, setSamples] = useState<SampleMap[] | null>(null);
  const [catalogError, setCatalogError] = useState('');
  const [requestRevision, setRequestRevision] = useState(0);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setCatalogError('');
    void fetchSampleMaps(fetch, controller.signal).then(
      result => {
        if (active) setSamples(result);
      },
      failure => {
        if (active) setCatalogError(String(failure));
      }
    );
    return () => {
      active = false;
      controller.abort();
    };
  }, [requestRevision]);
  return (
    <section aria-label="Kepler sample maps" aria-busy={Boolean(loadingId)}>
      <h2>Sample maps</h2>
      <p>
        Public Kepler demo maps, fetched from GitHub when loaded. Loading applies the sample’s
        layers, filters and camera to this map. Source data remains subject to its provider’s terms.
      </p>
      {!samples && !catalogError ? <p role="status">Loading sample catalog…</p> : null}
      {catalogError ? <p role="alert">Could not load the sample catalog: {catalogError}</p> : null}
      {catalogError ? (
        <button type="button" onClick={() => setRequestRevision(value => value + 1)}>
          Retry sample catalog
        </button>
      ) : null}
      {error ? <p role="alert">Sample map could not load: {error}</p> : null}
      {samples?.map(sample => {
        const unsupported = getSampleUnsupportedReason(sample);
        return (
          <article key={sample.id} data-testid={`sample-map-${sample.id}`}>
            <h3>{sample.label}</h3>
            <p>{sample.description}</p>
            <p>
              {sample.size === undefined
                ? 'Size unspecified'
                : `${sample.size.toLocaleString()} catalog entries`}
              {!sample.visible ? ' · Hidden in the upstream gallery' : ''}
              {!sample.dataUrl ? ' · Config-driven remote map' : ''}
            </p>
            <SampleAttribution sample={sample} />
            {unsupported ? <p>Unsupported: {unsupported}</p> : null}
            <button
              type="button"
              disabled={Boolean(loadingId) || Boolean(unsupported)}
              onClick={() => {
                void onLoad(sample).catch(() => {});
              }}
            >
              {loadingId === sample.id ? 'Loading…' : `Load ${sample.label}`}
            </button>
          </article>
        );
      })}
    </section>
  );
}
