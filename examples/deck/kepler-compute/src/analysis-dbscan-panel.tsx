// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {
  MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS,
  MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES,
  type KeplerDBSCANSettings
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {ResidentConsumerNotice} from './resident-consumer-notice';

/** Configure geographic density clustering as an atomic recipe update. */
export function DBSCANPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.dbscan;
  const summary = analysis.summary?.dbscan;
  return (
    <>
      <DBSCANControls key={JSON.stringify(settings)} analysis={analysis} settings={settings} />
      {summary && (
        <p className="hint">
          DBSCAN: {summary.clusterCount} clusters · {summary.noiseCount} noise points ·{' '}
          {summary.validCount} valid observations · {summary.excludedCount} excluded observations.
          {summary.complete
            ? ' Complete traversal.'
            : ' Incomplete traversal; results unavailable.'}
        </p>
      )}
    </>
  );
}

function DBSCANControls({
  analysis,
  settings
}: {
  analysis: AnalysisRuntime;
  settings?: KeplerDBSCANSettings;
}) {
  const [longitudeColumn, setLongitudeColumn] = useState(settings?.longitudeColumn ?? '');
  const [latitudeColumn, setLatitudeColumn] = useState(settings?.latitudeColumn ?? '');
  const [epsilon, setEpsilon] = useState(String(settings?.epsilon ?? 1000));
  const [minimumSamples, setMinimumSamples] = useState(String(settings?.minimumSamples ?? 5));
  const [error, setError] = useState('');
  return (
    <details className="analysis-dbscan-settings">
      <summary>Configure DBSCAN on geographic points</summary>
      <p className="hint">
        Group nearby dense observations using spherical great-circle distance. Coordinates are
        longitude and latitude in degrees; the neighborhood radius is in meters. The numeric
        analysis field does not affect clustering.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            analysis.configureDBSCAN({
              longitudeColumn,
              latitudeColumn,
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              units: 'meters',
              epsilon: Number(epsilon),
              minimumSamples: Number(minimumSamples)
            });
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <label htmlFor="analysis-dbscan-longitude">DBSCAN longitude field</label>
        <select
          id="analysis-dbscan-longitude"
          aria-label="DBSCAN longitude field"
          required
          value={longitudeColumn}
          onChange={event => setLongitudeColumn(event.target.value)}
        >
          <option value="">Choose longitude in degrees</option>
          {analysis.availableColumns.map(column => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        <label htmlFor="analysis-dbscan-latitude">DBSCAN latitude field</label>
        <select
          id="analysis-dbscan-latitude"
          aria-label="DBSCAN latitude field"
          required
          value={latitudeColumn}
          onChange={event => setLatitudeColumn(event.target.value)}
        >
          <option value="">Choose latitude in degrees</option>
          {analysis.availableColumns.map(column => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        <label htmlFor="analysis-dbscan-epsilon">DBSCAN neighborhood radius in meters</label>
        <input
          id="analysis-dbscan-epsilon"
          type="number"
          required
          min="0"
          step="any"
          value={epsilon}
          onChange={event => setEpsilon(event.target.value)}
        />
        <label htmlFor="analysis-dbscan-minimum">DBSCAN minimum samples including self</label>
        <input
          id="analysis-dbscan-minimum"
          type="number"
          required
          min="1"
          max={MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES}
          step="1"
          value={minimumSamples}
          onChange={event => setMinimumSamples(event.target.value)}
        />
        <p className="hint">
          Density counts include the focus point. Coordinates outside longitude/latitude bounds,
          missing coordinates and nonfinite coordinates are excluded. Selecting a population
          recomputes its clusters; excluded rows cannot connect clusters.
        </p>
        <p className="hint">
          Cluster IDs belong to original source rows. IDs repeat deterministically for the same
          source order, settings and population, but can change after filtering. Noise has value −1;
          excluded rows are missing. Gray indicates noise; cluster colors repeat after 15 clusters,
          so use the table IDs to distinguish them.
        </p>
        <p className="hint">
          At most {MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS.toLocaleString()} source rows and{' '}
          {MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES} minimum samples. This bounded exact computation
          uses OPTICS with DBSCAN extraction and completes every source-row visit. It does not
          perform native viewport clustering.
        </p>
        <button
          type="submit"
          disabled={analysis.rowCount < 1 || analysis.rowCount > MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS}
        >
          Apply radius and run DBSCAN
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
