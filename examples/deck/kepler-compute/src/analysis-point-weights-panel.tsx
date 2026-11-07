// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {AnalysisRuntime, PointWeightsSettings} from './analysis-runtime';
import {PointTopologyControls, usePointTopology} from './analysis-point-topology-controls';
import {PointCoordinateControls, usePointCoordinates} from './analysis-point-coordinate-controls';
import {validatePointCoordinateColumns} from './analysis-point-coordinate-settings';
import {ResidentConsumerNotice} from './resident-consumer-notice';

/** Coordinate fields and units are explicit: numeric columns do not imply a geographic CRS. */
export function PointWeightsPanel({analysis}: {analysis: AnalysisRuntime}) {
  const saved = analysis.settings.pointWeights;
  const coordinates = usePointCoordinates(saved, analysis.datasetId);
  const topology = usePointTopology(saved, analysis.rowCount, analysis.datasetId);
  const [scale, setScale] = useState(String(saved?.scale ?? 1));
  const [islandPolicy, setIslandPolicy] = useState<PointWeightsSettings['islandPolicy']>(
    saved?.islandPolicy ?? 'undefined'
  );
  const [error, setError] = useState('');
  useEffect(() => {
    setScale(String(saved?.scale ?? 1));
    setIslandPolicy(saved?.islandPolicy ?? 'undefined');
    setError('');
  }, [analysis.datasetId, saved?.scale, saved?.islandPolicy]);
  return (
    <details className="analysis-point-weights-settings">
      <summary>
        Configure {coordinates.draft.coordinateSpace === 'planar' ? 'projected' : 'geographic'}{' '}
        neighbors and spatial rates
      </summary>
      <p className="hint">
        Choose explicit geographic or projected meter coordinates for the neighbor graph. The
        numeric field above supplies events.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            const coordinateSettings = coordinates.getSettings();
            validatePointCoordinateColumns(coordinateSettings, analysis.availableColumns);
            analysis.configurePointWeights({
              ...coordinateSettings,
              ...topology.getSettings(),
              scale: Number(scale),
              islandPolicy
            });
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <label htmlFor="analysis-point-population">Spatial rate population field</label>
        <select
          id="analysis-point-population"
          value={analysis.settings.denominatorColumn}
          onChange={event => {
            try {
              analysis.setDenominatorColumn(event.target.value);
              setError('');
            } catch (failure) {
              setError(failure instanceof Error ? failure.message : String(failure));
            }
          }}
        >
          {analysis.availableColumns.map(column => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        <PointCoordinateControls
          value={coordinates.draft}
          onChange={coordinates.setDraft}
          columns={analysis.availableColumns}
          idPrefix="analysis-point"
          labelPrefix="Neighbor"
        />
        <PointTopologyControls
          topology={topology}
          coordinateSpace={coordinates.draft.coordinateSpace}
          idPrefix="analysis-point"
          labelPrefix="Neighbor"
          nearestLabel="Nearest neighbors"
        />
        <label htmlFor="analysis-point-scale">Rate scale</label>
        <input
          id="analysis-point-scale"
          type="number"
          min="0"
          step="any"
          required
          value={scale}
          onChange={event => setScale(event.target.value)}
        />
        <label htmlFor="analysis-point-islands">Rows without eligible neighbors</label>
        <select
          id="analysis-point-islands"
          value={islandPolicy}
          onChange={event =>
            setIslandPolicy(event.target.value === 'own-rate' ? 'own-rate' : 'undefined')
          }
        >
          <option value="undefined">Undefined</option>
          <option value="own-rate">Use the row's own rate</option>
        </select>
        <p className="hint">
          The directed neighbor graph uses all rows with valid coordinates. Each rate pools its own
          events and population with eligible neighbors, then multiplies by the scale. Filtering
          removes observations from selected-population rates without replacing their neighbors.
        </p>
        <p className="hint">
          Missing or invalid coordinates have no links and always produce undefined rates.
          Coordinate validity and distance precision follow the chosen analysis coordinate system.
        </p>
        <button type="submit">Apply neighbors and run spatial rates</button>
        <p className="hint">
          Dense clusters can make the first exact neighbor build slow. Filtering and compatible
          analysis settings reuse the completed neighbor graph.
        </p>
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
