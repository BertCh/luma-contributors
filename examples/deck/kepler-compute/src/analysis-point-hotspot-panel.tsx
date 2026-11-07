// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {AnalysisRuntime, PointHotspotSettings} from './analysis-runtime';
import {PointTopologyControls, usePointTopology} from './analysis-point-topology-controls';
import {PointCoordinateControls, usePointCoordinates} from './analysis-point-coordinate-controls';
import {validatePointCoordinateColumns} from './analysis-point-coordinate-settings';
import {ResidentConsumerNotice} from './resident-consumer-notice';

/** Point observations retain source row identity; no raster cells or lattice order are inferred. */
export function PointHotspotPanel({analysis}: {analysis: AnalysisRuntime}) {
  const saved = analysis.settings.pointHotspot;
  const rateWeights = analysis.settings.pointWeights;
  const coordinates = usePointCoordinates(saved ?? rateWeights, analysis.datasetId);
  const topology = usePointTopology(saved ?? rateWeights, analysis.rowCount, analysis.datasetId);
  const [minimumNeighbors, setMinimumNeighbors] = useState(String(saved?.minimumNeighbors ?? 8));
  const [selfWeight, setSelfWeight] = useState(String(saved?.selfWeight ?? 1));
  const [multipleTesting, setMultipleTesting] = useState<PointHotspotSettings['multipleTesting']>(
    saved?.multipleTesting ?? 'benjamini-hochberg'
  );
  const [islands, setIslands] = useState<PointHotspotSettings['islands']>(
    saved?.islands ?? 'exclude'
  );
  const [error, setError] = useState('');
  useEffect(() => {
    setMinimumNeighbors(String(saved?.minimumNeighbors ?? 8));
    setSelfWeight(String(saved?.selfWeight ?? 1));
    setMultipleTesting(saved?.multipleTesting ?? 'benjamini-hochberg');
    setIslands(saved?.islands ?? 'exclude');
    setError('');
  }, [
    analysis.datasetId,
    saved?.minimumNeighbors,
    saved?.selfWeight,
    saved?.multipleTesting,
    saved?.islands
  ]);
  return (
    <details className="analysis-point-hotspot-settings">
      <summary>
        Configure hot spots on{' '}
        {coordinates.draft.coordinateSpace === 'planar' ? 'projected' : 'geographic'} points
      </summary>
      <p className="hint">
        Analyze the numeric field at each original point using binary weights. Choose geographic
        coordinates or already projected meter coordinates for analysis.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            const coordinateSettings = coordinates.getSettings();
            validatePointCoordinateColumns(coordinateSettings, analysis.availableColumns);
            analysis.configurePointHotspots({
              ...coordinateSettings,
              ...topology.getSettings(),
              minimumNeighbors: Number(minimumNeighbors),
              selfWeight: Number(selfWeight),
              multipleTesting,
              islands
            });
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <PointCoordinateControls
          value={coordinates.draft}
          onChange={coordinates.setDraft}
          columns={analysis.availableColumns}
          idPrefix="analysis-point-hotspot"
          labelPrefix="Hot-spot"
        />
        <PointTopologyControls
          topology={topology}
          coordinateSpace={coordinates.draft.coordinateSpace}
          idPrefix="analysis-point-hotspot"
          labelPrefix="Hot-spot"
          nearestLabel="Hot-spot nearest neighbors"
        />
        <label htmlFor="analysis-point-hotspot-minimum">Hot-spot minimum neighbors</label>
        <input
          id="analysis-point-hotspot-minimum"
          type="number"
          min="1"
          step="1"
          required
          value={minimumNeighbors}
          onChange={event => setMinimumNeighbors(event.target.value)}
        />
        <label htmlFor="analysis-point-hotspot-self">Hot-spot self weight</label>
        <input
          id="analysis-point-hotspot-self"
          type="number"
          min="0"
          step="any"
          required
          value={selfWeight}
          onChange={event => setSelfWeight(event.target.value)}
        />
        <label htmlFor="analysis-point-hotspot-islands">Isolated points in global statistics</label>
        <select
          id="analysis-point-hotspot-islands"
          value={islands}
          onChange={event => setIslands(event.target.value === 'keep' ? 'keep' : 'exclude')}
        >
          <option value="exclude">Exclude from global moments</option>
          <option value="keep">Keep in global moments</option>
        </select>
        <label htmlFor="analysis-point-hotspot-correction">Point hot-spot correction</label>
        <select
          id="analysis-point-hotspot-correction"
          value={multipleTesting}
          onChange={event => {
            const value = event.target.value;
            if (
              value === 'none' ||
              value === 'bonferroni' ||
              value === 'holm' ||
              value === 'benjamini-hochberg' ||
              value === 'benjamini-yekutieli'
            )
              setMultipleTesting(value);
          }}
        >
          <option value="benjamini-hochberg">Benjamini–Hochberg</option>
          <option value="benjamini-yekutieli">Benjamini–Yekutieli</option>
          <option value="bonferroni">Bonferroni</option>
          <option value="holm">Holm</option>
          <option value="none">None</option>
        </select>
        <p className="hint">
          Selected-population analysis removes observations from the fixed source neighbor graph
          without replacing neighbors. Isolated points never receive a score. The minimum applies to
          eligible neighbors after filtering; eight is a common analytic-inference starting point.
        </p>
        <p className="hint">
          Self weight 1 gives Gi*; 0 excludes the focal value for Gi. Inference is analytic and
          two-sided. Table p-values are unadjusted; significance bins use the selected correction.
          Map colors show z-scores. Missing or out-of-range coordinates remain undefined.
        </p>
        <button type="submit">Apply neighbors and run point hot spots</button>
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
