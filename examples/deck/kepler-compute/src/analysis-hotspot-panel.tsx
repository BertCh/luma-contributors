// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {AnalysisRuntime, HotspotSettings} from './analysis-runtime';
import {ResidentConsumerNotice} from './resident-consumer-notice';

/** Lattice topology must be chosen explicitly; source coordinates never imply row adjacency. */
export function HotspotPanel({analysis}: {analysis: AnalysisRuntime}) {
  const saved = analysis.settings.hotspot;
  const [width, setWidth] = useState(saved ? String(saved.width) : '');
  const [height, setHeight] = useState(saved ? String(saved.height) : '');
  const [criterion, setCriterion] = useState<HotspotSettings['criterion']>(
    saved?.criterion ?? 'queen'
  );
  const [radius, setRadius] = useState(String(saved?.radius ?? 1));
  const [minimumNeighbors, setMinimumNeighbors] = useState(String(saved?.minimumNeighbors ?? 8));
  const [selfWeight, setSelfWeight] = useState(String(saved?.selfWeight ?? 1));
  const [multipleTesting, setMultipleTesting] = useState<HotspotSettings['multipleTesting']>(
    saved?.multipleTesting ?? 'benjamini-hochberg'
  );
  const [error, setError] = useState('');
  useEffect(() => {
    const settings = analysis.settings.hotspot;
    setWidth(settings ? String(settings.width) : '');
    setHeight(settings ? String(settings.height) : '');
    setCriterion(settings?.criterion ?? 'queen');
    setRadius(String(settings?.radius ?? 1));
    setMinimumNeighbors(String(settings?.minimumNeighbors ?? 8));
    setSelfWeight(String(settings?.selfWeight ?? 1));
    setMultipleTesting(settings?.multipleTesting ?? 'benjamini-hochberg');
    setError('');
  }, [
    analysis,
    saved?.width,
    saved?.height,
    saved?.criterion,
    saved?.radius,
    saved?.minimumNeighbors,
    saved?.selfWeight,
    saved?.multipleTesting
  ]);
  return (
    <details className="analysis-hotspot-settings">
      <summary>Configure hot spots on a row-major grid</summary>
      <p className="hint">
        Use a complete rectangular lattice ordered left to right, then top to bottom. Source row =
        row × width + column. Geographic coordinates do not define these neighbors.
      </p>
      <p className="hint">
        Width × height must equal all {analysis.rowCount.toLocaleString()} source rows. Filters
        remove rows from the analysis without reordering the grid.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            analysis.configureHotspots({
              width: Number(width),
              height: Number(height),
              criterion,
              radius: Number(radius),
              minimumNeighbors: Number(minimumNeighbors),
              selfWeight: Number(selfWeight),
              multipleTesting
            });
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <label htmlFor="analysis-grid-width">Grid width</label>
        <input
          id="analysis-grid-width"
          type="number"
          min="1"
          step="1"
          required
          value={width}
          onChange={event => setWidth(event.target.value)}
        />
        <label htmlFor="analysis-grid-height">Grid height</label>
        <input
          id="analysis-grid-height"
          type="number"
          min="1"
          step="1"
          required
          value={height}
          onChange={event => setHeight(event.target.value)}
        />
        <label htmlFor="analysis-grid-criterion">Neighborhood</label>
        <select
          id="analysis-grid-criterion"
          value={criterion}
          onChange={event => setCriterion(event.target.value === 'rook' ? 'rook' : 'queen')}
        >
          <option value="queen">Queen · includes diagonals</option>
          <option value="rook">Rook · Manhattan distance</option>
        </select>
        <label htmlFor="analysis-grid-radius">Radius in cells</label>
        <input
          id="analysis-grid-radius"
          type="number"
          min="1"
          max="32"
          step="1"
          required
          value={radius}
          onChange={event => setRadius(event.target.value)}
        />
        <label htmlFor="analysis-grid-minimum-neighbors">Minimum neighbors</label>
        <input
          id="analysis-grid-minimum-neighbors"
          type="number"
          min="0"
          step="1"
          required
          value={minimumNeighbors}
          onChange={event => setMinimumNeighbors(event.target.value)}
        />
        <label htmlFor="analysis-grid-self-weight">Self weight</label>
        <input
          id="analysis-grid-self-weight"
          type="number"
          min="0"
          step="any"
          required
          value={selfWeight}
          onChange={event => setSelfWeight(event.target.value)}
        />
        <label htmlFor="analysis-grid-correction">Multiple-testing correction</label>
        <select
          id="analysis-grid-correction"
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
          Analytic two-sided inference. Rows below the neighbor minimum are undefined. Table
          p-values are unadjusted; significance bins use the chosen correction. Map colors show the
          Gi* z-score.
        </p>
        <button type="submit">Apply grid and run hot spots</button>
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
