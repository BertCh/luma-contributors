// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  normalizeLocalMoranSettings,
  type LocalMoranSettings
} from './analysis-local-moran-settings';
import {PointTopologyControls, usePointTopology} from './analysis-point-topology-controls';
import {PointCoordinateControls, usePointCoordinates} from './analysis-point-coordinate-controls';
import {validatePointCoordinateColumns} from './analysis-point-coordinate-settings';
import {ResidentConsumerNotice} from './resident-consumer-notice';

/** Separates configuration from the resident runtime so applying settings is one atomic action. */
export function LocalMoranPanel({
  columns,
  rowCount,
  settings,
  onApply
}: {
  columns: readonly string[];
  rowCount: number;
  settings?: LocalMoranSettings;
  onApply(settings: LocalMoranSettings): void;
}) {
  const coordinates = usePointCoordinates(settings);
  const topology = usePointTopology(settings, rowCount);
  const [minimumNeighbors, setMinimumNeighbors] = useState(String(settings?.minimumNeighbors ?? 4));
  const [significanceLevel, setSignificanceLevel] = useState(
    String(settings?.significanceLevel ?? 0.05)
  );
  const [islands, setIslands] = useState<LocalMoranSettings['islands']>(
    settings?.islands ?? 'exclude'
  );
  const [multipleTesting, setMultipleTesting] = useState<LocalMoranSettings['multipleTesting']>(
    settings?.multipleTesting ?? 'benjamini-hochberg'
  );
  const [inference, setInference] = useState<NonNullable<LocalMoranSettings['inference']>>(
    settings?.inference ?? 'analytic-conditional-normal-two-sided'
  );
  const [permutationCount, setPermutationCount] = useState(
    String(settings?.permutationCount ?? 999)
  );
  const [seed, setSeed] = useState(String(settings?.seed ?? 0));
  const [secondColumn, setSecondColumn] = useState(settings?.secondColumn ?? '');
  const [variableColumns, setVariableColumns] = useState<string[]>(settings?.variableColumns ?? []);
  const multivariate = variableColumns.length > 0;
  const bivariate = secondColumn !== '' && !multivariate;
  const permutationInference = inference === 'permutation-conditional-two-sided';
  const [error, setError] = useState('');
  useEffect(() => {
    setMinimumNeighbors(String(settings?.minimumNeighbors ?? 4));
    setSignificanceLevel(String(settings?.significanceLevel ?? 0.05));
    setIslands(settings?.islands ?? 'exclude');
    setMultipleTesting(settings?.multipleTesting ?? 'benjamini-hochberg');
    setInference(settings?.inference ?? 'analytic-conditional-normal-two-sided');
    setPermutationCount(String(settings?.permutationCount ?? 999));
    setSeed(String(settings?.seed ?? 0));
    setSecondColumn(settings?.secondColumn ?? '');
    setVariableColumns(settings?.variableColumns ?? []);
    setError('');
  }, [
    settings?.variableColumns?.join('\u0000'),
    settings?.secondColumn,
    settings?.minimumNeighbors,
    settings?.significanceLevel,
    settings?.islands,
    settings?.multipleTesting,
    settings?.inference,
    settings?.permutationCount,
    settings?.seed
  ]);
  return (
    <details className="analysis-local-moran-settings">
      <summary>
        Configure Local Moran on{' '}
        {coordinates.draft.coordinateSpace === 'planar' ? 'projected' : 'geographic'} points
      </summary>
      <p className="hint">
        Analyze the numeric field at each original point. Choose geographic or projected meter
        coordinates for binary spatial weights.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            const coordinateSettings = coordinates.getSettings();
            validatePointCoordinateColumns(coordinateSettings, columns);
            onApply(
              normalizeLocalMoranSettings(
                {
                  ...coordinateSettings,
                  ...topology.getSettings(),
                  minimumNeighbors: Number(minimumNeighbors),
                  significanceLevel: Number(significanceLevel),
                  islands,
                  multipleTesting,
                  inference:
                    bivariate || multivariate ? 'permutation-conditional-two-sided' : inference,
                  ...(bivariate ? {secondColumn} : {}),
                  ...(multivariate ? {variableColumns} : {}),
                  ...(permutationInference || bivariate || multivariate
                    ? {permutationCount: Number(permutationCount), seed: Number(seed)}
                    : {})
                },
                rowCount
              )
            );
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <PointCoordinateControls
          value={coordinates.draft}
          onChange={coordinates.setDraft}
          columns={columns}
          idPrefix="analysis-moran"
          labelPrefix="Local Moran"
        />
        <label htmlFor="analysis-moran-mode">Local Moran variables</label>
        <select
          id="analysis-moran-mode"
          value={multivariate ? 'multivariate' : bivariate ? 'bivariate' : 'univariate'}
          onChange={event => {
            if (event.target.value === 'multivariate') {
              setSecondColumn('');
              setVariableColumns(columns.slice(0, 2));
              setInference('permutation-conditional-two-sided');
              setMinimumNeighbors('1');
              if (multipleTesting !== 'none' && multipleTesting !== 'bonferroni')
                setMultipleTesting('none');
            } else if (event.target.value === 'bivariate') {
              setVariableColumns([]);
              setSecondColumn(columns[0] ?? '');
              setInference('permutation-conditional-two-sided');
              setMinimumNeighbors('1');
            } else {
              setSecondColumn('');
              setVariableColumns([]);
            }
          }}
        >
          <option value="univariate">Univariate (selected field)</option>
          <option value="bivariate">Bivariate (lag of a second field)</option>
          <option value="multivariate">
            Multivariate local Geary (2 to 8 fields, tiled above 256 rows)
          </option>
        </select>
        {multivariate && (
          <fieldset id="analysis-moran-variables">
            <legend>Local Geary variables (2 to 8)</legend>
            {columns.map(column => (
              <label key={column}>
                <input
                  type="checkbox"
                  checked={variableColumns.includes(column)}
                  onChange={event =>
                    setVariableColumns(
                      event.target.checked
                        ? [...variableColumns, column].slice(0, 8)
                        : variableColumns.filter(name => name !== column)
                    )
                  }
                />{' '}
                {column}
              </label>
            ))}
            <p className="hint">
              Variables are standardized jointly; whole neighbor vectors are drawn in each
              conditional permutation. Outputs are the mean local Geary statistic, p-value and
              cluster class (1 similarity, 2 dissimilarity). Use at most 9999 permutations and no or
              Bonferroni correction; rows with more than 64 active neighbors stay untested.
            </p>
          </fieldset>
        )}
        {bivariate && (
          <>
            <label htmlFor="analysis-moran-second-column">Local Moran second variable</label>
            <select
              id="analysis-moran-second-column"
              value={secondColumn}
              onChange={event => setSecondColumn(event.target.value)}
            >
              {columns.map(column => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
            <p className="hint">
              Bivariate Local Moran compares the selected field at each point with the weighted
              neighbor average of the second field. Only conditional permutation inference is
              available; outputs are the statistic, p-values, quadrants and clusters.
            </p>
          </>
        )}
        <label htmlFor="analysis-moran-inference">Local Moran inference</label>
        <select
          id="analysis-moran-inference"
          value={inference}
          disabled={bivariate || multivariate}
          onChange={event => {
            const selected =
              event.target.value === 'permutation-conditional-two-sided'
                ? 'permutation-conditional-two-sided'
                : 'analytic-conditional-normal-two-sided';
            setInference(selected);
            if (selected === 'permutation-conditional-two-sided') setMinimumNeighbors('1');
          }}
        >
          <option value="analytic-conditional-normal-two-sided">
            Analytic conditional normal · two-sided
          </option>
          <option value="permutation-conditional-two-sided" disabled={rowCount >= 2 ** 24}>
            Conditional permutations · two-sided
          </option>
        </select>
        {permutationInference && (
          <>
            <label htmlFor="analysis-moran-permutations">Local Moran permutations</label>
            <input
              id="analysis-moran-permutations"
              type="number"
              min="1"
              max="9999"
              step="1"
              required
              value={permutationCount}
              onChange={event => setPermutationCount(event.target.value)}
            />
            <label htmlFor="analysis-moran-seed">Local Moran permutation seed</label>
            <input
              id="analysis-moran-seed"
              type="number"
              min="0"
              max="4294967295"
              step="1"
              required
              value={seed}
              onChange={event => setSeed(event.target.value)}
            />
          </>
        )}
        <PointTopologyControls
          topology={topology}
          coordinateSpace={coordinates.draft.coordinateSpace}
          idPrefix="analysis-moran"
          labelPrefix="Local Moran"
          nearestLabel="Local Moran nearest neighbors"
          permutationMaximumNeighbors={
            permutationInference || bivariate || multivariate ? 64 : undefined
          }
        />
        <label htmlFor="analysis-moran-minimum">Local Moran minimum neighbors</label>
        <input
          id="analysis-moran-minimum"
          type="number"
          min="1"
          step="1"
          required
          readOnly={permutationInference}
          value={minimumNeighbors}
          onChange={event => setMinimumNeighbors(event.target.value)}
        />
        <label htmlFor="analysis-moran-significance">Local Moran significance level</label>
        <input
          id="analysis-moran-significance"
          type="number"
          min="0"
          max="1"
          step="any"
          required
          value={significanceLevel}
          onChange={event => setSignificanceLevel(event.target.value)}
        />
        <label htmlFor="analysis-moran-islands">Local Moran isolated points</label>
        <select
          id="analysis-moran-islands"
          value={islands}
          onChange={event => setIslands(event.target.value === 'keep' ? 'keep' : 'exclude')}
        >
          <option value="exclude">Exclude from global moments</option>
          <option value="keep">Keep in global moments</option>
        </select>
        <label htmlFor="analysis-moran-correction">Local Moran correction</label>
        <select
          id="analysis-moran-correction"
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
        {permutationInference ? (
          <>
            <p className="hint">
              Two-sided conditional randomization keeps the focus value fixed and samples distinct
              other observations for its active neighbor slots. With P permutations, the raw p-value
              resolution has a floor of 2 / (P + 1); all-tied simulations give p = 1.
              Multiple-testing corrections can require a larger permutation count to resolve
              significance.
            </p>
            <p className="hint">
              The table z-score remains a separate analytic diagnostic. Permutation p-values and
              their corrected values determine the displayed quadrants. The seed reproduces this
              implementation's draws; GeoDa random-stream parity is not claimed.
            </p>
            <p className="hint">
              Permutation inference supports fewer than 16.7 million source rows (tiled GPU
              dispatches, results independent of tiling) and 9,999 permutations. Minimum neighbors
              is fixed at 1; rows with no included neighbors are not tested.
            </p>
          </>
        ) : (
          <p className="hint">
            Inference uses a two-sided normal approximation to the conditional randomization
            moments, without permutations. Small neighborhoods and skewed values can make this
            approximation poor. The table z-score is the analytic conditional-normal diagnostic.
          </p>
        )}
        <p className="hint">
          Quadrants identify high–high, low–high, low–low and high–low association only where the
          corrected p-value meets the significance level. Isolated points have no inferred quadrant;
          keeping them retains their values in global moments with local I and spatial lag zero.
        </p>
        <p className="hint">
          Weights stay binary and are not normalized to sum to one. Selection filters the fixed
          source-wide point graph without finding replacement neighbors. Constant values, too few
          neighbors, or undefined variance have no analytic score.
        </p>
        <button type="submit">Apply neighbors and run Local Moran</button>
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
