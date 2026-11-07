// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  MAXIMUM_KEPLER_POLYGON_ROWS,
  MAXIMUM_KEPLER_POLYGON_VERTICES,
  MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY,
  MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY,
  type KeplerPolygonAnalysisOperation,
  type KeplerPolygonWeightsSettings,
  type KeplerMultipleTestingMethod
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {
  PolygonQuantileControls,
  type PolygonQuantileDraft
} from './analysis-polygon-quantile-controls';
import {ResidentConsumerNotice} from './resident-consumer-notice';

import {clamp} from '@math.gl/core';

const CORRECTIONS = [
  {value: 'benjamini-hochberg', label: 'Benjamini–Hochberg'},
  {value: 'benjamini-yekutieli', label: 'Benjamini–Yekutieli'},
  {value: 'bonferroni', label: 'Bonferroni'},
  {value: 'holm', label: 'Holm'},
  {value: 'none', label: 'None'}
] as const;

type PolygonDraft = {
  kind: KeplerPolygonAnalysisOperation['kind'];
  geometryColumn: string;
  criterion: 'queen' | 'rook';
  neighborCapacity: string;
  pairCapacity: string;
  populationColumn: string;
  scale: string;
  islandPolicy: 'undefined' | 'own-rate';
  minimumNeighbors: string;
  selfWeight: string;
  multipleTesting: KeplerMultipleTestingMethod;
  islands: 'exclude' | 'keep';
  significanceLevel: string;
  inference: 'analytic-conditional-normal-two-sided' | 'permutation-conditional-two-sided';
  permutationCount: string;
  seed: string;
};

/** Explicit original-coordinate contiguity; all statistics keep original source feature identity. */
export function PolygonPanel({analysis}: {analysis: AnalysisRuntime}) {
  const saved = analysis.settings.polygon;
  const savedKind = saved?.kind ?? 'local-moran-polygons';
  const savedGeometry = saved?.weights.geometryColumn ?? analysis.availablePolygonColumns[0] ?? '';
  const savedCriterion = saved?.weights.criterion ?? 'queen';
  const savedNeighbors =
    saved?.weights.neighborCapacity ??
    clamp(analysis.rowCount * 8, 1, MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY);
  const savedPairs =
    saved?.weights.pairCapacity ??
    clamp(analysis.rowCount * 32, 1, MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY);
  const savedPopulation =
    saved?.kind === 'spatial-rate-polygons'
      ? saved.populations
      : analysis.settings.denominatorColumn;
  const savedScale = saved?.kind === 'spatial-rate-polygons' ? saved.scale : 1;
  const savedIslandPolicy =
    saved?.kind === 'spatial-rate-polygons' ? saved.islandPolicy : 'undefined';
  const savedHasAnalyticControls =
    saved?.kind === 'hot-spots-polygons' || saved?.kind === 'local-moran-polygons';
  const savedMinimum = savedHasAnalyticControls ? saved.minimumNeighbors : 4;
  const savedSelfWeight = saved?.kind === 'hot-spots-polygons' ? saved.selfWeight : 1;
  const savedCorrection = savedHasAnalyticControls ? saved.multipleTesting : 'benjamini-hochberg';
  const savedIslands = savedHasAnalyticControls ? saved.islands : 'exclude';
  const savedSignificance = saved?.kind === 'local-moran-polygons' ? saved.significanceLevel : 0.05;
  const savedInference =
    saved?.kind === 'local-moran-polygons'
      ? saved.inference
      : 'analytic-conditional-normal-two-sided';
  const savedPermutations =
    saved?.kind === 'local-moran-polygons' &&
    saved.inference === 'permutation-conditional-two-sided'
      ? saved.permutationCount
      : 999;
  const savedSeed =
    saved?.kind === 'local-moran-polygons' &&
    saved.inference === 'permutation-conditional-two-sided'
      ? saved.seed
      : 0;
  const savedQuantile = saved?.kind === 'quantile-lisa-polygons' ? saved : undefined;
  const savedQuantileCount = savedQuantile?.quantileCount ?? 2;
  const savedSelectedQuantile = savedQuantile?.selectedQuantile ?? 2;
  const savedQuantileSeed = savedQuantile?.seed ?? 314159;
  const savedQuantilePermutations = savedQuantile?.permutationCount ?? 199;
  const savedQuantileSignificance = savedQuantile?.significanceLevel ?? 0.05;
  const makeQuantileDraft = (): PolygonQuantileDraft => ({
    quantileCount: String(savedQuantileCount),
    selectedQuantile: String(savedSelectedQuantile),
    seed: String(savedQuantileSeed),
    permutationCount: String(savedQuantilePermutations),
    significanceLevel: String(savedQuantileSignificance)
  });
  const makeDraft = (): PolygonDraft => ({
    kind: savedKind,
    geometryColumn: savedGeometry,
    criterion: savedCriterion,
    neighborCapacity: String(savedNeighbors),
    pairCapacity: String(savedPairs),
    populationColumn: savedPopulation,
    scale: String(savedScale),
    islandPolicy: savedIslandPolicy,
    minimumNeighbors: String(savedMinimum),
    selfWeight: String(savedSelfWeight),
    multipleTesting: savedCorrection,
    islands: savedIslands,
    significanceLevel: String(savedSignificance),
    inference: savedInference,
    permutationCount: String(savedPermutations),
    seed: String(savedSeed)
  });
  const [draft, setDraft] = useState<PolygonDraft>(makeDraft);
  const [quantileDraft, setQuantileDraft] = useState<PolygonQuantileDraft>(makeQuantileDraft);
  const [error, setError] = useState('');
  useEffect(() => {
    setDraft(makeDraft());
    setError('');
  }, [
    analysis.datasetId,
    savedKind,
    savedGeometry,
    savedCriterion,
    savedNeighbors,
    savedPairs,
    savedPopulation,
    savedScale,
    savedIslandPolicy,
    savedMinimum,
    savedSelfWeight,
    savedCorrection,
    savedIslands,
    savedSignificance,
    savedInference,
    savedPermutations,
    savedSeed
  ]);
  useEffect(() => {
    setQuantileDraft(makeQuantileDraft());
    setError('');
  }, [
    analysis.datasetId,
    savedQuantileCount,
    savedSelectedQuantile,
    savedQuantileSeed,
    savedQuantilePermutations,
    savedQuantileSignificance
  ]);
  if (!analysis.availablePolygonColumns.length) return null;
  const isRate = draft.kind === 'spatial-rate-polygons';
  const isMoran = draft.kind === 'local-moran-polygons';
  const isQuantile = draft.kind === 'quantile-lisa-polygons';
  const isPermutation = isMoran && draft.inference === 'permutation-conditional-two-sided';
  const numberField = (
    field:
      | 'neighborCapacity'
      | 'pairCapacity'
      | 'scale'
      | 'minimumNeighbors'
      | 'selfWeight'
      | 'significanceLevel'
      | 'permutationCount'
      | 'seed',
    label: string,
    minimum: number,
    maximum?: number,
    step: string = '1',
    readOnly = false
  ) => (
    <label htmlFor={`analysis-polygon-${field}`}>
      {label}
      <input
        id={`analysis-polygon-${field}`}
        type="number"
        required
        min={minimum}
        max={maximum}
        step={step}
        readOnly={readOnly}
        value={draft[field]}
        onChange={event => setDraft({...draft, [field]: event.target.value})}
      />
    </label>
  );
  return (
    <details className="analysis-polygon-settings">
      <summary>Configure polygon contiguity analysis</summary>
      <p className="hint">
        Use the numeric field above as events for rates or values for hot spots, Local Moran and
        Quantile LISA. Each source row remains one feature, including all its polygon parts, holes
        and polygons within GeometryCollections.
      </p>
      <form
        onSubmit={event => {
          event.preventDefault();
          try {
            const weights: KeplerPolygonWeightsSettings = {
              kind: 'contiguity',
              geometryColumn: draft.geometryColumn,
              criterion: draft.criterion,
              equality: 'exact-source-xy',
              neighborCapacity: Number(draft.neighborCapacity),
              pairCapacity: Number(draft.pairCapacity)
            };
            const operation: KeplerPolygonAnalysisOperation =
              draft.kind === 'spatial-rate-polygons'
                ? {
                    kind: draft.kind,
                    events: analysis.settings.column,
                    populations: draft.populationColumn,
                    weights,
                    scale: Number(draft.scale),
                    islandPolicy: draft.islandPolicy
                  }
                : draft.kind === 'hot-spots-polygons'
                  ? {
                      kind: draft.kind,
                      column: analysis.settings.column,
                      weights,
                      minimumNeighbors: Number(draft.minimumNeighbors),
                      selfWeight: Number(draft.selfWeight),
                      multipleTesting: draft.multipleTesting,
                      islands: draft.islands,
                      inference: 'analytic-two-sided'
                    }
                  : draft.kind === 'quantile-lisa-polygons'
                    ? {
                        kind: draft.kind,
                        column: analysis.settings.column,
                        weights,
                        quantileCount: Number(quantileDraft.quantileCount),
                        selectedQuantile: Number(quantileDraft.selectedQuantile),
                        seed: Number(quantileDraft.seed),
                        permutationCount: Number(quantileDraft.permutationCount),
                        significanceLevel: Number(quantileDraft.significanceLevel),
                        inference: 'legacy-directed-pseudo',
                        multipleTesting: 'none',
                        islands: 'exclude',
                        quantileMethod: 'type-5'
                      }
                    : {
                        kind: draft.kind,
                        column: analysis.settings.column,
                        weights,
                        minimumNeighbors: Number(draft.minimumNeighbors),
                        multipleTesting: draft.multipleTesting,
                        islands: draft.islands,
                        significanceLevel: Number(draft.significanceLevel),
                        ...(draft.inference === 'permutation-conditional-two-sided'
                          ? {
                              inference: draft.inference,
                              permutationCount: Number(draft.permutationCount),
                              seed: Number(draft.seed)
                            }
                          : {inference: draft.inference})
                      };
            analysis.configurePolygon(operation);
            setError('');
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : String(failure));
          }
        }}
      >
        <label htmlFor="analysis-polygon-statistic">Polygon statistic</label>
        <select
          id="analysis-polygon-statistic"
          value={draft.kind}
          onChange={event => {
            const kind = event.target.value;
            if (
              kind === 'spatial-rate-polygons' ||
              kind === 'hot-spots-polygons' ||
              kind === 'local-moran-polygons' ||
              kind === 'quantile-lisa-polygons'
            )
              setDraft({
                ...draft,
                kind,
                minimumNeighbors:
                  kind === 'local-moran-polygons' &&
                  draft.inference === 'permutation-conditional-two-sided'
                    ? '1'
                    : draft.minimumNeighbors
              });
          }}
        >
          <option value="spatial-rate-polygons">Spatial rates</option>
          <option value="hot-spots-polygons">Hot spots · Gi / Gi*</option>
          <option value="local-moran-polygons">Local Moran</option>
          <option value="quantile-lisa-polygons">
            Quantile LISA · legacy directed pseudo p-values
          </option>
        </select>
        <label htmlFor="analysis-polygon-geometry">Polygon geometry field</label>
        <select
          id="analysis-polygon-geometry"
          required
          value={draft.geometryColumn}
          onChange={event => setDraft({...draft, geometryColumn: event.target.value})}
        >
          {analysis.availablePolygonColumns.map(column => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        <label htmlFor="analysis-polygon-criterion">Polygon contiguity rule</label>
        <select
          id="analysis-polygon-criterion"
          value={draft.criterion}
          onChange={event =>
            setDraft({...draft, criterion: event.target.value === 'rook' ? 'rook' : 'queen'})
          }
        >
          <option value="queen">Queen · identical shared vertex</option>
          <option value="rook">Rook · identical complete edge</option>
        </select>
        {numberField(
          'neighborCapacity',
          'Polygon directed neighbor capacity',
          1,
          MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY
        )}
        {numberField(
          'pairCapacity',
          'Polygon candidate pair capacity',
          1,
          MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY
        )}
        <p className="hint">
          Matching uses exact ORIGINAL source X/Y equality tokens, with Z ignored. Queen requires an
          identical shared vertex. Rook requires identical endpoints of a complete edge in either
          direction. There is no snapping, edge splitting, partial-overlap repair or antimeridian
          normalization. Differently segmented boundaries may not match; full parity with other
          contiguity implementations is not claimed.
        </p>
        <p className="hint">
          Directed neighbor capacity stores two slots per symmetric pair. Candidate pair capacity
          also covers repeated contacts before deduplication. Both must fit before results are
          published. If candidate pairs overflow, the reported required neighbor count is only a
          lower bound; increase capacity and apply again.
        </p>
        {isRate ? (
          <>
            <label htmlFor="analysis-polygon-population">Polygon rate population field</label>
            <select
              id="analysis-polygon-population"
              required
              value={draft.populationColumn}
              onChange={event => setDraft({...draft, populationColumn: event.target.value})}
            >
              {analysis.availableColumns.map(column => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
            {numberField('scale', 'Polygon rate scale', 0, undefined, 'any')}
            <label htmlFor="analysis-polygon-island-policy">
              Polygon rate rows without neighbors
            </label>
            <select
              id="analysis-polygon-island-policy"
              value={draft.islandPolicy}
              onChange={event =>
                setDraft({
                  ...draft,
                  islandPolicy: event.target.value === 'own-rate' ? 'own-rate' : 'undefined'
                })
              }
            >
              <option value="undefined">Undefined</option>
              <option value="own-rate">Use the row's own rate</option>
            </select>
            <p className="hint">
              Rates pool each row's own events and population with eligible neighbors, then apply
              the rate scale.
            </p>
          </>
        ) : isQuantile ? (
          <PolygonQuantileControls
            value={quantileDraft}
            onChange={setQuantileDraft}
            rowCount={analysis.rowCount}
          />
        ) : (
          <>
            {isMoran && (
              <>
                <label htmlFor="analysis-polygon-inference">Polygon Local Moran inference</label>
                <select
                  id="analysis-polygon-inference"
                  value={draft.inference}
                  onChange={event => {
                    const inference =
                      event.target.value === 'permutation-conditional-two-sided'
                        ? 'permutation-conditional-two-sided'
                        : 'analytic-conditional-normal-two-sided';
                    setDraft({
                      ...draft,
                      inference,
                      minimumNeighbors:
                        inference === 'permutation-conditional-two-sided'
                          ? '1'
                          : draft.minimumNeighbors
                    });
                  }}
                >
                  <option value="analytic-conditional-normal-two-sided">
                    Analytic conditional normal · two-sided
                  </option>
                  <option value="permutation-conditional-two-sided">
                    Conditional permutations · two-sided
                  </option>
                </select>
                {numberField(
                  'significanceLevel',
                  'Polygon Local Moran significance level',
                  0,
                  1,
                  'any'
                )}
              </>
            )}
            {numberField(
              'minimumNeighbors',
              'Polygon minimum neighbors',
              1,
              undefined,
              '1',
              isPermutation
            )}
            {!isMoran &&
              numberField('selfWeight', 'Polygon hot-spot self weight', 0, undefined, 'any')}
            <label htmlFor="analysis-polygon-islands">
              Polygon isolated rows in global moments
            </label>
            <select
              id="analysis-polygon-islands"
              value={draft.islands}
              onChange={event =>
                setDraft({...draft, islands: event.target.value === 'keep' ? 'keep' : 'exclude'})
              }
            >
              <option value="exclude">Exclude</option>
              <option value="keep">Keep</option>
            </select>
            <label htmlFor="analysis-polygon-correction">Polygon multiple-testing correction</label>
            <select
              id="analysis-polygon-correction"
              value={draft.multipleTesting}
              onChange={event => {
                const correction = CORRECTIONS.find(option => option.value === event.target.value);
                if (correction) setDraft({...draft, multipleTesting: correction.value});
              }}
            >
              {CORRECTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {isPermutation ? (
              <>
                {numberField('permutationCount', 'Polygon Local Moran permutations', 1, 9999)}
                {numberField('seed', 'Polygon Local Moran permutation seed', 0, 0xffffffff)}
                <p className="hint">
                  Conditional permutations fix the focus value and draw distinct other observations.
                  The raw two-sided p-value floor is 2 / (P + 1); all ties give p = 1. Seeded draws
                  reproduce this implementation, without GeoDa random-stream parity. Minimum
                  neighbors is fixed at 1. More than 64 active neighbors makes inference
                  unavailable. The z-score remains a separate analytic diagnostic.
                </p>
              </>
            ) : (
              <p className="hint">
                {isMoran
                  ? 'Local Moran uses a two-sided conditional-normal approximation; corrected p-values gate association quadrants.'
                  : 'Hot spots use analytic two-sided inference. Self weight 1 gives Gi*; 0 gives Gi. Corrected p-values set significance bins.'}{' '}
                Small neighborhoods and skewed values can make analytic approximations poor.
              </p>
            )}
          </>
        )}
        <p className="hint">
          At most {MAXIMUM_KEPLER_POLYGON_ROWS.toLocaleString()} source rows and{' '}
          {MAXIMUM_KEPLER_POLYGON_VERTICES.toLocaleString()} polygon vertices; vertex capacity is
          checked when applying. Empty, null and nonpolygon rows retain their source identity and
          are excluded. Selection filters the fixed source-wide binary graph, which is reused
          without rebuilding topology for each population.
        </p>
        <button
          type="submit"
          disabled={
            analysis.rowCount < (isQuantile ? 3 : 1) ||
            analysis.rowCount > MAXIMUM_KEPLER_POLYGON_ROWS
          }
        >
          Apply contiguity and run polygon analysis
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
