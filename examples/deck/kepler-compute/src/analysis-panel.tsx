// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {useAnalysisFormInitialization} from './analysis-form-initialization';
import {isAnalysisMode, isStandardizationMode, type AnalysisRuntime} from './analysis-runtime';
import {HotspotPanel} from './analysis-hotspot-panel';
import {PointWeightsPanel} from './analysis-point-weights-panel';
import {PointHotspotPanel} from './analysis-point-hotspot-panel';
import {LocalMoranPanel} from './analysis-local-moran-panel';
import {QuantileLisaPanel} from './analysis-quantile-lisa-panel';
import {getQuantileLisaPValueLabel} from './analysis-quantile-lisa-inference';
import {InterpolationPanel, InterpolationResults} from './interpolation-panel';
import {DBSCANPanel} from './analysis-dbscan-panel';
import {OLSPanel, OLSResults} from './analysis-ols-panel';
import {PolygonPanel} from './analysis-polygon-panel';
import {CartogramPanel, CartogramResults} from './analysis-cartogram-panel';
import {RateResults} from './analysis-rate-results';
import {KrigingPanel} from './analysis-kriging-panel';
import {ResidentConsumerNotice} from './resident-consumer-notice';
import {analysisDatasetContext} from './analysis-dataset-context';
import {AnalysisRowMeaning} from './analysis-profile-declarations';

export type AnalysisSource = {
  id: string;
  label: string;
  getAnalysis(): AnalysisRuntime;
  polygonAnalysisUnavailableReason?: string;
};

/** Embeds in the existing Kepler compute controls; its results stay in the map's GPU buffers. */
export function AnalysisPanel({
  analysis,
  sources,
  selectedDatasetId,
  onSelectDataset
}: {
  analysis: AnalysisRuntime;
  sources?: readonly AnalysisSource[];
  /** Host-owned selection survives native Kepler tabs unmounting this panel. */
  selectedDatasetId?: string;
  onSelectDataset?: (datasetId: string) => void;
}) {
  const [selected, setSelected] = useState<{id: string; analysis: AnalysisRuntime} | null>(null);
  const [sourceError, setSourceError] = useState('');
  const requestedDatasetId = selectedDatasetId ?? selected?.id ?? analysis.datasetId;
  const current =
    selected?.id === requestedDatasetId && sources?.some(source => source.id === selected.id)
      ? selected.analysis
      : analysis;
  const polygonUnavailableReason = sources?.find(
    source => source.id === current.datasetId
  )?.polygonAnalysisUnavailableReason;
  const [controls, setControls] = useState({analysis: current, revision: 0});
  if (controls.analysis !== current) {
    // A replacement may reuse its dataset ID. Reset drafts before committing the new controls.
    setControls({analysis: current, revision: controls.revision + 1});
  }
  useEffect(() => {
    if (requestedDatasetId === analysis.datasetId) {
      if (selected) setSelected(null);
      return;
    }
    const source = sources?.find(candidate => candidate.id === requestedDatasetId);
    if (!source) {
      setSelected(null);
      setSourceError('');
      onSelectDataset?.(analysis.datasetId);
      return;
    }
    try {
      // A restored source can retain its ID while replacing its resident runtime.
      const replacement = source.getAnalysis();
      if (source.id !== selected?.id || replacement !== selected?.analysis)
        setSelected({id: source.id, analysis: replacement});
    } catch (failure) {
      setSelected(null);
      setSourceError(failure instanceof Error ? failure.message : String(failure));
      onSelectDataset?.(analysis.datasetId);
    }
  }, [analysis, selected, sources, requestedDatasetId, onSelectDataset]);
  return (
    <section
      aria-label="GPU analysis"
      style={{borderTop: '1px solid #405168', marginTop: 18, paddingTop: 10}}
    >
      <h2 style={{fontSize: 16, marginBottom: 8}}>Analyze resident data</h2>
      {sources && sources.length > 1 && (
        <>
          <label htmlFor="analysis-dataset">Dataset</label>
          <select
            id="analysis-dataset"
            value={current.datasetId}
            style={{width: '100%'}}
            onChange={event => {
              const source = sources.find(candidate => candidate.id === event.target.value);
              if (!source) return;
              try {
                setSelected({id: source.id, analysis: source.getAnalysis()});
                setSourceError('');
                onSelectDataset?.(source.id);
              } catch (failure) {
                setSourceError(failure instanceof Error ? failure.message : String(failure));
              }
            }}
          >
            {sources.map(source => (
              <option key={source.id} value={source.id}>
                {source.label}
              </option>
            ))}
          </select>
        </>
      )}
      {sourceError && (
        <ResidentConsumerNotice
          consumer="analysis"
          status="error"
          className="error"
          reason={sourceError}
        />
      )}
      {polygonUnavailableReason && (
        <p className="hint">Polygon analysis unavailable: {polygonUnavailableReason}</p>
      )}
      <AnalysisControls key={controls.revision} analysis={current} />
    </section>
  );
}

function AnalysisControls({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const [, refresh] = useState(0);
  const [savedRecipe, setSavedRecipe] = useState('');
  const [error, setError] = useState('');
  useEffect(() => analysis.subscribe(() => refresh(value => value + 1)), [analysis]);
  const settings = analysis.settings;
  const summary = analysis.summary;
  const isInterpolationAnalysis =
    settings.mode === 'idw-points' || settings.mode === 'kriging-points';
  const isRateAnalysis =
    settings.mode === 'excess-risk' ||
    settings.mode === 'spatial-rate-points' ||
    settings.mode === 'spatial-rate-polygons';
  const invoke = (action: () => void) => {
    try {
      action();
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };
  return (
    <>
      {analysisDatasetContext.get(analysis.datasetId)?.profile && (
        <AnalysisRowMeaning
          profile={analysisDatasetContext.get(analysis.datasetId)!.profile}
          onDeclare={rowSemantics =>
            analysisDatasetContext.setDeclarations(analysis.datasetId, {
              ...analysisDatasetContext.getDeclarations(analysis.datasetId),
              rowSemantics
            })
          }
        />
      )}
      <label htmlFor="analysis-column">Numeric field</label>
      <select
        id="analysis-column"
        disabled={settings.mode === 'ols' || settings.mode === 'dorling-cartogram'}
        value={settings.column}
        onChange={event => invoke(() => analysis.setColumn(event.target.value))}
        style={{width: '100%'}}
      >
        {!settings.column && <option value="">Choose a value field for numeric analyses</option>}
        {analysis.availableColumns.map(column => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
      <label htmlFor="analysis-operation">Operation</label>
      <select
        id="analysis-operation"
        value={settings.mode}
        onChange={event => {
          const mode = event.target.value;
          if (isAnalysisMode(mode)) invoke(() => analysis.setMode(mode));
        }}
        style={{width: '100%'}}
      >
        <option value="off">Native layer colors</option>
        <option value="z-score">Standardize · z-score</option>
        <option value="deviation-from-mean">Standardize · deviation from mean</option>
        <option value="mad">Standardize · mean absolute deviation</option>
        <option value="range-adjust">Standardize · divide by range</option>
        <option value="range-standardize">Standardize · range from zero</option>
        <option value="quantile">Classify · quantiles</option>
        <option value="equal-interval">Classify · equal intervals</option>
        <option value="natural-breaks">Classify · natural breaks</option>
        <option value="excess-risk">Rates · observed / expected risk</option>
        <option value="spatial-rate-points" disabled={!settings.pointWeights}>
          Rates · configured point neighbors
        </option>
        <option value="hot-spots-lattice" disabled={!settings.hotspot}>
          Hot spots · configured row-major grid
        </option>
        <option value="dorling-cartogram" disabled={!settings.cartogram}>
          Cartogram · configured Dorling circles
        </option>
        <option value="ols" disabled={!settings.ols}>
          Regress · configured ordinary least squares
        </option>
        <option value="dbscan-points" disabled={!settings.dbscan}>
          Cluster · configured geographic DBSCAN
        </option>
        <option value="idw-points" disabled={!settings.idw}>
          Interpolate · resident IDW geographic grid
        </option>
        <option value="kriging-points" disabled={!settings.kriging}>
          Interpolate · resident kriging with fitting and validation
        </option>
        <option value="quantile-lisa-points" disabled={!settings.quantileLisa}>
          Quantile LISA · configured points
        </option>
        <option value="local-moran-points" disabled={!settings.localMoran}>
          Local Moran · configured points
        </option>
        <option value="hot-spots-points" disabled={!settings.pointHotspot}>
          Hot spots · configured points
        </option>
        <option
          value="spatial-rate-polygons"
          disabled={settings.polygon?.kind !== 'spatial-rate-polygons'}
        >
          Rates · configured polygon contiguity
        </option>
        <option
          value="hot-spots-polygons"
          disabled={settings.polygon?.kind !== 'hot-spots-polygons'}
        >
          Hot spots · configured polygon contiguity
        </option>
        <option
          value="quantile-lisa-polygons"
          disabled={settings.polygon?.kind !== 'quantile-lisa-polygons'}
        >
          Quantile LISA · configured polygon contiguity
        </option>
        <option
          value="local-moran-polygons"
          disabled={settings.polygon?.kind !== 'local-moran-polygons'}
        >
          Local Moran · configured polygon contiguity
        </option>
      </select>
      <PolygonPanel analysis={analysis} />
      <CartogramPanel analysis={analysis} />
      <OLSPanel analysis={analysis} />
      <DBSCANPanel analysis={analysis} />
      <InterpolationPanel analysis={analysis} />
      <KrigingPanel analysis={analysis} />
      <HotspotPanel analysis={analysis} />
      <PointWeightsPanel analysis={analysis} />
      <PointHotspotPanel analysis={analysis} />
      <LocalMoranPanel
        coordinateSource={initialization.coordinateSource}
        coordinateInitialization={initialization.coordinates}
        coordinateResetKey={analysis.coordinateRestorationKey}
        columns={analysis.availableColumns}
        rowCount={analysis.rowCount}
        settings={settings.localMoran}
        onApply={settings => analysis.configureLocalMoran(settings)}
      />
      <QuantileLisaPanel
        coordinateSource={initialization.coordinateSource}
        coordinateInitialization={initialization.coordinates}
        coordinateResetKey={analysis.coordinateRestorationKey}
        columns={analysis.availableColumns}
        rowCount={analysis.rowCount}
        settings={settings.quantileLisa}
        onApply={settings => analysis.configureQuantileLisa(settings)}
      />
      {settings.mode !== 'off' && (
        <>
          {(settings.mode === 'excess-risk' ||
            settings.mode === 'spatial-rate-points' ||
            settings.mode === 'spatial-rate-polygons') && (
            <>
              <p className="hint">The numeric field above supplies events.</p>
              <label htmlFor="analysis-denominator">Population field</label>
              <select
                id="analysis-denominator"
                value={settings.denominatorColumn}
                style={{width: '100%'}}
                onChange={event => invoke(() => analysis.setDenominatorColumn(event.target.value))}
              >
                {analysis.availableColumns.map(column => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
              {settings.mode === 'excess-risk' && (
                <p className="hint">
                  Expected events = row population × total events / total population. A result of 1
                  matches the pooled rate; values above 1 exceed it.
                </p>
              )}
              {(settings.mode === 'spatial-rate-points' ||
                settings.mode === 'spatial-rate-polygons') && (
                <p className="hint">
                  Spatial rate = pooled events / pooled population × scale, using the row and its
                  eligible neighbors. The configured island policy applies when no eligible
                  neighbors remain.
                </p>
              )}
              <p className="hint">
                Missing values, negative events and nonpositive populations are excluded. Zero
                events are eligible; source values are used as supplied.
              </p>
              {settings.mode === 'excess-risk' && (
                <p className="hint">All-zero events produce undefined ratios.</p>
              )}
            </>
          )}
          {settings.mode === 'z-score' && (
            <>
              <label htmlFor="analysis-variance">Variance convention</label>
              <select
                id="analysis-variance"
                value={settings.variance}
                style={{width: '100%'}}
                onChange={event =>
                  invoke(() =>
                    analysis.setVariance(
                      event.target.value === 'population' ? 'population' : 'sample'
                    )
                  )
                }
              >
                <option value="sample">Sample · divide by n − 1</option>
                <option value="population">Population · divide by n</option>
              </select>
            </>
          )}
          {settings.mode === 'mad' && (
            <p className="hint">
              Centers values on their mean, then divides by mean absolute deviation. This uses the
              arithmetic mean, not the median.
            </p>
          )}
          {settings.mode === 'range-adjust' && (
            <p className="hint">
              Divides each value by maximum − minimum without subtracting the minimum.
            </p>
          )}
          {settings.mode === 'range-standardize' && (
            <p className="hint">Subtracts the minimum, then divides by maximum − minimum.</p>
          )}
          {(settings.mode === 'mad' ||
            settings.mode === 'range-adjust' ||
            settings.mode === 'range-standardize') && (
            <p className="hint">
              When the reference population is constant, original values are retained.
            </p>
          )}
          <label htmlFor="analysis-population">Analysis population</label>
          <select
            id="analysis-population"
            value={settings.population}
            onChange={event =>
              invoke(() =>
                analysis.setPopulation(event.target.value === 'selected' ? 'selected' : 'all')
              )
            }
            style={{width: '100%'}}
          >
            <option value="all">All valid observations</option>
            <option value="selected">Recompute on selected observations</option>
          </select>
          <p className="hint">
            {isInterpolationAnalysis
              ? settings.population === 'all'
                ? 'The grid uses all valid samples and stays unchanged when source rows are filtered.'
                : 'Filters change contributing samples and recompute the grid.'
              : settings.mode === 'dorling-cartogram'
                ? settings.population === 'all'
                  ? 'The layout includes all eligible source rows, regardless of the current filters.'
                  : 'Filters change participating circles and recompute the layout.'
                : settings.population === 'all'
                  ? 'Filters hide results; the reference population stays fixed.'
                  : 'Filters change the reference population and recompute results.'}
          </p>
          {(settings.mode === 'quantile' ||
            settings.mode === 'equal-interval' ||
            settings.mode === 'natural-breaks') && (
            <label htmlFor="analysis-class-count">
              Classes <output>{settings.classCount}</output>
              <input
                id="analysis-class-count"
                type="range"
                min="2"
                max="8"
                step="1"
                value={settings.classCount}
                onChange={event => invoke(() => analysis.setClassCount(Number(event.target.value)))}
              />
            </label>
          )}
          {settings.mode === 'natural-breaks' && (
            <>
              <label htmlFor="analysis-natural-breaks">Natural breaks method</label>
              <select
                id="analysis-natural-breaks"
                value={settings.naturalBreaksMode}
                style={{width: '100%'}}
                onChange={event =>
                  invoke(() =>
                    analysis.setNaturalBreaksMode(
                      event.target.value === 'exact' ? 'exact' : 'approximate'
                    )
                  )
                }
              >
                <option value="approximate">Approximate refinement</option>
                <option value="exact" disabled={!analysis.supportsExactNaturalBreaks}>
                  Exact · at most 1,024 source rows
                </option>
              </select>
              <p className="hint">
                Exact mode minimizes the documented Float32 interval cost. Its limit applies to all
                source rows, even with a smaller selected population.
              </p>
            </>
          )}
          <OLSResults analysis={analysis} />
          {settings.mode === 'dorling-cartogram' ? (
            <CartogramResults analysis={analysis} />
          ) : summary ? (
            <div aria-live="polite" className="hint">
              {isRateAnalysis ? (
                <RateResults analysis={analysis} />
              ) : (
                <div>
                  {summary.count.toLocaleString()} analyzed · {summary.nullCount.toLocaleString()}{' '}
                  undefined
                </div>
              )}
              {settings.mode !== 'dbscan-points' && (
                <div>
                  {settings.mode === 'ols'
                    ? 'OLS residual'
                    : settings.mode === 'excess-risk'
                      ? 'Observed / expected'
                      : settings.mode === 'spatial-rate-points' ||
                          settings.mode === 'spatial-rate-polygons'
                        ? 'Spatial rate'
                        : isInterpolationAnalysis
                          ? settings.mode === 'kriging-points'
                            ? 'Kriging prediction'
                            : 'Interpolated grid value'
                          : settings.mode === 'quantile-lisa-points' ||
                              settings.mode === 'quantile-lisa-polygons'
                            ? getQuantileLisaPValueLabel(summary.quantileLisa?.alternative)
                            : settings.mode === 'local-moran-points' ||
                                settings.mode === 'local-moran-polygons'
                              ? 'Local Moran I'
                              : settings.mode === 'hot-spots-lattice' ||
                                  settings.mode === 'hot-spots-points' ||
                                  settings.mode === 'hot-spots-polygons'
                                ? 'Hot-spot z-score'
                                : isStandardizationMode(settings.mode)
                                  ? 'Transformed value'
                                  : 'Value'}{' '}
                  range: {formatNumber(summary.minimum)} to {formatNumber(summary.maximum)}
                </div>
              )}
              {settings.mode !== 'dbscan-points' && <div>Mean: {formatNumber(summary.mean)}</div>}
              {summary.dbscan && (
                <div>
                  {summary.dbscan.clusterCount} clusters · {summary.dbscan.noiseCount} noise rows ·{' '}
                  {summary.dbscan.excludedCount} excluded rows. Gray is noise; colors repeat every
                  15 clusters. Cluster IDs identify this source order and population only.
                </div>
              )}
              {summary.pointWeights && !isRateAnalysis && (
                <div>
                  {summary.pointWeights.weightsRequiredCount.toLocaleString()} point neighbor links
                  · {summary.pointWeights.invalidCount.toLocaleString()} invalid coordinate rows
                </div>
              )}
              {summary.pointWeightsPath && !isRateAnalysis && (
                <div data-testid="point-weights-path">
                  Point weights path:{' '}
                  {summary.pointWeightsPath.path === 'streamed'
                    ? `streamed row chunks (${summary.pointWeightsPath.chunkCount.toLocaleString()} chunk${summary.pointWeightsPath.chunkCount === 1 ? '' : 's'}${summary.pointWeightsPath.reason === 'forced' ? ', forced' : ', single CSR exceeds the device storage binding limit'})`
                    : 'single CSR'}
                </div>
              )}
              {summary.polygonWeights && !isRateAnalysis && (
                <div>
                  {summary.polygonWeights.weightsRequiredCount.toLocaleString()} polygon neighbor
                  links · {summary.polygonWeights.invalidCount.toLocaleString()} rows without
                  eligible polygon parts
                </div>
              )}
              {summary.quantileLisa && (
                <>
                  <div>
                    Quantile interval: [
                    {summary.quantileLisa.boundaries.map(formatBoundary).join(', ')}) ·{' '}
                    {summary.quantileLisa.validCount.toLocaleString()} finite{' '}
                    {settings.mode === 'quantile-lisa-polygons'
                      ? 'polygon-valid'
                      : 'coordinate-valid'}{' '}
                    observations · {summary.quantileLisa.islandCount.toLocaleString()} isolated
                  </div>
                  <p>
                    Reported pseudo p-values use{' '}
                    {summary.quantileLisa.alternative ?? 'geoda-legacy'} without correction.
                    Undefined values include nonmembers, zero local joins, excluded observations and
                    islands. Class 0 combines nonsignificant and untested rows; class 1 is
                    significant under this rule; class 2 is undefined input; class 3 is isolated.
                    Untested tail counts use 4294967295.
                  </p>
                </>
              )}
              {summary.localMoran && (
                <>
                  <div>
                    {summary.localMoran.invalidCount.toLocaleString()} included rows without{' '}
                    {(settings.mode === 'local-moran-polygons' &&
                    settings.polygon?.kind === 'local-moran-polygons'
                      ? settings.polygon.inference
                      : settings.localMoran?.inference) === 'permutation-conditional-two-sided'
                      ? 'permutation'
                      : 'analytic'}{' '}
                    inference
                  </div>
                  <p>
                    Map: gray = unreported; red = high–high; light blue = low–high; blue = low–low;
                    orange = high–low. Quadrants pass the selected correction and significance
                    threshold. Unreported combines nonsignificant and undefined inference; inspect
                    p-values in the table. Z-scores always describe the analytic approximation; the
                    selected inference method determines p-values and quadrant significance.
                  </p>
                </>
              )}
              {summary.hotspot && (
                <>
                  <div>
                    {summary.hotspot.weightsRequiredCount.toLocaleString()} neighbor links ·{' '}
                    {summary.hotspot.invalidCount.toLocaleString()} undefined rows
                  </div>
                  {summary.hotspot.weightsOverflow > 0 && (
                    <p role="alert" className="error">
                      Neighborhood capacity was exceeded; these results are incomplete.
                    </p>
                  )}
                  <p>
                    Bin −1/−2/−3 indicates cold spots; +1/+2/+3 indicates hot spots at 90/95/99%
                    confidence after the selected correction. Bin 0 is not significant or undefined;
                    inspect its z-score and p-value.
                  </p>
                </>
              )}
              {summary.breaks.length > 0 && (
                <>
                  <div>Edges: {summary.breaks.map(formatNumber).join(' · ')}</div>
                  <div>
                    Class counts:{' '}
                    {summary.classCounts.map(count => count.toLocaleString()).join(' · ')}
                  </div>
                </>
              )}
              <div>
                Source revision {summary.sourceRevision}
                {summary.selectionRevision === undefined
                  ? ''
                  : ` · selection ${summary.selectionRevision}`}
              </div>
            </div>
          ) : (
            <p className="hint" aria-live="polite">
              Computing GPU analysis…
            </p>
          )}
          <p className="hint">
            Float32 analysis of {settings.column}. Colors and results stay on the GPU. Source
            batches stay intact; global analysis uses explicitly reserved working memory. Only
            compact summaries and requested rows return to the CPU.
          </p>
          {isInterpolationAnalysis ? (
            <InterpolationResults analysis={analysis} />
          ) : (
            <p className="hint">Table result: {analysis.resultColumn}.</p>
          )}
          <div className="actions">
            <button
              type="button"
              onClick={() => invoke(() => setSavedRecipe(analysis.serializeRecipe()))}
            >
              Save recipe
            </button>
            <button
              type="button"
              disabled={!savedRecipe.trim()}
              onClick={() => invoke(() => analysis.restoreRecipe(savedRecipe))}
            >
              Restore recipe
            </button>
          </div>
          <details>
            <summary>Saved analysis recipe</summary>
            <textarea
              aria-label="Saved analysis recipe"
              value={savedRecipe}
              onChange={event => setSavedRecipe(event.target.value)}
              rows={5}
              style={{boxSizing: 'border-box', width: '100%', marginTop: 8, fontSize: 11}}
            />
            <p>
              Selected recipes require the same dataset and selection revision. Their population
              cannot silently change on restoration.
            </p>
          </details>
        </>
      )}
      {analysis.isBuildingStreamedWeights && (
        <p role="status" data-testid="point-weights-building">
          Building streamed point weights (count pass, then one build per chunk)…
        </p>
      )}
      {(error || analysis.error) && (
        <ResidentConsumerNotice
          consumer="analysis"
          status="error"
          className="error"
          reason={error || analysis.error}
        />
      )}
    </>
  );
}

function formatNumber(value: number): string {
  return Number.isFinite(value)
    ? value.toLocaleString(undefined, {maximumFractionDigits: 3})
    : 'undefined';
}

function formatBoundary(value: number): string {
  return value === Infinity ? '∞' : value === -Infinity ? '−∞' : formatNumber(value);
}
