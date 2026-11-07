// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useState} from 'react';
import type {Feature, Geometry} from 'geojson';
import type {ResidentConsumerState} from './resident-consumer-state';
import {makeDrawingLayerSource, type ExtractedRows} from './resident-drawing-actions';
import {
  findDrawnFeatureAtPoint,
  measureDrawnGeometry,
  validateDrawnPolygon,
  type DrawingIssue
} from './resident-drawing-geometry';
import {
  describeDrawingMaskProblems,
  planDrawingLayerMasks,
  type DrawingMaskLayer
} from './resident-drawing-layer-masks';
import type {KeplerComputeFilter} from './kepler-table-bridge';
import {ResidentDrawingReplay} from './resident-drawing-replay-panel';
import type {DrawingReplayAnnotation, DrawingReplayStep} from './resident-drawing-replay';

export type ResidentDrawingPanelProps = {
  /** Native visState.editor.features: drawn shapes that are not (yet) filters. */
  features: readonly Feature<Geometry>[];
  selectedFeature: Feature<Geometry> | null;
  filters: readonly KeplerComputeFilter[];
  layers: readonly (DrawingMaskLayer & {label: string})[];
  zoom: number;
  /** Last pointer position on the native map in [longitude, latitude], when known. */
  readPointerCoordinate(): [number, number] | null;
  onSelectFeature(feature: Feature<Geometry> | null): void;
  onCreateLayer(source: {table: unknown; label: string; geometryColumn: string}): void;
  onUseAsFilter(feature: Feature<Geometry>, layerIds: readonly string[]): void;
  /** Extracts the selected rows of one dataset and publishes them through the source workspace. */
  onExtractRows(datasetId: string): Promise<ResidentConsumerState<ExtractedRows>>;
  /** Native visState.annotations (empty while the host leaves enableAnnotations off). */
  annotations?: readonly DrawingReplayAnnotation[];
  /** Map pixel size used to fit each replay step. */
  viewportSize?: {width: number; height: number};
  /** Replay: fly the map to the step and highlight it. */
  onFocusReplayStep?(step: DrawingReplayStep): void;
};

function formatIssues(issues: readonly DrawingIssue[]): string {
  return issues.map(issue => issue.message).join(' ');
}

/** Native editor companion: validation reasons, select-by-click and the drawing actions. */
export function ResidentDrawingPanel(props: ResidentDrawingPanelProps) {
  const {features, selectedFeature, filters, layers, zoom} = props;
  const [notice, setNotice] = useState('');
  const [extractDatasetId, setExtractDatasetId] = useState('');
  const issues = useMemo(
    () =>
      features.flatMap((feature, index) => {
        if (feature.geometry?.type !== 'Polygon' && feature.geometry?.type !== 'MultiPolygon')
          return [];
        const validation = validateDrawnPolygon(feature);
        return validation.status === 'invalid'
          ? [{featureId: String(feature.id ?? index), reasons: validation.reasons}]
          : [];
      }),
    [features]
  );
  const maskProblems = useMemo(
    () => describeDrawingMaskProblems(planDrawingLayerMasks(layers, filters)),
    [layers, filters]
  );
  const datasetIds = useMemo(() => [...new Set(layers.map(layer => layer.dataId))], [layers]);

  useEffect(() => {
    if (!features.length) return undefined;
    const onClick = (event: MouseEvent) => {
      if (!(event.target instanceof HTMLCanvasElement)) return;
      const point = props.readPointerCoordinate();
      if (!point) return;
      // About eight screen pixels in degrees at the current zoom.
      const toleranceDegrees = (360 / (512 * 2 ** zoom)) * 8;
      const index = findDrawnFeatureAtPoint(features, point, toleranceDegrees);
      if (index >= 0 && features[index].id !== selectedFeature?.id)
        props.onSelectFeature(features[index]);
    };
    document.addEventListener('click', onClick);
    return () => document.removeEventListener('click', onClick);
  }, [features, selectedFeature?.id, zoom]);

  if (!features.length && !selectedFeature && !maskProblems.length) return null;
  const selected = selectedFeature?.geometry ? selectedFeature : null;
  const selectedValidation = selected ? validateDrawnPolygon(selected) : null;
  const measurement = selected ? measureDrawnGeometry(selected.geometry) : null;
  const filterableLayers = layers.filter(layer =>
    ['point', 'icon', 'heatmap'].includes(layer.type)
  );

  return (
    <section aria-label="Drawing" data-testid="resident-drawing-panel">
      <h2>Drawing</h2>
      <p className="hint">{features.length} drawn shapes</p>
      {issues.map(issue => (
        <p key={issue.featureId} role="alert" data-testid="drawing-invalid-reason">
          Shape {issue.featureId}: {formatIssues(issue.reasons)}
        </p>
      ))}
      {maskProblems.map(problem => (
        <p key={problem} role="alert" data-testid="drawing-mask-problem">
          {problem}
        </p>
      ))}
      {selected && measurement ? (
        <p className="hint" data-testid="drawing-selected">
          Selected {selected.geometry.type}: {Math.round(measurement.lengthMeters)} m,{' '}
          {Math.round(measurement.areaSquareMeters)} m2
          {selectedValidation?.status === 'valid' && selectedValidation.warnings.length
            ? ` (${formatIssues(selectedValidation.warnings)})`
            : ''}
        </p>
      ) : null}
      <button
        type="button"
        disabled={!features.length}
        onClick={() => {
          const result = makeDrawingLayerSource(features, {
            id: `drawing-${Date.now()}`,
            label: 'Drawing'
          });
          if (result.status === 'ready') {
            props.onCreateLayer({
              table: result.source.source,
              label: result.source.label,
              geometryColumn: result.source.geometryColumn
            });
            setNotice(`Prepared ${result.featureCount} drawn features in the Sources panel.`);
          } else if (result.status === 'invalid') {
            setNotice(
              result.rejected
                .map(item => `Shape ${item.featureId}: ${formatIssues(item.reasons)}`)
                .join(' ')
            );
          } else setNotice(result.reason);
        }}
      >
        Create layer from drawing
      </button>
      <button
        type="button"
        disabled={!selected || !filterableLayers.length}
        onClick={() => {
          if (!selected) return;
          if (selectedValidation?.status === 'invalid') {
            setNotice(formatIssues(selectedValidation.reasons));
            return;
          }
          props.onUseAsFilter(
            selected,
            filterableLayers.map(layer => layer.id)
          );
          setNotice('Drawing is now a polygon filter on the point layers.');
        }}
      >
        Use drawing as filter
      </button>
      <label>
        Extract from{' '}
        <select
          value={extractDatasetId}
          onChange={event => setExtractDatasetId(event.target.value)}
        >
          <option value="">Choose dataset</option>
          {datasetIds.map(datasetId => (
            <option key={datasetId} value={datasetId}>
              {datasetId}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        disabled={!extractDatasetId}
        onClick={() => {
          setNotice('Extracting selected rows…');
          void props.onExtractRows(extractDatasetId).then(
            state =>
              setNotice(
                state.status === 'ready'
                  ? `Extracted ${state.value.selectedCount} rows to the Sources panel.`
                  : state.reason
              ),
            error => setNotice(error instanceof Error ? error.message : String(error))
          );
        }}
      >
        Extract selected rows
      </button>
      {props.onFocusReplayStep ? (
        <ResidentDrawingReplay
          features={features}
          annotations={props.annotations}
          viewport={props.viewportSize ?? {width: 800, height: 600}}
          onFocusStep={props.onFocusReplayStep}
        />
      ) : null}
      {notice ? (
        <p role="status" data-testid="drawing-notice">
          {notice}
        </p>
      ) : null}
    </section>
  );
}
