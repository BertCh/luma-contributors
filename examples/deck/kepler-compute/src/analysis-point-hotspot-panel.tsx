// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useAnalysisFormInitialization} from './analysis-form-initialization';
import type {AnalysisRuntime} from './analysis-runtime';
import {
  normalizePointHotspotSettings,
  type PointHotspotSettings
} from './analysis-point-hotspot-settings';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  makePointSheetDefaults,
  usePointAnalysisSheet,
  createPointSheetFields,
  getPointSheetTopology,
  getPointSheetChecks,
  POINT_SHEET_CORRECTIONS,
  type PointSheetValues
} from './analysis-point-weights-panel';

export function getPointHotspotSheetSettings(
  values: PointSheetValues,
  rowCount: number
): PointHotspotSettings {
  return normalizePointHotspotSettings(
    {
      ...getPointSheetTopology(values, rowCount),
      minimumNeighbors: Number(values['minimumNeighbors']),
      selfWeight: Number(values['selfWeight']),
      multipleTesting: values['multipleTesting'] as PointHotspotSettings['multipleTesting'],
      islands: values['islands'] as PointHotspotSettings['islands']
    },
    rowCount
  );
}

/** The original binary-weight analytic point method keeps its resident executor and result views. */
export function PointHotspotPanel({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const saved = analysis.settings.pointHotspot;
  const topology = saved ?? analysis.settings.pointWeights;
  const sheet = usePointAnalysisSheet(
    makePointSheetDefaults(topology, initialization.coordinates, analysis.rowCount, {
      minimumNeighbors: saved?.minimumNeighbors ?? 8,
      selfWeight: saved?.selfWeight ?? 1,
      multipleTesting: saved?.multipleTesting ?? 'benjamini-hochberg',
      islands: saved?.islands ?? 'exclude'
    }),
    {saved, topology},
    analysis.coordinateRestorationKey ?? analysis.datasetId,
    initialization.coordinateSource,
    Boolean(saved)
  );
  const source = (name: string) =>
    !saved &&
    topology &&
    name in makePointSheetDefaults(topology, undefined, analysis.rowCount, {}) &&
    !sheet.source(name).startsWith('User')
      ? 'Inherited spatial-rate weights'
      : sheet.source(name);
  const fields = createPointSheetFields(
    sheet.values,
    analysis.availableColumns,
    'analysis-point-hotspot',
    'Hot-spot',
    source,
    [
      {
        name: 'minimumNeighbors',
        label: 'Hot-spot minimum neighbors',
        controlId: 'analysis-point-hotspot-minimum',
        kind: 'integer',
        minimum: 1,
        maximum: 2 ** 24
      },
      {
        name: 'selfWeight',
        label: 'Hot-spot self weight',
        controlId: 'analysis-point-hotspot-self',
        kind: 'number',
        minimum: 0
      },
      {
        name: 'islands',
        label: 'Isolated points in global statistics',
        controlId: 'analysis-point-hotspot-islands',
        kind: 'enum',
        options: ['exclude', 'keep'],
        level: 'advanced'
      },
      {
        name: 'multipleTesting',
        label: 'Point hot-spot correction',
        controlId: 'analysis-point-hotspot-correction',
        kind: 'enum',
        options: POINT_SHEET_CORRECTIONS,
        level: 'advanced'
      }
    ]
  );
  return (
    <details className="analysis-point-hotspot-settings">
      <summary>
        Configure hot spots on{' '}
        {sheet.values['coordinateSpace'] === 'planar' ? 'projected' : 'geographic'} points
      </summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([analysis.coordinateRestorationKey, saved, topology])}
        fields={fields}
        checks={getPointSheetChecks(fields, () =>
          getPointHotspotSheetSettings(sheet.values, analysis.rowCount)
        )}
        method="Resident point hot spots use binary weights with analytic, two-sided inference. Self weight 1 gives Gi*; 0 gives Gi. Coordinate fields and the source-wide neighbor graph are explicit."
        submitLabel="Apply neighbors and run point hot spots"
        onChange={(name, value) => sheet.change(name, value)}
        onApply={current =>
          analysis.configurePointHotspots(
            getPointHotspotSheetSettings({...sheet.values, ...current}, analysis.rowCount)
          )
        }
      />
      <p className="hint">
        Selected-population analysis removes observations from the fixed source neighbor graph
        without replacing neighbors. Isolated points never receive a score. The minimum applies to
        eligible neighbors after filtering; eight is a common analytic-inference starting point.
      </p>
      <p className="hint">
        Self weight 1 gives Gi*; 0 excludes the focal value for Gi. Inference is analytic and
        two-sided. Table p-values are unadjusted; significance bins use the selected correction. Map
        colors show z-scores. Missing or out-of-range coordinates remain undefined.
      </p>
      <p className="hint">
        Dense clusters can make the first exact neighbor build slow. Filtering and compatible
        settings reuse the completed graph.
      </p>
    </details>
  );
}
