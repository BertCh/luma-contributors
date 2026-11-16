// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '@luma-contributors/kepler';
import {useAnalysisFormInitialization} from './analysis-form-initialization';
import type {AnalysisRuntime} from './analysis-runtime';
import {normalizeHotspotSettings, type HotspotSettings} from './analysis-hotspot-settings';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  createResidentSheetFields,
  getResidentSheetChecks,
  readResidentSheetNumber
} from './analysis-sheet-model';
import {
  usePointAnalysisSheet,
  POINT_SHEET_CORRECTIONS,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';

export function makeHotspotSheetDefaults(
  saved: HotspotSettings | undefined,
  lattice: {width: string; height: string} | undefined
): Record<string, CatalogParameterValue> {
  return {
    width: saved?.width ?? lattice?.width ?? '',
    height: saved?.height ?? lattice?.height ?? '',
    criterion: saved?.criterion ?? 'queen',
    radius: saved?.radius ?? 1,
    minimumNeighbors: saved?.minimumNeighbors ?? 8,
    selfWeight: saved?.selfWeight ?? 1,
    multipleTesting: saved?.multipleTesting ?? 'benjamini-hochberg',
    editedFields: '[]'
  };
}

export function getHotspotSheetSettings(
  values: PointSheetValues,
  rowCount: number,
  saved?: HotspotSettings
): HotspotSettings {
  const number = (name: string) => readResidentSheetNumber(values, name);
  const settings: HotspotSettings = {
    width: number('width'),
    height: number('height'),
    criterion: values['criterion'] as HotspotSettings['criterion'],
    radius: number('radius'),
    minimumNeighbors: number('minimumNeighbors'),
    selfWeight: number('selfWeight'),
    multipleTesting: values['multipleTesting'] as HotspotSettings['multipleTesting']
  };
  // Preserve explicit restored capacity only while the topology is unchanged.
  if (
    saved &&
    ['width', 'height', 'criterion', 'radius'].every(
      name => values[name] === saved[name as keyof HotspotSettings]
    )
  )
    settings.neighborCapacity = saved.neighborCapacity;
  return normalizeHotspotSettings(settings, rowCount);
}

/** Only declared row-major metadata or explicit choices establish lattice adjacency. */
export function HotspotPanel({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const saved = analysis.settings.hotspot;
  const sheet = usePointAnalysisSheet(
    makeHotspotSheetDefaults(saved, initialization.lattice),
    saved,
    analysis.coordinateRestorationKey ?? analysis.datasetId
  );
  const specifications: PointSheetSpecification[] = [
    {
      name: 'width',
      label: 'Grid width',
      controlId: 'analysis-grid-width',
      kind: 'integer',
      minimum: 1
    },
    {
      name: 'height',
      label: 'Grid height',
      controlId: 'analysis-grid-height',
      kind: 'integer',
      minimum: 1
    },
    {
      name: 'criterion',
      label: 'Neighborhood',
      controlId: 'analysis-grid-criterion',
      kind: 'enum',
      options: ['queen', 'rook']
    },
    {
      name: 'radius',
      label: 'Radius in cells',
      controlId: 'analysis-grid-radius',
      kind: 'integer',
      minimum: 1,
      maximum: 32
    },
    {
      name: 'minimumNeighbors',
      label: 'Minimum neighbors',
      controlId: 'analysis-grid-minimum-neighbors',
      kind: 'integer',
      minimum: 0,
      level: 'advanced'
    },
    {
      name: 'selfWeight',
      label: 'Self weight',
      controlId: 'analysis-grid-self-weight',
      kind: 'number',
      minimum: 0,
      level: 'advanced'
    },
    {
      name: 'multipleTesting',
      label: 'Multiple-testing correction',
      controlId: 'analysis-grid-correction',
      kind: 'enum',
      options: POINT_SHEET_CORRECTIONS,
      level: 'advanced'
    }
  ];
  const fields = createResidentSheetFields(
    sheet.values,
    specifications,
    name =>
      !saved &&
      (name === 'width' || name === 'height') &&
      initialization.lattice &&
      !JSON.parse(String(sheet.values['editedFields'])).includes(name)
        ? 'Declared row-major lattice metadata'
        : sheet.source(name),
    () => []
  );
  const checks = getResidentSheetChecks(fields, () =>
    getHotspotSheetSettings(sheet.values, analysis.rowCount, saved)
  );
  if (!analysis.availableColumns.includes(analysis.settings.column))
    checks.push('Choose an available numeric value field above.');
  return (
    <details className="analysis-hotspot-settings">
      <summary>Configure hot spots on a row-major grid</summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([analysis.coordinateRestorationKey, saved])}
        fields={fields}
        checks={checks}
        method={`Resident lattice hot spots use the numeric field above (${analysis.settings.column}) and analytic two-sided inference. Source row = row × width + column. Width × height must equal all ${analysis.rowCount.toLocaleString()} source rows. Geographic coordinates do not define these neighbors. Queen includes diagonals; rook uses Manhattan distance. Filters remove rows without reordering the grid. Self weight 1 gives Gi*; 0 gives Gi. Table p-values are unadjusted; significance bins use the chosen correction. Map colors show the z-score.`}
        submitLabel="Apply grid and run hot spots"
        onDraftChange={sheet.markEdited}
        onChange={sheet.change}
        onApply={current =>
          analysis.configureHotspots(
            getHotspotSheetSettings({...sheet.values, ...current}, analysis.rowCount, saved)
          )
        }
      />
    </details>
  );
}
