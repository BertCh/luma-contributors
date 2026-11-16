// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY,
  MAXIMUM_KEPLER_DISTANCE_BAND_ROWS,
  type CatalogParameterValue,
  type WorkflowParameterSpec
} from '@luma-contributors/kepler';
import {AnalysisWeightsPanel} from './analysis-weights-panel';
import {
  makePointWeightsRecipe,
  normalizePointTopologySettings,
  normalizePointWeightsSettings,
  type PointTopologySettings,
  type PointWeightsSettings
} from './analysis-point-weights-settings';
import {useAnalysisFormDraft, useAnalysisFormInitialization} from './analysis-form-initialization';
import type {AnalysisRuntime} from './analysis-runtime';
import type {PointCoordinateDraft} from './analysis-point-coordinate-controls';
import {
  makePointCoordinateDraft,
  getPointCoordinateDraftSettings,
  validatePointCoordinateColumns
} from './analysis-point-coordinate-settings';
import {ResidentAnalysisSheetForm, type ResidentAnalysisSheetField} from './analysis-sheet';

export type PointSheetValues = Readonly<Record<string, CatalogParameterValue>>;
export type PointSheetSpecification = WorkflowParameterSpec & {controlId: string};
export const POINT_SHEET_CORRECTIONS = [
  'benjamini-hochberg',
  'benjamini-yekutieli',
  'bonferroni',
  'holm',
  'none'
] as const;

/** These are resident recipe fields. Catalog methods with different weighting retain their descriptors. */
export function makePointSheetDefaults(
  settings: PointTopologySettings | undefined,
  initialization: PointCoordinateDraft | undefined,
  rowCount: number,
  parameters: PointSheetValues
): Record<string, CatalogParameterValue> {
  return {
    ...(settings
      ? makePointCoordinateDraft(settings)
      : (initialization ?? makePointCoordinateDraft())),
    kind: settings?.kind === 'distance-band' ? 'distance-band' : 'knn',
    k: settings && settings.kind !== 'distance-band' ? settings.k : 8,
    radiusMeters: settings?.kind === 'distance-band' ? settings.radiusMeters : 1000,
    neighborCapacity:
      settings?.kind === 'distance-band'
        ? settings.neighborCapacity
        : Math.max(1, Math.min(rowCount * 32, MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY)),
    permutationDegree: settings?.kind === 'distance-band' ? (settings.permutationDegree ?? 64) : 64,
    ...parameters,
    editedFields: '[]'
  };
}

/** Shared AX draft reconciliation preserves edits as profile metadata improves and resets on replay. */
export function usePointAnalysisSheet(
  defaults: Record<string, CatalogParameterValue>,
  saved: unknown,
  scope: unknown,
  coordinateSource = 'Publication or guessed coordinate roles; check the fields and coordinate system',
  hasSavedSettings = Boolean(saved)
) {
  const [values, setValues, markEdited] = useAnalysisFormDraft(
    defaults,
    JSON.stringify(saved),
    scope
  );
  return {
    values,
    /** A pending or cleared text draft is an edit even when its committed value has not changed. */
    markEdited(name: string) {
      markEdited(name);
      setValues(previous => ({
        ...previous,
        editedFields: JSON.stringify([
          ...new Set([...(JSON.parse(String(previous['editedFields'])) as string[]), name])
        ])
      }));
    },
    change(name: string, value: CatalogParameterValue, changes: PointSheetValues = {}) {
      setValues(previous => ({
        ...previous,
        [name]: value,
        ...changes,
        editedFields: JSON.stringify([
          ...new Set([
            ...(JSON.parse(String(previous['editedFields'])) as string[]),
            name,
            ...Object.keys(changes)
          ])
        ])
      }));
    },
    source(name: string) {
      if ((JSON.parse(String(values['editedFields'])) as string[]).includes(name))
        return 'User override';
      if (hasSavedSettings) return 'Saved resident settings';
      if (name in makePointCoordinateDraft()) return coordinateSource;
      return `Declared resident default: ${String(defaults[name])}`;
    }
  };
}

/** Coordinate field names share the recipe's 1,024-character limit, including projected roles. */
export function makePointSheetColumn(
  name: string,
  label: string,
  controlId: string
): PointSheetSpecification {
  return {
    name,
    label,
    controlId,
    kind: 'column',
    maximumLength: 1024,
    columnsFrom: 'points',
    fieldRole: {parameter: name, role: name, label, accepts: ['number', 'integer'], required: true}
  };
}

export function createPointSheetFields(
  values: PointSheetValues,
  columns: readonly string[],
  prefix: string,
  label: string,
  source: (name: string) => string,
  parameters: readonly PointSheetSpecification[],
  options: {
    longitudeId?: string;
    latitudeId?: string;
    nearestId?: string;
    nearestLabel?: string;
    permutations?: boolean;
  } = {}
): ResidentAnalysisSheetField[] {
  const projected = values['coordinateSpace'] === 'planar';
  const specifications: PointSheetSpecification[] = [
    {
      name: 'coordinateSpace',
      label: `${label} coordinate system`,
      controlId: `${prefix}-coordinate-space`,
      kind: 'enum',
      options: ['longitude-latitude', 'planar'],
      description:
        'Geographic fields are longitude/latitude degrees with spherical distances in meters. Projected fields must already use a declared CRS with meter units; choosing them does not transform or verify the source projection. Keep separate geographic geometry for map display.'
    },
    ...(projected
      ? [
          makePointSheetColumn('xColumn', `${label} projected X field`, `${prefix}-x`),
          makePointSheetColumn('yColumn', `${label} projected Y field`, `${prefix}-y`),
          {
            name: 'crs',
            label: `${label} declared projected CRS`,
            controlId: `${prefix}-declared-crs`,
            kind: 'string' as const,
            maximumLength: 1024
          },
          ...(['minX', 'minY', 'maxX', 'maxY'] as const).map((name, index) => ({
            name,
            label: `${label} projected ${['minimum X', 'minimum Y', 'maximum X', 'maximum Y'][index]} in meters`,
            controlId: `${prefix}-${name}`,
            kind: 'number' as const
          }))
        ]
      : [
          makePointSheetColumn(
            'longitudeColumn',
            `${label} longitude field`,
            options.longitudeId ?? `${prefix}-longitude`
          ),
          makePointSheetColumn(
            'latitudeColumn',
            `${label} latitude field`,
            options.latitudeId ?? `${prefix}-latitude`
          )
        ]),
    {
      name: 'kind',
      label: `${label} topology`,
      controlId: `${prefix}-topology`,
      kind: 'enum',
      options: ['knn', 'distance-band'],
      description:
        'kNN selects nearest neighbors; distance bands include every coordinate-valid neighbor within the radius. Selection filters the source-wide graph without finding replacement neighbors. Float32 precision can affect neighbors near distance boundaries.'
    },
    ...(values['kind'] === 'distance-band'
      ? [
          {
            name: 'radiusMeters',
            label: `${label} radius in meters`,
            controlId: `${prefix}-radius`,
            kind: 'number' as const,
            minimum: 0
          },
          {
            name: 'neighborCapacity',
            label: `${label} directed neighbor capacity`,
            controlId: `${prefix}-capacity`,
            kind: 'integer' as const,
            minimum: 1,
            maximum: MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY,
            level: 'advanced' as const,
            description: `Distance bands support up to ${MAXIMUM_KEPLER_DISTANCE_BAND_ROWS.toLocaleString()} source rows. Capacity counts total directed slots; a symmetric pair uses two. Overflow makes results unavailable and reports the required count; neighbors are never silently truncated.`
          },
          ...(options.permutations
            ? [
                {
                  name: 'permutationDegree',
                  label: `${label} permutation neighbor slots per row`,
                  controlId: `${prefix}-permutation-degree`,
                  kind: 'integer' as const,
                  minimum: 64,
                  maximum: 512,
                  level: 'advanced' as const,
                  description:
                    'Raise scratch slots up to 512 for dense neighborhoods. Cost grows quadratically. Exceeding the active-neighbor limit makes inference unavailable even when total storage capacity is sufficient.'
                }
              ]
            : [])
        ]
      : [
          {
            name: 'k',
            label: options.nearestLabel ?? `${label} nearest neighbors`,
            controlId: options.nearestId ?? `${prefix}-neighbors`,
            kind: 'integer' as const,
            minimum: 1,
            maximum: 32
          }
        ]),
    ...parameters
  ];
  return specifications.map(({controlId, ...specification}) => ({
    specification,
    controlId,
    value: values[specification.name] ?? '',
    source: source(specification.name),
    columns: specification.fieldRole ? columns : undefined
  }));
}

/** Reuse the coordinate and topology recipe converters rather than inventing another settings shape. */
export function getPointSheetTopology(
  values: PointSheetValues,
  rowCount: number
): PointTopologySettings {
  const coordinates = makePointCoordinateDraft();
  for (const name of Object.keys(coordinates) as (keyof PointCoordinateDraft)[]) {
    if (name !== 'coordinateSpace') coordinates[name] = String(values[name] ?? '');
  }
  coordinates.coordinateSpace =
    values['coordinateSpace'] === 'planar' ? 'planar' : 'longitude-latitude';
  return normalizePointTopologySettings(
    {
      ...getPointCoordinateDraftSettings(coordinates),
      ...(values['kind'] === 'distance-band'
        ? {
            kind: 'distance-band',
            radiusMeters: Number(values['radiusMeters']),
            neighborCapacity: Number(values['neighborCapacity']),
            ...(Number(values['permutationDegree']) === 64
              ? {}
              : {permutationDegree: Number(values['permutationDegree'])})
          }
        : {k: Number(values['k'])})
    },
    rowCount
  );
}

export function getPointSheetChecks(
  fields: readonly ResidentAnalysisSheetField[],
  readSettings: () => PointTopologySettings
): string[] {
  const failures = fields.flatMap(field =>
    field.specification.fieldRole && !field.columns?.includes(String(field.value))
      ? [`${field.specification.label}: choose an available prepared numeric column.`]
      : []
  );
  try {
    readSettings();
  } catch (failure) {
    failures.push(
      `${failure instanceof Error ? failure.message : String(failure)}. Review the coordinate declarations, source-row limits and settings before applying.`
    );
  }
  return failures;
}

export function getPointWeightsSheetSettings(
  values: PointSheetValues,
  rowCount: number
): PointWeightsSettings {
  return normalizePointWeightsSettings(
    {
      ...getPointSheetTopology(values, rowCount),
      scale: Number(values['scale']),
      islandPolicy: values['islandPolicy'] as PointWeightsSettings['islandPolicy']
    },
    rowCount
  );
}

/** Coordinate fields and units are explicit: numeric columns do not imply a geographic CRS. */
export function PointWeightsPanel({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const saved = analysis.settings.pointWeights;
  const sheet = usePointAnalysisSheet(
    makePointSheetDefaults(saved, initialization.coordinates, analysis.rowCount, {
      scale: saved?.scale ?? 1,
      islandPolicy: saved?.islandPolicy ?? 'undefined',
      denominatorColumn: analysis.settings.denominatorColumn
    }),
    saved,
    analysis.coordinateRestorationKey ?? analysis.datasetId,
    initialization.coordinateSource
  );
  const fields = createPointSheetFields(
    sheet.values,
    analysis.availableColumns,
    'analysis-point',
    'Neighbor',
    sheet.source,
    [
      makePointSheetColumn(
        'denominatorColumn',
        'Spatial rate population field',
        'analysis-point-population'
      ),
      {
        name: 'scale',
        label: 'Rate scale',
        controlId: 'analysis-point-scale',
        kind: 'number',
        minimum: 0
      },
      {
        name: 'islandPolicy',
        label: 'Rows without eligible neighbors',
        controlId: 'analysis-point-islands',
        kind: 'enum',
        options: ['undefined', 'own-rate'],
        level: 'advanced'
      }
    ],
    {nearestLabel: 'Nearest neighbors'}
  );
  return (
    <details className="analysis-point-weights-settings">
      <summary>
        Configure {sheet.values['coordinateSpace'] === 'planar' ? 'projected' : 'geographic'}{' '}
        neighbors and spatial rates
      </summary>
      {analysis.weightsStore && (
        <AnalysisWeightsPanel
          analysis={analysis}
          family="point"
          getSettings={() => {
            const settings = getPointSheetTopology(sheet.values, analysis.rowCount);
            validatePointCoordinateColumns(settings, analysis.availableColumns);
            return makePointWeightsRecipe(settings);
          }}
        />
      )}
      <ResidentAnalysisSheetForm
        key={JSON.stringify([analysis.coordinateRestorationKey, saved])}
        fields={fields}
        checks={getPointSheetChecks(fields, () =>
          getPointWeightsSheetSettings(sheet.values, analysis.rowCount)
        )}
        method="Resident spatial rates pool each row's events and population with eligible neighbors, then multiply by the scale. The numeric field above supplies events. Filtering removes observations without replacing their neighbors."
        submitLabel="Apply neighbors and run spatial rates"
        onChange={sheet.change}
        onApply={current =>
          analysis.configurePointWeights(
            getPointWeightsSheetSettings({...sheet.values, ...current}, analysis.rowCount),
            String(current['denominatorColumn'])
          )
        }
      />
      <p className="hint">
        Missing or invalid coordinates have no links and produce undefined rates. Coordinates and
        distances use Float32. Dense clusters can make the first exact neighbor build slow;
        compatible analyses reuse the completed graph.
      </p>
    </details>
  );
}
