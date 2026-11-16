// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {useAnalysisFormDraft, useAnalysisFormInitialization} from './analysis-form-initialization';

import {
  MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS,
  MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES,
  parseKeplerDBSCANSettings,
  type CatalogParameterValue,
  type WorkflowParameterSpec,
  type KeplerDBSCANSettings
} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {ResidentAnalysisSheetForm, type ResidentAnalysisSheetField} from './analysis-sheet';

/** Configure geographic density clustering as an atomic recipe update. */
export function DBSCANPanel({analysis}: {analysis: AnalysisRuntime}) {
  const settings = analysis.settings.dbscan;
  const summary = analysis.summary?.dbscan;
  return (
    <>
      <DBSCANControls
        key={JSON.stringify([analysis.datasetId, analysis.coordinateRestorationKey, settings])}
        analysis={analysis}
        settings={settings}
      />
      {summary && (
        <p className="hint">
          DBSCAN: {summary.clusterCount} clusters · {summary.noiseCount} noise points ·{' '}
          {summary.validCount} valid observations · {summary.excludedCount} excluded observations.
          {summary.complete
            ? ' Complete traversal.'
            : ' Incomplete traversal; results unavailable.'}
        </p>
      )}
    </>
  );
}

function DBSCANControls({
  analysis,
  settings
}: {
  analysis: AnalysisRuntime;
  settings?: KeplerDBSCANSettings;
}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const [values, setValues] = useAnalysisFormDraft<Record<string, CatalogParameterValue>>(
    {
      longitudeColumn: settings?.longitudeColumn ?? initialization.interpolation.longitudeColumn,
      latitudeColumn: settings?.latitudeColumn ?? initialization.interpolation.latitudeColumn,
      epsilon: settings?.epsilon ?? 1000,
      minimumSamples: settings?.minimumSamples ?? 5
    },
    JSON.stringify(settings),
    analysis.coordinateRestorationKey ?? analysis
  );
  const [edited, setEdited] = useState<ReadonlySet<string>>(() => new Set());
  const fields = createDBSCANSheetFields(values, analysis.availableColumns, name =>
    edited.has(name)
      ? 'User override'
      : settings
        ? 'Saved resident DBSCAN settings'
        : name === 'longitudeColumn' || name === 'latitudeColumn'
          ? initialization.coordinateSource
          : name === 'epsilon'
            ? 'Declared resident default: 1,000 meters; no k-distance knee has been computed'
            : 'Declared resident default: 5 samples including self'
  );
  return (
    <details className="analysis-dbscan-settings">
      <summary>Configure DBSCAN on geographic points</summary>
      <p className="hint">
        Group nearby dense observations using spherical great-circle distance. Coordinates are
        longitude and latitude in degrees; the neighborhood radius is in meters. The numeric
        analysis field does not affect clustering.
      </p>
      <ResidentAnalysisSheetForm
        fields={fields}
        checks={getDBSCANSheetChecks(values, analysis.availableColumns, analysis.rowCount)}
        method="Resident DBSCAN uses spherical great-circle distance and bounded exact OPTICS extraction. Catalog density clustering uses a local planar frame and canonical core-root labels; the methods can differ at distance boundaries and for border-point assignment."
        submitLabel="Apply radius and run DBSCAN"
        onChange={(name, value) => {
          setValues(previous => ({...previous, [name]: value}));
          setEdited(previous => new Set([...previous, name]));
        }}
        onApply={current => analysis.configureDBSCAN(getDBSCANSheetSettings(current))}
      />
      <p className="hint">
        Density counts include the focus point. Coordinates outside longitude/latitude bounds,
        missing coordinates and nonfinite coordinates are excluded. Selecting a population
        recomputes its clusters; excluded rows cannot connect clusters.
      </p>
      <p className="hint">
        Cluster IDs belong to original source rows. IDs repeat deterministically for the same source
        order, settings and population, but can change after filtering. Noise has value −1; excluded
        rows are missing. Gray indicates noise; cluster colors repeat after 15 clusters, so use the
        table IDs to distinguish them.
      </p>
      <p className="hint">
        At most {MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS.toLocaleString()} source rows and{' '}
        {MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES} minimum samples. This bounded exact computation uses
        OPTICS with DBSCAN extraction and completes every source-row visit. It does not perform
        native viewport clustering.
      </p>
    </details>
  );
}

/** Resident method metadata: field names and caps match the saved recipe, not the catalog executor. */
const DBSCAN_SHEET_PARAMETERS: readonly (WorkflowParameterSpec & {controlId: string})[] = [
  ...(['longitudeColumn', 'latitudeColumn'] as const).map(name => ({
    name,
    label: name === 'longitudeColumn' ? 'DBSCAN longitude field' : 'DBSCAN latitude field',
    controlId:
      name === 'longitudeColumn' ? 'analysis-dbscan-longitude' : 'analysis-dbscan-latitude',
    kind: 'column' as const,
    maximumLength: 1024,
    columnsFrom: 'points',
    fieldRole: {
      parameter: name,
      role: name === 'longitudeColumn' ? 'longitude' : 'latitude',
      label: name === 'longitudeColumn' ? 'Longitude in degrees' : 'Latitude in degrees',
      accepts: ['number', 'integer'] as const,
      required: true
    }
  })),
  {
    name: 'epsilon',
    label: 'DBSCAN neighborhood radius in meters',
    controlId: 'analysis-dbscan-epsilon',
    kind: 'number',
    minimum: 0,
    unit: 'meters',
    defaultValue: 1000
  },
  {
    name: 'minimumSamples',
    label: 'DBSCAN minimum samples including self',
    controlId: 'analysis-dbscan-minimum',
    kind: 'integer',
    minimum: 1,
    maximum: MAXIMUM_KEPLER_DBSCAN_MINIMUM_SAMPLES,
    defaultValue: 5
  }
];

export function createDBSCANSheetFields(
  values: Readonly<Record<string, CatalogParameterValue>>,
  columns: readonly string[],
  source: (name: string) => string
): ResidentAnalysisSheetField[] {
  return DBSCAN_SHEET_PARAMETERS.map(({controlId, ...specification}) => ({
    specification,
    controlId,
    value: values[specification.name] ?? '',
    columns: specification.fieldRole ? columns : undefined,
    source: source(specification.name)
  }));
}

/** Reuse the strict recipe parser for Float32 radius semantics and the geographic-only contract. */
export function getDBSCANSheetSettings(
  values: Readonly<Record<string, CatalogParameterValue>>
): KeplerDBSCANSettings {
  return parseKeplerDBSCANSettings({
    longitudeColumn: values['longitudeColumn'],
    latitudeColumn: values['latitudeColumn'],
    epsilon: values['epsilon'],
    minimumSamples: values['minimumSamples'],
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  });
}

export function getDBSCANSheetChecks(
  values: Readonly<Record<string, CatalogParameterValue>>,
  columns: readonly string[],
  rowCount: number
): string[] {
  const failures: string[] = [];
  if (
    !Number.isSafeInteger(rowCount) ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS
  )
    failures.push(
      `DBSCAN needs 1 to ${MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS.toLocaleString()} source rows. Choose a smaller source dataset; filtering does not reduce the source-row capacity.`
    );
  for (const name of ['longitudeColumn', 'latitudeColumn']) {
    if (!columns.includes(String(values[name] ?? '')))
      failures.push(
        `DBSCAN ${name}: choose an available prepared numeric coordinate column in degrees.`
      );
  }
  try {
    getDBSCANSheetSettings(values);
  } catch (failure) {
    failures.push(
      `${failure instanceof Error ? failure.message : String(failure)}. Choose coordinate fields, a positive Float32 radius and a supported minimum sample count.`
    );
  }
  return failures;
}
