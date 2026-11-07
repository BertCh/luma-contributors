// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  getLensScenario,
  listLensScenarios,
  listLensUserInputRoles,
  runLensScenarioWithUserInput,
  type LensScenarioResult
} from '@luma-contributors/lenses/computation-lenses';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogTable,
  type CatalogDiagnostic
} from './catalog-geojson';
import type {CatalogRunOutput} from './catalog-binding-types';

/** A registered, self-contained lens scenario: it runs a real contributor on its own embedded dataset. */
export type CatalogLensScenario = {
  id: string;
  entry: string;
  /** Documented export the scenario covers, such as `GPULocalMoran`. */
  element: string;
  title: string;
  audience: 'analyst' | 'engineer';
  datasetId?: string;
  /** True when the scenario can run on a mapped user table instead of its embedded dataset. */
  acceptsUserData?: boolean;
};

/** One typed role of a user-data run: an embedded column name the user maps a numeric field to. */
export type CatalogLensFieldRole = {role: string; kind: 'number'; example: number | null};
/** Role name to the user's numeric column; null entries are missing values. */
export type CatalogLensUserData = {
  rowCount: number;
  columns: Readonly<Record<string, ArrayLike<number | null>>>;
};

/**
 * Lists the lens scenarios registered so far. Scenario modules register themselves when they are
 * imported (`@luma.gl/graph` computation-lenses scenarios index); the host decides when to pay
 * for that import, so an empty list means the registry has not been loaded.
 */
export function listCatalogLensScenarios(): CatalogLensScenario[] {
  return listLensScenarios().map(descriptor => ({
    id: descriptor.id,
    entry: descriptor.entry,
    element: descriptor.element,
    title: descriptor.title,
    audience: descriptor.audience,
    datasetId: descriptor.dataset?.id,
    acceptsUserData: acceptsUserData(descriptor)
  }));
}

function acceptsUserData(descriptor: Parameters<typeof listLensUserInputRoles>[0]): boolean {
  return (
    (descriptor.entry === 'gpu-dataframe' || descriptor.entry === 'gpu-sql') &&
    descriptor.dataset?.id !== undefined &&
    ['f-chicago-tracts', 'f-chicago-nature', 'f-ghcn-stations'].includes(descriptor.dataset.id)
  );
}

/** Roles the field-mapping form offers for a scenario; empty when it only runs on embedded data. */
export async function listCatalogLensFieldRoles(
  scenarioId: string
): Promise<CatalogLensFieldRole[]> {
  const descriptor = getLensScenario(scenarioId);
  if (!descriptor) return [];
  const listing = await listLensUserInputRoles(descriptor);
  return (listing?.roles ?? []).map(role => ({...role, kind: 'number' as const}));
}

/**
 * Builds the user-data table from a field mapping (role to field name) and a reader of numeric
 * columns. Non-finite values become null. Throws when a mapped field has no values.
 */
export function mapCatalogLensUserData(
  mapping: Readonly<Record<string, string>>,
  rowCount: number,
  readColumn: (fieldName: string) => ArrayLike<number | null> | undefined
): CatalogLensUserData {
  const columns: Record<string, ArrayLike<number | null>> = {};
  for (const [role, fieldName] of Object.entries(mapping)) {
    if (!fieldName) continue;
    const values = readColumn(fieldName);
    // The mapped field must exist and be numeric.
    if (!values || values.length !== rowCount) throw new Error(`Field ${fieldName} is not numeric`);
    columns[role] = Array.from(values, value =>
      typeof value === 'number' && Number.isFinite(value) ? value : null
    );
  }
  return {rowCount, columns};
}

/** Groups scenarios by the documented export they cover. */
export function groupCatalogLensScenarios(
  scenarios: readonly CatalogLensScenario[]
): Map<string, CatalogLensScenario[]> {
  const grouped = new Map<string, CatalogLensScenario[]>();
  for (const scenario of scenarios) {
    const list = grouped.get(scenario.element) ?? [];
    list.push(scenario);
    grouped.set(scenario.element, list);
  }
  return grouped;
}

/**
 * Runs one lens scenario on the device and summarizes the recorded computation: graph forms,
 * compiled nodes, measurements and analyst choices. The scenario's own dataset is used; catalog
 * inputs are not read.
 */
export async function runCatalogLensScenario(
  device: Device,
  scenarioId: string,
  options: {signal?: AbortSignal; userData?: CatalogLensUserData} = {}
): Promise<CatalogRunOutput> {
  const descriptor = getLensScenario(scenarioId);
  // Scenario not registered; import the scenario registry first.
  if (!descriptor) throw new Error('Unknown lens scenario');
  let record: LensScenarioResult;
  if (options.userData) {
    const datasetId = descriptor.dataset?.id;
    // Only scenarios backed by an embedded dataframe dataset accept a field mapping.
    if (!datasetId || !acceptsUserData(descriptor))
      throw new Error('This scenario only runs on its embedded dataset');
    ({record} = await runLensScenarioWithUserInput(
      descriptor,
      device,
      {datasetId, rowCount: options.userData.rowCount, columns: options.userData.columns},
      {signal: options.signal}
    ));
  } else {
    const loaded = await descriptor.load();
    record = await loaded.run(device, options);
  }
  const diagnostics: CatalogDiagnostic[] = [];
  const nodes = makeCatalogTable('lens-nodes', `Recorded nodes of ${descriptor.element}`, {
    graph_id: 'string',
    form: 'string',
    node_id: 'string',
    label: 'string',
    kind: 'string',
    operation: 'string',
    workgroups: 'string'
  });
  let nodeCount = 0;
  for (const graph of record.graphs)
    for (const node of graph.nodes) {
      nodeCount++;
      appendCatalogRow(nodes, {
        graph_id: graph.id,
        form: graph.form,
        node_id: node.id,
        label: node.label,
        kind: node.kind,
        operation: node.op ?? null,
        workgroups: node.dispatch ? node.dispatch.workgroups.join('x') : null
      });
    }
  const choices = makeCatalogTable('lens-choices', 'Analyst choices', {
    stage: 'string',
    name: 'string',
    value: 'string',
    label: 'string'
  });
  for (const choice of record.choices)
    appendCatalogRow(choices, {
      stage: choice.stage,
      name: choice.name,
      value: String(choice.value),
      label: choice.label ?? null
    });
  addCatalogDiagnostic(
    diagnostics,
    'lens-embedded-dataset',
    'info',
    options.userData
      ? `Ran on ${options.userData.rowCount} rows of your table through the mapped fields ${Object.keys(options.userData.columns).join(', ')}.`
      : descriptor.dataset
        ? `Ran on the embedded dataset ${descriptor.dataset.id} (${descriptor.dataset.licence}), not on catalog inputs.`
        : 'Ran on data embedded in the scenario, not on catalog inputs.'
  );
  return {
    tables: [nodes, choices],
    diagnostics,
    summary: {
      scenarioId: descriptor.id,
      element: descriptor.element,
      runId: record.runId,
      adapter: `${record.adapter.backend}${record.adapter.description ? ` ${record.adapter.description}` : ''}`,
      graphCount: record.graphs.length,
      nodeCount,
      measuredGraphCount: record.metrics?.length ?? 0
    },
    dispatchedContributors: [descriptor.element]
  };
}
