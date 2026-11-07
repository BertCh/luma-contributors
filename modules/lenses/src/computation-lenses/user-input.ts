// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {LensScenarioDescriptor, LensScenarioResult} from './types';

/**
 * Lets a lens scenario run on the caller's own table instead of its embedded dataset. A scenario
 * that loads one of the embedded dataframe datasets (see `loadDataFrameLensData`) reads its
 * columns by name; while user input is active those names resolve to the caller's mapped columns.
 * Reading a column that was not mapped throws `LensUserInputRoleError`, which names the role so a
 * host can ask for that field. Runs are serialized because the input is module scoped.
 */
export type LensUserColumn = ArrayLike<number | null>;
export type LensUserInput = {
  /** Embedded dataset the mapping replaces, such as `f-chicago-tracts`. */
  datasetId: string;
  rowCount: number;
  /** Embedded column name (role) to the caller's numeric values; null is a missing value. */
  columns: Readonly<Record<string, LensUserColumn>>;
};

export class LensUserInputRoleError extends Error {
  readonly role: string;
  constructor(role: string) {
    super(`Map a numeric field to "${role}" to run this scenario on your data`);
    this.name = 'LensUserInputRoleError';
    this.role = role;
  }
}

type ActiveInput = {input: LensUserInput; accessedRoles: Set<string>};
let activeInput: ActiveInput | null = null;
let queue: Promise<unknown> = Promise.resolve();

/**
 * Returns the embedded dataset, or while user input is active for it a copy whose `rowCount` and
 * `columns` come from the caller. Other embedded fields (projection, units) are kept.
 */
export function applyLensUserInput<Dataset extends {id: string; rowCount: number; columns: object}>(
  dataset: Dataset
): Dataset {
  const active = activeInput;
  if (!active || active.input.datasetId !== dataset.id) return dataset;
  const columns = new Proxy(
    {},
    {
      get(_target, property) {
        if (typeof property !== 'string') return undefined;
        active.accessedRoles.add(property);
        const values = active.input.columns[property];
        // The scenario needs this column and the caller did not map it.
        if (!values) throw new LensUserInputRoleError(property);
        return Array.from(values);
      },
      has: (_target, property) => typeof property === 'string' && property in active.input.columns,
      ownKeys: () => Object.keys(active.input.columns),
      getOwnPropertyDescriptor: (_target, property) =>
        typeof property === 'string' && property in active.input.columns
          ? {enumerable: true, configurable: true}
          : undefined
    }
  );
  return {...dataset, rowCount: active.input.rowCount, columns: columns as Dataset['columns']};
}

/** Embedded dataset ids a scenario can replace; `undefined` when it only runs on embedded data. */
export function getLensUserInputDatasetId(descriptor: LensScenarioDescriptor): string | undefined {
  const datasetId = descriptor.dataset?.id;
  return descriptor.entry === 'gpu-dataframe' || descriptor.entry === 'gpu-sql'
    ? datasetId === 'f-chicago-tracts' ||
      datasetId === 'f-chicago-nature' ||
      datasetId === 'f-ghcn-stations'
      ? datasetId
      : undefined
    : undefined;
}

/**
 * Runs a scenario with `input` standing in for its embedded dataset. Resolves with the record and
 * the roles the scenario read; on failure the error carries `accessedRoles` for host feedback.
 */
export async function runLensScenarioWithUserInput(
  descriptor: LensScenarioDescriptor,
  device: Device,
  input: LensUserInput,
  options: {signal?: AbortSignal} = {}
): Promise<{record: LensScenarioResult; accessedRoles: string[]}> {
  if (getLensUserInputDatasetId(descriptor) !== input.datasetId)
    // The scenario does not read this embedded dataset, so a mapping cannot apply.
    throw new Error('Scenario cannot run on user data for this dataset');
  const run = async () => {
    const active: ActiveInput = {input, accessedRoles: new Set()};
    activeInput = active;
    try {
      const loaded = await descriptor.load();
      const record = await loaded.run(device, options);
      return {record, accessedRoles: [...active.accessedRoles]};
    } catch (error) {
      if (error instanceof Error)
        (error as Error & {accessedRoles?: string[]}).accessedRoles = [...active.accessedRoles];
      throw error;
    } finally {
      activeInput = null;
    }
  };
  const result = queue.then(run, run);
  queue = result.catch(() => undefined);
  return result;
}

const EMBEDDED_DATASETS = {
  'f-chicago-tracts': () => import('./data/f-chicago-tracts.json'),
  'f-chicago-nature': () => import('./data/f-chicago-nature.json'),
  'f-ghcn-stations': () => import('./data/f-ghcn-stations.json')
} as const;

export type LensUserInputRole = {
  /** Column name the scenario reads from its embedded dataset. */
  role: string;
  /** First embedded value, shown as a hint for the expected unit and range. */
  example: number | null;
};

/**
 * Candidate roles of a scenario that can run on user data: every column of its embedded dataset.
 * A scenario reads only some of them; the run reports the ones it needs but were not mapped.
 */
export async function listLensUserInputRoles(
  descriptor: LensScenarioDescriptor
): Promise<{datasetId: string; roles: LensUserInputRole[]} | undefined> {
  const datasetId = getLensUserInputDatasetId(descriptor) as keyof typeof EMBEDDED_DATASETS;
  if (!datasetId) return undefined;
  const dataset = (await EMBEDDED_DATASETS[datasetId]()).default as {
    columns: Record<string, readonly (number | null)[]>;
  };
  return {
    datasetId,
    roles: Object.entries(dataset.columns).map(([role, values]) => ({
      role,
      example: values[0] ?? null
    }))
  };
}
