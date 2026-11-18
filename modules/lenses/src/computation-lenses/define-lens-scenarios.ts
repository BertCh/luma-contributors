// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from './registry';
import type {LensScenarioDescriptor} from './types';

/** Fields every scenario in a registration group shares; individual entries may override them. */
export type LensScenarioGroupDefaults = Partial<
  Pick<
    LensScenarioDescriptor,
    'entry' | 'audience' | 'rungs' | 'defaultRung' | 'dataset' | 'example'
  >
>;

/** One scenario in a group; `id` is the full `<entry>.<name>[.<variant>]` identifier. */
export type LensScenarioGroupEntry = Pick<LensScenarioDescriptor, 'id' | 'element' | 'title'> &
  Partial<Omit<LensScenarioDescriptor, 'load'>> & {load: LensScenarioDescriptor['load']};

/** Default rung ladder shared by nearly every scenario. */
export const DEFAULT_LENS_RUNGS = ['result', 'datum', 'kernel', 'graph'] as const;

/** Rung ladder for scenarios that document a formula. */
export const FORMULA_LENS_RUNGS = ['result', 'datum', 'formula', 'kernel', 'graph'] as const;

/** Rung ladder for formula scenarios that also expose a table. */
export const TABLE_LENS_RUNGS = ['result', 'datum', 'formula', 'kernel', 'graph', 'table'] as const;

/** A scenario declared as data: only `id`, `element`, `title` and `load` are required. */
export type LensScenarioRow = Pick<LensScenarioDescriptor, 'id' | 'element' | 'title' | 'load'> &
  Partial<Omit<LensScenarioDescriptor, 'id' | 'element' | 'title' | 'load'>>;

/**
 * Registers one scenario from a data row. `entry` defaults to the first id segment, `audience` to
 * `'analyst'`, `rungs` to {@link DEFAULT_LENS_RUNGS} and `defaultRung` to `'result'`.
 */
export function defineLensScenario(row: LensScenarioRow): void {
  registerLensScenario(createLensScenarioDescriptor(row));
}

/** Build the same defaulted descriptor without mutating the registry. @internal */
export function createLensScenarioDescriptor(row: LensScenarioRow): LensScenarioDescriptor {
  return {
    entry: row.id.split('.')[0],
    audience: 'analyst',
    rungs: DEFAULT_LENS_RUNGS,
    defaultRung: 'result',
    ...row
  };
}

/**
 * Registers a group of scenarios that share an entry, dataset and rung defaults. Per-entry fields
 * win over the group defaults; ids, elements and titles stay explicit at the call site so they
 * remain greppable.
 */
export function registerLensScenarios(
  defaults: LensScenarioGroupDefaults,
  entries: readonly LensScenarioGroupEntry[]
): void {
  for (const entry of entries) {
    defineLensScenario({...defaults, ...entry});
  }
}
