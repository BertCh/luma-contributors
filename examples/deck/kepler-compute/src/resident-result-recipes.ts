// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentJoinRequest, ResidentSpatialJoinRequest} from './resident-join-actions';
import type {ResidentSuitabilityRequest} from './resident-suitability-actions';

/**
 * The operation that produced a GPU-resident (over-cap) result. Parameters are the exact request
 * the result was computed from, so re-running it on equal inputs recomputes the same rows.
 */
export type ResidentResultOperation =
  | Readonly<{kind: 'join'; request: ResidentJoinRequest}>
  | Readonly<{kind: 'spatial-join'; request: ResidentSpatialJoinRequest}>
  | Readonly<{kind: 'suitability'; request: ResidentSuitabilityRequest}>;

/** CPU-only lineage of one resident result: operation, input dataset ids and revisions, rows. */
export type ResidentResultRecipe = Readonly<{
  resultId: string;
  rowCount: number;
  inputs: readonly Readonly<{datasetId: string; revision: number}>[];
  operation: ResidentResultOperation;
}>;

/** Latest recipe per result dataset ID. Entries for datasets that no longer exist are ignored. */
export class ResidentResultRecipes {
  private readonly recipes = new Map<string, ResidentResultRecipe>();

  /** Detaches the recipe from the caller's mutable request objects. */
  record(recipe: ResidentResultRecipe): void {
    this.recipes.set(recipe.resultId, structuredClone(recipe));
  }

  get(resultId: string): ResidentResultRecipe | undefined {
    return this.recipes.get(resultId);
  }

  /** Result-to-input links, for visibility closure in eviction. */
  getLineage(): {resultId: string; inputIds: string[]}[] {
    return [...this.recipes.values()].map(recipe => ({
      resultId: recipe.resultId,
      inputIds: recipe.inputs.map(input => input.datasetId)
    }));
  }

  forget(resultId: string): void {
    this.recipes.delete(resultId);
  }

  /**
   * Recipes for exactly the given resident result IDs, ordered so every recipe follows the recipes
   * of the resident results it consumes. Throws when a result has no recipe or on a cycle.
   */
  collect(residentResultIds: Iterable<string>): ResidentResultRecipe[] {
    const pending = new Map<string, ResidentResultRecipe>();
    for (const resultId of residentResultIds) {
      const recipe = this.recipes.get(resultId);
      // A resident result without lineage cannot be recomputed after device loss.
      if (!recipe) throw new Error(`Resident result ${resultId} has no recovery lineage`);
      pending.set(resultId, recipe);
    }
    const ordered: ResidentResultRecipe[] = [];
    while (pending.size) {
      const ready = [...pending.values()].filter(recipe =>
        recipe.inputs.every(input => !pending.has(input.datasetId))
      );
      // Equal dataset IDs in a chain would never become ready.
      if (!ready.length) throw new Error('Resident result lineage is cyclic');
      for (const recipe of ready) {
        ordered.push(recipe);
        pending.delete(recipe.resultId);
      }
    }
    return ordered;
  }
}
