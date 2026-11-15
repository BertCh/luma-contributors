// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  parseWorkflowCollection,
  type KeplerJSONValue,
  type WorkflowCollection
} from '@luma-contributors/kepler';

/** Name of the member under `settings.extensions` of the sandbox state. */
export const SANDBOX_EXTENSION_WORKFLOWS = 'workflows';

/** The part of the workbench store that saving and replay need. */
export type WorkbenchExtensionBinding = {
  serialize(): KeplerJSONValue | undefined;
  restore(collection: WorkflowCollection): void;
  clear(): Promise<void>;
};

let binding: WorkbenchExtensionBinding | null = null;
let pending: WorkflowCollection | null = null;

/**
 * Connects the workbench store to sandbox saves. This module stays free of React and of the app
 * runtime so the sandbox state code (and its node specs) can import it; the store binds itself
 * when it is created. A member replayed before that is applied on binding.
 */
export function bindWorkbenchExtension(next: WorkbenchExtensionBinding | null): void {
  binding = next;
  if (binding && pending) {
    binding.restore(pending);
    pending = null;
  }
}

/** The saved member, or undefined when there are no workflows. */
export function collectWorkbenchExtension(): KeplerJSONValue | undefined {
  return binding?.serialize();
}

/** Strict check before replay changes anything; throws on a malformed member. */
export function validateWorkbenchExtension(value: unknown): WorkflowCollection | undefined {
  return value === undefined ? undefined : parseWorkflowCollection(value);
}

/** Applies a validated member after the workspace is rebuilt; an absent member clears the workbench. */
export function applyWorkbenchExtension(collection: WorkflowCollection | undefined): void {
  if (!binding) pending = collection ?? null;
  else if (collection) binding.restore(collection);
  else void binding.clear();
}
