// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WorkflowError, type WorkflowDocument} from './workflow-types';

export type WorkflowHistoryOptions = {
  /** Undo steps kept; older ones are dropped. Default 100. */
  maximumEntries?: number;
  /** Edits with the same tag inside this window form one undo step. Default 500. */
  coalesceMilliseconds?: number;
  /** Non-finite clock readings record separate undo entries and break coalescing. */
  now?: () => number;
};

/** Undo and redo by document snapshot. Documents are immutable, so snapshots share structure. */
export class WorkflowHistory {
  private currentDocument: WorkflowDocument;
  private readonly undoDocuments: WorkflowDocument[] = [];
  private readonly redoDocuments: WorkflowDocument[] = [];
  private readonly maximumEntries: number;
  private readonly coalesceMilliseconds: number;
  private readonly now: () => number;
  private lastTag: string | undefined;
  private lastAppliedAt = -Infinity;

  constructor(document: WorkflowDocument, options: WorkflowHistoryOptions = {}) {
    for (const name of ['maximumEntries', 'coalesceMilliseconds'] as const) {
      if (options[name] !== undefined && !Number.isFinite(options[name]))
        throw new WorkflowError(
          'parameter-out-of-range',
          `Workflow history: ${name} must be a finite number; got ${options[name]}.`
        );
    }
    this.currentDocument = document;
    this.maximumEntries = Math.max(0, Math.floor(options.maximumEntries ?? 100));
    this.coalesceMilliseconds = Math.max(0, options.coalesceMilliseconds ?? 500);
    this.now = options.now ?? Date.now;
  }

  get document(): WorkflowDocument {
    return this.currentDocument;
  }
  get canUndo(): boolean {
    return this.undoDocuments.length > 0;
  }
  get canRedo(): boolean {
    return this.redoDocuments.length > 0;
  }

  /** Records an edit. The same non-empty tag inside the coalesce window replaces the last entry. */
  apply(document: WorkflowDocument, tag?: string): void {
    if (document === this.currentDocument) return;
    const appliedAt = this.now();
    const elapsed = appliedAt - this.lastAppliedAt;
    if (
      !tag ||
      tag !== this.lastTag ||
      !Number.isFinite(appliedAt) ||
      !Number.isFinite(this.lastAppliedAt) ||
      elapsed < 0 ||
      elapsed > this.coalesceMilliseconds
    ) {
      this.undoDocuments.push(this.currentDocument);
      if (this.undoDocuments.length > this.maximumEntries) this.undoDocuments.shift();
    }
    this.currentDocument = document;
    this.redoDocuments.length = 0;
    // An invalid clock cannot establish a window, but the edit still needs an undo entry.
    if (Number.isFinite(appliedAt)) {
      this.lastTag = tag || undefined;
      this.lastAppliedAt = appliedAt;
    } else {
      this.clearCoalescing();
    }
  }

  /** Returns false when there is nothing to undo. */
  undo(): boolean {
    const document = this.undoDocuments.pop();
    if (!document) return false;
    this.redoDocuments.push(this.currentDocument);
    this.currentDocument = document;
    this.clearCoalescing();
    return true;
  }

  redo(): boolean {
    const document = this.redoDocuments.pop();
    if (!document) return false;
    this.undoDocuments.push(this.currentDocument);
    this.currentDocument = document;
    this.clearCoalescing();
    return true;
  }

  /** Replaces the document and clears both stacks. */
  reset(document: WorkflowDocument): void {
    this.currentDocument = document;
    this.undoDocuments.length = 0;
    this.redoDocuments.length = 0;
    this.clearCoalescing();
  }

  private clearCoalescing(): void {
    this.lastTag = undefined;
    this.lastAppliedAt = -Infinity;
  }
}
