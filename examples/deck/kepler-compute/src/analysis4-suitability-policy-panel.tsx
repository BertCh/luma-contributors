// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentSuitabilityMissingValuePolicy} from './resident-suitability-plan';

export type SuitabilityPolicyRecord = Readonly<{
  id: string;
  label: string;
  missingValues: ResidentSuitabilityMissingValuePolicy;
  categoryScoreText: string;
}>;

type SuitabilityOperationLike = {
  id: string;
  resultLabel?: string;
  missingValues?: string;
  categoryScores?: Readonly<Record<string, Readonly<Record<string, number>>>>;
};

export type SuitabilityCategoryScores = Readonly<Record<string, Readonly<Record<string, number>>>>;

/**
 * Category scores are typed one field per line as `field: label=score, label=score`. Labels may
 * not contain commas or equals signs; every score must be finite, otherwise the text is refused.
 */
export function parseSuitabilityCategoryScoreText(text: string): {
  scores?: SuitabilityCategoryScores;
  error?: string;
} {
  const scores: Record<string, Record<string, number>> = {};
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const separator = line.indexOf(':');
    if (separator < 1) return {error: `Line "${line}" needs "field: label=score, ..."`};
    const field = line.slice(0, separator).trim();
    const entries: Record<string, number> = {};
    for (const part of line.slice(separator + 1).split(',')) {
      if (!part.trim()) continue;
      const equals = part.lastIndexOf('=');
      const label = part.slice(0, equals).trim();
      const score = Number(part.slice(equals + 1));
      if (equals < 1 || !label || part.slice(equals + 1).trim() === '' || !Number.isFinite(score))
        return {error: `"${part.trim()}" needs label=score with a finite score`};
      entries[label] = score;
    }
    scores[field] = entries;
  }
  return {scores};
}

export function formatSuitabilityCategoryScoreText(
  scores: SuitabilityCategoryScores | undefined
): string {
  return Object.entries(scores ?? {})
    .map(
      ([field, entries]) =>
        `${field}: ${Object.entries(entries)
          .map(([label, score]) => `${label}=${score}`)
          .join(', ')}`
    )
    .join('\n');
}

/** Pending Suitability operations with their current missing-value policy. */
export function listSuitabilityPolicyRecords(
  operations: readonly SuitabilityOperationLike[] | undefined
): SuitabilityPolicyRecord[] {
  return (operations ?? []).map(operation => ({
    id: operation.id,
    label: operation.resultLabel || operation.id,
    missingValues:
      operation.missingValues === 'exclude' || operation.missingValues === 'renormalize'
        ? operation.missingValues
        : 'refuse',
    categoryScoreText: formatSuitabilityCategoryScoreText(operation.categoryScores)
  }));
}

export type SuitabilityPolicyPanelProps = Readonly<{
  suitabilities: readonly SuitabilityPolicyRecord[];
  setPolicy(id: string, missingValues: ResidentSuitabilityMissingValuePolicy): void;
  /** Stores per-field category scores on the operation (an empty object restores dictionary order). */
  setCategoryScores?(id: string, categoryScores: SuitabilityCategoryScores): void;
}>;

/**
 * Companion to Kepler's native Suitability panel: choose how rows missing in only some weighted
 * fields are scored. The value is stored on the operation, so the native Run button executes it.
 */
export function SuitabilityPolicyPanel({
  suitabilities,
  setPolicy,
  setCategoryScores
}: SuitabilityPolicyPanelProps) {
  if (suitabilities.length === 0) return null;
  return (
    <section aria-label="Suitability missing values" data-testid="suitability-policy-panel">
      <h2>Suitability missing values</h2>
      {suitabilities.map(operation => (
        <label key={operation.id}>
          {operation.label}{' '}
          <select
            value={operation.missingValues}
            data-testid={`suitability-missing-${operation.id}`}
            onChange={event =>
              setPolicy(operation.id, event.target.value as ResidentSuitabilityMissingValuePolicy)
            }
          >
            <option value="refuse">Stop when rows are missing in only some fields</option>
            <option value="exclude">Exclude those rows (score null)</option>
            <option value="renormalize">Renormalize weights over the defined fields</option>
          </select>
        </label>
      ))}
      {setCategoryScores
        ? suitabilities.map(operation => (
            <label key={`${operation.id}-categories`}>
              {operation.label} category scores (one field per line, field: label=score, ...){' '}
              <textarea
                defaultValue={operation.categoryScoreText}
                data-testid={`suitability-categories-${operation.id}`}
                onBlur={event => {
                  const parsed = parseSuitabilityCategoryScoreText(event.target.value);
                  if (parsed.scores) setCategoryScores(operation.id, parsed.scores);
                  event.target.setCustomValidity(parsed.error ?? '');
                  if (parsed.error) event.target.reportValidity();
                }}
              />
            </label>
          ))
        : null}
    </section>
  );
}
