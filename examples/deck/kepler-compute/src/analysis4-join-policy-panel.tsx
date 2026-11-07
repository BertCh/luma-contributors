// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type JoinPolicyRecord = Readonly<{
  id: string;
  label: string;
  duplicateRightMatches: 'all' | 'first' | 'last';
  /** undefined keeps the native behavior: duplicate names are kept as they are. */
  rightColumnSuffix: string | undefined;
  dropRightKey: boolean;
}>;

export type JoinPolicyChange = Partial<{
  duplicateRightMatches: 'all' | 'first' | 'last';
  rightColumnSuffix: string | undefined;
  dropRightKey: boolean;
}>;

type JoinOperationLike = {
  id: string;
  implementation?: string;
  resultLabel?: string;
  duplicateRightMatches?: 'all' | 'first' | 'last';
  rightColumnSuffix?: string;
  dropRightKey?: boolean;
};

/** Pending attribute joins (not spatial) with their current host policy values. */
export function listJoinPolicyRecords(
  joins: readonly JoinOperationLike[] | undefined
): JoinPolicyRecord[] {
  return (joins ?? [])
    .filter(operation => operation.implementation !== 'spatial')
    .map(operation => ({
      id: operation.id,
      label: operation.resultLabel || operation.id,
      duplicateRightMatches: operation.duplicateRightMatches ?? 'all',
      rightColumnSuffix: operation.rightColumnSuffix,
      dropRightKey: Boolean(operation.dropRightKey)
    }));
}

export type JoinPolicyPanelProps = Readonly<{
  joins: readonly JoinPolicyRecord[];
  setPolicy(id: string, change: JoinPolicyChange): void;
}>;

/**
 * Companion to Kepler's native Join panel, which has no such settings: choose the duplicate-match,
 * suffix-naming and drop-key policies of a pending join. The values are stored on the join
 * operation, so the Run button of the native panel executes them. Renders nothing without a join.
 */
export function JoinPolicyPanel({joins, setPolicy}: JoinPolicyPanelProps) {
  if (joins.length === 0) return null;
  return (
    <section aria-label="Join policies" data-testid="join-policy-panel">
      <h2>Join policies</h2>
      {joins.map(join => {
        const isSuffixEnabled = join.rightColumnSuffix !== undefined;
        return (
          <fieldset key={join.id} data-testid={`join-policy-${join.id}`}>
            <legend>{join.label}</legend>
            <label>
              Duplicate right matches{' '}
              <select
                value={join.duplicateRightMatches}
                data-testid="join-duplicate-select"
                onChange={event =>
                  setPolicy(join.id, {
                    duplicateRightMatches: event.target.value as 'all' | 'first' | 'last'
                  })
                }
              >
                <option value="all">Keep every match (one output row per match)</option>
                <option value="first">First match only</option>
                <option value="last">Last match only</option>
              </select>
            </label>
            <label>
              <input
                type="checkbox"
                checked={isSuffixEnabled}
                data-testid="join-suffix-enabled"
                onChange={event =>
                  setPolicy(join.id, {
                    rightColumnSuffix: event.target.checked ? '_right' : undefined,
                    ...(event.target.checked ? {} : {dropRightKey: false})
                  })
                }
              />{' '}
              Rename colliding right columns with a suffix
            </label>
            {isSuffixEnabled ? (
              <>
                <label>
                  Suffix{' '}
                  <input
                    type="text"
                    value={join.rightColumnSuffix}
                    data-testid="join-suffix-input"
                    onChange={event => setPolicy(join.id, {rightColumnSuffix: event.target.value})}
                  />
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={join.dropRightKey}
                    data-testid="join-drop-key"
                    onChange={event => setPolicy(join.id, {dropRightKey: event.target.checked})}
                  />{' '}
                  Drop the right key when it has the same name as the left key
                </label>
              </>
            ) : (
              <p className="hint">
                Without a suffix, colliding column names are kept as the native Join keeps them.
              </p>
            )}
          </fieldset>
        );
      })}
    </section>
  );
}
