// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Pure accounting shared by the browser recorder and Node qualification specs. */
export const GALLERY_SAMPLE_IDENTIFIERS = Object.freeze([
  'h3-cells',
  'points-time',
  'districts',
  'connections',
  'trips',
  'trip-observations',
  'numeric-grid',
  'directed-network'
]);

export const AX_SURFACE_IDENTIFIERS = Object.freeze([
  'ax-suggestions',
  'ax-analysis-sheet',
  'ax-workflow-gallery',
  'ax-command-palette',
  'ax-result-card'
]);

export function accountJourney({steps, startedAt, setupFinishedAt, finishedAt}) {
  const setupActionCount = steps.filter(step => step.phase === 'setup').length;
  const duration = (start, finish) =>
    start !== null && finish !== null ? Math.round(finish - start) : null;
  return {
    actionCount: steps.length,
    setupActionCount,
    analysisActionCount: steps.length - setupActionCount,
    milliseconds: duration(startedAt, finishedAt),
    setupMilliseconds: duration(startedAt, setupFinishedAt),
    // Includes waits after setup, even when no subsequent action succeeds.
    analysisMilliseconds: duration(setupFinishedAt ?? startedAt, finishedAt)
  };
}

/** Missing and failed records never become zero-cost improvements. */
export function compareJourneys(records, baselineRecords) {
  return records.map(record => {
    const baseline = baselineRecords.find(candidate => candidate.id === record.id);
    const comparable =
      baseline?.outcome === 'result-visible' && record.outcome === 'result-visible';
    const delta = key =>
      comparable && Number.isFinite(record[key]) && Number.isFinite(baseline[key])
        ? record[key] - baseline[key]
        : null;
    return {
      id: record.id,
      baselineOutcome: baseline?.outcome ?? null,
      outcome: record.outcome,
      actionCountDelta: delta('actionCount'),
      analysisActionCountDelta: delta('analysisActionCount'),
      millisecondsDelta: delta('milliseconds')
    };
  });
}

/** A visible card is evidence only with explicit completion, identity and freshness metadata. */
export function getSuggestionResultFailure(result, expected) {
  if (!expected || !expected.sampleIdentifier || !expected.suggestionIdentifier)
    return 'Selected sample and suggestion identity are required';
  if (!result || result.status !== 'done')
    return 'Result computation has not completed successfully';
  if (result.hasError || /^\s*(?:error:|failed:)|GPU computation failed/i.test(result.text ?? ''))
    return 'Result reports a computation error';
  if (result.stale !== false || result.busy !== false) return 'Result is stale or still loading';
  if (result.sampleIdentifier !== expected.sampleIdentifier)
    return 'Result belongs to another sample';
  if (result.suggestionIdentifier !== expected.suggestionIdentifier)
    return 'Result belongs to another suggestion';
  if (!result.runIdentifier || result.runIdentifier === expected.previousRunIdentifier)
    return 'Result does not identify a fresh completed run';
  if (!result.text?.trim()) return 'Completed result card is empty';
  return null;
}

/** Optional absent surfaces are explicit not-run gates; actual failures always propagate. */
export function evaluateJourneyTargets(records, targets) {
  validateJourneyTargets(targets);
  const gates = [];
  for (const sampleIdentifier of GALLERY_SAMPLE_IDENTIFIERS) {
    const identifier = `first-result:${sampleIdentifier}`;
    const matches = records.filter(record => record.id === identifier);
    gates.push({
      id: `coverage:${sampleIdentifier}`,
      required: true,
      status: matches.length === 1 && matches[0].outcome === 'result-visible' ? 'passed' : 'failed',
      reason:
        matches.length === 1
          ? matches[0].outcome
          : `Expected one record, received ${matches.length}`
    });
  }
  for (const target of targets.journeys) {
    const matches = records.filter(record => record.id === target.id);
    const record = matches[0];
    let status = 'passed';
    let reason = 'Target met';
    if (matches.length > 1) {
      status = 'failed';
      reason = 'Duplicate journey records';
    } else if (!record || record.outcome === 'surface-not-present') {
      status = target.required ? 'failed' : 'not-run';
      reason = record?.reason ?? 'Journey not selected';
    } else if (record.outcome !== target.outcome) {
      status = 'failed';
      reason = `Expected ${target.outcome}, received ${record.outcome}`;
    } else {
      for (const [field, maximum] of Object.entries(target.maximum ?? {})) {
        if (!Number.isFinite(record[field]) || record[field] > maximum) {
          status = 'failed';
          reason = `${field}: expected at most ${maximum}, received ${record[field]}`;
          break;
        }
      }
    }
    if (status === 'passed' && target.id.startsWith('suggestion-result:')) {
      const expected = record.observations?.suggestionEvidence;
      const failure = getSuggestionResultFailure(record.observations?.resultEvidence, expected);
      if (failure || expected?.sampleIdentifier !== target.id.slice('suggestion-result:'.length)) {
        status = 'failed';
        reason = failure ?? 'Result evidence does not match the journey sample';
      }
    }
    gates.push({id: target.id, required: target.required, status, reason});
  }
  // Also propagate failures in untargeted journeys and partial --only runs.
  for (const record of records.filter(candidate => candidate.outcome === 'failed')) {
    gates.push({
      id: `execution:${record.id}`,
      required: true,
      status: 'failed',
      reason: record.error
    });
  }
  return {
    status: gates.some(gate => gate.status === 'failed')
      ? 'failed'
      : gates.some(gate => gate.status === 'not-run')
        ? 'incomplete'
        : 'passed',
    qualified: gates.every(gate => gate.status === 'passed'),
    gates
  };
}

function validatePolicyRecord(value, allowedFields, path) {
  if (
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    ![Object.prototype, null].includes(Object.getPrototypeOf(value))
  )
    throw new Error(`AX-70 targets: ${path} must be a plain object`);
  for (const field of Object.keys(value)) {
    if (!allowedFields.includes(field))
      throw new Error(`AX-70 targets: unknown ${path} field ${field}`);
  }
}

/** Validate the release policy before launching Chromium. No silent typo/no-op assertions. */
export function validateJourneyTargets(targets) {
  validatePolicyRecord(targets, ['schemaVersion', 'description', 'journeys'], 'policy');
  if (targets?.schemaVersion !== 1 || !Array.isArray(targets.journeys)) {
    throw new Error('AX-70 targets: schemaVersion must be 1 and journeys must be an array');
  }
  const identifiers = new Set();
  for (const target of targets.journeys) {
    validatePolicyRecord(target, ['id', 'required', 'outcome', 'maximum'], 'journey');
    if (
      typeof target.id !== 'string' ||
      !target.id.trim() ||
      identifiers.has(target.id) ||
      typeof target.required !== 'boolean' ||
      !['result-visible', 'observed'].includes(target.outcome)
    ) {
      throw new Error(`AX-70 targets: invalid or duplicate journey ${target.id}`);
    }
    identifiers.add(target.id);
    if (Object.hasOwn(target, 'maximum')) {
      validatePolicyRecord(
        target.maximum,
        ['actionCount', 'analysisActionCount', 'milliseconds', 'analysisMilliseconds'],
        `${target.id}.maximum`
      );
    }
    for (const [field, maximum] of Object.entries(target.maximum ?? {})) {
      if (
        !['actionCount', 'analysisActionCount', 'milliseconds', 'analysisMilliseconds'].includes(
          field
        ) ||
        !Number.isFinite(maximum) ||
        maximum < 0
      ) {
        throw new Error(
          `AX-70 targets: ${target.id} maximum ${field} must be a nonnegative finite limit`
        );
      }
    }
  }
  return targets;
}
