// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {planJoinKeyPrecision, planSpatialJoinSemantics} from './analysis-join-policy';
import {planKeplerGroupByPasses} from '@luma-contributors/kepler';
import {
  classifyRateInvalidCauses,
  diagnoseFloat32Narrowing,
  type NumericNarrowingReport
} from './analysis-numeric-diagnostics';
import type {ResidentGroupByPlan} from './resident-group-by-plan';
import type {ResidentJoinPlan} from './resident-join-plan';
import type {ResidentSpatialJoinPlan} from './resident-spatial-join-plan';

/**
 * Binds the policy planners (join, spatial join, group-by) to the resident plans the GPU runtime
 * executes. The planners describe the contract and cross-check the plan; they do not change what
 * runs. Each function returns one human-readable line for panels and workflow notices.
 */

export function describeResidentJoinPolicy(plan: ResidentJoinPlan): string {
  // Resident keys are dictionary strings or Int32/Uint32/integral Float32, all exact on the GPU.
  const precision = planJoinKeyPrecision(plan.keyKind === 'string' ? 'string' : 'float32');
  const duplicates =
    plan.duplicateRightMatches === 'all'
      ? 'every duplicate right match is joined'
      : `only the ${plan.duplicateRightMatches} duplicate right match is joined`;
  const naming =
    plan.provenance.rightColumnSuffix === undefined
      ? 'colliding right columns are prefixed with the right dataset name'
      : `colliding right columns receive the suffix "${plan.provenance.rightColumnSuffix}"`;
  return (
    `Join policy: ${plan.joinType}, ${plan.keyKind} key (${precision.mode}); null keys never match; ` +
    `${duplicates}; rows follow left order then unmatched right; ${naming}.`
  );
}

/**
 * Cross-checks the plan's boundary policy against the semantics planner. Returns the description
 * line, or throws when the two disagree (an invariant break, not a user error).
 */
export function describeResidentSpatialJoinPolicy(plan: ResidentSpatialJoinPlan): string {
  const {semantics, refusal} = planSpatialJoinSemantics(plan.predicate, 'EPSG:4326', 'EPSG:4326');
  // Planner and runtime must agree on closed/open boundaries.
  if (
    !semantics ||
    (semantics.boundary === 'inclusive') !== (plan.provenance.boundaryPolicy === 'closed')
  )
    throw new Error(refusal ?? 'Spatial Join boundary policy mismatch');
  return (
    `Spatial Join policy: ${semantics.predicate}, boundary ${semantics.boundary}, ${semantics.crs} ` +
    `longitude/latitude on both sides (no reprojection); one result row per point in source order; ` +
    `polygons without geometry never match.`
  );
}

export function describeResidentGroupByPolicy(plan: ResidentGroupByPlan, rowCount = 0): string {
  // The GPU runtime executes exactly this pass plan: key windows crossed with measure batches.
  const passPlan = planKeplerGroupByPasses({
    keySlotCount: plan.keyCapacity,
    measureCount: plan.measures.length,
    rowCount
  });
  const count = passPlan.passes.length;
  return (
    `Group By policy: ${plan.keyCapacity} key slots in ${passPlan.windowCount} window${passPlan.windowCount === 1 ? '' : 's'}, ` +
    `${plan.measures.length} measures in ${passPlan.batchCount} batch${passPlan.batchCount === 1 ? '' : 'es'}; ` +
    `${count === 1 ? 'single GPU pass' : `${count} GPU passes merged in first-occurrence order`}; ` +
    `Float32 accumulation.`
  );
}

/** One-line numeric narrowing verdict for a column that is about to be stored as Float32. */
export function describeNumericNarrowing(
  format: Parameters<typeof diagnoseFloat32Narrowing>[0],
  values: ArrayLike<number | bigint | null>
): string {
  return describeNumericNarrowingReport(diagnoseFloat32Narrowing(format, values));
}

/** One-line verdict for an already computed (possibly chunk-merged) narrowing report. */
export function describeNumericNarrowingReport(report: NumericNarrowingReport): string {
  const {format} = report;
  return report.verdict === 'exact'
    ? `${format} column narrows to Float32 exactly (${report.rowCount} rows).`
    : report.verdict === 'lossy'
      ? `${format} column is rounded in ${report.inexactCount} of ${report.rowCount} rows (max error ${report.maximumAbsoluteError}).`
      : `${format} column overflows Float32 in ${report.overflowCount} rows; narrowing refused.`;
}

/** One-line explanation of why rate rows were excluded, by cause. */
export function describeRateInvalidCauses(
  events: ArrayLike<number | null>,
  populations: ArrayLike<number | null>,
  mask?: ArrayLike<number | boolean>
): string {
  return describeRateCauseCounts(classifyRateInvalidCauses(events, populations, mask).counts);
}

/** Cause-count line shared by the whole-array and block-merged rate diagnostics. */
export function describeRateCauseCounts(counts: Readonly<Record<string, number>>): string {
  const excluded = Object.entries(counts).filter(
    ([cause, count]) => cause !== 'valid' && count > 0
  );
  return excluded.length
    ? `Excluded rows: ${excluded.map(([cause, count]) => `${count} ${cause}`).join(', ')}.`
    : 'No rows excluded.';
}
