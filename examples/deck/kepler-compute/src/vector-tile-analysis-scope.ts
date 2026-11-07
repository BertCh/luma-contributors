// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Viewport-versus-complete-dataset analysis scope for vector tiles. API FOR THE ANALYSIS LEAD.
 *
 * A vector tile dataset is a lazily fetched, zoom-dependent, geometry-clipped view of a tileset.
 * Whatever tiles are resident at one moment are a VIEWPORT WORKING SET, never the dataset. This
 * module turns a working-set snapshot into a typed descriptor, and gives analysis bindings one
 * call that either refuses or labels:
 *
 *   const scope = describeVectorTileAnalysisScope({...});
 *   const decision = decideVectorTileAnalysis(scope, {requiresCompleteDataset: true});
 *   if (!decision.allowed) show(decision.reason); else label(decision.resultLabel);
 *
 * Rules enforced here:
 * - The scope is `complete-dataset` only when the caller states the source was fully materialized
 *   (`isSourceFullyMaterialized`, for example a tileset converted to a resident table); resident
 *   tiles alone never qualify, even when every visible tile is loaded.
 * - Viewport scopes are refused for analyses that require the complete dataset (global
 *   statistics, spatial weights, regression, kriging fitting, hotspot inference).
 * - Viewport scopes that are allowed carry the label `Viewport working set` plus tile and zoom
 *   context and flags: `isTileCoverageComplete` (every visible tile loaded, none rejected),
 *   `hasSplitFeatureDoubleCountRisk` (features without a stable id may be counted per fragment).
 */

import type {VectorTileFeatureRegistrySummary} from './vector-tile-feature-ids';

export type VectorTileAnalysisScopeKind = 'viewport-working-set' | 'complete-dataset';

export type VectorTileAnalysisScopeInputs = Readonly<{
  datasetId: string;
  zoom: number;
  visibleTileKeys: readonly string[];
  loadedTileKeys: readonly string[];
  rejectedTileKeys?: readonly string[];
  /** Feature registry summary from the working set; omit when features carry no tracked ids. */
  featureSummary?: VectorTileFeatureRegistrySummary;
  /** True only when the whole tileset was materialized into a resident table. */
  isSourceFullyMaterialized?: boolean;
}>;

export type VectorTileAnalysisScope = Readonly<{
  kind: VectorTileAnalysisScopeKind;
  datasetId: string;
  zoom: number;
  visibleTileCount: number;
  loadedVisibleTileCount: number;
  missingTileKeys: readonly string[];
  rejectedTileKeys: readonly string[];
  /** Every visible tile loaded and none rejected; still a viewport population. */
  isTileCoverageComplete: boolean;
  /** Counts per fragment may exceed counts per feature (no stable id, or property-hash only). */
  hasSplitFeatureDoubleCountRisk: boolean;
  /** Geometry is generalized per zoom, so results change with zoom. */
  isZoomDependent: boolean;
  /** Always false for viewport working sets. */
  canInferGlobally: boolean;
  label: string;
}>;

export type VectorTileAnalysisRequest = Readonly<{
  /** Global statistics, weights, model fitting and inference require every feature. */
  requiresCompleteDataset: boolean;
  /** Counting analyses that must not double count split features. */
  requiresStableFeatureIds?: boolean;
}>;

export type VectorTileAnalysisDecision = Readonly<{
  allowed: boolean;
  /** Present when allowed: stamp this on result tables and legends. */
  resultLabel?: string;
  /** Present when refused. */
  reason?: string;
}>;

export function describeVectorTileAnalysisScope(
  inputs: VectorTileAnalysisScopeInputs
): VectorTileAnalysisScope {
  const loaded = new Set(inputs.loadedTileKeys);
  const rejectedTileKeys = inputs.rejectedTileKeys ?? [];
  const missingTileKeys = inputs.visibleTileKeys.filter(tileKey => !loaded.has(tileKey));
  const isComplete = Boolean(inputs.isSourceFullyMaterialized);
  const isTileCoverageComplete = missingTileKeys.length === 0 && rejectedTileKeys.length === 0;
  const summary = inputs.featureSummary;
  const hasSplitFeatureDoubleCountRisk = !summary || summary.hasCollisionRisk;
  const loadedVisibleTileCount = inputs.visibleTileKeys.length - missingTileKeys.length;
  const kind: VectorTileAnalysisScopeKind = isComplete
    ? 'complete-dataset'
    : 'viewport-working-set';
  const label = isComplete
    ? 'Complete dataset'
    : `Viewport working set (${loadedVisibleTileCount} of ${inputs.visibleTileKeys.length} visible tiles` +
      `, zoom ${Math.round(inputs.zoom * 100) / 100}${isTileCoverageComplete ? '' : ', incomplete'})`;
  return {
    kind,
    datasetId: inputs.datasetId,
    zoom: inputs.zoom,
    visibleTileCount: inputs.visibleTileKeys.length,
    loadedVisibleTileCount,
    missingTileKeys,
    rejectedTileKeys,
    isTileCoverageComplete,
    hasSplitFeatureDoubleCountRisk,
    isZoomDependent: !isComplete,
    canInferGlobally: isComplete,
    label
  };
}

export function decideVectorTileAnalysis(
  scope: VectorTileAnalysisScope,
  request: VectorTileAnalysisRequest
): VectorTileAnalysisDecision {
  if (scope.kind === 'complete-dataset') return {allowed: true, resultLabel: scope.label};
  if (request.requiresCompleteDataset)
    return {
      allowed: false,
      reason: `${scope.label} is not the complete dataset; this analysis needs every feature.`
    };
  if (request.requiresStableFeatureIds && scope.hasSplitFeatureDoubleCountRisk)
    return {
      allowed: false,
      reason: `${scope.label} contains features without stable ids, so split features could be counted more than once.`
    };
  return {allowed: true, resultLabel: scope.label};
}
