// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as dataframeAnalysis from '../../../src/dataframe-analysis/index';
import * as analysisCore from '../../../src/analysis-core/index';
import * as geospatialAnalysis from '../../../src/geospatial-analysis/index';
import * as network from '../../../src/gpu-network/index';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as terrain from '../../../src/gpu-terrain/index';
import * as pointCloudAnalysis from '../../../src/pointcloud-analysis/index';
import * as rasterAnalysis from '../../../src/raster-analysis/index';
import type {KernelAuditCase} from './kernel-audit-context';

/** Analysis entry barrels whose contributors the kernel audit builds. */
export const ANALYSIS_BARRELS: Readonly<Record<string, Record<string, unknown>>> = {
  'analysis-core': analysisCore,
  'geospatial-analysis': geospatialAnalysis,
  'dataframe-analysis': dataframeAnalysis,
  'raster-analysis': rasterAnalysis,
  'pointcloud-analysis': pointCloudAnalysis,
  'gpu-terrain': terrain,
  'gpu-spatial-analysis': spatialAnalysis,
  'gpu-network': network
};

/**
 * Public contributors of the given barrels: classes with `getCommandNodes` or `addToGraph`, and
 * `add*Recipe` functions. Maps each name to its entry.
 */
export function getContributorNames(
  barrels: Readonly<Record<string, Record<string, unknown>>>
): Map<string, string> {
  const contributors = new Map<string, string>();
  for (const [entry, barrel] of Object.entries(barrels)) {
    for (const [name, value] of Object.entries(barrel)) {
      if (typeof value !== 'function') continue;
      const prototype = (value as {prototype?: {getCommandNodes?: unknown; addToGraph?: unknown}})
        .prototype;
      const isContributor =
        typeof prototype?.getCommandNodes === 'function' ||
        typeof prototype?.addToGraph === 'function' ||
        /^add\w+Recipe$/.test(name);
      if (isContributor && !contributors.has(name)) contributors.set(name, entry);
    }
  }
  return contributors;
}

let analysisContributors: Map<string, string> | undefined;

/** Contributors exported by the analysis entries. */
export function getAnalysisContributors(): Map<string, string> {
  analysisContributors ??= getContributorNames(ANALYSIS_BARRELS);
  return analysisContributors;
}

/**
 * Keeps the cases of analysis contributors. Cases that name contributors owned by published
 * `@luma.gl/graph` are not this package's to audit and are skipped.
 */
export function selectAnalysisCases(cases: readonly KernelAuditCase[]): KernelAuditCase[] {
  const contributors = getAnalysisContributors();
  return cases.filter(auditCase => contributors.has(auditCase.exportName));
}
