// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as dataframeAnalysis from '../../../src/dataframe-analysis/index';
import * as analysisCore from '../../../src/analysis-core/index';
import * as computationRecord from '../../../src/computation-record/index';
import * as geospatialAnalysis from '../../../src/geospatial-analysis/index';
import * as network from '../../../src/gpu-network/index';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as terrain from '../../../src/gpu-terrain/index';
import * as pointCloudAnalysis from '../../../src/pointcloud-analysis/index';
import * as rasterAnalysis from '../../../src/raster-analysis/index';
import {createPublicDeclarationInventory} from '../../../../../scripts/public-declaration-inventory.mjs';
import type {KernelAuditCase} from './kernel-audit-context';

/** Analysis entry barrels whose contributors the kernel audit builds. */
export const ANALYSIS_BARRELS: Readonly<Record<string, Record<string, unknown>>> = {
  'analysis-core': analysisCore,
  // Record producers (GPUTransferMask, GPUSnapshotTap) add command nodes too (CORE-115).
  'computation-record': computationRecord,
  'geospatial-analysis': geospatialAnalysis,
  'dataframe-analysis': dataframeAnalysis,
  'raster-analysis': rasterAnalysis,
  'pointcloud-analysis': pointCloudAnalysis,
  'gpu-terrain': terrain,
  'gpu-spatial-analysis': spatialAnalysis,
  'gpu-network': network
};

/** Inventory key of an export whose name more than one entry exports as a distinct function. */
export function getContributorKey(entry: string, name: string): string {
  return `${entry}/${name}`;
}

/**
 * Public contributors of the given barrels: classes with `getCommandNodes` or `addToGraph`, and
 * `add*Recipe` functions. Maps each key to its entry. The key is the export name, or
 * `entry/name` (see {@link getContributorKey}) when several entries export different functions
 * under that name, so each of them needs its own case (TOOL-112). One function re-exported by
 * several entries stays a single contributor keyed by name.
 */
export function getContributorNames(
  barrels: Readonly<Record<string, Record<string, unknown>>>
): Map<string, string> {
  const exportsByName = new Map<string, {entry: string; value: unknown}[]>();
  const entries = new Map(
    Object.keys(barrels).map(entry => [`@luma-contributors/analysis/${entry}`, entry])
  );
  const inventory = createPublicDeclarationInventory(
    [...entries].map(([entryPoint, entry]) => ({
      entryPoint,
      exports: Object.entries(barrels[entry])
    }))
  );
  for (const {entryPoint, exportName: name, declaration: value} of inventory) {
    const entry = entries.get(entryPoint)!;
    if (typeof value !== 'function') continue;
    const prototype = (value as {prototype?: {getCommandNodes?: unknown; addToGraph?: unknown}})
      .prototype;
    const isContributor =
      typeof prototype?.getCommandNodes === 'function' ||
      typeof prototype?.addToGraph === 'function' ||
      /^add\w+Recipe$/.test(name);
    if (!isContributor) continue;
    const exports = exportsByName.get(name) ?? [];
    if (!exports.some(existing => existing.value === value)) exports.push({entry, value});
    exportsByName.set(name, exports);
  }
  const contributors = new Map<string, string>();
  for (const [name, exports] of exportsByName) {
    if (exports.length === 1) {
      contributors.set(name, exports[0].entry);
    } else {
      for (const {entry} of exports) contributors.set(getContributorKey(entry, name), entry);
    }
  }
  return contributors;
}

/** Inventory key of a case: `entry/exportName` when the case names its entry, else the name. */
export function getKernelAuditCaseKey(
  auditCase: Pick<KernelAuditCase, 'entry' | 'exportName'>
): string {
  return auditCase.entry
    ? getContributorKey(auditCase.entry, auditCase.exportName)
    : auditCase.exportName;
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
  return cases.filter(auditCase => contributors.has(getKernelAuditCaseKey(auditCase)));
}

export type RuntimePublicDeclaration = {
  entryPoint: string;
  exportName: string;
  declaration: string;
  runtimeKind?: 'function' | 'value';
};

/** Read-only parity gate: all admitted manifest entries and value bindings must be audited. */
export function getRuntimeInventoryIssues(
  inventory: readonly RuntimePublicDeclaration[],
  barrels: Readonly<Record<string, Record<string, unknown>>>
): string[] {
  const packageName = '@luma-contributors/analysis';
  // Root forwarding and ontology metadata are not contributor-family runtime entries.
  const excluded = new Set([packageName, packageName + '/gpu-spatial-analysis/ontology']);
  const issues = new Set<string>();
  const origins = new Map<string, unknown>();
  for (const record of inventory) {
    if (excluded.has(record.entryPoint)) continue;
    const entry = record.entryPoint.slice(packageName.length + 1);
    const barrel = barrels[entry];
    if (!barrel) {
      issues.add(`missing runtime audit entry ${record.entryPoint}`);
      continue;
    }
    if (!record.runtimeKind) continue;
    let value: unknown = barrel;
    let present = true;
    for (const part of record.exportName.split('.')) {
      if (!value || !['object', 'function'].includes(typeof value) || !Object.hasOwn(value, part)) {
        present = false;
        break;
      }
      value = (value as Record<string, unknown>)[part];
    }
    const identity = `${record.entryPoint}:${record.exportName}`;
    if (!present) {
      issues.add(`missing runtime audit binding ${identity}`);
      continue;
    }
    if (record.runtimeKind === 'function' && typeof value !== 'function') {
      issues.add(`runtime audit binding ${identity} must be callable`);
    }
    if (origins.has(record.declaration) && !Object.is(origins.get(record.declaration), value)) {
      issues.add(
        `runtime audit binding ${identity} disagrees with shared declaration ${record.declaration}`
      );
    } else origins.set(record.declaration, value);
  }
  return [...issues].sort();
}
