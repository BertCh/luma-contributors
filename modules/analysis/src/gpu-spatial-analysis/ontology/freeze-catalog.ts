// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisStability, GPUSpatialAnalysisCapability} from './spatial-analysis-ontology';

/** Authored catalog data may omit stability and receive the experimental default. */
export type CatalogRecordDefinition<Value> = Omit<Value, 'stability'> & {
  readonly stability?: GPUAnalysisStability;
};

export type CatalogCapabilityDefinition = CatalogRecordDefinition<GPUSpatialAnalysisCapability>;

export function withCatalogStability<Value extends object>(
  record: Value & {readonly stability?: GPUAnalysisStability}
): Value & {readonly stability: GPUAnalysisStability} {
  return {...record, stability: record.stability ?? 'experimental'};
}

/** Freeze catalog data once at assembly, including nested evidence, limits and member arrays. */
export function freezeCatalog<Value>(value: Value): Value {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freezeCatalog(child);
    Object.freeze(value);
  }
  return value;
}
