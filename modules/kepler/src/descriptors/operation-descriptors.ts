// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CURATED_DRIVE_TIME_BANDS_DESCRIPTOR} from './descriptors-curated-drive-time-bands';
import {CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR} from './descriptors-curated-cell-flow-topology';
import {CURATED_NETWORK_DISRUPTION_DESCRIPTOR} from './descriptors-curated-network-disruption';
import {CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR} from './descriptors-curated-regional-raster-change';
import {CURATED_ZONAL_EXPOSURE_DESCRIPTOR} from './descriptors-curated-zonal-exposure';
import {CURATED_SURFACE_PREPARATION_DESCRIPTOR} from './descriptors-curated-surface-preparation';
import {CURATED_DRAINAGE_NETWORK_DESCRIPTOR} from './descriptors-curated-drainage-network';
import {CURATED_FLEET_DWELL_DESCRIPTOR} from './descriptors-curated-fleet-dwell';
import {CURATED_LANDSCAPE_METRICS_DESCRIPTOR} from './descriptors-curated-landscape-metrics';
import {CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR} from './descriptors-curated-emerging-hot-spots';
import {CURATED_TRACK_QUALITY_DESCRIPTOR} from './descriptors-curated-track-quality';
import {CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR} from './descriptors-curated-dasymetric-redistribution';
import {
  getGPUAnalysisDataKindForCatalogGeometry,
  type GPUAnalysisQuestion
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogOperationBinding} from '../analysis/catalog-binding-types';
import {AGGREGATION_OPERATION_DESCRIPTORS} from './descriptors-aggregation';
import {CLUSTERING_OPERATION_DESCRIPTORS} from './descriptors-clustering';
import {CURATED_OPERATION_DESCRIPTORS} from './descriptors-curated';
import {GEOMETRY_A_OPERATION_DESCRIPTORS} from './descriptors-geometry-a';
import {GEOMETRY_B_OPERATION_DESCRIPTORS} from './descriptors-geometry-b';
import {INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS} from './descriptors-interpolation-density';
import {NETWORK_OPERATION_DESCRIPTORS} from './descriptors-network';
import {POINT_CLOUD_OPERATION_DESCRIPTORS} from './descriptors-pointcloud';
import {NETWORK_CATEGORICAL_OPERATION_DESCRIPTORS} from './descriptors-network-categorical';
import type {
  KeplerOperationDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';
import {
  MAXIMUM_OPERATION_ACTION_LENGTH,
  validateKeplerOperationDescriptor
} from './validate-operation-descriptor';
import {RASTER_OPERATION_DESCRIPTORS} from './descriptors-raster';
import {RASTER_EMBEDDING_OPERATION_DESCRIPTORS} from './descriptors-raster-embedding';
import {STATISTICS_OPERATION_DESCRIPTORS} from './descriptors-statistics';
import {VALIDATION_OPERATION_DESCRIPTORS} from './descriptors-validation';
import {TRAJECTORY_OPERATION_DESCRIPTORS} from './descriptors-trajectory';
import {SPACETIME_OPERATION_DESCRIPTORS} from './descriptors-spacetime';

// The registry of hand-written descriptors, and a conservative descriptor for any binding that has
// none. Another session adds bindings while descriptors are written, so every consumer asks
// `resolveKeplerOperationDescriptor` and never assumes a descriptor exists.

/** All hand-written descriptors, one per registered binding, in family order. */
export const KEPLER_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  ...GEOMETRY_A_OPERATION_DESCRIPTORS,
  ...GEOMETRY_B_OPERATION_DESCRIPTORS,
  ...NETWORK_OPERATION_DESCRIPTORS,
  ...NETWORK_CATEGORICAL_OPERATION_DESCRIPTORS,
  ...AGGREGATION_OPERATION_DESCRIPTORS,
  ...RASTER_OPERATION_DESCRIPTORS,
  ...RASTER_EMBEDDING_OPERATION_DESCRIPTORS,
  ...STATISTICS_OPERATION_DESCRIPTORS,
  ...CLUSTERING_OPERATION_DESCRIPTORS,
  ...INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS,
  ...CURATED_OPERATION_DESCRIPTORS,
  CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR,
  CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR,
  CURATED_TRACK_QUALITY_DESCRIPTOR,
  CURATED_LANDSCAPE_METRICS_DESCRIPTOR,
  CURATED_DRAINAGE_NETWORK_DESCRIPTOR,
  CURATED_SURFACE_PREPARATION_DESCRIPTOR,
  CURATED_ZONAL_EXPOSURE_DESCRIPTOR,
  CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR,
  CURATED_NETWORK_DISRUPTION_DESCRIPTOR,
  CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR,
  CURATED_DRIVE_TIME_BANDS_DESCRIPTOR,
  CURATED_FLEET_DWELL_DESCRIPTOR,
  ...POINT_CLOUD_OPERATION_DESCRIPTORS,
  ...VALIDATION_OPERATION_DESCRIPTORS,
  ...TRAJECTORY_OPERATION_DESCRIPTORS,
  ...SPACETIME_OPERATION_DESCRIPTORS
];

const DESCRIPTORS_BY_BINDING_ID: ReadonlyMap<string, KeplerOperationDescriptor> = new Map(
  KEPLER_OPERATION_DESCRIPTORS.map(descriptor => [descriptor.bindingId, descriptor])
);

/** Returns the hand-written descriptor of a binding, or undefined when it has none. */
export function getKeplerOperationDescriptor(
  bindingId: string
): KeplerOperationDescriptor | undefined {
  return DESCRIPTORS_BY_BINDING_ID.get(bindingId);
}

/**
 * How many leading parameters of a derived descriptor are `basic`. A binding that nobody has
 * described gets a short settings list; the rest is folded away so that an undescribed operation
 * does not open with a wall of numbers.
 */
const DERIVED_BASIC_PARAMETER_COUNT = 3;

/** Derived descriptors are specialist until someone describes the operation. */
const DERIVED_PROMINENCE = 3;

/** Question a derived descriptor assigns, by the binding's domain. */
const DOMAIN_QUESTIONS: Readonly<Record<CatalogOperationBinding['domain'], GPUAnalysisQuestion>> = {
  geometry: 'prepare-data',
  network: 'what-is-near-or-reachable',
  aggregation: 'how-much-per-area',
  interpolation: 'value-where-unmeasured',
  density: 'where-concentrated',
  clustering: 'which-groups-or-regions',
  raster: 'shape-of-the-surface',
  analysis: 'where-significant'
};

/** Cuts a title to at most `limit` characters at the last space, or hard when it has none. */
function truncateAtWordBoundary(title: string, limit: number): string {
  const trimmed = title.trim();
  if (trimmed.length <= limit) {
    return trimmed;
  }
  const cut = trimmed.slice(0, limit + 1);
  const lastSpace = cut.lastIndexOf(' ');
  return (lastSpace > 0 ? cut.slice(0, lastSpace) : trimmed.slice(0, limit)).trimEnd();
}

/**
 * Builds a descriptor from the binding alone. It claims nothing the binding does not state:
 * the question comes from the domain, the input kind from the first accepted geometry, every
 * parameter keeps its default (the first three are shown first), the only requirements are the
 * binding's own capacities, no column is mapped to a role (string parameters stay plain
 * parameters), every apply policy is conservatively rebuild until graph semantics are described,
 * and prominence is 3, so the operation is listed under "Browse all" and never leads
 * a suggestion.
 */
export function deriveKeplerOperationDescriptor(
  binding: CatalogOperationBinding
): KeplerOperationDescriptor {
  const requirements: KeplerOperationRequirement[] = binding.inputs.flatMap(input => [
    ...(input.dataKind === 'raster'
      ? []
      : [
          {
            input: input.name,
            id: 'maximum-rows' as const,
            severity: 'block' as const,
            limit: input.maximumFeatures,
            remedy: 'Filter the layer to the area of interest'
          }
        ]),
    ...(input.dataKind === 'table' || input.dataKind === 'raster'
      ? []
      : [
          {
            input: input.name,
            id: 'maximum-vertices' as const,
            severity: 'block' as const,
            limit: input.maximumVertices,
            remedy: 'Simplify the geometry or split the layer'
          }
        ])
  ]);
  return {
    bindingId: binding.id,
    questions: [DOMAIN_QUESTIONS[binding.domain]],
    action: truncateAtWordBoundary(binding.title, MAXIMUM_OPERATION_ACTION_LENGTH),
    prominence: DERIVED_PROMINENCE,
    inputs: binding.inputs.map(input => ({
      name: input.name,
      kind:
        input.dataKind === 'raster'
          ? 'surface'
          : input.dataKind === 'table'
            ? 'attributes'
            : getGPUAnalysisDataKindForCatalogGeometry(input.accepts[0] ?? 'point'),
      fields: []
    })),
    parameters: binding.parameters.map((parameter, index) => ({
      name: parameter.name,
      apply: 'rebuild' as const,
      defaultPolicy: {from: 'constant' as const},
      level: index < DERIVED_BASIC_PARAMETER_COUNT ? ('basic' as const) : ('advanced' as const)
    })),
    requirements,
    outputs: binding.outputs.map(output => ({
      tableId: output.tableId,
      kind:
        output.geometry === 'none'
          ? 'summary'
          : output.geometry === 'mixed'
            ? 'render-geometry'
            : getGPUAnalysisDataKindForCatalogGeometry(output.geometry)
    }))
  };
}

/** A descriptor for a binding and whether a person wrote it. */
export type KeplerResolvedOperationDescriptor = {
  descriptor: KeplerOperationDescriptor;
  /** `hand-written`: from the registry. `derived`: built from the binding alone. */
  source: 'hand-written' | 'derived';
};

/** Returns the hand-written descriptor of a binding, or derives one, and says which. */
export function resolveKeplerOperationDescriptor(
  binding: CatalogOperationBinding
): KeplerResolvedOperationDescriptor {
  const handWritten = getKeplerOperationDescriptor(binding.id);
  return handWritten
    ? {descriptor: handWritten, source: 'hand-written'}
    : {descriptor: deriveKeplerOperationDescriptor(binding), source: 'derived'};
}

/** How well hand-written descriptors cover a set of bindings. */
export type KeplerOperationDescriptorCoverage = {
  /** Binding ids that have a hand-written descriptor. */
  handWrittenBindingIds: readonly string[];
  /** Binding ids that only a derived descriptor describes. */
  derivedBindingIds: readonly string[];
  /** Hand-written descriptors whose binding is not in the set. */
  orphanedDescriptorIds: readonly string[];
  /** One sentence per mismatch between a hand-written descriptor and its binding. */
  validationIssues: readonly string[];
};

/** Reports which bindings are described by hand, which are not, and whether the descriptors agree. */
export function getKeplerOperationDescriptorCoverage(
  bindings: readonly CatalogOperationBinding[]
): KeplerOperationDescriptorCoverage {
  const bindingIds = new Set(bindings.map(binding => binding.id));
  const handWrittenBindingIds: string[] = [];
  const derivedBindingIds: string[] = [];
  const validationIssues: string[] = [];
  for (const binding of bindings) {
    const descriptor = getKeplerOperationDescriptor(binding.id);
    if (descriptor) {
      handWrittenBindingIds.push(binding.id);
      validationIssues.push(...validateKeplerOperationDescriptor(descriptor, binding));
    } else {
      derivedBindingIds.push(binding.id);
    }
  }
  return {
    handWrittenBindingIds,
    derivedBindingIds,
    orphanedDescriptorIds: KEPLER_OPERATION_DESCRIPTORS.filter(
      descriptor => !bindingIds.has(descriptor.bindingId)
    ).map(descriptor => descriptor.bindingId),
    validationIssues
  };
}
