// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addWorkflowStep} from './workflow-edit';
import {getWorkflowPortCompatibility} from './workflow-operations';
import type {
  WorkflowDatasetProfile,
  WorkflowDocument,
  WorkflowOperationRegistry,
  WorkflowPortKind
} from './workflow-types';

/** A next step worth offering for some data, with the reason in plain words. */
export type WorkflowRecommendation = {
  /** Stable id: `<operationId>` or `<operationId>:<variant>`. */
  id: string;
  operationId: string;
  /** Short action label, at most 40 characters: `Find clusters`, `Count points per polygon`. */
  label: string;
  /** One sentence: what the user gets and what it needs (`Needs a numeric column`). */
  reason: string;
  /** Input port of the operation that reads the data. */
  inputName: string;
  /** Parameter values to preset (for example a numeric column for `valueProperty`). */
  parameters: Record<string, string | number | boolean>;
};

/** What is known about some data when recommending: its kind and its columns. */
export type WorkflowRecommendationSubject = {
  kind: WorkflowPortKind;
  rowCount?: number;
  /** Non-geometry columns with their declared type. */
  columns: readonly {name: string; type: 'string' | 'float64' | 'int32' | 'boolean'}[];
};

type ColumnRole = 'numeric' | 'predictor' | 'category' | 'longitude' | 'latitude' | 'first';

/** Ordered rules; column roles describe requirements and the parameters they populate. */
export type WorkflowRecommendationCandidate = {
  operationId: string;
  label: string;
  reason: string;
  kinds: readonly WorkflowPortKind[];
  inputName: string;
  /** Per-kind ordering where a general operation is especially useful. */
  priority?: Partial<Record<WorkflowPortKind, number>>;
  parameters?: Readonly<Record<string, string | number | boolean>>;
  columnParameters?: Readonly<Record<string, ColumnRole>>;
  /** Only offered when every otherwise eligible rule exceeds its input row limit. */
  sample?: boolean;
};

export const WORKFLOW_RECOMMENDATION_CANDIDATES: readonly WorkflowRecommendationCandidate[] = [
  {
    operationId: 'density-clustering',
    label: 'Find clusters',
    kinds: ['point'],
    reason: 'Find dense groups of points and flag isolated observations.',
    inputName: 'points'
  },
  {
    operationId: 'kernel-density-grid',
    label: 'Create a density surface',
    kinds: ['point'],
    reason: 'Turn point concentrations into a smooth density grid.',
    inputName: 'points'
  },
  {
    operationId: 'recipe-hot-spot-lattice',
    label: 'Find hot spots',
    kinds: ['point'],
    reason: 'Count points on a lattice and identify significant hot and cold spots.',
    inputName: 'points'
  },
  {
    operationId: 'hotspot-gi-star',
    priority: {polygon: 6},
    label: 'Find value hot spots',
    kinds: ['point', 'polygon'],
    reason: 'Find unusually high or low neighboring values; needs a numeric column.',
    inputName: 'features',
    columnParameters: {valueProperty: 'numeric'}
  },
  {
    operationId: 'voronoi-diagram',
    label: 'Create Voronoi catchments',
    kinds: ['point'],
    reason: 'Divide the surrounding area into the nearest catchment for each point.',
    inputName: 'points'
  },
  {
    operationId: 'group-convex-hull',
    label: 'Draw hulls per group',
    kinds: ['point'],
    reason: 'Outline each group of points; needs a string or integer category column.',
    inputName: 'points',
    columnParameters: {groupProperty: 'category'}
  },
  {
    operationId: 'local-outlier-factor',
    label: 'Find local outliers',
    kinds: ['point'],
    reason: 'Find points that are isolated relative to their local neighbors.',
    inputName: 'points'
  },
  {
    operationId: 'k-means',
    label: 'Group nearby points',
    kinds: ['point'],
    reason: 'Partition points into compact geographic clusters.',
    inputName: 'points'
  },
  {
    operationId: 'dggs-cell-aggregation',
    label: 'Count points in grid cells',
    kinds: ['point'],
    reason: 'Aggregate points into a regular Quadbin grid.',
    inputName: 'points',
    parameters: {family: 'quadbin'}
  },
  {
    operationId: 'geographic-distribution',
    label: 'Measure geographic spread',
    priority: {features: 1},
    kinds: ['point', 'line', 'polygon'],
    inputName: 'features',
    reason: 'Calculate centers, standard distance and a distribution ellipse.'
  },
  {
    operationId: 'idw-grid',
    label: 'Interpolate a value surface',
    kinds: ['point'],
    reason: 'Estimate a numeric column between observations using inverse distance weighting.',
    inputName: 'points',
    columnParameters: {valueProperty: 'numeric'}
  },
  {
    operationId: 'kriging-grid',
    label: 'Estimate a kriging surface',
    kinds: ['point'],
    reason: 'Estimate a value surface and uncertainty; needs a numeric column.',
    inputName: 'points',
    columnParameters: {valueProperty: 'numeric'}
  },
  {
    operationId: 'regression-ols',
    label: 'Explain a numeric value',
    kinds: ['point'],
    reason:
      'Fit ordinary least squares; needs a numeric response and a different numeric predictor.',
    inputName: 'points',
    columnParameters: {responseProperty: 'numeric', predictorProperties: 'predictor'}
  },
  {
    operationId: 'line-simplification',
    priority: {line: 0},
    label: 'Simplify lines',
    kinds: ['line'],
    reason: 'Remove unnecessary vertices while retaining the overall shape of each line.',
    inputName: 'lines'
  },
  {
    operationId: 'line-length-per-polygon',
    priority: {line: 1},
    label: 'Measure length per polygon',
    kinds: ['line'],
    reason: 'Total line length inside each area; also needs a connected polygon dataset.',
    inputName: 'lines'
  },
  {
    operationId: 'network-service-area',
    priority: {line: 2},
    label: 'Find a network service area',
    kinds: ['line'],
    reason: 'Find reachable network edges; also needs a connected origin point.',
    inputName: 'lines'
  },
  {
    operationId: 'network-centrality',
    priority: {line: 3},
    label: 'Find central network nodes',
    kinds: ['line'],
    reason: 'Measure node degree and closeness in a bounded line network.',
    inputName: 'lines'
  },
  {
    operationId: 'line-density-grid',
    priority: {line: 4},
    label: 'Create a line density grid',
    kinds: ['line'],
    reason: 'Measure how strongly lines concentrate across a grid.',
    inputName: 'lines'
  },
  {
    operationId: 'geometry-measures',
    label: 'Measure geometry',
    priority: {polygon: 0, features: 0, line: 5},
    kinds: ['point', 'line', 'polygon'],
    reason: 'Calculate area, length or perimeter, bounds and vertex count for each feature.',
    inputName: 'features'
  },
  {
    operationId: 'geometry-validity',
    priority: {polygon: 1},
    label: 'Check polygon validity',
    kinds: ['polygon'],
    reason: 'Report invalid polygon rings and intersections before further analysis.',
    inputName: 'polygons'
  },
  {
    operationId: 'geometry-topology-repair',
    priority: {polygon: 2},
    label: 'Repair polygon topology',
    kinds: ['polygon'],
    reason: 'Rebuild valid polygon rings; needs a GPU device.',
    inputName: 'polygons'
  },
  {
    operationId: 'geometry-label-point',
    priority: {polygon: 3},
    label: 'Place labels inside polygons',
    kinds: ['polygon'],
    reason: 'Create a label point in the interior of each polygon.',
    inputName: 'polygons'
  },
  {
    operationId: 'geometry-overlay-intersection',
    priority: {polygon: 4},
    label: 'Intersect polygon layers',
    kinds: ['polygon'],
    reason: 'Keep overlapping areas; also needs a second polygon dataset and a GPU device.',
    inputName: 'left'
  },
  {
    operationId: 'spatial-weights',
    priority: {polygon: 5},
    label: 'Build neighbor weights',
    kinds: ['point', 'polygon'],
    reason: 'Create weighted neighbor links for spatial analysis.',
    inputName: 'features'
  },
  {
    operationId: 'local-moran-statistics',
    priority: {polygon: 7},
    label: 'Find value clusters',
    kinds: ['point', 'polygon'],
    reason: 'Identify local clusters and outliers of a numeric column with local Moran statistics.',
    inputName: 'features',
    columnParameters: {valueProperty: 'numeric'}
  },
  {
    operationId: 'table-points-from-columns',
    label: 'Make points from coordinates',
    kinds: ['table'],
    reason: 'Create point geometry from numeric longitude and latitude columns.',
    inputName: 'rows',
    columnParameters: {longitudeColumn: 'longitude', latitudeColumn: 'latitude'}
  },
  {
    operationId: 'table-summarize',
    label: 'Summarise rows',
    kinds: ['table'],
    reason: 'Count the rows in the table.',
    inputName: 'rows'
  },
  {
    operationId: 'table-filter-rows',
    label: 'Filter rows',
    kinds: ['table'],
    reason: 'Keep rows matching an expression; edit the initial expression to choose your subset.',
    inputName: 'rows',
    parameters: {expression: 'true'}
  },
  {
    operationId: 'table-sort-rows',
    label: 'Take a sample first',
    kinds: ['point', 'line', 'polygon', 'table'],
    inputName: 'rows',
    sample: true,
    reason: 'Keep a bounded subset sorted by a column before applying smaller analysis operations.',
    parameters: {limit: 1000},
    columnParameters: {column: 'first'}
  }
];

/** The subject for a host dataset profile (geometry field, point columns or neither). */
export function getWorkflowRecommendationSubject(
  profile: WorkflowDatasetProfile
): WorkflowRecommendationSubject {
  const geometryFields = new Set(profile.geometryFields);
  const pointColumns = profile.pointColumns;
  return {
    kind: geometryFields.size ? 'features' : pointColumns ? 'point' : 'table',
    rowCount: profile.rowCount,
    columns: profile.fields
      .filter(
        field =>
          !geometryFields.has(field.name) &&
          field.name !== pointColumns?.longitude &&
          field.name !== pointColumns?.latitude
      )
      .map(field => ({
        name: field.name,
        type:
          field.type === 'real'
            ? 'float64'
            : field.type === 'integer'
              ? 'int32'
              : field.type === 'boolean'
                ? 'boolean'
                : 'string'
      }))
  };
}

function isNumericColumn(column: WorkflowRecommendationSubject['columns'][number]): boolean {
  return (
    (column.type === 'float64' || column.type === 'int32') && !/(id|index)$/i.test(column.name)
  );
}

function findCoordinateColumn(
  subject: WorkflowRecommendationSubject,
  axis: 'longitude' | 'latitude'
) {
  const pattern =
    axis === 'longitude'
      ? /(?:^|[_\s-])(longitude|long|lon|lng|x)$/i
      : /(?:^|[_\s-])(latitude|lat|y)$/i;
  return subject.columns.find(
    column =>
      isNumericColumn(column) && pattern.test(column.name.replace(/([a-z])([A-Z])/g, '$1_$2'))
  )?.name;
}

function getCandidateParameters(
  candidate: WorkflowRecommendationCandidate,
  subject: WorkflowRecommendationSubject
): WorkflowRecommendation['parameters'] | undefined {
  const numericColumns = subject.columns.filter(isNumericColumn);
  const columns: Record<ColumnRole, string | undefined> = {
    numeric: numericColumns[0]?.name,
    predictor: numericColumns[1]?.name,
    category: subject.columns.find(
      column => column.type === 'string' || (column.type === 'int32' && isNumericColumn(column))
    )?.name,
    longitude: findCoordinateColumn(subject, 'longitude'),
    latitude: findCoordinateColumn(subject, 'latitude'),
    first: subject.columns[0]?.name
  };
  const parameters = {...candidate.parameters};
  for (const [parameter, role] of Object.entries(candidate.columnParameters ?? {})) {
    const column = columns[role];
    if (column === undefined) return undefined;
    parameters[parameter] = column;
  }
  return parameters;
}

function getRecommendationLabel(
  candidate: WorkflowRecommendationCandidate,
  kind: WorkflowPortKind
) {
  if (kind !== 'features' || candidate.kinds.length === 3) return candidate.label;
  const requirement = candidate.kinds
    .map(accepted => (accepted === 'point' ? 'points' : accepted === 'line' ? 'lines' : 'polygons'))
    .join('/');
  return `${candidate.label} (${requirement})`;
}

/**
 * Up to `limit` (default 6) next steps for some data, most generally useful first. Rule-based and
 * deterministic: a curated list per kind, kept only when the operation exists in the registry,
 * can read the kind, accepts the row count, and has what it needs (a numeric column, a category).
 */
export function recommendWorkflowOperations(
  registry: WorkflowOperationRegistry,
  subject: WorkflowRecommendationSubject,
  options: {limit?: number} = {}
): WorkflowRecommendation[] {
  const limit = Math.max(0, Math.floor(options.limit ?? 6));
  if (!limit) return [];
  const candidates = WORKFLOW_RECOMMENDATION_CANDIDATES.filter(
    candidate =>
      !candidate.sample &&
      (subject.kind === 'features'
        ? !candidate.kinds.includes('table')
        : candidate.kinds.includes(subject.kind))
  );
  // More geometry kinds means fewer assumptions when the geometry has not yet been read.
  candidates.sort(
    (first, second) =>
      (first.priority?.[subject.kind] ?? 10) - (second.priority?.[subject.kind] ?? 10) ||
      (subject.kind === 'features' ? second.kinds.length - first.kinds.length : 0)
  );
  const recommendations: WorkflowRecommendation[] = [];
  const skippedRowLimits: number[] = [];
  for (const candidate of candidates) {
    const operation = registry.get(candidate.operationId);
    const input = operation?.inputs.find(
      port =>
        port.name === candidate.inputName &&
        getWorkflowPortCompatibility(subject.kind, port) !== 'no'
    );
    const parameters = getCandidateParameters(candidate, subject);
    if (!input || !parameters) continue;
    if (subject.rowCount !== undefined && subject.rowCount > input.maximumRows) {
      skippedRowLimits.push(input.maximumRows);
      continue;
    }
    recommendations.push({
      id: candidate.operationId,
      operationId: candidate.operationId,
      label: getRecommendationLabel(candidate, subject.kind),
      reason: candidate.reason,
      inputName: input.name,
      parameters
    });
    if (recommendations.length >= limit) return recommendations;
  }
  if (recommendations.length || !skippedRowLimits.length) return recommendations;
  const sample = WORKFLOW_RECOMMENDATION_CANDIDATES.find(candidate => candidate.sample);
  if (!sample) return [];
  const operation = registry.get(sample.operationId);
  const input = operation?.inputs.find(
    port =>
      port.name === sample.inputName && getWorkflowPortCompatibility(subject.kind, port) !== 'no'
  );
  const parameters = getCandidateParameters(sample, subject);
  if (!input || !parameters) return [];
  // Sampling is the explicit size fallback; its own read capacity still needs host-side bounding.
  parameters['limit'] = Math.min(1000, ...skippedRowLimits);
  return [
    {
      id: sample.operationId,
      operationId: sample.operationId,
      label: sample.label,
      reason: sample.reason,
      inputName: input.name,
      parameters
    }
  ];
}

/** Adds the recommended step reading `connectFrom`, with its preset parameters, in one edit. */
export function applyWorkflowRecommendation(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  recommendation: WorkflowRecommendation,
  connectFrom: {stepId: string; output: string}
): {document: WorkflowDocument; stepId: string} {
  return addWorkflowStep(document, registry, {
    operationId: recommendation.operationId,
    connectFrom,
    connectTo: recommendation.inputName,
    parameters: recommendation.parameters
  });
}
