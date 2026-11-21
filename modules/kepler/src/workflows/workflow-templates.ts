// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {CatalogFeature, CatalogFeatureCollection} from '../analysis/catalog-geojson';
import {
  createWorkflowDocument,
  makeUniqueWorkflowId,
  parseWorkflowDocument,
  serializeWorkflowDocument
} from './workflow-document';
import {
  addWorkflowStep,
  connectWorkflowSteps,
  layoutWorkflow,
  promoteWorkflowParameter,
  updateWorkflowStep,
  updateWorkflowInput
} from './workflow-edit';
import {getWorkflowOperationRegistry} from './workflow-operations';
import {planWorkflow} from './workflow-plan';
import {WorkflowError, type WorkflowDocument, type WorkflowConnection} from './workflow-types';
import {
  createWorkflowAnalysisTemplateFromDocument,
  listCuratedWorkflowAnalysisTemplates,
  type WorkflowAnalysisTemplate
} from './workflow-analysis-template';

export type WorkflowTemplate = {
  id: string;
  title: string;
  summary: string;
  /** What the chain demonstrates, shown in the template gallery. */
  category: 'points' | 'network' | 'polygons' | 'statistics' | 'glue';
  /** Self-contained: reads embedded GeoJSON, so it runs without any host dataset. */
  document: WorkflowDocument;
  /** Reusable AX envelope over the same executable document. */
  analysisTemplate: WorkflowAnalysisTemplate;
};

// Definitions and their generated geometry are built once, only when the gallery is used.
let workflowTemplates: readonly WorkflowTemplate[] | undefined;

const LONGITUDE = -87.64;
const LATITUDE = 41.88;
const COORDINATE_SYSTEM = {coordinateSystem: 'EPSG:4326'};

type GeneratedFeature = CatalogFeature & {type: 'Feature'};

function makeCollection(features: CatalogFeature[]): string {
  const collection: CatalogFeatureCollection = {type: 'FeatureCollection', features};
  return JSON.stringify(collection);
}

function makePoint(id: string, x: number, y: number, value = 1): GeneratedFeature {
  return {
    type: 'Feature',
    id,
    properties: {value},
    geometry: {type: 'Point', coordinates: [LONGITUDE + x, LATITUDE + y]}
  };
}

function makeJitter(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state / 4294967296 - 0.5) * 0.00012;
  };
}

function makeObservations(): string {
  const jitter = makeJitter(17);
  return makeCollection(
    Array.from({length: 4}, (_, group) =>
      Array.from({length: 7 + group * 3}, (_, member) =>
        makePoint(
          `observation-${group * 16 + member}`,
          (group % 2) * 0.008 + 0.001 + (member % 4) * 0.002 + jitter(),
          Math.floor(group / 2) * 0.008 + 0.001 + Math.floor(member / 4) * 0.002 + jitter(),
          1 + ((group * 16 + member) % 7)
        )
      )
    ).flat()
  );
}

function makeClusters(): string {
  const jitter = makeJitter(29);
  const points = Array.from({length: 48}, (_, index) => {
    const group = Math.floor(index / 16);
    const member = index % 16;
    return makePoint(
      `sample-${index}`,
      0.002 + group * 0.006 + (member % 4) * 0.0003 + jitter(),
      0.003 + (group % 2) * 0.006 + Math.floor(member / 4) * 0.0003 + jitter()
    );
  });
  points.push(makePoint('noise-west', -0.005, -0.005), makePoint('noise-east', 0.025, 0.02));
  return makeCollection(points);
}

function makeFacilities(): string {
  return makeCollection(
    Array.from({length: 4}, (_, index) =>
      makePoint(
        `facility-${index}`,
        0.004 + (index % 2) * 0.008,
        0.004 + Math.floor(index / 2) * 0.008
      )
    )
  );
}

function makeDistricts(): string {
  return makeCollection(
    Array.from({length: 4}, (_, index): GeneratedFeature => {
      const west = LONGITUDE + (index % 2) * 0.008;
      const south = LATITUDE + Math.floor(index / 2) * 0.008;
      return {
        type: 'Feature',
        id: `district-${index}`,
        properties: {district: `District ${index + 1}`},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [west, south],
              [west + 0.008, south],
              [west + 0.008, south + 0.008],
              [west, south + 0.008],
              [west, south]
            ]
          ]
        }
      };
    })
  );
}

function makeStreets(): string {
  return makeCollection(
    Array.from({length: 16}, (_, index): GeneratedFeature => {
      const offset = 0.001 + (index % 8) * 0.002;
      const coordinates =
        index < 8
          ? [
              [LONGITUDE, LATITUDE + offset],
              [LONGITUDE + 0.008, LATITUDE + offset],
              [LONGITUDE + 0.016, LATITUDE + offset]
            ]
          : [
              [LONGITUDE + offset, LATITUDE],
              [LONGITUDE + offset, LATITUDE + 0.008],
              [LONGITUDE + offset, LATITUDE + 0.016]
            ];
      return {
        type: 'Feature',
        id: `street-${index}`,
        properties: {},
        geometry: {type: 'LineString', coordinates}
      };
    })
  );
}

type TemplateStep = {
  id: string;
  operationId: string;
  label: string;
  parameters?: Record<string, CatalogParameterValue>;
  inputs?: Record<string, WorkflowConnection>;
  promote?: {parameter: string; minimum: number; maximum: number};
  display?: string;
};

function buildTemplate(
  metadata: Omit<WorkflowTemplate, 'document' | 'analysisTemplate'>,
  steps: readonly TemplateStep[]
): WorkflowTemplate {
  const registry = getWorkflowOperationRegistry();
  let document = createWorkflowDocument({
    id: metadata.id,
    title: metadata.title,
    description: metadata.summary
  });
  for (const step of steps) {
    document = addWorkflowStep(
      document,
      registry,
      step.operationId === 'source-geojson'
        ? {...step, parameters: {coordinateSystem: 'EPSG:4326', ...step.parameters}}
        : step
    ).document;
    for (const [input, from] of Object.entries(step.inputs ?? {})) {
      document = connectWorkflowSteps(document, registry, {from, to: {stepId: step.id, input}});
    }
    if (step.promote) {
      const promoted = promoteWorkflowParameter(
        document,
        registry,
        step.id,
        step.promote.parameter
      );
      // Gallery inputs describe useful ranges for this fixed sample, not the whole catalog.
      document = updateWorkflowInput(promoted.document, promoted.inputId, {
        minimum: step.promote.minimum,
        maximum: step.promote.maximum
      });
    }
    if (step.display) document = updateWorkflowStep(document, step.id, {display: [step.display]});
  }
  document = parseWorkflowDocument(serializeWorkflowDocument(layoutWorkflow(document)));
  const plan = planWorkflow(document, {registry});
  const issues = [...plan.issues, ...Object.values(plan.steps).flatMap(step => step.issues)];
  const error = issues.find(issue => issue.severity === 'error');
  if (error) throw new WorkflowError(error.code, `${metadata.title}: ${error.message}`);
  return {
    ...metadata,
    document,
    analysisTemplate: createWorkflowAnalysisTemplateFromDocument({...metadata, document})
  };
}

function source(id: string, label: string, text: string): TemplateStep {
  return {id, label, operationId: 'source-geojson', parameters: {text, name: label}};
}

function output(stepId: string, name: string): WorkflowConnection {
  return {stepId, output: name};
}

function freezeTemplate<Value>(value: Value): Value {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freezeTemplate(child);
    Object.freeze(value);
  }
  return value;
}

/** Ready-made workflows over embedded sample geometry. */
export function listWorkflowTemplates(): readonly WorkflowTemplate[] {
  if (workflowTemplates) return workflowTemplates;
  const observations = makeObservations();
  const facilities = makeFacilities();
  const districts = makeDistricts();
  workflowTemplates = freezeTemplate([
    buildTemplate(
      {
        id: 'cluster-footprints',
        title: 'Cluster footprints',
        category: 'points',
        summary:
          'Find three dense groups, remove isolated noise and draw a convex footprint for each cluster.'
      },
      [
        source('samples', 'Cluster samples', makeClusters()),
        {
          id: 'clusters',
          operationId: 'density-clustering',
          label: 'Find clusters',
          parameters: {...COORDINATE_SYSTEM, epsilon: 80, minimumSamples: 4},
          inputs: {points: output('samples', 'data')},
          promote: {parameter: 'epsilon', minimum: 60, maximum: 200}
        },
        {
          id: 'members',
          operationId: 'table-filter-rows',
          label: 'Remove noise',
          parameters: {expression: 'cluster >= 0'},
          inputs: {rows: output('clusters', 'points')}
        },
        {
          id: 'footprints',
          operationId: 'group-convex-hull',
          label: 'Cluster outlines',
          parameters: {...COORDINATE_SYSTEM, groupProperty: 'cluster'},
          inputs: {points: output('members', 'filtered')},
          display: 'group-hulls'
        }
      ]
    ),
    buildTemplate(
      {
        id: 'buffer-areas',
        title: 'Buffer areas',
        category: 'polygons',
        summary:
          'Buffer four facilities, measure the footprints and calculate their area in hectares.'
      },
      [
        source('facilities', 'Facilities', facilities),
        {
          id: 'buffers',
          operationId: 'geometry-point-buffer',
          label: 'Walking radius',
          parameters: {...COORDINATE_SYSTEM, radius: 180, segmentsPerQuarter: 4},
          inputs: {points: output('facilities', 'data')},
          promote: {parameter: 'radius', minimum: 50, maximum: 500}
        },
        {
          id: 'measures',
          operationId: 'geometry-measures',
          label: 'Measure footprints',
          parameters: COORDINATE_SYSTEM,
          inputs: {features: output('buffers', 'buffers')}
        },
        {
          id: 'hectares',
          operationId: 'table-calculate-column',
          label: 'Area in hectares',
          parameters: {columnName: 'hectares', expression: 'area / 10000'},
          inputs: {rows: output('measures', 'measures')},
          display: 'calculated'
        }
      ]
    ),
    buildTemplate(
      {
        id: 'nearest-facilities',
        title: 'Nearest facilities',
        category: 'glue',
        summary:
          'Assign observations to the nearest facility, outline each service footprint and join its demand totals.'
      },
      [
        source('observations', 'Observations', observations),
        source('facilities', 'Facilities', facilities),
        {
          id: 'nearest',
          operationId: 'nearest-feature-join',
          label: 'Nearest facility',
          parameters: {...COORDINATE_SYSTEM, maximumDistance: 3000},
          inputs: {points: output('observations', 'data'), targets: output('facilities', 'data')}
        },
        {
          id: 'demand',
          operationId: 'table-summarize',
          label: 'Facility demand',
          parameters: {groupBy: 'target_source_id', valueColumn: 'distance'},
          inputs: {rows: output('nearest', 'nearest-feature-join')}
        },
        {
          id: 'footprints',
          operationId: 'group-convex-hull',
          label: 'Service footprints',
          parameters: {...COORDINATE_SYSTEM, groupProperty: 'target_source_id'},
          inputs: {points: output('nearest', 'nearest-feature-join')}
        },
        {
          id: 'mapped',
          operationId: 'table-join-attributes',
          label: 'Map facility demand',
          parameters: {
            targetKey: 'group_value',
            attributesKey: 'target_source_id',
            columns: 'row_count,value_mean',
            prefix: 'demand_'
          },
          inputs: {
            target: output('footprints', 'group-hulls'),
            attributes: output('demand', 'summary')
          },
          display: 'joined'
        }
      ]
    ),
    buildTemplate(
      {
        id: 'district-counts',
        title: 'District counts',
        category: 'glue',
        summary:
          'Assign observations to districts, keep matches and join district totals to measured polygon areas.'
      },
      [
        source('observations', 'Observations', observations),
        source('districts', 'Districts', districts),
        {
          id: 'assignment',
          operationId: 'point-in-polygon-join',
          label: 'Assign districts',
          inputs: {points: output('observations', 'data'), polygons: output('districts', 'data')}
        },
        {
          id: 'inside',
          operationId: 'table-filter-rows',
          label: 'Keep district matches',
          parameters: {expression: 'match_count > 0'},
          inputs: {rows: output('assignment', 'point-in-polygon-join')}
        },
        {
          id: 'counts',
          operationId: 'table-summarize',
          label: 'Count by district',
          parameters: {groupBy: 'polygon_source_id'},
          inputs: {rows: output('inside', 'filtered')}
        },
        {
          id: 'measures',
          operationId: 'geometry-measures',
          label: 'Measure districts',
          parameters: COORDINATE_SYSTEM,
          inputs: {features: output('districts', 'data')}
        },
        {
          id: 'mapped',
          operationId: 'table-join-attributes',
          label: 'Map district counts',
          parameters: {
            targetKey: 'source_id',
            attributesKey: 'polygon_source_id',
            columns: 'row_count'
          },
          inputs: {target: output('measures', 'measures'), attributes: output('counts', 'summary')},
          display: 'joined'
        }
      ]
    ),
    buildTemplate(
      {
        id: 'street-coverage',
        title: 'Street coverage',
        category: 'network',
        summary:
          'Simplify redundant street vertices, clip the grid to districts and convert total lengths to kilometers.'
      },
      [
        source('streets', 'Street grid', makeStreets()),
        source('districts', 'Districts', districts),
        {
          id: 'simplified',
          operationId: 'line-simplification',
          label: 'Simplify streets',
          parameters: {...COORDINATE_SYSTEM, tolerance: 2},
          inputs: {lines: output('streets', 'data')}
        },
        {
          id: 'lengths',
          operationId: 'line-length-per-polygon',
          label: 'Streets per district',
          parameters: COORDINATE_SYSTEM,
          inputs: {
            lines: output('simplified', 'simplified-lines'),
            polygons: output('districts', 'data')
          }
        },
        {
          id: 'kilometers',
          operationId: 'table-calculate-column',
          label: 'Street kilometers',
          parameters: {columnName: 'kilometers', expression: 'line_length / 1000'},
          inputs: {rows: output('lengths', 'polygon-lengths')},
          display: 'calculated'
        }
      ]
    ),
    buildTemplate(
      {
        id: 'voronoi-catchments',
        title: 'Voronoi catchments',
        category: 'statistics',
        summary:
          'Build facility catchments, count and sum observations, then rank demand alongside measured perimeters.'
      },
      [
        source('facilities', 'Facilities', facilities),
        source('observations', 'Observations', observations),
        {
          id: 'catchments',
          operationId: 'voronoi-diagram',
          label: 'Facility catchments',
          parameters: {...COORDINATE_SYSTEM, padding: 0.5},
          inputs: {points: output('facilities', 'data')}
        },
        {
          id: 'statistics',
          operationId: 'zonal-point-statistics',
          label: 'Catchment demand',
          parameters: {valueProperty: 'value'},
          inputs: {
            polygons: output('catchments', 'voronoi-cells'),
            points: output('observations', 'data')
          }
        },
        {
          id: 'measures',
          operationId: 'geometry-measures',
          label: 'Measure catchments',
          parameters: COORDINATE_SYSTEM,
          inputs: {features: output('catchments', 'voronoi-cells')}
        },
        {
          id: 'perimeters',
          operationId: 'table-join-attributes',
          label: 'Join perimeters',
          parameters: {
            targetKey: 'polygon_id',
            attributesKey: 'source_id',
            columns: 'length',
            prefix: 'shape_'
          },
          inputs: {
            target: output('statistics', 'zones'),
            attributes: output('measures', 'measures')
          }
        },
        {
          id: 'ranked',
          operationId: 'table-sort-rows',
          label: 'Rank demand',
          parameters: {column: 'sum', order: 'descending'},
          inputs: {rows: output('perimeters', 'joined')},
          display: 'sorted'
        }
      ]
    )
  ]);
  return workflowTemplates;
}

/** A copy of the template document with an id and title not in `takenIds`. */
export function instantiateWorkflowTemplate(
  templateId: string,
  takenIds: Iterable<string>
): WorkflowDocument {
  const template = listWorkflowTemplates().find(candidate => candidate.id === templateId);
  if (!template)
    throw new WorkflowError('malformed', `Workflow template "${templateId}" does not exist.`);
  const taken = new Set(takenIds);
  const document = parseWorkflowDocument(serializeWorkflowDocument(template.document));
  document.id = makeUniqueWorkflowId(document.id, taken);
  const title = document.title;
  for (
    let suffix = 2;
    taken.has(document.title) || taken.has(makeUniqueWorkflowId(document.title, []));
    suffix++
  ) {
    document.title = `${title} ${suffix}`;
  }
  return document;
}

/** Shared question gallery: the curated registry and the existing multi-step sample chains. */
export function listWorkflowAnalysisTemplates(): readonly WorkflowAnalysisTemplate[] {
  return [
    ...listCuratedWorkflowAnalysisTemplates(),
    ...listWorkflowTemplates().map(template => template.analysisTemplate)
  ];
}
