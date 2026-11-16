// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity, validateMaximumCount} from '@luma-contributors/analysis/analysis-core';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {KeplerDatasetProfile} from '../profile/dataset-profile-types';
import {
  getWorkflowAnalysisProfiles,
  getWorkflowTopologicalOrder,
  resolveWorkflowParameters
} from './workflow-plan';
import type {
  WorkflowDocument,
  WorkflowHost,
  WorkflowOperationRegistry,
  WorkflowRunnerState
} from './workflow-types';

/** Counts supplied by revision-current metadata or the shared AX source-profile service. */
export type WorkflowPreflightShape = {
  rowCount: number;
  /** Features accepted by the catalog packer; output rows can include empty geometry. */
  packedFeatureCount?: number;
  /** Homogeneous geometry family. */
  geometry?: 'point' | 'line' | 'polygon';
  /** Total packed positions, including polygon closing vertices. */
  vertexCount?: number;
};

export type WorkflowStepEstimate = {
  stepId: string;
  operationId: string;
  status: 'estimated' | 'unknown' | 'disabled';
  /** Logical GPU transient bytes; excludes input/output buffers, staging and driver overhead. */
  scratchBytes: number | null;
  commandNodeCount: number | null;
  outputs: Readonly<Record<string, WorkflowPreflightShape>>;
  reason: string;
};

export type WorkflowMaterializationBoundary = {
  producer: string;
  output: string;
  consumer: string;
  input: string;
  materialization: 'cpu-table' | 'cpu-raster';
  reason: string;
};

export type WorkflowPreflight = {
  steps: WorkflowStepEstimate[];
  boundaries: WorkflowMaterializationBoundary[];
  /** Sequential runner releases each GPU session before the next step: max, not sum. */
  peakScratchBytes: number | null;
  commandNodeCount: number | null;
  knownPeakScratchBytes: number;
  knownCommandNodeCount: number;
  complete: boolean;
};

/** Qualification policy, not measured accuracy. Zero scratch must match exactly. */
export const WORKFLOW_PREFLIGHT_TOLERANCE = Object.freeze({scratchRelative: 0.25, commandNodes: 0});

// GPUGeometryMeasures' feature statistics are sixteen uint32 words. The catalog requests feature
// outputs only, without grouping/status. Above 512 packed positions it adds a cooperative node.
const GEOMETRY_MEASURES_BYTES_PER_FEATURE = 64;
const GEOMETRY_MEASURES_COOPERATIVE_ROWS = 512;

function validateShape(shape: WorkflowPreflightShape): void {
  validateCapacity('WorkflowPreflight', 'rowCount', shape.rowCount, 0, Number.MAX_SAFE_INTEGER);
  if (shape.packedFeatureCount !== undefined) {
    validateCapacity(
      'WorkflowPreflight',
      'packedFeatureCount',
      shape.packedFeatureCount,
      0,
      shape.rowCount
    );
  }
  if (shape.vertexCount !== undefined) {
    validateCapacity(
      'WorkflowPreflight',
      'vertexCount',
      shape.vertexCount,
      0,
      Number.MAX_SAFE_INTEGER
    );
  }
}

function getProfileShape(
  profile: KeplerDatasetProfile | undefined
): WorkflowPreflightShape | undefined {
  if (
    !profile?.catalogGeometry ||
    profile.vertexCount === undefined ||
    profile.mixedGeometry ||
    profile.geometryFeatureCounts?.[profile.catalogGeometry] !== profile.rowCount ||
    profile.geometryFeatureCounts.withoutGeometry !== 0
  )
    return undefined;
  const shape = {
    rowCount: profile.rowCount,
    packedFeatureCount: profile.geometryPackedFeatureCounts?.[profile.catalogGeometry],
    geometry: profile.catalogGeometry,
    vertexCount: profile.vertexCount
  };
  validateShape(shape);
  return shape;
}

/**
 * Cold-run estimate of the current CPU-materialised plan, independent of cache hits. Unknown
 * operations and missing metadata stay unknown. This is not a memory reservation or a total
 * device-memory estimate. Source shapes must describe the document/revision being planned.
 * Optional source profiling reuses the shared AX profile cache; the first bounded embedded
 * GeoJSON profile may parse the source.
 */
export function estimateWorkflowPreflight(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  options: {
    sourceShapes?: Readonly<Record<string, Readonly<Record<string, WorkflowPreflightShape>>>>;
    state?: WorkflowRunnerState | null;
    profileSources?: boolean;
    host?: WorkflowHost;
  } = {}
): WorkflowPreflight {
  const {order, cyclic} = getWorkflowTopologicalOrder(document);
  const definitions = new Map(document.steps.map(step => [step.id, step]));
  const estimates = new Map<string, WorkflowStepEstimate>();
  const boundaries: WorkflowMaterializationBoundary[] = [];
  for (const stepId of order) {
    const step = definitions.get(stepId)!;
    const operation = registry.get(step.operationId);
    const estimate: WorkflowStepEstimate = {
      stepId,
      operationId: step.operationId,
      status: 'unknown',
      scratchBytes: null,
      commandNodeCount: null,
      outputs: {},
      reason: 'No cost model for this operation.'
    };
    estimates.set(stepId, estimate);
    for (const [input, connection] of Object.entries(step.inputs)) {
      const producer = definitions.get(connection.stepId);
      const raster =
        producer &&
        registry
          .get(producer.operationId)
          ?.outputs.find(output => output.name === connection.output)?.kind === 'raster';
      boundaries.push({
        producer: connection.stepId,
        output: connection.output,
        consumer: stepId,
        input,
        materialization: raster ? 'cpu-raster' : 'cpu-table',
        reason: raster
          ? 'The runner hands off a decoded CPU raster band and grid; GPU steps upload again.'
          : 'The runner hands off a CPU table; GPU results are read back and later GPU steps upload again.'
      });
    }
    if (step.disabled) {
      Object.assign(estimate, {
        status: 'disabled',
        scratchBytes: 0,
        commandNodeCount: 0,
        reason: 'Disabled step does not execute.'
      });
      continue;
    }
    if (!operation || cyclic.includes(stepId)) continue;
    let parameters: Record<string, CatalogParameterValue>;
    try {
      parameters = operation.normalizeParameters(
        resolveWorkflowParameters(document, step, operation)
      );
    } catch {
      estimate.reason = 'Resolve invalid parameters before estimating this step.';
      continue;
    }
    const profiles = options.profileSources
      ? getWorkflowAnalysisProfiles(document, step, registry, options.host)
      : {};
    const inputs = Object.fromEntries(
      Object.entries(step.inputs).map(([name, connection]) => [
        name,
        estimates.get(connection.stepId)?.outputs[connection.output] ??
          getProfileShape(profiles[name])
      ])
    );
    const sourceShapes = options.sourceShapes?.[stepId];
    if (sourceShapes) Object.values(sourceShapes).forEach(validateShape);
    if (operation.origin === 'builtin' && operation.category === 'source') {
      Object.assign(estimate, {
        status: 'estimated',
        scratchBytes: 0,
        commandNodeCount: 0,
        outputs: sourceShapes ?? {},
        reason: 'Host source; no GPU command graph.'
      });
    } else if (operation.origin === 'builtin' && operation.category === 'table') {
      // Only proven single-point rows have an exact position count after a truncating sort.
      const preservesRows = ['table-calculate-column', 'table-sort-rows'].includes(operation.id);
      let outputShape = preservesRows ? inputs['rows'] : undefined;
      const limit = Number(parameters['limit']);
      if (
        operation.id === 'table-sort-rows' &&
        outputShape &&
        limit > 0 &&
        limit < outputShape.rowCount
      ) {
        const singlePoints =
          outputShape.geometry === 'point' &&
          outputShape.packedFeatureCount === outputShape.rowCount &&
          outputShape.vertexCount === outputShape.rowCount;
        outputShape = singlePoints
          ? {...outputShape, rowCount: limit, packedFeatureCount: limit, vertexCount: limit}
          : undefined;
      }
      Object.assign(estimate, {
        status: 'estimated',
        scratchBytes: 0,
        commandNodeCount: 0,
        outputs: outputShape ? {[operation.outputs[0].name]: outputShape} : {},
        reason: 'CPU table operation; JavaScript heap is outside the GPU scratch estimate.'
      });
    } else if (
      ((step.engine ?? document.settings.engine) === 'cpu' && operation.engine !== 'gpu') ||
      operation.engine === 'cpu-reference'
    ) {
      Object.assign(estimate, {
        status: 'estimated',
        scratchBytes: 0,
        commandNodeCount: 0,
        reason: 'CPU execution requested; no GPU scratch.'
      });
    } else if (
      operation.id === 'geometry-point-buffer' &&
      inputs['points']?.geometry === 'point' &&
      inputs['points'].vertexCount === inputs['points'].rowCount &&
      parameters['coordinateSystem'] === 'planar'
    ) {
      const rowCount = inputs['points'].rowCount;
      const vertexCount = rowCount * (Number(parameters['segmentsPerQuarter']) * 4 + 1);
      validateMaximumCount(
        'WorkflowPreflight',
        'output vertexCount',
        vertexCount,
        Number.MAX_SAFE_INTEGER
      );
      Object.assign(estimate, {
        status: 'estimated',
        scratchBytes: 0,
        commandNodeCount: rowCount ? 1 : 0,
        outputs: {
          buffers: {rowCount, packedFeatureCount: rowCount, geometry: 'polygon', vertexCount}
        },
        reason:
          'GPUPointBuffer writes caller-owned offsets directly; no graph transients. Planar single points only.'
      });
    } else if (
      operation.id === 'geometry-measures' &&
      inputs['features']?.geometry &&
      inputs['features'].packedFeatureCount !== undefined &&
      inputs['features'].vertexCount !== undefined
    ) {
      const shape = inputs['features'];
      validateMaximumCount(
        'WorkflowPreflight',
        'scratchBytes',
        shape.packedFeatureCount! * GEOMETRY_MEASURES_BYTES_PER_FEATURE,
        Number.MAX_SAFE_INTEGER
      );
      Object.assign(estimate, {
        status: 'estimated',
        scratchBytes: shape.packedFeatureCount! * GEOMETRY_MEASURES_BYTES_PER_FEATURE,
        commandNodeCount: !shape.packedFeatureCount
          ? 0
          : 2 + (shape.vertexCount! > GEOMETRY_MEASURES_COOPERATIVE_ROWS ? 1 : 0),
        outputs: {measures: shape},
        reason:
          'Homogeneous packed features: 16 uint32 statistics per packed feature; feature/scatter nodes and optional cooperative node.'
      });
    } else if (['geometry-point-buffer', 'geometry-measures'].includes(operation.id)) {
      estimate.reason =
        'Needs homogeneous packed feature and vertex counts from source metadata (planar single points for buffering).';
    }
    // Current result profiles can seed later steps, but never make an unmodelled cost zero.
    const current = options.state?.steps[stepId];
    if (current?.status === 'done' && current.key === current.result?.key) {
      const outputs = {...estimate.outputs};
      for (const [name, data] of Object.entries(current.result!.outputs)) {
        const shape = getProfileShape(data.analysisProfile);
        if (shape && shape.rowCount === data.table.rowCount) outputs[name] = shape;
      }
      estimate.outputs = outputs;
    }
  }
  const steps = [...estimates.values()];
  const complete = !cyclic.length && steps.every(step => step.status !== 'unknown');
  const knownPeakScratchBytes = Math.max(0, ...steps.map(step => step.scratchBytes ?? 0));
  const knownCommandNodeCount = steps.reduce(
    (total, step) => total + (step.commandNodeCount ?? 0),
    0
  );
  return {
    steps,
    boundaries,
    complete,
    knownPeakScratchBytes,
    knownCommandNodeCount,
    peakScratchBytes: complete ? knownPeakScratchBytes : null,
    commandNodeCount: complete ? knownCommandNodeCount : null
  };
}

/** Compare external observations in the same scope; missing estimates never qualify. */
export function compareWorkflowPreflightMeasurement(
  estimate: WorkflowPreflight,
  measured: {peakScratchBytes: number; commandNodeCount: number}
): {passed: boolean; scratchRelativeError: number | null; commandNodeDifference: number | null} {
  validateCapacity(
    'WorkflowPreflight',
    'measured peakScratchBytes',
    measured.peakScratchBytes,
    0,
    Number.MAX_SAFE_INTEGER
  );
  validateCapacity(
    'WorkflowPreflight',
    'measured commandNodeCount',
    measured.commandNodeCount,
    0,
    Number.MAX_SAFE_INTEGER
  );
  const scratchRelativeError =
    estimate.peakScratchBytes === null
      ? null
      : measured.peakScratchBytes === 0
        ? estimate.peakScratchBytes === 0
          ? 0
          : Infinity
        : Math.abs(estimate.peakScratchBytes - measured.peakScratchBytes) /
          measured.peakScratchBytes;
  const commandNodeDifference =
    estimate.commandNodeCount === null
      ? null
      : Math.abs(estimate.commandNodeCount - measured.commandNodeCount);
  return {
    passed:
      estimate.complete &&
      scratchRelativeError !== null &&
      scratchRelativeError <= WORKFLOW_PREFLIGHT_TOLERANCE.scratchRelative &&
      commandNodeDifference === WORKFLOW_PREFLIGHT_TOLERANCE.commandNodes,
    scratchRelativeError,
    commandNodeDifference
  };
}

/** Three deterministic chains for external WebGPU qualification; not measured by construction. */
export function createWorkflowPreflightMeasurementChains(): Array<{
  document: WorkflowDocument;
  sourceShapes: Record<string, Record<string, WorkflowPreflightShape>>;
}> {
  return [
    {id: 'point-measures', rowCount: 4, branches: 1, buffer: false},
    {id: 'buffer-measures', rowCount: 128, branches: 1, buffer: true},
    {id: 'branched-buffer-measures', rowCount: 64, branches: 2, buffer: true}
  ].map(configuration => {
    const document: WorkflowDocument = {
      schemaVersion: 1,
      id: configuration.id,
      title: configuration.id,
      inputs: [],
      settings: {engine: 'auto', autoRun: false},
      steps: []
    };
    const text = JSON.stringify({
      type: 'FeatureCollection',
      features: Array.from({length: configuration.rowCount}, (_, index) => ({
        type: 'Feature',
        id: `point-${index}`,
        properties: {},
        geometry: {type: 'Point', coordinates: [index * 20, index % 3]}
      }))
    });
    document.steps.push({
      id: 'source',
      operationId: 'source-geojson',
      position: {x: 0, y: 0},
      inputs: {},
      parameters: {
        text: {from: 'static', value: text},
        coordinateSystem: {from: 'static', value: 'planar'}
      }
    });
    for (let branch = 0; branch < configuration.branches; branch++) {
      const bufferId = `buffer-${branch}`;
      const measuresId = `measures-${branch}`;
      if (configuration.buffer)
        document.steps.push({
          id: bufferId,
          operationId: 'geometry-point-buffer',
          position: {x: 100, y: branch * 100},
          inputs: {points: {stepId: 'source', output: 'data'}},
          parameters: {
            coordinateSystem: {from: 'static', value: 'planar'},
            radius: {from: 'static', value: 5 + branch},
            segmentsPerQuarter: {from: 'static', value: 4}
          }
        });
      document.steps.push({
        id: measuresId,
        operationId: 'geometry-measures',
        position: {x: 200, y: branch * 100},
        inputs: {
          features: {
            stepId: configuration.buffer ? bufferId : 'source',
            output: configuration.buffer ? 'buffers' : 'data'
          }
        },
        parameters: {coordinateSystem: {from: 'static', value: 'planar'}}
      });
      document.steps.push({
        id: `calculate-${branch}`,
        operationId: 'table-calculate-column',
        position: {x: 300, y: branch * 100},
        inputs: {rows: {stepId: measuresId, output: 'measures'}},
        parameters: {
          columnName: {from: 'static', value: 'double_length'},
          expression: {from: 'static', value: 'length * 2'}
        }
      });
    }
    return {
      document,
      sourceShapes: {
        source: {
          data: {
            rowCount: configuration.rowCount,
            packedFeatureCount: configuration.rowCount,
            geometry: 'point',
            vertexCount: configuration.rowCount
          }
        }
      }
    };
  });
}
