// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addKeplerLinearKernel as addKernel} from './analysis-scaling';
import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUOPTICS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS,
  parseKeplerDBSCANSettings,
  type KeplerDBSCANSettings
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';

export type KeplerPointDBSCANOutputs = {
  values?: GraphDataView<'float32'>;
  labels?: GraphDataView<'uint32'>;
  coreDistances?: GraphDataView<'float32'>;
  reachability?: GraphDataView<'float32'>;
  predecessors?: GraphDataView<'uint32'>;
  /** Visit rank to source row; this is not a source-row field. */
  ordering?: GraphDataView<'uint32'>;
  status?: {
    /** [visitedCount, complete, excludedCount], where excluded includes unselected rows. */
    optics?: GraphDataView<'uint32'>;
    /** [clusterCount, noiseCount, validCount]; noise excludes invalid/unselected rows. */
    summary?: GraphDataView<'uint32'>;
  };
};
export type KeplerPointDBSCANProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerDBSCANSettings;
  longitude: GraphDataView<'float32'>;
  latitude: GraphDataView<'float32'>;
  /** Declared population and source validity; selection rebuilds clusters. */
  mask?: GraphDataView<'uint32'>;
  coordinateValidityMask?: GraphDataView<'uint32'>;
  outputs?: KeplerPointDBSCANOutputs;
};

/**
 * DBSCAN extraction from exact bounded OPTICS with spherical distances in meters. Original row
 * order determines ties; labels are deterministic for a fixed population, not persistent cluster
 * identities across selection changes. minimumSamples includes self. No weights cache is used.
 *
 * Primary values encode compact labels as Float32 and noise/excluded as -1; raw labels use
 * 0xffffffff. effectiveMask distinguishes excluded rows from valid noise. Check optics[1] before
 * publication: incomplete OPTICS extraction leaves every label noise. ordering is rank-to-row,
 * whereas distances, predecessors, labels and primary values retain original source alignment.
 * All settings are graph constants. Caller retains inputs/outputs; omitted outputs belong to the
 * compiled graph. No parameter buffers are allocated by this adapter.
 */
export function addKeplerPointDBSCAN<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointDBSCANProps
) {
  const settings = parseKeplerDBSCANSettings(props.settings);
  const id = props.id ?? 'kepler-point-dbscan';
  const rowCount = props.longitude.length;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_DBSCAN_SOURCE_ROWS ||
    (props.population !== 'all' && props.population !== 'selected') ||
    (props.population === 'selected'
      ? !props.mask ||
        !Number.isSafeInteger(props.selectionRevision) ||
        (props.selectionRevision ?? 0) < 1
      : props.selectionRevision !== undefined)
  )
    throw new Error('DBSCAN requires a current bounded source and explicit population');
  const inputs = [props.longitude, props.latitude, props.mask, props.coordinateValidityMask].filter(
    view => view !== undefined
  );
  for (const view of inputs) {
    validatePackedView(
      view,
      view === props.mask || view === props.coordinateValidityMask ? ['uint32'] : ['float32'],
      id
    );
    if (view.length !== rowCount || view.buffer.graph !== graph)
      throw new Error('DBSCAN source rows and graph must match');
  }
  const provided = props.outputs ?? {};
  const specifications = [
    [provided.values, 'float32', rowCount],
    [provided.labels, 'uint32', rowCount],
    [provided.coreDistances, 'float32', rowCount],
    [provided.reachability, 'float32', rowCount],
    [provided.predecessors, 'uint32', rowCount],
    [provided.ordering, 'uint32', rowCount],
    [provided.status?.optics, 'uint32', 3],
    [provided.status?.summary, 'uint32', 3]
  ] as const;
  const suppliedBuffers = new Set(inputs.map(view => view.buffer));
  for (const [view, format, length] of specifications) {
    if (!view) continue;
    validatePackedView(view, [format], id);
    if (view.length !== length || view.buffer.graph !== graph || suppliedBuffers.has(view.buffer))
      throw new Error('DBSCAN outputs require distinct matching graph buffers');
    suppliedBuffers.add(view.buffer);
  }
  const values =
    provided.values ?? createTransientView(graph, `${id}-values-output`, 'float32', rowCount);
  const labels =
    provided.labels ?? createTransientView(graph, `${id}-labels-output`, 'uint32', rowCount);
  const coreDistances =
    provided.coreDistances ?? createTransientView(graph, `${id}-core-output`, 'float32', rowCount);
  const reachability =
    provided.reachability ??
    createTransientView(graph, `${id}-reachability-output`, 'float32', rowCount);
  const predecessors =
    provided.predecessors ??
    createTransientView(graph, `${id}-predecessors-output`, 'uint32', rowCount);
  const ordering =
    provided.ordering ?? createTransientView(graph, `${id}-ordering-output`, 'uint32', rowCount);
  const status = {
    optics:
      provided.status?.optics ??
      createTransientView(graph, `${id}-optics-status-output`, 'uint32', 3),
    summary:
      provided.status?.summary ?? createTransientView(graph, `${id}-summary-output`, 'uint32', 3)
  };
  const positions = createTransientView(graph, `${id}-positions`, 'float32x2', rowCount);
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const spatialContext = {
    coordinateSpace: settings.coordinateSpace,
    metric: settings.metric,
    units: settings.units
  };
  const contributor = new GPUOPTICS({
    id,
    spatialContext,
    positions,
    minimumSamples: settings.minimumSamples,
    maximumDistance: settings.epsilon,
    extractionDistance: settings.epsilon,
    extractionMethod: 'dbscan',
    maximumIterations: rowCount,
    output: {labels, coreDistances, reachability, predecessors, ordering, status: status.optics}
  });
  addKernel(
    graph,
    `${id}-prepare`,
    rowCount,
    [
      {name: 'longitude', view: props.longitude},
      {name: 'latitude', view: props.latitude},
      ...(props.mask ? [{name: 'populationMask', view: props.mask}] : []),
      ...(props.coordinateValidityMask
        ? [{name: 'coordinateValidity', view: props.coordinateValidityMask}]
        : []),
      {name: 'positions', view: positions, write: true},
      {name: 'effectiveMask', view: effectiveMask, write: true}
    ],
    `let longitudeValue = longitude[longitudeOffset + index];
let latitudeValue = latitude[latitudeOffset + index];
let included = isFiniteValue(longitudeValue) && isFiniteValue(latitudeValue) && abs(longitudeValue) <= 180.0 && abs(latitudeValue) <= 90.0
${props.mask ? '&& populationMask[populationMaskOffset + index] != 0u' : ''}
${props.coordinateValidityMask ? '&& coordinateValidity[coordinateValidityOffset + index] != 0u' : ''};
let missing = bitcast<f32>(0x7f800000u | (index & 0u));
positions[positionsOffset + index * 2u] = select(missing, longitudeValue, included);
positions[positionsOffset + index * 2u + 1u] = select(missing, latitudeValue, included);
effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, included);`
  );
  graph.add(contributor);
  addKernel(
    graph,
    `${id}-summary`,
    1,
    [
      {name: 'labels', view: labels},
      {name: 'effectiveMask', view: effectiveMask},
      {name: 'values', view: values, write: true},
      {name: 'summary', view: status.summary, write: true}
    ],
    `var clusterCount = 0u; var noiseCount = 0u; var validCount = 0u;
for (var row = 0u; row < ${rowCount}u; row++) {
  let label = labels[labelsOffset + row];
  let valid = effectiveMask[effectiveMaskOffset + row] != 0u;
  let noise = label == 0xffffffffu;
  values[valuesOffset + row] = select(f32(label), -1.0, noise);
  validCount += u32(valid); noiseCount += u32(valid && noise);
  if (!noise) { clusterCount = max(clusterCount, label + 1u); }
}
summary[summaryOffset] = clusterCount; summary[summaryOffset + 1u] = noiseCount; summary[summaryOffset + 2u] = validCount;`
  );
  return {
    operation: 'dbscan-points' as const,
    source: Object.freeze({...props.source}),
    population: props.population,
    selectionRevision: props.selectionRevision,
    settings,
    spatialContext,
    numericalPolicy: 'float32' as const,
    effectiveMask,
    outputs: {values, labels, coreDistances, reachability, predecessors, ordering, status},
    status,
    parameterByteLength: 0,
    destroyParameters() {}
  };
}
