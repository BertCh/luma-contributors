// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Kernel} from '@luma-contributors/graph';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  GPUReduction,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUContiguityWeights,
  type GPUPolygonGeometryPort,
  type GPUPackedGeometryPort,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {
  parseKeplerPolygonWeightsSettings,
  MAXIMUM_KEPLER_POLYGON_ROWS,
  MAXIMUM_KEPLER_POLYGON_VERTICES,
  type KeplerDatasetReference,
  type KeplerPolygonWeightsSettings
} from '../recipes/analysis-recipe';

export type KeplerPolygonWeightsStatus = {
  overflow: GraphDataView<'uint32'>;
  /** A lower bound if candidateOverflow is nonzero. */
  requiredCount: GraphDataView<'uint32'>;
  candidateOverflow: GraphDataView<'uint32'>;
  invalidCount: GraphDataView<'uint32'>;
};
export type KeplerPolygonWeightsBinding = {
  source: KeplerDatasetReference;
  settings: KeplerPolygonWeightsSettings;
  weights: GPUSpatialWeights;
  geometryMask: GraphDataView<'uint32'>;
  status: KeplerPolygonWeightsStatus;
};
export type KeplerPolygonWeightsOutput = GPUSpatialWeightsOutput & {
  status: KeplerPolygonWeightsStatus;
};
export type KeplerPolygonWeightsProps = {
  id?: string;
  source: KeplerDatasetReference;
  settings: KeplerPolygonWeightsSettings;
  /** Private equality tokens [exact-source-XY vertex ID, 0], never rendering coordinates. */
  polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  /** Prepared source geometry validity, independent of numeric validity and selection. */
  geometryMask: GraphDataView<'uint32'>;
  output?: KeplerPolygonWeightsOutput;
  status?: KeplerPolygonWeightsStatus;
};

export function getKeplerPolygonWeightsCapacity(
  settings: KeplerPolygonWeightsSettings,
  rowCount: number
): number {
  if (!Number.isSafeInteger(rowCount) || rowCount < 1 || rowCount > MAXIMUM_KEPLER_POLYGON_ROWS)
    throw new Error('Polygon weights require 1 to 4096 original feature rows');
  return parseKeplerPolygonWeightsSettings(settings).neighborCapacity;
}

/** Validate imported reusable topology before recording dependent work. */
export function validateKeplerPolygonWeightsBinding<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  binding: KeplerPolygonWeightsBinding,
  source: KeplerDatasetReference,
  rowCount: number,
  expectedSettings: KeplerPolygonWeightsSettings = binding.settings
): void {
  const settings = parseKeplerPolygonWeightsSettings(binding.settings);
  const expected = parseKeplerPolygonWeightsSettings(expectedSettings);
  const capacity = getKeplerPolygonWeightsCapacity(settings, rowCount);
  if (
    !source.datasetId ||
    !Number.isSafeInteger(source.revision) ||
    source.revision < 1 ||
    binding.source.datasetId !== source.datasetId ||
    binding.source.revision !== source.revision ||
    Object.keys(expected).some(
      key =>
        settings[key as keyof KeplerPolygonWeightsSettings] !==
        expected[key as keyof KeplerPolygonWeightsSettings]
    )
  )
    throw new Error('Polygon weights source or settings are stale');
  const information = binding.weights.information;
  if (
    binding.geometryMask.length !== rowCount ||
    binding.weights.offsets.length !== rowCount + 1 ||
    binding.weights.neighbors.length !== capacity ||
    binding.weights.weights.length !== capacity ||
    information?.rowCount !== rowCount ||
    information.transform !== 'binary' ||
    information.symmetric !== true ||
    information.hasSelfWeights !== false ||
    information.provenance?.source !== 'GPUContiguityWeights' ||
    information.provenance.criterion !== settings.criterion ||
    binding.weights.spatialContext !== undefined ||
    binding.weights.distances !== undefined
  )
    throw new Error('Polygon weights require complete source-aligned binary contiguity metadata');
  for (const view of [
    binding.geometryMask,
    binding.weights.offsets,
    binding.weights.neighbors,
    ...Object.values(binding.status)
  ])
    validatePackedView(view, ['uint32'], 'Polygon weights integer view');
  validatePackedView(binding.weights.weights, ['float32'], 'Polygon weights values');
  for (const name of ['overflow', 'requiredCount', 'candidateOverflow', 'invalidCount'] as const)
    if (!binding.status[name] || binding.status[name].length !== 1)
      throw new Error('Polygon weights require scalar completeness diagnostics');
  for (const view of [
    binding.geometryMask,
    binding.weights.offsets,
    binding.weights.neighbors,
    binding.weights.weights,
    ...Object.values(binding.status)
  ])
    if (view.buffer.graph !== graph)
      throw new Error('Polygon weights views must belong to the consumer graph');
}

/** Source-wide queen/rook equality topology; no selection, snapping or metric transformation. */
export function addKeplerPolygonWeights<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPolygonWeightsProps
) {
  const id = props.id ?? 'kepler-polygon-weights';
  const settings = parseKeplerPolygonWeightsSettings(props.settings);
  const {polygons, geometryMask} = props;
  const rowCount = polygons.featureOffsets.length - 1;
  const capacity = getKeplerPolygonWeightsCapacity(settings, rowCount);
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1
  )
    throw new Error('Polygon weights require current source identity');
  if (
    polygons.positions.length > MAXIMUM_KEPLER_POLYGON_VERTICES ||
    geometryMask.length !== rowCount ||
    polygons.ringOffsets.length < 1 ||
    polygons.polygonOffsets.length < 1 ||
    polygons.ringOffsets.length > MAXIMUM_KEPLER_POLYGON_VERTICES + 1 ||
    polygons.polygonOffsets.length > MAXIMUM_KEPLER_POLYGON_VERTICES + 1
  )
    throw new KeplerAllocationError('capacity', {
      message: 'Polygon topology exceeds feature or vertex capacity'
    });
  validatePackedView(polygons.positions, ['float32x2'], 'Polygon equality tokens');
  for (const view of [
    polygons.featureOffsets,
    polygons.polygonOffsets,
    polygons.ringOffsets,
    geometryMask
  ])
    validatePackedView(view, ['uint32'], 'Polygon topology offsets/mask');
  const inputViews = [
    polygons.positions,
    polygons.featureOffsets,
    polygons.polygonOffsets,
    polygons.ringOffsets,
    geometryMask
  ];
  if (inputViews.some(view => view.buffer.graph !== graph))
    throw new Error('Polygon topology belongs to another graph');
  if (props.output && props.status) throw new Error('Provide polygon output or status, not both');
  const status = props.output?.status ??
    props.status ?? {
      overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1),
      requiredCount: createTransientView(graph, `${id}-required-count`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `${id}-candidate-overflow`, 'uint32', 1),
      invalidCount: createTransientView(graph, `${id}-invalid-count`, 'uint32', 1)
    };
  const output: KeplerPolygonWeightsOutput = props.output ?? {
    offsets: createTransientView(graph, `${id}-offsets`, 'uint32', rowCount + 1),
    neighbors: createTransientView(graph, `${id}-neighbors`, 'uint32', capacity),
    weights: createTransientView(graph, `${id}-weights`, 'float32', capacity),
    status
  };
  const information = {
    rowCount,
    transform: 'binary' as const,
    symmetric: true,
    hasSelfWeights: false,
    provenance: {source: 'GPUContiguityWeights', criterion: settings.criterion}
  };
  const weights: GPUSpatialWeights = {...output, information, status};
  const binding = {
    source: Object.freeze({...props.source}),
    settings,
    weights,
    geometryMask,
    status
  };
  validateKeplerPolygonWeightsBinding(graph, binding, props.source, rowCount);
  const outputs = [output.offsets, output.neighbors, output.weights, ...Object.values(status)];
  for (let index = 0; index < outputs.length; index++) {
    if (
      [...inputViews, ...outputs.slice(index + 1)].some(view =>
        doGraphDataViewsOverlap(outputs[index], view)
      )
    )
      throw new Error('Polygon output ports must not overlap inputs or each other');
  }
  if (polygons.positions.length === 0) {
    for (const view of [
      output.offsets,
      output.neighbors,
      output.weights,
      status.overflow,
      status.requiredCount,
      status.candidateOverflow
    ])
      addFill(graph, `${id}-empty-${view.buffer.id}-${view.byteOffset}`, view, 0);
    addFill(graph, `${id}-empty-invalid`, status.invalidCount, rowCount);
  } else {
    graph.add(
      new GPUContiguityWeights({
        id,
        polygons,
        criterion: settings.criterion,
        snapTolerance: 0,
        pairCapacity: settings.pairCapacity,
        output
      })
    );
    const invalid = createTransientView(graph, `${id}-invalid-flags`, 'uint32', rowCount);
    addInvalidFlags(graph, `${id}-invalid-flags`, geometryMask, invalid);
    graph.add(
      new GPUReduction({
        id: `${id}-invalid-total`,
        input: invalid,
        output: status.invalidCount,
        operation: 'sum'
      })
    );
  }
  return {...binding, parameterByteLength: 0, destroyParameters() {}};
}

function addFill<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  output: GraphDataView<'uint32'> | GraphDataView<'float32'>,
  value: number
): void {
  const dispatch = getBoundedDispatchLayout(
    id,
    output.length,
    64,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id,
    resources: [{buffer: output, usage: 'storage-read-write'}],
    compile: ({device}) => {
      const type = output.format === 'float32' ? 'f32' : 'u32';
      const kernel = new Kernel(device, {
        id,
        source: `
@group(0) @binding(0) var<storage, read_write> output: array<${type}>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32) {
 ${getBoundedInvocationIndexSource(dispatch, 64)}
 if (index < ${output.length}u) { output[${getViewElementOffset(output)}u + index] = ${type}(${value}); }
}`,
        shaderLayout: {bindings: [{name: 'output', type: 'storage', group: 0, location: 0}]}
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {output: getViewBinding(output, getBuffer)},
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}
function addInvalidFlags<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): void {
  const dispatch = getBoundedDispatchLayout(
    id,
    input.length,
    64,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id,
    resources: [
      {buffer: input, usage: 'storage-read'},
      {buffer: output, usage: 'storage-read-write'}
    ],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id,
        source: `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32) {
 ${getBoundedInvocationIndexSource(dispatch, 64)}
 if (index < ${input.length}u) { output[${getViewElementOffset(output)}u + index] = select(0u, 1u, input[${getViewElementOffset(input)}u + index] == 0u); }
}`,
        shaderLayout: {
          bindings: [
            {name: 'input', type: 'storage', group: 0, location: 0},
            {name: 'output', type: 'storage', group: 0, location: 1}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {
              input: getViewBinding(input, getBuffer),
              output: getViewBinding(output, getBuffer)
            },
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}
