// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma-contributors/graph';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewBinding,
  getViewElementOffset,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUDorlingCartogram,
  type GPUDorlingCartogramOutput
} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {
  getKeplerPointWeightCoordinateKeys,
  getKeplerPointWeightSpatialContext,
  parseKeplerPointWeightCoordinates,
  type KeplerProjectedPointWeightCoordinates
} from './point-weight-coordinates';

/** Exact all-pairs relaxation, executed in row tiles of {@link KEPLER_DORLING_ROW_TILE_SIZE}. */
export const MAXIMUM_KEPLER_DORLING_SOURCE_ROWS = 65536;
export const KEPLER_DORLING_ROW_TILE_SIZE = 4096;
export const MAXIMUM_KEPLER_DORLING_ITERATIONS = 256;

export type KeplerDorlingSettings = KeplerProjectedPointWeightCoordinates &
  Readonly<{
    valueColumn: string;
    /** Square meters of circle area per source value unit. */
    areaScale: number;
    maximumIterations: number;
    /** Maximum residual penetration in meters accepted as convergence. */
    overlapTolerance: number;
    /** Per-pass attraction toward original centers, in [0, 1]. */
    attraction: number;
  }>;

export type KeplerDorlingOutputs = Partial<GPUDorlingCartogramOutput>;
export type KeplerDorlingProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerDorlingSettings;
  x: GraphDataView<'float32'>;
  y: GraphDataView<'float32'>;
  values: GraphDataView<'float32'>;
  /** Combined source null validity for X, Y and value, independent of selection. */
  validityMask?: GraphDataView<'uint32'>;
  /** Selected population mask; ignored for the all-source population. */
  mask?: GraphDataView<'uint32'>;
  effectiveMask?: GraphDataView<'uint32'>;
  outputs?: KeplerDorlingOutputs;
};

/** Detach normalized Float32 settings before allocating or recording graph work. */
export function parseKeplerDorlingSettings(value: unknown): KeplerDorlingSettings {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Dorling settings must be an object');
  const settings = value as Record<string, unknown>;
  const coordinateKeys = getKeplerPointWeightCoordinateKeys('planar');
  const keys = [
    ...coordinateKeys,
    'valueColumn',
    'areaScale',
    'maximumIterations',
    'overlapTolerance',
    'attraction'
  ];
  if (Object.keys(settings).some(key => !keys.includes(key)))
    throw new Error('Unexpected Dorling setting');
  const coordinates = parseKeplerPointWeightCoordinates(
    Object.fromEntries(coordinateKeys.map(key => [key, settings[key]]))
  );
  if (coordinates.coordinateSpace !== 'planar')
    throw new Error('Dorling requires projected meter coordinates');
  const valueColumn = settings['valueColumn'];
  if (typeof valueColumn !== 'string' || !valueColumn.length || valueColumn.length > 1024)
    throw new Error('Dorling requires a value column');
  const maximumIterations = settings['maximumIterations'];
  if (
    typeof maximumIterations !== 'number' ||
    !Number.isSafeInteger(maximumIterations) ||
    maximumIterations < 0 ||
    maximumIterations > MAXIMUM_KEPLER_DORLING_ITERATIONS
  )
    throw new Error('Dorling iteration budget exceeds capacity');
  return Object.freeze({
    ...coordinates,
    valueColumn,
    maximumIterations,
    areaScale: readParameter(settings['areaScale'], 1e15),
    overlapTolerance: readParameter(settings['overlapTolerance'], 1e15),
    attraction: readParameter(settings['attraction'], 1)
  });
}

/** Conservative GPU output/scratch reservation; excludes borrowed source buffers and CPU graph nodes. */
export function estimateKeplerDorlingByteLength(rowCount: number): number {
  validateRowCount(rowCount);
  return rowCount * 64 + 4096;
}

/**
 * Source-aligned circle relaxation in an already-projected meter CRS, without reprojection.
 * Circle area is value * areaScale. Zero values remain stationary zero-radius points. Invalid,
 * negative, out-of-bounds, source-null and unselected rows emit NaN geometry/radii, retaining IDs.
 * Coordinates and radii beyond the core's 1e15 limit are also excluded. Float32 rounding can erase
 * small separations at large coordinate origins; this adapter adds no precision or parity claim.
 *
 * metrics = [maximum penetration, maximum displacement from original center] in meters;
 * status = [iterations used, converged, exhausted, excluded rows]. Exhausted finite geometry is
 * a valid bounded layout, with residual overlap disclosed. A stationary overlap is not convergence.
 * Source rows/chunks are borrowed; callers explicitly pack inputs and retain any supplied outputs.
 * Static settings require no parameter buffer. All omitted outputs/scratch belong to the graph.
 */
export function addKeplerDorlingCartogram<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerDorlingProps
) {
  const settings = parseKeplerDorlingSettings(props.settings);
  const id = props.id ?? 'kepler-dorling-cartogram';
  const rowCount = props.values.length;
  validateRowCount(rowCount);
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    (props.population !== 'all' && props.population !== 'selected') ||
    (props.population === 'selected'
      ? !props.mask ||
        !Number.isSafeInteger(props.selectionRevision) ||
        (props.selectionRevision ?? 0) < 1
      : props.selectionRevision !== undefined)
  )
    throw new Error('Dorling requires a current source and explicit population');
  const selectedMask = props.population === 'selected' ? props.mask : undefined;
  const inputs = [props.x, props.y, props.values, props.validityMask, selectedMask].filter(
    view => view !== undefined
  );
  for (const view of inputs) {
    validatePackedView(
      view,
      view === props.validityMask || view === selectedMask ? ['uint32'] : ['float32'],
      id
    );
    if (view.length !== rowCount || view.buffer.graph !== graph)
      throw new Error('Dorling inputs must match source rows and graph');
  }
  const provided = props.outputs ?? {};
  const specifications = [
    [provided.positions, 'float32x2', rowCount],
    [provided.radii, 'float32', rowCount],
    [provided.sourceIds, 'uint32', rowCount],
    [provided.metrics, 'float32x2', 1],
    [provided.status, 'uint32', 4],
    [props.effectiveMask, 'uint32', rowCount]
  ] as const;
  const priorViews: GraphDataView[] = [...inputs];
  for (const [view, format, length] of specifications) {
    if (!view) continue;
    validatePackedView(view, [format], id);
    if (
      view.length !== length ||
      view.buffer.graph !== graph ||
      priorViews.some(input => doGraphDataViewsOverlap(view, input))
    )
      throw new Error('Dorling outputs require disjoint matching graph views');
    priorViews.push(view);
  }
  const outputs: GPUDorlingCartogramOutput = {
    positions:
      provided.positions ??
      createTransientView(graph, `${id}-positions-output`, 'float32x2', rowCount),
    radii: provided.radii ?? createTransientView(graph, `${id}-radii-output`, 'float32', rowCount),
    sourceIds:
      provided.sourceIds ??
      createTransientView(graph, `${id}-source-ids-output`, 'uint32', rowCount),
    metrics: provided.metrics ?? createTransientView(graph, `${id}-metrics-output`, 'float32x2', 1),
    status: provided.status ?? createTransientView(graph, `${id}-status-output`, 'uint32', 4)
  };
  const effectiveMask =
    props.effectiveMask ?? createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const initialPositions = createTransientView(
    graph,
    `${id}-initial-positions`,
    'float32x2',
    rowCount
  );
  const spatialContext = getKeplerPointWeightSpatialContext(settings);
  const contributor = new GPUDorlingCartogram({
    id,
    spatialContext,
    positions: initialPositions,
    values: props.values,
    areaScale: settings.areaScale,
    maximumIterations: settings.maximumIterations,
    overlapTolerance: settings.overlapTolerance,
    attraction: settings.attraction,
    rowTileSize: KEPLER_DORLING_ROW_TILE_SIZE,
    output: outputs
  });
  const bindings = [
    {name: 'x', view: props.x},
    {name: 'y', view: props.y},
    {name: 'values', view: props.values},
    ...(props.validityMask ? [{name: 'validity', view: props.validityMask}] : []),
    ...(selectedMask ? [{name: 'selection', view: selectedMask}] : []),
    {name: 'positions', view: initialPositions, write: true},
    {name: 'effectiveMask', view: effectiveMask, write: true}
  ];
  const bounds = settings.bounds.map(number => number.toPrecision(17));
  graph.addComputePass({
    id: `${id}-prepare`,
    resources: bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.write ? 'storage-read-write' : 'storage-read'
    })),
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: `${id}-prepare`,
        source: /* wgsl */ `
${bindings.map((binding, index) => `@group(0) @binding(${index}) var<storage, ${binding.write ? 'read_write' : 'read'}> ${binding.name}: array<${binding.view.format === 'uint32' ? 'u32' : 'f32'}>;\nconst ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;`).join('\n')}
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= ${rowCount}u) { return; }
  let center = vec2f(x[xOffset + index], y[yOffset + index]);
  let value = values[valuesOffset + index];
  let radius = sqrt(value) * ${Math.sqrt(settings.areaScale / Math.PI).toPrecision(17)};
  let included = finite(center.x) && finite(center.y) && all(abs(center) <= vec2f(1e15))
    && center.x >= ${bounds[0]} && center.x <= ${bounds[2]} && center.y >= ${bounds[1]} && center.y <= ${bounds[3]}
    && finite(value) && value >= 0.0 && finite(radius) && radius <= 1e15
    ${props.validityMask ? '&& validity[validityOffset + index] != 0u' : ''}
    ${selectedMask ? '&& selection[selectionOffset + index] != 0u' : ''};
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  positions[positionsOffset + index * 2u] = select(missing, center.x, included);
  positions[positionsOffset + index * 2u + 1u] = select(missing, center.y, included);
  effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, included);
}`,
        shaderLayout: {
          bindings: bindings.map((binding, index) => ({
            name: binding.name,
            type: 'storage',
            group: 0,
            location: index
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
            ),
            x: Math.ceil(rowCount / 64)
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
  graph.add(contributor);
  return {
    ...outputs,
    outputs,
    effectiveMask,
    spatialContext,
    settings,
    source: Object.freeze({...props.source}),
    population: props.population,
    selectionRevision: props.selectionRevision,
    parameterByteLength: 0,
    destroyParameters() {}
  };
}

function validateRowCount(rowCount: number): void {
  if (
    !Number.isSafeInteger(rowCount) ||
    rowCount < 1 ||
    rowCount > MAXIMUM_KEPLER_DORLING_SOURCE_ROWS
  )
    throw new Error('Dorling source row count exceeds capacity');
}
function readParameter(value: unknown, maximum: number): number {
  if (
    typeof value !== 'number' ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > maximum ||
    !Number.isFinite(Math.fround(value)) ||
    (value !== 0 && Math.fround(value) === 0)
  )
    throw new Error('Dorling parameter requires a bounded nonnegative Float32 value');
  return Math.fround(value);
}
