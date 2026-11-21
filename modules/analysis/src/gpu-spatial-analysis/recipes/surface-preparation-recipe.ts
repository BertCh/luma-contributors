// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveRecipeDestinations} from './recipe-utils';

import {addPreparedRecipe} from './prepared-recipe';
import {snapshotRecipeRasterMetadata} from './recipe-raster-metadata';

import {validateRecipeBoundary, getRecipeView, getOrCreateView} from './recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPURasterFillNoData,
  type GPURasterFillNoDataProps
} from '../../raster-analysis/gpu-raster-fill-nodata';
import {
  GPURasterRankFilter,
  type GPURasterRankFilterProps
} from '../../raster-analysis/gpu-raster-rank-filter';
import {
  getRasterBandViews,
  snapshotRasterBand,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph
} from '../../raster-analysis/raster-band-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {createRecipeKernelNode as createWGSLKernelNode} from './recipe-utils';
import {addRecipeCountNodes, RecipeBuilder, type GPURecipeResult} from './recipe-utils';

export type GPUSurfacePreparationOutput = {
  values: GraphDataView<'float32'>;
  validity: GraphDataView<'uint32'>;
  originalValidity: GraphDataView<'uint32'>;
  filled: GraphDataView<'uint32'>;
  filtered: GraphDataView<'uint32'>;
  unresolvedCount: GraphDataView<'uint32'>;
};
export type GPUSurfacePreparationRecipeProps = Omit<GPURasterFillNoDataProps, 'output'> & {
  filter?: Pick<GPURasterRankFilterProps, 'radius' | 'quantile' | 'borderMode' | 'noDataPolicy'>;
  /** Caller-owned terminal destinations. */
  output?: Partial<GPUSurfacePreparationOutput>;
};
/** Continuous-surface repair with original validity and independent fill/filter provenance. */
export function addSurfacePreparationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSurfacePreparationRecipeProps
): GPURecipeResult<GPUSurfacePreparationOutput> & {band: GPURasterBufferBand<'float32'>} {
  return addPreparedRecipe(graph, props, declareSurfacePreparationRecipe);
}

function declareSurfacePreparationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSurfacePreparationRecipeProps
): GPURecipeResult<GPUSurfacePreparationOutput> & {band: GPURasterBufferBand<'float32'>} {
  const destinations = resolveRecipeDestinations(props.id ?? 'surface-preparation-recipe', props);
  props = snapshotConstructionFields(props, [
    'id',
    'grid',
    'input',
    'maximumDistance',
    'method',
    'filter',
    'output'
  ]);
  const provided = snapshotConstructionFields(destinations ?? {}, [
    'values',
    'validity',
    'originalValidity',
    'filled',
    'filtered',
    'unresolvedCount'
  ]);
  props = {
    ...props,
    input: snapshotRasterBand(props.input),
    output: provided,
    filter:
      props.filter &&
      snapshotConstructionFields(props.filter, ['radius', 'quantile', 'borderMode', 'noDataPolicy'])
  };
  const id = props.id ?? 'surface-preparation-recipe';
  validateRasterTextureBand(id, props.input, props.grid);
  validateRasterBandBelongsToGraph(id, graph, props.input, []);
  // Defaults use fresh graph-owned buffers: validate every caller-owned boundary before registering them.
  validateRecipeBoundary(graph, id, getRasterBandViews(props.input), Object.values(provided));
  const count = props.grid.width * props.grid.height;
  for (const [name, format, length] of [
    ['values', 'float32', count],
    ['validity', 'uint32', count],
    ['originalValidity', 'uint32', count],
    ['filled', 'uint32', count],
    ['filtered', 'uint32', count],
    ['unresolvedCount', 'uint32', 1]
  ] as const) {
    const view = provided[name];
    // With a supplied view getRecipeView only validates; it cannot register a default resource.
    if (view) getRecipeView(graph, `${id} outputs.${name}`, format, length, view);
  }
  const builder = new RecipeBuilder(graph);
  const outputs: GPUSurfacePreparationOutput = {
    values: getOrCreateView(graph, `${id}-values`, 'float32', count, provided.values),
    validity: getOrCreateView(graph, `${id}-validity`, 'uint32', count, provided.validity),
    originalValidity: getOrCreateView(
      graph,
      `${id}-original-validity`,
      'uint32',
      count,
      provided.originalValidity
    ),
    filled: getOrCreateView(graph, `${id}-filled`, 'uint32', count, provided.filled),
    filtered: getOrCreateView(graph, `${id}-filtered`, 'uint32', count, provided.filtered),
    unresolvedCount: getOrCreateView(
      graph,
      `${id}-unresolved`,
      'uint32',
      1,
      provided.unresolvedCount
    )
  };
  const fillValues = props.filter
    ? getOrCreateView(graph, `${id}-fill-values`, 'float32', count)
    : outputs.values;
  const fillValidity = props.filter
    ? getOrCreateView(graph, `${id}-fill-validity`, 'uint32', count)
    : outputs.validity;
  const fillUnresolved = getOrCreateView(graph, `${id}-fill-unresolved`, 'uint32', 1);
  builder.add(
    new GPURasterFillNoData({
      ...props,
      id: `${id}-fill`,
      output: {
        values: fillValues,
        validity: fillValidity,
        filled: outputs.filled,
        status: {invalidCount: fillUnresolved}
      }
    })
  );
  builder.addGlue(
    'provenance',
    createWGSLKernelNode(graph, {
      id: `${id}-provenance`,
      operation: id,
      invocationCount: count,
      bindings: [
        {name: 'validity', view: fillValidity, type: 'u32', access: 'read'},
        {name: 'filled', view: outputs.filled, type: 'u32', access: 'read'},
        {name: 'original', view: outputs.originalValidity, type: 'u32', access: 'read_write'},
        ...(!props.filter
          ? [
              {
                name: 'filtered',
                view: outputs.filtered,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      body: `original[originalOffset + index] = select(0u, 1u, validity[validityOffset + index] != 0u && filled[filledOffset + index] == 0u);
${!props.filter ? 'filtered[filteredOffset + index] = 0u;' : ''}`
    })
  );
  if (props.filter)
    builder.add(
      new GPURasterRankFilter({
        ...props.filter,
        id: `${id}-filter`,
        grid: props.grid,
        input: {
          id: `${id}-filled-band`,
          format: 'float32',
          storage: {kind: 'buffer', values: fillValues},
          validity: fillValidity
        },
        output: {values: outputs.values, validity: outputs.validity, changed: outputs.filtered}
      })
    );
  addRecipeCountNodes(builder, {
    id: `${id}-unresolved`,
    operation: id,
    rowCount: count,
    bindings: [{name: 'validity', view: outputs.validity, type: 'u32', access: 'read'}],
    counts: [
      {
        name: 'missing',
        output: outputs.unresolvedCount,
        value: 'u32(validity[validityOffset + index] == 0u)'
      }
    ]
  });
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {fillValues, fillValidity, grid: snapshotRecipeRasterMetadata(props.grid)},
    band: {
      id,
      format: 'float32',
      storage: {kind: 'buffer', values: outputs.values},
      validity: outputs.validity
    },
    status: {
      stages: [
        {stage: 'fill-nodata', status: {invalidCount: fillUnresolved}},
        {stage: 'prepared-surface', status: {invalidCount: outputs.unresolvedCount}}
      ]
    }
  };
}
