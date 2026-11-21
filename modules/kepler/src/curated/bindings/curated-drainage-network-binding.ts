// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {addDrainageNetworkRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUTerrainFlowParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import {appendCatalogRow, makeCatalogTable} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {
  validateCatalogRasterInput,
  type CatalogRasterBand,
  type CatalogRasterInput
} from '../../analysis/catalog-raster-input';

const BINDING_ID = 'curated-drainage-network';
/** Bounds the cell polygon publication and its CPU serialization, as well as GPU scratch. */
export const MAXIMUM_DRAINAGE_CELLS = 65536;
/** The underlying fill/flat/accumulation relaxation contract permits at most 1024 rounds. */
const MAXIMUM_DRAINAGE_ITERATIONS = 1024;
const NO_CELL = 0xffffffff;
const UNRESOLVED_CELL = 0xfffffffe;

type DrainageValues = Readonly<Record<string, CatalogParameterValue>>;

type PreparedDrainageBand = {
  validity: Uint32Array;
} & (
  | {format: 'float32'; values: Float32Array}
  | {format: 'sint32'; values: Int32Array}
  | {format: 'uint32'; values: Uint32Array}
);

/** Normalize decoded samples once at upload, comparing nodata in the original source domain. */
function prepareDrainageBand(band: CatalogRasterBand): PreparedDrainageBand {
  const validity = Uint32Array.from(band.values, (sample, cell) =>
    Number(band.validity[cell] !== 0 && Number.isFinite(sample) && sample !== band.noDataValue)
  );
  const {values} = band;
  if (values instanceof Int32Array || values instanceof Int16Array || values instanceof Int8Array) {
    return {
      format: 'sint32',
      values: values instanceof Int32Array ? values : Int32Array.from(values),
      validity
    };
  }
  if (
    values instanceof Uint32Array ||
    values instanceof Uint16Array ||
    values instanceof Uint8Array
  ) {
    return {
      format: 'uint32',
      values: values instanceof Uint32Array ? values : Uint32Array.from(values),
      validity
    };
  }
  return {
    format: 'float32',
    values: values instanceof Float32Array ? values : Float32Array.from(values),
    validity
  };
}

/** Construction-time preparation: preserves the decoded band, validity and original cell indices. */
export function prepareCuratedDrainageNetwork(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  values: DrainageValues
) {
  const elevation = inputs['elevation'];
  validateRequirement(BINDING_ID, elevation?.type === 'raster', 'elevation must be a raster');
  const raster = elevation as CatalogRasterInput;
  validateCatalogRasterInput(BINDING_ID, raster, MAXIMUM_DRAINAGE_CELLS);
  validateRequirement(
    BINDING_ID,
    raster.crs === 'planar' && raster.cellSizeMode === 'uniform',
    'elevation.crs must be planar with uniform cell sizes and consistent linear units'
  );
  const {affine} = raster.grid;
  validateRequirement(
    BINDING_ID,
    affine[1] === 0 && affine[3] === 0,
    'elevation.grid.affine must be axis aligned for the complete planar D8 domain'
  );
  const cellCount = raster.grid.width * raster.grid.height;
  const streamThreshold = values['streamThreshold'] ?? 3;
  validateRequirement(
    BINDING_ID,
    typeof streamThreshold === 'number',
    'streamThreshold must be a number'
  );
  validateFiniteNumber(
    BINDING_ID,
    'streamThreshold',
    streamThreshold as number,
    1,
    MAXIMUM_DRAINAGE_CELLS
  );
  const readBound = (name: string, fallback: number, maximum: number, automatic = false) => {
    const value = values[name] ?? fallback;
    validateRequirement(BINDING_ID, typeof value === 'number', `${name} must be an integer`);
    validateCapacity(BINDING_ID, name, value as number, automatic ? 0 : 1, maximum);
    return automatic && value === 0 ? fallback : (value as number);
  };
  const readBoolean = (name: string) => {
    const value = values[name] ?? true;
    validateRequirement(BINDING_ID, typeof value === 'boolean', `${name} must be boolean`);
    return value as boolean;
  };
  return {
    raster,
    cellCount,
    // Keep float32 stream geometry near zero. Restore the original float64 translation on publication.
    grid: {...raster.grid, affine: [affine[0], 0, 0, 0, affine[4], 0] as const},
    streamThreshold: streamThreshold as number,
    fillDepressions: readBoolean('fillDepressions'),
    resolveFlats: readBoolean('resolveFlats'),
    maximumFillIterations: readBound('maximumFillIterations', 512, MAXIMUM_DRAINAGE_ITERATIONS),
    maximumFlatIterations: readBound('maximumFlatIterations', 512, MAXIMUM_DRAINAGE_ITERATIONS),
    maximumAccumulationIterations: readBound(
      'maximumAccumulationIterations',
      64,
      MAXIMUM_DRAINAGE_ITERATIONS
    ),
    maximumSteps: readBound('maximumSteps', cellCount, MAXIMUM_DRAINAGE_CELLS, true),
    reachCapacity: readBound('reachCapacity', cellCount, MAXIMUM_DRAINAGE_CELLS, true),
    vertexCapacity: readBound('vertexCapacity', 2 * cellCount, 2 * MAXIMUM_DRAINAGE_CELLS, true),
    elevationBand: prepareDrainageBand(raster.band)
  };
}

/** Readbacks of recipe outputs; no terrain computation is repeated by the publication adapter. */
export type CuratedDrainageReadback = {
  flowDirections: ArrayLike<number>;
  streams: ArrayLike<number>;
  accumulation: ArrayLike<number>;
  streamOrder: ArrayLike<number>;
  catchments: ArrayLike<number>;
  distance: ArrayLike<number>;
  validity: ArrayLike<number>;
  terminalCells: ArrayLike<number>;
  sourceCells: ArrayLike<number>;
  targetCells: ArrayLike<number>;
  vertexOffsets: ArrayLike<number>;
  positions: ArrayLike<number>;
  lengths: ArrayLike<number>;
  reachCount: number;
  vertexCount: number;
  catchmentIds: ArrayLike<number>;
  cellCounts: ArrayLike<number>;
  meanDownstreamDistance: ArrayLike<number>;
  summaryCount: number;
  stages: readonly {
    stage: string;
    converged?: number;
    overflow?: number;
    unresolvedCount?: number;
  }[];
};

/** Reject incomplete results before any geometry is published; include the stage status in the error. */
export function publishCuratedDrainageNetwork(
  prepared: ReturnType<typeof prepareCuratedDrainageNetwork>,
  result: CuratedDrainageReadback
): CatalogRunOutput {
  const {stages} = result;
  for (const status of stages) {
    validateRequirement(
      BINDING_ID,
      !('uncertainCount' in status),
      `${status.stage}.uncertainCount was removed; use unresolvedCount`
    );
  }
  const failedStages = stages.filter(
    stage =>
      stage.converged === 0 || (stage.overflow ?? 0) !== 0 || (stage.unresolvedCount ?? 0) !== 0
  );
  validateRequirement(
    BINDING_ID,
    failedStages.length === 0,
    `recipe status must converge without overflow or unresolved paths; ${JSON.stringify(failedStages)}`
  );
  validateCapacity(BINDING_ID, 'network.reachCount', result.reachCount, 0, prepared.reachCapacity);
  validateCapacity(
    BINDING_ID,
    'network.vertexCount',
    result.vertexCount,
    0,
    prepared.vertexCapacity
  );
  validateCapacity(BINDING_ID, 'summary.count', result.summaryCount, 0, prepared.cellCount);
  const {raster, cellCount} = prepared;
  const cells = makeCatalogTable(
    'catchment-cells',
    'Drainage catchment cells',
    {
      source_id: 'string',
      band_index: 'int32',
      cell_index: 'int32',
      catchment_id: 'int32',
      flow_direction: 'int32',
      accumulation: 'float64',
      stream: 'boolean',
      stream_order: 'int32',
      downstream_distance: 'float64',
      terminal_cell: 'int32',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  const reaches = makeCatalogTable(
    'streams',
    'Drainage stream reaches',
    {
      source_id: 'string',
      band_index: 'int32',
      source_cell: 'int32',
      target_cell: 'int32',
      catchment_id: 'int32',
      stream_order: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const summary = makeCatalogTable('catchments', 'Drainage catchment summaries', {
    source_id: 'string',
    band_index: 'int32',
    catchment_id: 'int32',
    cell_count: 'int32',
    mean_downstream_distance: 'float64'
  });
  for (const table of [cells, reaches]) {
    table.coordinateSystem = 'planar';
  }
  const [scaleX, , originX, , scaleY, originY] = raster.grid.affine;
  const position = (column: number, row: number) => [
    originX + scaleX * column,
    originY + scaleY * row
  ];
  const nullableCell = (cell: number) => (cell >= UNRESOLVED_CELL ? null : cell);
  let invalidCellCount = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    const invalid = result.flowDirections[cell] === NO_CELL;
    if (invalid) invalidCellCount++;
    const column = cell % raster.grid.width;
    const row = Math.floor(cell / raster.grid.width);
    // Point-interpreted rasters use sample-centered footprints, area rasters use pixel edges.
    const offset = raster.grid.pixelInterpretation === 'point' ? -0.5 : 0;
    const left = column + offset;
    const top = row + offset;
    appendCatalogRow(cells, {
      source_id: raster.id,
      band_index: raster.band.index,
      cell_index: cell,
      catchment_id: nullableCell(result.catchments[cell]),
      flow_direction: nullableCell(result.flowDirections[cell]),
      accumulation: invalid ? null : result.accumulation[cell],
      stream: invalid ? null : result.streams[cell] === 1,
      stream_order: invalid ? null : result.streamOrder[cell],
      downstream_distance: result.validity[cell] ? result.distance[cell] : null,
      terminal_cell: nullableCell(result.terminalCells[cell]),
      status: invalid ? 'no-data' : 'analysed',
      geometry: JSON.stringify({
        type: 'Polygon',
        coordinates: [
          [
            position(left, top),
            position(left + 1, top),
            position(left + 1, top + 1),
            position(left, top + 1),
            position(left, top)
          ]
        ]
      })
    });
  }
  for (let reach = 0; reach < result.reachCount; reach++) {
    const coordinates: number[][] = [];
    for (
      let vertex = result.vertexOffsets[reach];
      vertex < result.vertexOffsets[reach + 1];
      vertex++
    ) {
      coordinates.push([
        originX + result.positions[2 * vertex],
        originY + result.positions[2 * vertex + 1]
      ]);
    }
    const sourceCell = result.sourceCells[reach];
    appendCatalogRow(reaches, {
      source_id: raster.id,
      band_index: raster.band.index,
      source_cell: sourceCell,
      target_cell: result.targetCells[reach],
      catchment_id: nullableCell(result.catchments[sourceCell]),
      stream_order: result.streamOrder[sourceCell],
      length: result.lengths[reach],
      geometry: JSON.stringify({type: 'LineString', coordinates})
    });
  }
  for (let group = 0; group < result.summaryCount; group++) {
    appendCatalogRow(summary, {
      source_id: raster.id,
      band_index: raster.band.index,
      catchment_id: nullableCell(result.catchmentIds[group]),
      cell_count: result.cellCounts[group],
      mean_downstream_distance: result.meanDownstreamDistance[group]
    });
  }
  return {
    tables: [reaches, cells, summary],
    diagnostics: [
      {
        code: 'planar-geometry-transform-required',
        severity: 'info',
        count: 1,
        message:
          'Planar geometry is retained as text. Map publication requires a geographic CRS transform before publishing geometry to a geographic map.'
      },
      {
        code: 'bounded-planar-domain',
        severity: 'info',
        count: 1,
        message:
          'Complete supplied D8 domain; cells beyond its boundary contribute no flow. Catchments identify terminal outlets or pits, not individual stream reaches.'
      },
      ...(invalidCellCount
        ? [
            {
              code: 'no-data',
              severity: 'warning' as const,
              count: invalidCellCount,
              message:
                'No-data cells are retained with null results and excluded from catchment summaries.'
            }
          ]
        : [])
    ],
    summary: {
      cellCount,
      crs: raster.crs,
      invalidCellCount,
      reachCount: result.reachCount,
      catchmentCount: result.summaryCount,
      streamThreshold: prepared.streamThreshold,
      accumulationUnit: 'cells',
      distanceUnit: 'coordinate-unit',
      fillDepressions: Number(prepared.fillDepressions),
      resolveFlats: Number(prepared.resolveFlats),
      converged: 1,
      overflow: 0,
      ...(stages.some(stage => stage.unresolvedCount !== undefined) ? {unresolvedCount: 0} : {}),
      recipeStatus: JSON.stringify(stages)
    },
    dispatchedContributors: CURATED_DRAINAGE_NETWORK_BINDING.gpuContributorNames
  };
}

export const CURATED_DRAINAGE_NETWORK_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Drainage network and catchments',
  domain: 'raster',
  summary:
    'Planar D8 drainage, stream reaches, terminal-outlet catchments and downstream distance.',
  capabilityIds: ['terrain-hydrology'],
  contributorNames: [
    'GPUTerrainFlow',
    'GPUTerrainStreamOrder',
    'GPUTerrainWatersheds',
    'GPUTerrainFlowDistance',
    'GPUTerrainStreamNetwork',
    'GPUGroupStatistics'
  ],
  gpuContributorNames: [
    'GPUTerrainFlow',
    'GPUTerrainStreamOrder',
    'GPUTerrainWatersheds',
    'GPUTerrainFlowDistance',
    'GPUTerrainStreamNetwork',
    'GPUGroupStatistics'
  ],
  engine: 'gpu',
  inputs: [
    {
      name: 'elevation',
      label: 'Elevation raster (selected numeric band)',
      dataKind: 'raster',
      accepts: [],
      maximumFeatures: 0,
      maximumVertices: 0,
      maximumCells: MAXIMUM_DRAINAGE_CELLS
    }
  ],
  parameters: [
    {
      name: 'streamThreshold',
      label: 'Minimum contributing cells',
      kind: 'number',
      unit: 'cells',
      minimum: 1,
      maximum: MAXIMUM_DRAINAGE_CELLS,
      defaultValue: 3
    },
    {name: 'fillDepressions', label: 'Fill depressions', kind: 'boolean', defaultValue: true},
    {name: 'resolveFlats', label: 'Route across flats', kind: 'boolean', defaultValue: true},
    ...['maximumFillIterations', 'maximumFlatIterations', 'maximumAccumulationIterations'].map(
      (name, index) => ({
        name,
        label: name,
        kind: 'integer' as const,
        minimum: 1,
        maximum: MAXIMUM_DRAINAGE_ITERATIONS,
        defaultValue: index === 2 ? 64 : 512
      })
    ),
    {
      name: 'maximumSteps',
      label: 'Path bound (0 uses cell count)',
      kind: 'integer',
      minimum: 0,
      maximum: MAXIMUM_DRAINAGE_CELLS,
      defaultValue: 0
    },
    {
      name: 'reachCapacity',
      label: 'Reach capacity (0 uses cell count)',
      kind: 'integer',
      minimum: 0,
      maximum: MAXIMUM_DRAINAGE_CELLS,
      defaultValue: 0
    },
    {
      name: 'vertexCapacity',
      label: 'Vertex capacity (0 uses twice cell count)',
      kind: 'integer',
      minimum: 0,
      maximum: 2 * MAXIMUM_DRAINAGE_CELLS,
      defaultValue: 0
    }
  ],
  outputs: [
    {
      tableId: 'streams',
      label: 'Stream reaches',
      geometry: 'line',
      identityColumns: ['source_id', 'band_index', 'source_cell', 'target_cell'],
      maximumRows: MAXIMUM_DRAINAGE_CELLS
    },
    {
      tableId: 'catchment-cells',
      label: 'Catchment cells',
      geometry: 'polygon',
      identityColumns: ['source_id', 'band_index', 'cell_index'],
      maximumRows: MAXIMUM_DRAINAGE_CELLS
    },
    {
      tableId: 'catchments',
      label: 'Catchment summaries',
      geometry: 'none',
      identityColumns: ['source_id', 'band_index', 'catchment_id'],
      maximumRows: MAXIMUM_DRAINAGE_CELLS
    }
  ],
  cleanup:
    'Session releases compiled graph and buffers on success or failure; plain result arrays publish through native Sources.',
  declaredBehavior: [
    'Requires a complete, axis-aligned planar domain; no implicit reprojection or resampling.',
    'No-data is classified in the source sample domain before conversion. Integer bands retain exact 32-bit storage; terrain calculations use float32.',
    'Nonconvergence, unresolved paths or overflow rejects publication with the failing stage status. Downstream-distance and stream-network report unresolvedCount. Successful recipeStatus retains the named stage evidence. Omitted optional unresolved evidence remains omitted; a canonical zero is reported only when a count was supplied.'
  ],
  limitation:
    'No tiled/global watershed stitching. Distances use declared planar units. Catchment polygons are individual cells, not dissolved basin boundaries.',
  async runDataGpu(context, inputs, values) {
    const prepared = prepareCuratedDrainageNetwork(inputs, values);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const {raster, grid} = prepared;
      const validity = session.input('validity', prepared.elevationBand.validity, 'uint32');
      const uploadElevation = (): GPURasterBufferBand => {
        const band = prepared.elevationBand;
        // The original sentinel is already represented in validity. Forwarding a rounded
        // sentinel would reject distinct valid Float64 samples that share a float32 value.
        switch (band.format) {
          case 'sint32':
            return {
              id: raster.id,
              format: 'sint32',
              validity,
              storage: {kind: 'buffer', values: session.input('elevation', band.values, 'sint32')}
            };
          case 'uint32':
            return {
              id: raster.id,
              format: 'uint32',
              validity,
              storage: {kind: 'buffer', values: session.input('elevation', band.values, 'uint32')}
            };
          case 'float32':
            return {
              id: raster.id,
              format: 'float32',
              validity,
              storage: {kind: 'buffer', values: session.input('elevation', band.values, 'float32')}
            };
        }
      };
      const recipe = addDrainageNetworkRecipe(session.graph, {
        flow: {
          grid,
          elevation: uploadElevation(),
          parameters: session.input(
            'parameters',
            getGPUTerrainFlowParameterValues({streamThreshold: prepared.streamThreshold}, grid),
            'float32'
          ),
          cellSizeMode: 'uniform',
          fillDepressions: prepared.fillDepressions,
          resolveFlats: prepared.resolveFlats,
          maximumFillIterations: prepared.maximumFillIterations,
          maximumFlatIterations: prepared.maximumFlatIterations,
          maximumAccumulationIterations: prepared.maximumAccumulationIterations
        },
        maximumSteps: prepared.maximumSteps,
        reachCapacity: prepared.reachCapacity,
        vertexCapacity: prepared.vertexCapacity
      });
      const readbacks: (() => Promise<void>)[] = [];
      function capture<Format extends 'uint32' | 'float32' | 'float32x2'>(
        name: string,
        source: GraphDataView<Format>
      ) {
        const output = session.output(name, source.format, source.length);
        const components = source.format === 'float32x2' ? 2 : 1;
        const format = source.format === 'uint32' ? 'uint32' : 'float32';
        const sourceWords = session.graph.createDataView(source.buffer, {
          format,
          byteOffset: source.byteOffset,
          length: source.length * components
        });
        const destinationWords = session.graph.createDataView(output.view.buffer, {
          format,
          byteOffset: output.view.byteOffset,
          length: source.length * components
        });
        const type = source.format === 'uint32' ? 'u32' : 'f32';
        session.graph.add(
          createWGSLKernelNode(session.graph, {
            id: `publish-${name}`,
            operation: BINDING_ID,
            invocationCount: source.length * components,
            bindings: [
              {name: 'source', view: sourceWords, type, access: 'read'},
              {name: 'destination', view: destinationWords, type, access: 'read_write'}
            ],
            body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
          })
        );
        const captured = {values: new Float32Array(0) as Float32Array | Uint32Array};
        readbacks.push(async () => {
          captured.values =
            source.format === 'uint32'
              ? await session.readUint32(output)
              : await session.readFloat32(output);
        });
        return captured;
      }
      const {outputs} = recipe;
      const captured = {
        flowDirections: capture('directions', outputs.flowDirections),
        streams: capture('streams', outputs.streams),
        accumulation: capture('accumulation', outputs.accumulation),
        streamOrder: capture('order', outputs.streamOrder),
        catchments: capture('catchments', outputs.catchments),
        distance: capture('distance', outputs.distance.distance),
        validity: capture('distance-validity', outputs.distance.validity),
        terminalCells: capture('terminals', outputs.distance.terminalCells),
        sourceCells: capture('sources', outputs.network.sourceCells),
        targetCells: capture('targets', outputs.network.targetCells),
        vertexOffsets: capture('offsets', outputs.network.vertexOffsets),
        positions: capture('positions', outputs.network.positions),
        lengths: capture('lengths', outputs.network.lengths),
        reachCount: capture('reach-count', outputs.network.reachCount),
        vertexCount: capture('vertex-count', outputs.network.vertexCount),
        catchmentIds: capture('summary-ids', outputs.summary.catchmentIds),
        cellCounts: capture('summary-counts', outputs.summary.cellCounts),
        meanDownstreamDistance: capture(
          'summary-distances',
          outputs.summary.meanDownstreamDistance
        ),
        summaryCount: capture('summary-count', outputs.summary.count)
      };
      const stages = recipe.status.stages.map(({stage, status}) => ({
        stage,
        flags: Object.entries(status)
          .filter(([name]) => ['converged', 'overflow', 'unresolvedCount'].includes(name))
          .map(([name, view]) => ({
            name,
            captured: capture(`${stage}-${name}`, view as GraphDataView<'uint32'>)
          }))
      }));
      session.run();
      await Promise.all(readbacks.map(read => read()));
      return publishCuratedDrainageNetwork(prepared, {
        flowDirections: captured.flowDirections.values,
        streams: captured.streams.values,
        accumulation: captured.accumulation.values,
        streamOrder: captured.streamOrder.values,
        catchments: captured.catchments.values,
        distance: captured.distance.values,
        validity: captured.validity.values,
        terminalCells: captured.terminalCells.values,
        sourceCells: captured.sourceCells.values,
        targetCells: captured.targetCells.values,
        vertexOffsets: captured.vertexOffsets.values,
        positions: captured.positions.values,
        lengths: captured.lengths.values,
        catchmentIds: captured.catchmentIds.values,
        cellCounts: captured.cellCounts.values,
        meanDownstreamDistance: captured.meanDownstreamDistance.values,
        reachCount: captured.reachCount.values[0],
        vertexCount: captured.vertexCount.values[0],
        summaryCount: captured.summaryCount.values[0],
        stages: stages.map(({stage, flags}) => ({
          stage,
          ...Object.fromEntries(flags.map(({name, captured: flag}) => [name, flag.values[0]]))
        }))
      });
    });
  }
};
