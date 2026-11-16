// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPURasterConnectedComponents} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterValue
} from '../../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import {validateCatalogRasterInput} from '../../analysis/catalog-raster-input';
import {appendCatalogRow, makeCatalogTable} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';

const BINDING_ID = 'curated-landscape-metrics';
const MISSING_PATCH = 0xffffffff;
/** Keep every started mapping alive until all sibling readbacks settle. */
async function readLandscapeBuffers<const Values extends readonly unknown[]>(
  reads: {[Index in keyof Values]: () => Promise<Values[Index]>}
): Promise<Values> {
  const results = await Promise.allSettled(reads.map(read => Promise.resolve().then(read)));
  for (const result of results) if (result.status === 'rejected') throw result.reason;
  return results.map(
    result => (result as PromiseFulfilledResult<unknown>).value
  ) as unknown as Values;
}

/** Bound publication and quadratic recipe work; two distance fields use at most 8 MiB. */
export const LANDSCAPE_METRICS_LIMITS = {
  cells: 4096,
  classes: 32,
  patches: 256,
  patchCells: 1048576
} as const;
/** Positive shader literals must remain normal Float32 values; GPU subnormals may flush to zero. */
export const LANDSCAPE_METRICS_SHADER_LIMITS = {
  minimumPositive: 2 ** -126,
  maximum: (2 - 2 ** -23) * 2 ** 127
} as const;

function validateLandscapeShaderConstant(name: string, value: number): void {
  validateFiniteNumber(
    BINDING_ID,
    name,
    value,
    LANDSCAPE_METRICS_SHADER_LIMITS.minimumPositive,
    LANDSCAPE_METRICS_SHADER_LIMITS.maximum
  );
}

/** Encode the selected native band once; zero is a class and original codes survive publication. */
export function prepareLandscapeMetrics(
  input: CatalogDataInput,
  values: Readonly<Record<string, CatalogParameterValue>> = {}
) {
  validateRequirement(BINDING_ID, input?.type === 'raster', 'landcover must be a native raster');
  if (input.type !== 'raster') throw new Error(`${BINDING_ID}: landcover must be a native raster`);
  validateCatalogRasterInput(BINDING_ID, input, LANDSCAPE_METRICS_LIMITS.cells);
  validateRequirement(
    BINDING_ID,
    input.crs === 'planar' && input.cellSizeMode === 'uniform',
    'landcover.crs must be planar with caller-confirmed linear units; reproject geographic or Web Mercator grids'
  );
  const affine = input.grid.affine;
  validateRequirement(
    BINDING_ID,
    input.grid.pixelInterpretation === 'area' && affine[1] === 0 && affine[3] === 0,
    'landcover.grid must use area pixels and an axis-aligned affine (no rotation or shear)'
  );
  const cellSize = [Math.abs(affine[0]), Math.abs(affine[4])] as const;
  // The recipes interpolate these products into Float32 WGSL. Reject unsupported constants
  // before the first labeling session allocates resources, rather than relying on readback.
  validateLandscapeShaderConstant('landcover.grid.cellSize[0]', cellSize[0]);
  validateLandscapeShaderConstant('landcover.grid.cellSize[1]', cellSize[1]);
  validateLandscapeShaderConstant('landcover.grid.cellArea', cellSize[0] * cellSize[1]);
  validateLandscapeShaderConstant('landcover.grid.domainWidth', input.grid.width * cellSize[0]);
  validateLandscapeShaderConstant('landcover.grid.domainHeight', input.grid.height * cellSize[1]);
  const connectivityValue = values['connectivity'] ?? '4';
  validateRequirement(
    BINDING_ID,
    connectivityValue === '4' || connectivityValue === '8',
    'connectivity must be 4 or 8'
  );
  const connectivity: 4 | 8 = connectivityValue === '8' ? 8 : 4;
  const edgeDepth = values['edgeDepth'] ?? 0;
  const searchRadius = values['searchRadius'] ?? 1000;
  validateRequirement(
    BINDING_ID,
    typeof edgeDepth === 'number' && typeof searchRadius === 'number',
    'edgeDepth and searchRadius must be numbers in [0, 1000000]'
  );
  validateFiniteNumber(BINDING_ID, 'edgeDepth', edgeDepth as number, 0, 1e6);
  validateFiniteNumber(BINDING_ID, 'searchRadius', searchRadius as number, 0, 1e6);
  if (edgeDepth !== 0) validateLandscapeShaderConstant('edgeDepth', edgeDepth as number);
  if (searchRadius !== 0) validateLandscapeShaderConstant('searchRadius', searchRadius as number);
  const distanceModel = values['distanceModel'] ?? 'cell-boundary';
  validateRequirement(
    BINDING_ID,
    distanceModel === 'cell-boundary' || distanceModel === 'cell-center',
    'distanceModel must be cell-boundary or cell-center'
  );
  const excludeClass = values['excludeClass'] ?? false;
  const excludedCode = values['excludedClass'] ?? 0;
  validateRequirement(
    BINDING_ID,
    typeof excludeClass === 'boolean',
    'excludeClass must be boolean'
  );
  validateRequirement(
    BINDING_ID,
    typeof excludedCode === 'number' && Number.isSafeInteger(excludedCode),
    'excludedClass must be a safe integer code'
  );
  const excludedClass = excludeClass ? excludedCode : undefined;
  const codes = new Set<number>();
  const retained = new Uint8Array(input.band.values.length);
  let invalidClassCount = 0;
  let excludedCellCount = 0;
  for (let cell = 0; cell < retained.length; cell++) {
    const value = input.band.values[cell];
    if (!input.band.validity[cell] || value === input.band.noDataValue || value === excludedClass) {
      excludedCellCount++;
      continue;
    }
    if (!Number.isSafeInteger(value)) invalidClassCount++;
    else {
      codes.add(value);
      retained[cell] = 1;
    }
  }
  validateRequirement(
    BINDING_ID,
    invalidClassCount === 0,
    'landcover.band valid class values must be safe integers (no continuous measurements)'
  );
  validateCapacity(BINDING_ID, 'classCount', codes.size, 1, LANDSCAPE_METRICS_LIMITS.classes);
  const classCodes = [...codes].sort((first, second) => first - second);
  const codeIndices = new Map(classCodes.map((value, index) => [value, index]));
  const classes = Uint32Array.from(input.band.values, (value, cell) =>
    retained[cell] ? codeIndices.get(value)! : MISSING_PATCH
  );
  return {
    input,
    classes,
    classCodes,
    excludedCellCount,
    cellSize,
    connectivity,
    edgeDepth: edgeDepth as number,
    searchRadius: searchRadius as number,
    distanceModel: distanceModel as 'cell-boundary' | 'cell-center'
  };
}

/** Compact published GPU minimum-cell roots in source row order, without doing CPU labeling. */
export function compactLandscapePatches(
  prepared: ReturnType<typeof prepareLandscapeMetrics>,
  labelsByClass: readonly Uint32Array[]
) {
  validateRequirement(
    BINDING_ID,
    labelsByClass.length === prepared.classCodes.length &&
      labelsByClass.every(labels => labels.length === prepared.classes.length),
    'labelsByClass must contain one complete label grid per class'
  );
  const patches = new Uint32Array(prepared.classes.length).fill(MISSING_PATCH);
  const patchClasses: number[] = [];
  const rootCells: number[] = [];
  const roots = new Map<string, number>();
  let missingLabels = 0;
  for (let cell = 0; cell < patches.length; cell++) {
    const category = prepared.classes[cell];
    if (category === MISSING_PATCH) continue;
    const label = labelsByClass[category][cell];
    if (!label || label > patches.length) {
      missingLabels++;
      continue;
    }
    const key = `${category}:${label}`;
    let patch = roots.get(key);
    if (patch === undefined) {
      patch = roots.size;
      roots.set(key, patch);
      patchClasses.push(category);
      rootCells.push(label - 1);
    }
    patches[cell] = patch;
  }
  validateRequirement(
    BINDING_ID,
    missingLabels === 0,
    'connected-component labels must cover every retained cell; incomplete segmentation cannot publish'
  );
  validateCapacity(BINDING_ID, 'patchCount', roots.size, 1, LANDSCAPE_METRICS_LIMITS.patches);
  validateMaximumCount(
    BINDING_ID,
    'patchCount times cellCount',
    roots.size * patches.length,
    LANDSCAPE_METRICS_LIMITS.patchCells
  );
  return {patches, patchClasses: Uint32Array.from(patchClasses), rootCells};
}

/** An unreferenced planar affine cannot produce geographic GeoJSON: retain cell attributes only. */
export function makeLandscapeCellsTable(
  prepared: ReturnType<typeof prepareLandscapeMetrics>,
  compacted: ReturnType<typeof compactLandscapePatches>,
  coreValues: Uint32Array
) {
  const {input, classes, classCodes} = prepared;
  const table = makeCatalogTable('cells', 'Land-cover cell attributes', {
    source_id: 'string',
    band_index: 'int32',
    cell_index: 'int32',
    cell_column: 'int32',
    cell_row: 'int32',
    patch_id: 'int32',
    patch_root_cell: 'int32',
    class_code: 'float64',
    core: 'boolean'
  });
  for (let cell = 0; cell < classes.length; cell++) {
    const patch = compacted.patches[cell];
    if (patch === MISSING_PATCH) continue;
    appendCatalogRow(table, {
      source_id: input.id,
      band_index: input.band.index,
      cell_index: cell,
      cell_column: cell % input.grid.width,
      cell_row: Math.floor(cell / input.grid.width),
      patch_id: patch,
      patch_root_cell: compacted.rootCells[patch],
      class_code: classCodes[classes[cell]],
      core: Boolean(coreValues[cell])
    });
  }
  return table;
}

const OUTPUTS = [
  [
    'cells',
    'Land-cover cell attributes',
    ['source_id', 'band_index', 'cell_index'],
    LANDSCAPE_METRICS_LIMITS.cells
  ],
  ['patches', 'Patch metrics and proximity', ['patch_id'], LANDSCAPE_METRICS_LIMITS.patches],
  ['classes', 'Class metrics', ['class_code'], LANDSCAPE_METRICS_LIMITS.classes],
  [
    'adjacency',
    'Directed orthogonal class adjacency',
    ['source_class', 'target_class'],
    LANDSCAPE_METRICS_LIMITS.classes ** 2
  ],
  [
    'distances',
    'Patch pair distances',
    ['source_patch', 'target_patch'],
    LANDSCAPE_METRICS_LIMITS.patches ** 2
  ],
  ['landscape', 'Landscape metrics', ['source_id', 'band_index'], 1]
] as const;

export const CURATED_LANDSCAPE_METRICS_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Landscape metrics and patch proximity',
  domain: 'raster',
  summary:
    'Label categorical land cover and measure patch, class and landscape fragmentation, core area and isolation.',
  capabilityIds: ['coverage-landscape-ecology'],
  contributorNames: ['addLandscapeMetricsRecipe', 'addPatchProximityRecipe'],
  gpuContributorNames: [
    'GPURasterConnectedComponents',
    'addLandscapeMetricsRecipe',
    'addPatchProximityRecipe'
  ],
  engine: 'gpu',
  inputs: [
    {
      name: 'landcover',
      label: 'Selected land-cover band',
      dataKind: 'raster',
      accepts: [],
      maximumFeatures: 0,
      maximumVertices: 0,
      maximumCells: LANDSCAPE_METRICS_LIMITS.cells
    }
  ],
  parameters: [
    {
      name: 'connectivity',
      label: 'Patch connectivity',
      kind: 'enum',
      options: ['4', '8'],
      defaultValue: '4'
    },
    {
      name: 'edgeDepth',
      label: 'Core edge depth',
      kind: 'number',
      minimum: 0,
      maximum: 1e6,
      unit: 'crs',
      defaultValue: 0
    },
    {
      name: 'searchRadius',
      label: 'Proximity search radius',
      kind: 'number',
      minimum: 0,
      maximum: 1e6,
      unit: 'crs',
      defaultValue: 1000
    },
    {
      name: 'distanceModel',
      label: 'Patch separation',
      kind: 'enum',
      options: ['cell-boundary', 'cell-center'],
      defaultValue: 'cell-boundary'
    },
    {name: 'excludeClass', label: 'Exclude one class', kind: 'boolean', defaultValue: false},
    {
      name: 'excludedClass',
      label: 'Excluded class code',
      kind: 'number',
      minimum: -Number.MAX_SAFE_INTEGER,
      maximum: Number.MAX_SAFE_INTEGER,
      defaultValue: 0
    }
  ],
  outputs: OUTPUTS.map(([tableId, label, identityColumns, maximumRows]) => ({
    tableId,
    label,
    geometry: 'none',
    identityColumns,
    maximumRows
  })),
  cleanup:
    'Owns and destroys both GPU sessions, compiled graphs and buffers; output tables contain copied scalars.',
  declaredBehavior: [
    'One complete native categorical band; valid safe integer codes are sorted and densely encoded without changing published class codes. Zero remains a class. Invalid, nodata and explicitly excluded cells form barriers.',
    'Only planar, axis-aligned area grids with confirmed consistent linear units. Areas use squared CRS units; perimeter, core depth and distances use CRS units. No reprojection is inferred. Map publication is withheld because no longitude/latitude transform is supplied; cell outputs contain table attributes and grid indices only.',
    'Published GPU connected components labels each class. Convergence failure rejects all results. Dense patch rows follow the first retained row-major cell; patch root identities are scoped to source id, band and grid.',
    'Landscape metrics count outer and background edges, four orthogonal directed class contacts and area-weighted contagion. Proximity excludes self, uses physical core depth and treats the outer boundary as an edge.',
    'At most 4096 cells, 32 classes, 256 patches and 1048576 patch-cell pairs; capacities reject before proximity allocation. Missing nearest neighbours and nonfinite metric values publish null with diagnostics; zero-distance proximity is undefined.'
  ],
  limitation:
    'GPU only; complete bounded grids, Float32 metrics, no geographic distances or map geometry. Positive shader literals (cell sizes, cell area, domain extents, depth and radius) must fit normal Float32; zero depth and radius are supported. Search radius is an explicit constant in confirmed CRS units. Patch identity changes when the grid or connectivity changes.',
  async runDataGpu(context, inputs, values) {
    const prepared = prepareLandscapeMetrics(inputs['landcover'], values);
    const {
      input,
      classes,
      classCodes,
      cellSize,
      connectivity,
      edgeDepth,
      searchRadius,
      distanceModel
    } = prepared;
    const {width, height} = input.grid;
    const labelsByClass = await withCatalogGpuSession(
      context.device,
      `${BINDING_ID}-labels`,
      async session => {
        const outputs = classCodes.map((_, category) => {
          const labels = session.output(`labels-${category}`, 'uint32', classes.length);
          const validity = session.output(`validity-${category}`, 'uint32', classes.length);
          const converged = session.output(`converged-${category}`, 'uint32', 1);
          const foreground = Uint32Array.from(classes, value => Number(value === category));
          new GPURasterConnectedComponents({
            id: `${BINDING_ID}-class-${category}`,
            width,
            height,
            connectivity,
            input: makeGPURasterBand(
              session.input(`foreground-${category}`, foreground, 'uint32'),
              input.grid
            ),
            output: labels.view,
            outputValidity: validity.view,
            converged: converged.view
          }).addToGraph(session.graph);
          return {labels, converged};
        });
        session.run();
        const convergence = await readLandscapeBuffers(
          outputs.map(output => () => session.readScalar(output.converged))
        );
        validateRequirement(
          BINDING_ID,
          convergence.every(value => value === 1),
          'connected-component converged must equal 1 for every class; no result was published'
        );
        return await readLandscapeBuffers(
          outputs.map(output => () => session.readUint32(output.labels))
        );
      }
    );
    const compacted = compactLandscapePatches(prepared, labelsByClass);
    const patchCount = compacted.patchClasses.length;
    return await withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const patches = session.input('patches', compacted.patches, 'uint32');
      const patchClasses = session.input('patch-classes', compacted.patchClasses, 'uint32');
      const patchMetrics = session.output('patch-metrics', 'float32x4', patchCount);
      const classMetrics = session.output('class-metrics', 'float32x4', classCodes.length);
      const classAdjacency = session.output('adjacency', 'uint32', classCodes.length ** 2);
      const aggregation = session.output('aggregation', 'float32', classCodes.length);
      const landscape = session.output('landscape', 'float32x4', 1);
      const landscapeAggregation = session.output('landscape-aggregation', 'float32', 1);
      const landscapeCohesion = session.output('landscape-cohesion', 'float32', 1);
      const core = session.output('core', 'uint32', classes.length);
      const distances = session.output('distances', 'float32', patchCount ** 2);
      const proximity = session.output('proximity', 'float32x4', patchCount);
      addLandscapeMetricsRecipe(session.graph, {
        id: `${BINDING_ID}-metrics`,
        patches,
        patchClasses,
        width,
        height,
        cellSize,
        connectivity,
        classCount: classCodes.length,
        output: {
          patchMetrics: patchMetrics.view,
          classMetrics: classMetrics.view,
          classAdjacency: classAdjacency.view,
          aggregation: aggregation.view,
          landscape: landscape.view,
          landscapeAggregation: landscapeAggregation.view,
          landscapeCohesion: landscapeCohesion.view
        }
      });
      addPatchProximityRecipe(session.graph, {
        id: `${BINDING_ID}-proximity`,
        patches,
        patchClasses,
        width,
        height,
        cellSize,
        edgeDepth,
        searchRadius,
        distanceModel,
        output: {core: core.view, distances: distances.view, metrics: proximity.view}
      });
      session.run();
      const [
        patchValues,
        classValues,
        adjacencyValues,
        aggregationValues,
        landscapeValues,
        landscapeAggregationValues,
        landscapeCohesionValues,
        coreValues,
        distanceValues,
        proximityValues
      ] = await readLandscapeBuffers([
        () => session.readFloat32(patchMetrics),
        () => session.readFloat32(classMetrics),
        () => session.readUint32(classAdjacency),
        () => session.readFloat32(aggregation),
        () => session.readFloat32(landscape),
        () => session.readFloat32(landscapeAggregation),
        () => session.readFloat32(landscapeCohesion),
        () => session.readUint32(core),
        () => session.readFloat32(distances),
        () => session.readFloat32(proximity)
      ]);
      let undefinedMetricCount = 0;
      const finiteMetric = (value: number) => {
        if (Number.isFinite(value)) return value;
        undefinedMetricCount++;
        return null;
      };
      const cellsTable = makeLandscapeCellsTable(prepared, compacted, coreValues);
      const patchTable = makeCatalogTable('patches', 'Patch metrics and proximity', {
        patch_id: 'int32',
        patch_root_cell: 'int32',
        class_code: 'float64',
        area: 'float64',
        perimeter: 'float64',
        cell_count: 'int32',
        like_sides: 'int32',
        core_area: 'float64',
        nearest_same_class: 'float64',
        proximity: 'float64'
      });
      const classTable = makeCatalogTable('classes', 'Class metrics', {
        class_code: 'float64',
        area: 'float64',
        perimeter: 'float64',
        mesh: 'float64',
        cohesion: 'float64',
        aggregation: 'float64'
      });
      const adjacencyTable = makeCatalogTable('adjacency', 'Directed class adjacency', {
        source_class: 'float64',
        target_class: 'float64',
        side_count: 'int32'
      });
      const distanceTable = makeCatalogTable('distances', 'Patch distances', {
        source_patch: 'int32',
        target_patch: 'int32',
        distance: 'float64'
      });
      const landscapeTable = makeCatalogTable('landscape', 'Landscape metrics', {
        source_id: 'string',
        band_index: 'int32',
        contagion: 'float64',
        mesh: 'float64',
        area: 'float64',
        richness: 'int32',
        aggregation: 'float64',
        cohesion: 'float64'
      });
      for (let patch = 0; patch < patchCount; patch++) {
        appendCatalogRow(patchTable, {
          patch_id: patch,
          patch_root_cell: compacted.rootCells[patch],
          class_code: classCodes[compacted.patchClasses[patch]],
          area: finiteMetric(patchValues[4 * patch]),
          perimeter: finiteMetric(patchValues[4 * patch + 1]),
          cell_count: patchValues[4 * patch + 2],
          like_sides: patchValues[4 * patch + 3],
          core_area: finiteMetric(proximityValues[4 * patch + 1]),
          nearest_same_class: finiteMetric(proximityValues[4 * patch + 2]),
          proximity: finiteMetric(proximityValues[4 * patch + 3])
        });
        for (let target = 0; target < patchCount; target++)
          if (target !== patch)
            appendCatalogRow(distanceTable, {
              source_patch: patch,
              target_patch: target,
              distance: finiteMetric(distanceValues[patch * patchCount + target])
            });
      }
      for (let category = 0; category < classCodes.length; category++) {
        appendCatalogRow(classTable, {
          class_code: classCodes[category],
          area: finiteMetric(classValues[4 * category]),
          perimeter: finiteMetric(classValues[4 * category + 1]),
          mesh: finiteMetric(classValues[4 * category + 2]),
          cohesion: finiteMetric(classValues[4 * category + 3]),
          aggregation: finiteMetric(aggregationValues[category])
        });
        for (let target = 0; target < classCodes.length; target++)
          appendCatalogRow(adjacencyTable, {
            source_class: classCodes[category],
            target_class: classCodes[target],
            side_count: adjacencyValues[category * classCodes.length + target]
          });
      }
      appendCatalogRow(landscapeTable, {
        source_id: input.id,
        band_index: input.band.index,
        contagion: finiteMetric(landscapeValues[0]),
        mesh: finiteMetric(landscapeValues[1]),
        area: finiteMetric(landscapeValues[2]),
        richness: landscapeValues[3],
        aggregation: finiteMetric(landscapeAggregationValues[0]),
        cohesion: finiteMetric(landscapeCohesionValues[0])
      });
      return {
        tables: [cellsTable, patchTable, classTable, adjacencyTable, distanceTable, landscapeTable],
        diagnostics: [
          {
            code: 'unreferenced-planar-grid',
            severity: 'info',
            count: 1,
            message:
              'Map publication withheld: the planar grid has no explicit transform to longitude/latitude. Cell identities and grid indices remain available as table attributes.'
          },
          ...(prepared.excludedCellCount
            ? [
                {
                  code: 'excluded-cells',
                  severity: 'info' as const,
                  count: prepared.excludedCellCount,
                  message:
                    'Masked, nodata or excluded-class cells are barriers and do not contribute area.'
                }
              ]
            : []),
          ...(undefinedMetricCount
            ? [
                {
                  code: 'undefined-metrics',
                  severity: 'warning' as const,
                  count: undefinedMetricCount,
                  message:
                    'Null metrics denote no same-class neighbour, zero-distance proximity, or an undefined small-domain aggregation/cohesion/contagion.'
                }
              ]
            : [])
        ],
        dispatchedContributors: [...CURATED_LANDSCAPE_METRICS_BINDING.gpuContributorNames],
        summary: {
          sourceId: input.id,
          bandIndex: input.band.index,
          cellCount: classes.length,
          retainedCellCount: cellsTable.rowCount,
          excludedCellCount: prepared.excludedCellCount,
          classCount: classCodes.length,
          patchCount,
          connectivity,
          edgeDepth,
          searchRadius,
          distanceModel,
          coreModel: 'physical',
          coreBoundary: 'edge',
          contagionWeighting: 'area',
          countBoundary: 1,
          countBackground: 1,
          coordinateSystem: 'planar',
          mapPublication: 'withheld-unreferenced-planar',
          gridWidth: width,
          gridHeight: height,
          gridAffine: JSON.stringify(input.grid.affine),
          lengthUnit: 'declared CRS linear unit',
          areaUnit: 'squared declared CRS linear unit',
          labelsConverged: 1,
          undefinedMetricCount
        }
      };
    });
  }
};
