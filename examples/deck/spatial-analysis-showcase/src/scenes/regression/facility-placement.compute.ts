// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUBoundedFacilityAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {LocalMetricProjection} from '../../engine/projection';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {PolygonFillLayer} from './b6-fill-layer';
import {FACILITY_PLACEMENT_PALETTE} from './facility-placement-palette';
import {
  buildOutlineSegments,
  computeCentroids,
  createFeatureLocator,
  formatInteger,
  formatNumber,
  readPolygonGeometry,
  triangulatePolygonGeometry
} from './b6-polygons';
import {createNamedReader} from './b6-reader';

const CANDIDATE_COUNT = 8;
const SELECT_COUNT = 3;
export type FacilityPlacementOptions = {
  map: 'demand' | 'assignments';
  objective: 'average' | 'worst';
  capacity: number;
  showExisting: boolean;
};

type AllocationBuffers = {
  selected: Buffer;
  assignments: Buffer;
  objective: Buffer;
  status: Buffer;
};

/** Small candidate-set siting study with GPU p-center and capacitated p-median solutions. */
export async function createFacilityPlacement(
  ctx: SceneContext<FacilityPlacementOptions>
): Promise<SceneInstance<FacilityPlacementOptions>> {
  const {device} = ctx;
  const tracts = ctx.datasets.get('chicago-tracts');
  const libraries = ctx.datasets.get('chicago-facilities');
  const origin = tracts.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const resources = new SpatialAnalysisResources(device, 'facility-placement');

  const geometry = readPolygonGeometry(tracts);
  const tractCount = geometry.featureCount;
  const locateTract = createFeatureLocator(geometry);
  const tractCentroids = computeCentroids(geometry);
  const projectedCentroids = new Float32Array(tractCount * 2);
  for (let tract = 0; tract < tractCount; tract++) {
    const [x, y] = projection.project(tractCentroids[tract * 2], tractCentroids[tract * 2 + 1]);
    projectedCentroids[tract * 2] = x;
    projectedCentroids[tract * 2 + 1] = y;
  }
  const populationColumn = tracts.column<Float32Array>('population');
  const population = Float32Array.from(populationColumn, value =>
    Number.isFinite(value) && value > 0 ? value : 0
  );
  const tractGeoids = (tracts.manifest as unknown as {geoid: string[]}).geoid;

  // The bounded candidate set is spatially spread across populated tracts, starting at the most
  // populous tract. The transparent sampling rule avoids implying that these are available parcels.
  const candidateTracts = selectFarthestCandidates(projectedCentroids, population, CANDIDATE_COUNT);
  const candidatePositions = new Float32Array(CANDIDATE_COUNT * 2);
  candidateTracts.forEach((tract, candidate) =>
    candidatePositions.set(projectedCentroids.subarray(tract * 2, tract * 2 + 2), candidate * 2)
  );
  const demandOrder = Array.from({length: tractCount}, (_, index) => index).sort(
    (left, right) => population[right] - population[left] || left - right
  );
  const rowOfTract = new Int32Array(tractCount);
  demandOrder.forEach((tract, row) => (rowOfTract[tract] = row));
  const orderedDemand = Float32Array.from(demandOrder, tract => population[tract]);
  const costs = new Float32Array(tractCount * CANDIDATE_COUNT);
  for (let row = 0; row < tractCount; row++) {
    const tract = demandOrder[row];
    const x = projectedCentroids[tract * 2];
    const y = projectedCentroids[tract * 2 + 1];
    for (let candidate = 0; candidate < CANDIDATE_COUNT; candidate++) {
      const dx = x - candidatePositions[candidate * 2];
      const dy = y - candidatePositions[candidate * 2 + 1];
      costs[row * CANDIDATE_COUNT + candidate] = Math.hypot(dx, dy);
    }
  }
  const totalDemand = orderedDemand.reduce((sum, value) => sum + value, 0);

  const facilityPositionsAll = libraries.projectColumn('position', origin);
  const libraryKinds = libraries.column<Uint8Array>('category');
  const libraryCategoryNames = libraries.categories('category');
  const libraryIndex = libraryCategoryNames.indexOf('library');
  const libraryCoordinates: number[] = [];
  for (let row = 0; row < libraryKinds.length; row++) {
    if (libraryKinds[row] === libraryIndex)
      libraryCoordinates.push(facilityPositionsAll[row * 2], facilityPositionsAll[row * 2 + 1]);
  }

  const triangleData = triangulatePolygonGeometry(geometry);
  const outlineSegments = buildOutlineSegments(geometry);
  const trianglesBuffer = resources.createBuffer('tract-triangles', triangleData.corners);
  const ownersBuffer = resources.createBuffer('tract-owners', triangleData.owners);
  const outlineBuffer = resources.createBuffer('tract-outline', outlineSegments);
  const candidatePositionBuffer = resources.createBuffer('candidates', candidatePositions);
  const existingLibraryBuffer = resources.createBuffer(
    'existing-libraries',
    Float32Array.from(libraryCoordinates)
  );
  const averageMap = resources.createBuffer(
    'average-map',
    new Uint32Array(tractCount).fill(0xffffffff)
  );
  const worstMap = resources.createBuffer(
    'worst-map',
    new Uint32Array(tractCount).fill(0xffffffff)
  );
  const populationMap = resources.createBuffer('population-map', population);

  const demandBuffer = resources.createBuffer('demand', orderedDemand);
  const costBuffer = resources.createBuffer('costs', costs);
  const capacityValues = new Float32Array(CANDIDATE_COUNT).fill(ctx.options.capacity);
  const capacityBuffer = resources.createBuffer('capacities', capacityValues);
  const makeOutput = (prefix: string): AllocationBuffers => ({
    selected: resources.createBuffer(`${prefix}-selected`, new Uint32Array(CANDIDATE_COUNT)),
    assignments: resources.createBuffer(`${prefix}-assignments`, new Uint32Array(tractCount)),
    objective: resources.createBuffer(`${prefix}-objective`, new Float32Array(1)),
    status: resources.createBuffer(`${prefix}-status`, new Uint32Array(3))
  });
  const average = makeOutput('average');
  const worst = makeOutput('worst');

  const graph = new GPUCommandGraph<void>(device, {id: 'facility-placement'});
  const view = <Format extends GPUVectorFormat>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  const costsView = view('costs', costBuffer, 'float32', tractCount * CANDIDATE_COUNT);
  const demandView = view('demand', demandBuffer, 'float32', tractCount);
  const capacitiesView = view('capacities', capacityBuffer, 'float32', CANDIDATE_COUNT);
  const sourceRowsView = view(
    'demand-source-rows',
    resources.createBuffer('demand-source-rows', Uint32Array.from(demandOrder)),
    'uint32',
    tractCount
  );
  const averageMapView = view('average-map', averageMap, 'uint32', tractCount);
  const worstMapView = view('worst-map', worstMap, 'uint32', tractCount);
  const makeOutputViews = (name: string, output: AllocationBuffers) => ({
    selectedFacilities: view(`${name}-selected`, output.selected, 'uint32', CANDIDATE_COUNT),
    assignments: view(`${name}-assignments`, output.assignments, 'uint32', tractCount),
    objective: view(`${name}-objective`, output.objective, 'float32', 1),
    status: view(`${name}-status`, output.status, 'uint32', 3)
  });
  const averageViews = makeOutputViews('average', average);
  graph.add(
    new GPUBoundedFacilityAllocation({
      id: 'capacity-limited-average-access',
      operation: 'capacitated-p-median',
      facilityCount: CANDIDATE_COUNT,
      selectCount: SELECT_COUNT,
      costs: costsView,
      demand: demandView,
      capacities: capacitiesView,
      output: averageViews
    })
  );
  addAllocationMapPass(
    graph,
    'average',
    averageViews.assignments,
    averageViews.selectedFacilities,
    sourceRowsView,
    averageMapView
  );
  const worstViews = makeOutputViews('worst', worst);
  graph.add(
    new GPUBoundedFacilityAllocation({
      id: 'worst-case-access',
      operation: 'p-center',
      facilityCount: CANDIDATE_COUNT,
      selectCount: SELECT_COUNT,
      costs: costsView,
      demand: demandView,
      capacities: capacitiesView,
      output: worstViews
    })
  );
  addAllocationMapPass(
    graph,
    'worst',
    worstViews.assignments,
    worstViews.selectedFacilities,
    sourceRowsView,
    worstMapView
  );
  const compiled = resources.track(await graph.compileAsync());

  let dirty = true;
  let revision = 0;
  let submittedRevision = -1;
  let readbackRevision = -1;
  let inFlightRevision = -1;
  let latest: {
    averageAssignments: Uint32Array;
    averageSelected: Uint32Array;
    averageStatus: Uint32Array;
    worstAssignments: Uint32Array;
    worstSelected: Uint32Array;
    worstObjective: number;
    worstStatus: Uint32Array;
  } | null = null;
  const reader = createNamedReader(
    resources,
    'facility-placement',
    [
      {name: 'averageSelected', buffer: average.selected, bytes: CANDIDATE_COUNT * 4},
      {name: 'averageAssignments', buffer: average.assignments, bytes: tractCount * 4},
      {name: 'averageStatus', buffer: average.status, bytes: 12},
      {name: 'worstSelected', buffer: worst.selected, bytes: CANDIDATE_COUNT * 4},
      {name: 'worstAssignments', buffer: worst.assignments, bytes: tractCount * 4},
      {name: 'worstObjective', buffer: worst.objective, bytes: 4},
      {name: 'worstStatus', buffer: worst.status, bytes: 12}
    ],
    get => {
      const averageAssignments = get('averageAssignments').u32;
      const worstAssignments = get('worstAssignments').u32;
      const averageSelected = get('averageSelected').u32;
      const worstSelected = get('worstSelected').u32;
      const averageStatus = get('averageStatus').u32;
      const worstObjective = get('worstObjective').f32[0];
      const worstStatus = get('worstStatus').u32;
      const responseRevision = inFlightRevision;
      inFlightRevision = -1;
      if (responseRevision !== revision) {
        dirty = true;
        return;
      }
      latest = {
        averageAssignments,
        averageSelected,
        averageStatus,
        worstAssignments,
        worstSelected,
        worstObjective,
        worstStatus
      };
      publishSummary(
        ctx,
        latest,
        demandOrder,
        orderedDemand,
        costs,
        tractGeoids,
        candidateTracts,
        totalDemand
      );
      dirty = false;
      ctx.requestLayers();
    }
  );

  const candidateCoordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  return {
    getCompiledGraphs: () => [compiled],
    setOption(id) {
      if (id === 'map') ctx.requestLayers();
      if (id === 'capacity') {
        capacityBuffer.write(new Float32Array(CANDIDATE_COUNT).fill(ctx.options.capacity));
        revision++;
        dirty = true;
      }
      if (id === 'objective') {
        if (latest)
          publishSummary(
            ctx,
            latest,
            demandOrder,
            orderedDemand,
            costs,
            tractGeoids,
            candidateTracts,
            totalDemand
          );
        else dirty = true;
      }
      ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (dirty && submittedRevision !== revision) {
        compiled.encode(commandEncoder, {parameters: undefined});
        submittedRevision = revision;
      }
      if (reader.isPending) {
        if (submittedRevision !== inFlightRevision) reader.request(commandEncoder);
        else reader.flush(commandEncoder);
      } else if (readbackRevision !== submittedRevision) {
        // Track only a copy that can start now. Requests made while a read is pending merely mark
        // the reader stale; their eventual copy will contain the latest submitted graph output.
        inFlightRevision = submittedRevision;
        readbackRevision = submittedRevision;
        reader.request(commandEncoder);
      } else reader.flush(commandEncoder);
    },
    getLayers() {
      const isDark = ctx.theme() === 'dark';
      const layers: Layer[] = [
        new PolygonFillLayer({
          id: `facility-placement-assignments-${ctx.options.objective}`,
          triangles: trianglesBuffer,
          owners: ownersBuffer,
          triangleCount: triangleData.triangleCount,
          values:
            ctx.options.map === 'demand'
              ? populationMap
              : ctx.options.objective === 'average'
                ? averageMap
                : worstMap,
          valueFormat: ctx.options.map === 'demand' ? 'float32' : 'uint32',
          colormap: ctx.options.map === 'demand' ? 'cividis' : 'category',
          valueRange: [0, 12000],
          palette: FACILITY_PLACEMENT_PALETTE,
          noDataColor: [80, 86, 96, 105],
          opacity: 0.84
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'facility-placement-tract-outline',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          segments: outlineBuffer,
          instanceCount: outlineSegments.length / 4,
          widthPixels: 0.45,
          color: isDark ? [228, 232, 240, 65] : [42, 48, 60, 80]
        }),
        new SpatialAnalysisPointLayer({
          id: 'facility-placement-candidates',
          coordinateOrigin: candidateCoordinateOrigin,
          positions: candidatePositionBuffer,
          instanceCount: CANDIDATE_COUNT,
          shape: 'ring',
          radiusPixels: 7,
          outlineWidthPixels: 1.6,
          color: [91, 105, 122, 245]
        })
      ];
      if (ctx.options.map === 'assignments') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `facility-placement-selected-${ctx.options.objective}`,
            coordinateOrigin: candidateCoordinateOrigin,
            positions: candidatePositionBuffer,
            instanceCount: CANDIDATE_COUNT,
            values: ctx.options.objective === 'average' ? average.selected : worst.selected,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [226, 153, 54, 255],
            noDataColor: [0, 0, 0, 0],
            shape: 'star',
            radiusPixels: 9,
            outlineColor: isDark ? [28, 32, 38, 255] : [255, 250, 230, 255],
            outlineWidthPixels: 1
          })
        );
      }
      if (ctx.options.showExisting && libraryCoordinates.length) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'facility-placement-existing-libraries',
            coordinateOrigin: candidateCoordinateOrigin,
            positions: existingLibraryBuffer,
            instanceCount: libraryCoordinates.length / 2,
            color: [76, 129, 210, 240],
            noDataColor: [0, 0, 0, 0],
            outlineColor: [242, 247, 255, 235],
            outlineWidthPixels: 0.8,
            radiusPixels: 4.5
          })
        );
      }
      return layers;
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    getTooltip(event) {
      if (!event.coordinate || !latest) return null;
      const closest = locateTract(event.coordinate[0], event.coordinate[1]);
      if (closest < 0) return null;
      const active =
        ctx.options.objective === 'average' ? latest.averageAssignments : latest.worstAssignments;
      const row = rowOfTract[closest];
      const assignedCandidate = row >= 0 ? active[row] : 0xffffffff;
      const assignmentText =
        assignedCandidate < CANDIDATE_COUNT
          ? `Candidate tract ${tractGeoids[candidateTracts[assignedCandidate]]}`
          : 'Unassigned under current capacity';
      return [
        `Tract ${tractGeoids[closest]}`,
        `Population ${formatInteger(population[closest])}`,
        assignmentText
      ].join('\n');
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}

function selectFarthestCandidates(
  positions: Float32Array,
  population: Float32Array,
  count: number
): number[] {
  const selected: number[] = [];
  let first = 0;
  for (let tract = 1; tract < population.length; tract++) {
    if (population[tract] > population[first]) first = tract;
  }
  selected.push(first);
  while (selected.length < count) {
    let bestTract = -1;
    let bestDistance = -1;
    for (let tract = 0; tract < population.length; tract++) {
      if (population[tract] <= 0 || selected.includes(tract)) continue;
      let nearestDistance = Infinity;
      for (const chosen of selected) {
        const dx = positions[tract * 2] - positions[chosen * 2];
        const dy = positions[tract * 2 + 1] - positions[chosen * 2 + 1];
        nearestDistance = Math.min(nearestDistance, dx * dx + dy * dy);
      }
      if (nearestDistance > bestDistance) {
        bestDistance = nearestDistance;
        bestTract = tract;
      }
    }
    if (bestTract < 0) break;
    selected.push(bestTract);
  }
  while (selected.length < count) selected.push(selected[selected.length - 1]);
  return selected;
}

function addAllocationMapPass(
  graph: GPUCommandGraph<void>,
  id: string,
  assignments: GraphDataView<'uint32'>,
  selected: GraphDataView<'uint32'>,
  sourceRows: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): void {
  addKernelPass(graph, {
    id: `${id}-map-assignments`,
    invocationCount: output.length,
    bindings: [
      {name: 'assignments', view: assignments, type: 'u32', access: 'read'},
      {name: 'selected', view: selected, type: 'u32', access: 'read'},
      {name: 'sourceRows', view: sourceRows, type: 'u32', access: 'read'},
      {name: 'mapValues', view: output, type: 'u32', access: 'read_write'}
    ],
    declarations: `const CANDIDATE_COUNT: u32 = ${CANDIDATE_COUNT}u;`,
    body: `let sourceRow = sourceRows[sourceRowsOffset + index];
  let candidate = assignments[assignmentsOffset + index];
  var rank = 0xffffffffu;
  if (candidate < CANDIDATE_COUNT && selected[selectedOffset + candidate] != 0u) {
    rank = 0u;
    for (var other = 0u; other < candidate; other++) {
      if (selected[selectedOffset + other] != 0u) { rank++; }
    }
  }
  mapValues[mapValuesOffset + sourceRow] = rank;`
  });
}

function publishSummary(
  ctx: SceneContext<FacilityPlacementOptions>,
  summary: {
    averageAssignments: Uint32Array;
    averageSelected: Uint32Array;
    averageStatus: Uint32Array;
    worstAssignments: Uint32Array;
    worstSelected: Uint32Array;
    worstObjective: number;
    worstStatus: Uint32Array;
  },
  demandOrder: number[],
  orderedDemand: Float32Array,
  costs: Float32Array,
  tractGeoids: string[],
  candidateTracts: number[],
  totalDemand: number
): void {
  const selected =
    ctx.options.objective === 'average' ? summary.averageSelected : summary.worstSelected;
  const selectedNames = Array.from(selected, (value, candidate) =>
    value ? tractGeoids[candidateTracts[candidate]] : null
  ).filter((value): value is string => value !== null);
  let unassignedDemand = 0;
  for (let row = 0; row < summary.averageAssignments.length; row++) {
    if (summary.averageAssignments[row] === 0xffffffff) unassignedDemand += orderedDemand[row];
  }
  const measureMeanDistance = (assignments: Uint32Array, commonSupport: boolean) => {
    let assignedDemand = 0;
    let weightedDistance = 0;
    for (let row = 0; row < assignments.length; row++) {
      const candidate = assignments[row];
      const weight = orderedDemand[row];
      if (candidate === 0xffffffff) continue;
      if (commonSupport && summary.averageAssignments[row] === 0xffffffff) continue;
      assignedDemand += weight;
      weightedDistance += weight * costs[row * CANDIDATE_COUNT + candidate];
    }
    return assignedDemand > 0 ? weightedDistance / assignedDemand / 1000 : NaN;
  };
  const averageMeanKm = measureMeanDistance(summary.averageAssignments, false);
  const worstMeanKm = measureMeanDistance(summary.worstAssignments, true);
  const averageUnassignedDemand = unassignedDemand;
  const averageServedDemand = totalDemand - averageUnassignedDemand;
  ctx.setReadout(
    'candidates',
    `${CANDIDATE_COUNT} tract centroids; choose ${SELECT_COUNT} (${combinationCount(CANDIDATE_COUNT, SELECT_COUNT)} subsets)`
  );
  ctx.setReadout(
    'demand',
    `${formatInteger(totalDemand)} residents across ${formatInteger(orderedDemand.length)} tracts`
  );
  ctx.setReadout(
    'selected',
    selectedNames.length ? selectedNames.join(', ') : 'GPU result pending'
  );
  ctx.setReadout(
    'averageDistance',
    Number.isFinite(averageMeanKm) && averageServedDemand > 0
      ? `${formatNumber(averageMeanKm, 2)} km across assigned population`
      : 'n/a'
  );
  ctx.setReadout(
    'unassigned',
    `${formatInteger(averageUnassignedDemand)} people in ${formatInteger(summary.averageStatus[0])} tracts`
  );
  ctx.setReadout(
    'worstDistance',
    Number.isFinite(summary.worstObjective) && summary.worstStatus[0] === 0
      ? `${formatNumber(summary.worstObjective / 1000, 2)} km maximum centroid distance`
      : 'n/a'
  );
  ctx.setChart('comparison', {
    kind: 'bars',
    title: 'Mean distance on population assigned by both solutions',
    values: [averageMeanKm, worstMeanKm],
    labels: ['Average objective', 'Worst-case objective'],
    colors: [
      [72, 201, 176, 255],
      [115, 147, 255, 255]
    ],
    yLabel: 'population-weighted distance (km)',
    description:
      'Both bars use the same population support: tracts assigned in the capacity-limited solution. The p-center solution’s distance is measured for those same tracts. Unassigned tracts are excluded from both bars and shown separately.'
  });
}

function combinationCount(count: number, selected: number): number {
  let result = 1;
  for (let index = 1; index <= selected; index++) result = (result * (count - index + 1)) / index;
  return Math.round(result);
}
