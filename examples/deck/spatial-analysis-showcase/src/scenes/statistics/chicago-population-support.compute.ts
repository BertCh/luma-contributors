// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getGPURasterFractionalCoverageParameterValues,
  GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {
  addDasymetricRedistributionRecipe,
  addExposureByZoneRecipe,
  addPopulationSupportAlignmentRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {Buffer} from '@luma.gl/core';
import {fetchBytes, fetchJson, getDataFileUrl} from '../../data/loaders';
import type {GeoJsonCollection, GeoJsonGeometry} from '../../data/loaders';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

export type ChicagoPopulationSupportOptions = {
  display: 'source-counts' | 'allocation' | 'built-up-fraction' | 'tree-cover';
  support: 'uniform-area' | 'built-up';
  treeThreshold: '10' | '25' | '50';
};

const GRID_SIZE = 64;
const TARGET_SIZE = 8;
const SOURCE_COUNT = 12;
const FINE_CELL_COUNT = GRID_SIZE * GRID_SIZE;
const TARGET_COUNT = TARGET_SIZE * TARGET_SIZE;
const TUPLE_CAPACITY = SOURCE_COUNT * 4096;
const PAIR_CAPACITY = SOURCE_COUNT * TARGET_COUNT;
const FRAME_ORIGIN: [number, number] = [-87.59, 41.795];
const ANALYSIS_FRAME = {kind: 'lambert-equal-area' as const, origin: FRAME_ORIGIN};
const GRID = makeGPURasterMetadata({
  width: GRID_SIZE,
  height: GRID_SIZE,
  cellSize: 62.5,
  origin: [-2000, 2000],
  rowDirection: 'south'
});
const TARGET_GRID = makeGPURasterMetadata({
  width: TARGET_SIZE,
  height: TARGET_SIZE,
  cellSize: 500,
  origin: [-2000, 2000],
  rowDirection: 'south'
});

type SupportManifest = {
  properties: {populationSum: number; tractCount: number};
};

function getPolygons(geometry: GeoJsonGeometry | null): number[][][][] {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates as number[][][]];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates as number[][][][];
  return [];
}

function buildPolygonFields(
  geojson: GeoJsonCollection,
  project: (longitude: number, latitude: number) => [number, number]
) {
  const positions: number[] = [];
  const featureOffsets: number[] = [0];
  const polygonOffsets: number[] = [0];
  const ringOffsets: number[] = [0];
  const outlines: number[] = [];
  for (const feature of geojson.features) {
    for (const polygon of getPolygons(feature.geometry)) {
      for (const ring of polygon) {
        const openRing =
          ring.length > 1 &&
          ring[0][0] === ring[ring.length - 1][0] &&
          ring[0][1] === ring[ring.length - 1][1]
            ? ring.slice(0, -1)
            : ring;
        const projected = openRing.map(([longitude, latitude]) => project(longitude, latitude));
        for (let index = 0; index < projected.length; index++) {
          const first = projected[index];
          const second = projected[(index + 1) % projected.length];
          positions.push(first[0], first[1]);
          outlines.push(first[0], first[1], second[0], second[1]);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    outlines: Float32Array.from(outlines)
  };
}

export async function createChicagoPopulationSupport(
  context: SceneContext<ChicagoPopulationSupportOptions>
): Promise<SceneInstance<ChicagoPopulationSupportOptions>> {
  const supportDataset = context.datasets.get('chicago-population-support');
  if (!supportDataset?.raster) throw new Error('Bundled Hyde Park support assets are required.');
  const [populationBytes, blackBytes, geoidBytes, manifest, supportGeoJson] = await Promise.all([
    fetchBytes(getDataFileUrl('chicago-population-support', 'population.bin'), context.signal),
    fetchBytes(getDataFileUrl('chicago-population-support', 'nhBlack.bin'), context.signal),
    fetchBytes(getDataFileUrl('chicago-population-support', 'tract-geoid.txt'), context.signal),
    fetchJson<SupportManifest>(
      getDataFileUrl('chicago-population-support', 'manifest.json'),
      context.signal
    ),
    fetchJson<GeoJsonCollection>(
      getDataFileUrl('chicago-population-support', 'tracts.geojson'),
      context.signal
    )
  ]);
  const population = new Float32Array(populationBytes);
  const nhBlack = new Float32Array(blackBytes);
  const tractGeoids = new TextDecoder().decode(geoidBytes).trim().split(/\r?\n/);
  const populationTotal = population.reduce((total, value) => total + value, 0);
  if (
    population.length !== SOURCE_COUNT ||
    nhBlack.length !== SOURCE_COUNT ||
    tractGeoids.length !== SOURCE_COUNT ||
    supportGeoJson?.features.length !== SOURCE_COUNT ||
    manifest.properties.tractCount !== SOURCE_COUNT ||
    manifest.properties.populationSum !== 29559 ||
    populationTotal !== 29559 ||
    population.some(value => !Number.isFinite(value) || value < 0) ||
    nhBlack.some(
      (value, index) => !Number.isFinite(value) || value < 0 || value > population[index]
    )
  ) {
    throw new Error('Hyde Park support rows do not match the bundled 12-tract source contract.');
  }
  for (let sourceIndex = 0; sourceIndex < SOURCE_COUNT; sourceIndex++) {
    const feature = supportGeoJson.features[sourceIndex];
    if (
      feature.properties?.GEOID !== tractGeoids[sourceIndex] ||
      population[sourceIndex] !== Number(feature.properties?.population)
    ) {
      throw new Error(
        'Hyde Park GEOID, geometry and population rows must preserve one source order.'
      );
    }
  }
  const projection = createAnalysisFrameProjection(ANALYSIS_FRAME);
  const polygonFields = buildPolygonFields(supportGeoJson, projection.project);
  const polygonMesh = buildPolygonMesh(supportGeoJson, projection.project);
  const resources = new SpatialAnalysisResources(context.device, 'chicago-population-support');
  const segments = resources.createBuffer('tract-source-outlines', polygonFields.outlines);
  const polygonBuffers = createPolygonMeshBuffers(resources, polygonMesh, 'tract-source-fill');
  const sourceValues = resources.createBuffer('observed-population', population);
  const fineFractions = supportDataset.raster.values;
  if (fineFractions.length !== FINE_CELL_COUNT * 3)
    throw new Error('Expected three 64 × 64 WorldCover fraction bands.');
  const builtUp = Float32Array.from(fineFractions.slice(0, FINE_CELL_COUNT));
  const uniform = new Float32Array(FINE_CELL_COUNT).fill(1);
  if (
    fineFractions.some(value => !Number.isFinite(value) || value < 0 || value > 1) ||
    fineFractions.slice(FINE_CELL_COUNT * 2).some(value => value !== 1)
  ) {
    throw new Error('WorldCover fractions must be finite in [0, 1] with complete valid support.');
  }
  const ancillaryBuffer = resources.createBuffer(
    'ancillary-weights',
    context.options.support === 'built-up' ? builtUp : uniform
  );
  const builtUpFractionBuffer = resources.createBuffer('built-up-fraction-input', builtUp);
  const quadrantBoundaryBuffer = resources.createBuffer(
    'tree-quadrant-boundaries',
    new Float32Array([-2000, 0, 2000, 0, 0, -2000, 0, 2000])
  );
  const positionsBuffer = resources.createBuffer('polygon-positions', polygonFields.positions);
  const featureOffsetsBuffer = resources.createBuffer(
    'feature-offsets',
    polygonFields.featureOffsets
  );
  const polygonOffsetsBuffer = resources.createBuffer(
    'polygon-offsets',
    polygonFields.polygonOffsets
  );
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', polygonFields.ringOffsets);
  const sourceTotalsBuffer = resources.createBuffer('source-totals', SOURCE_COUNT * 4);
  const targetTotalsBuffer = resources.createBuffer('target-totals', TARGET_COUNT * 4);
  const allocatedTotalsBuffer = resources.createBuffer('allocated-totals', SOURCE_COUNT * 4);
  const unallocatedTotalsBuffer = resources.createBuffer('unallocated-totals', SOURCE_COUNT * 4);
  const sourceValidityBuffer = resources.createBuffer(
    'source-validity',
    new Uint32Array(SOURCE_COUNT)
  );
  const statusBuffer = resources.createBuffer('ledger-status', new Uint32Array(7));
  const pairStatusBuffer = resources.createBuffer('pair-status', new Uint32Array(2));
  const targetZonesArray = new Uint32Array(FINE_CELL_COUNT);
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      targetZonesArray[row * GRID_SIZE + column] =
        Math.floor(row / 8) * TARGET_SIZE + Math.floor(column / 8);
    }
  }
  const targetZonesBuffer = resources.createBuffer('target-zones', targetZonesArray);
  const graph = new GPUCommandGraph(context.device, {id: 'chicago-population-support'});
  const importBuffer = <Format extends 'float32' | 'float32x2' | 'uint32' | 'uint32x4'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, buffer.id, buffer, format, length);
  const zones = {
    positions: importBuffer(positionsBuffer, 'float32x2', polygonFields.positions.length / 2),
    featureOffsets: importBuffer(
      featureOffsetsBuffer,
      'uint32',
      polygonFields.featureOffsets.length
    ),
    polygonOffsets: importBuffer(
      polygonOffsetsBuffer,
      'uint32',
      polygonFields.polygonOffsets.length
    ),
    ringOffsets: importBuffer(ringOffsetsBuffer, 'uint32', polygonFields.ringOffsets.length)
  };
  const shuffledPopulations = new Float32Array(SOURCE_COUNT * 2);
  const shuffledSourceIds = new Uint32Array(SOURCE_COUNT);
  const targetIds = new Uint32Array(SOURCE_COUNT);
  for (let row = 0; row < SOURCE_COUNT; row++) {
    const sourceIndex = SOURCE_COUNT - 1 - row;
    shuffledSourceIds[row] = Number(tractGeoids[sourceIndex].slice(5));
    shuffledPopulations[row * 2] = nhBlack[sourceIndex];
    shuffledPopulations[row * 2 + 1] = population[sourceIndex] - nhBlack[sourceIndex];
    targetIds[row] = Number(tractGeoids[row].slice(5));
  }
  const populationGroupsBuffer = resources.createBuffer(
    'population-groups-shuffled',
    shuffledPopulations
  );
  const sourceIdsBuffer = resources.createBuffer('source-tract-ids-shuffled', shuffledSourceIds);
  const targetIdsBuffer = resources.createBuffer('target-tract-ids-geometry-order', targetIds);
  const populationGroups = importBuffer(populationGroupsBuffer, 'float32x2', SOURCE_COUNT);
  const sourceIds = importBuffer(sourceIdsBuffer, 'uint32', SOURCE_COUNT);
  const alignedTargetIds = importBuffer(targetIdsBuffer, 'uint32', SOURCE_COUNT);
  const alignment = addPopulationSupportAlignmentRecipe(graph, {
    id: 'hyde-park-population-alignment',
    populations: populationGroups,
    sourceIds,
    targetIds: alignedTargetIds
  });
  const alignedGroupsBuffer = resources.createBuffer(
    'aligned-population-groups',
    SOURCE_COUNT * 2 * 4
  );
  const alignmentStatusBuffer = resources.createBuffer('alignment-status', 4 * 4);
  const alignedGroups = importBuffer(alignedGroupsBuffer, 'float32x2', SOURCE_COUNT);
  const alignmentStatus = importBuffer(alignmentStatusBuffer, 'uint32x4', 1);
  addKernelPass(graph, {
    id: 'aligned-population-export',
    invocationCount: SOURCE_COUNT,
    bindings: [
      {name: 'groups', view: alignment.outputs.populations, type: 'f32', access: 'read'},
      {name: 'totals', view: alignedGroups, type: 'f32', access: 'read_write'}
    ],
    body: 'totals[totalsOffset+2u*index]=groups[groupsOffset+2u*index];totals[totalsOffset+2u*index+1u]=groups[groupsOffset+2u*index+1u];'
  });
  addKernelPass(graph, {
    id: 'alignment-status-export',
    invocationCount: 1,
    bindings: [
      {name: 'status', view: alignment.packedStatus, type: 'u32', access: 'read'},
      {name: 'output', view: alignmentStatus, type: 'u32', access: 'read_write'}
    ],
    body: 'for(var word=0u;word<4u;word++){output[outputOffset+word]=status[statusOffset+word];}'
  });
  const totals = importBuffer(sourceTotalsBuffer, 'float32', SOURCE_COUNT);
  addKernelPass(graph, {
    id: 'aligned-population-group-sum',
    invocationCount: SOURCE_COUNT,
    bindings: [
      {name: 'groups', view: alignment.outputs.populations, type: 'f32', access: 'read'},
      {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
    ],
    body: 'totals[totalsOffset+index]=groups[groupsOffset+2u*index]+groups[groupsOffset+2u*index+1u];'
  });
  const targetZones = importBuffer(targetZonesBuffer, 'uint32', FINE_CELL_COUNT);
  const ancillaryValues = importBuffer(ancillaryBuffer, 'float32', FINE_CELL_COUNT);
  const targetTotalsView = importBuffer(targetTotalsBuffer, 'float32', TARGET_COUNT);
  const allocatedTotalsView = importBuffer(allocatedTotalsBuffer, 'float32', SOURCE_COUNT);
  const unallocatedTotalsView = importBuffer(unallocatedTotalsBuffer, 'float32', SOURCE_COUNT);
  const eligibleWeightsBuffer = resources.createBuffer('eligible-source-weights', SOURCE_COUNT * 4);
  const transferredFractionsBuffer = resources.createBuffer(
    'transferred-source-fractions',
    SOURCE_COUNT * 4
  );
  const eligibleWeightsView = importBuffer(eligibleWeightsBuffer, 'float32', SOURCE_COUNT);
  const transferredFractionsView = importBuffer(
    transferredFractionsBuffer,
    'float32',
    SOURCE_COUNT
  );
  const sourceValidityView = importBuffer(sourceValidityBuffer, 'uint32', SOURCE_COUNT);
  const statusView = importBuffer(statusBuffer, 'uint32', 7);
  const pairStatusView = importBuffer(pairStatusBuffer, 'uint32', 2);
  const ancillary = {
    id: 'selected-support',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: ancillaryValues}
  };
  const recipe = addDasymetricRedistributionRecipe(graph, {
    id: 'hyde-park-redistribution',
    grid: GRID,
    source: {
      kind: 'polygons',
      zones,
      tupleCapacity: TUPLE_CAPACITY,
      parameters: resources
        .createParameterBuffer(
          'coverage-parameters',
          'float32',
          GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH,
          getGPURasterFractionalCoverageParameterValues(GRID)
        )
        .importToGraph(graph)
    },
    sourceTotals: totals,
    targetZones,
    targetZoneCount: TARGET_COUNT,
    ancillary,
    pairCapacity: PAIR_CAPACITY,
    outputs: {
      targetTotals: targetTotalsView,
      sourceAllocatedTotals: allocatedTotalsView,
      sourceUnallocatedTotals: unallocatedTotalsView,
      sourceEligibleWeights: eligibleWeightsView,
      sourceTransferredFractions: transferredFractionsView,
      sourceValidity: sourceValidityView
    }
  });
  const treeScore = new Float32Array(TARGET_COUNT);
  const treeCover = new Float32Array(TARGET_COUNT);
  const treeCoverFine = fineFractions.slice(FINE_CELL_COUNT, FINE_CELL_COUNT * 2);
  const validFine = fineFractions.slice(FINE_CELL_COUNT * 2);
  const treeZones = new Uint32Array(TARGET_COUNT);
  for (let targetIndex = 0; targetIndex < TARGET_COUNT; targetIndex++) {
    let treeSum = 0;
    let validSum = 0;
    for (let fineRow = 0; fineRow < 8; fineRow++) {
      for (let fineColumn = 0; fineColumn < 8; fineColumn++) {
        const row = Math.floor(targetIndex / TARGET_SIZE) * 8 + fineRow;
        const column = (targetIndex % TARGET_SIZE) * 8 + fineColumn;
        const fineIndex = row * GRID_SIZE + column;
        treeSum += treeCoverFine[fineIndex];
        validSum += validFine[fineIndex];
      }
    }
    treeCover[targetIndex] = treeSum / 64;
    treeScore[targetIndex] = 1 - treeCover[targetIndex];
    treeZones[targetIndex] =
      (Math.floor(targetIndex / TARGET_SIZE) >= 4 ? 2 : 0) +
      (targetIndex % TARGET_SIZE >= 4 ? 1 : 0);
    if (validSum !== 64) treeScore[targetIndex] = Number.NaN;
  }
  const treeScoreBuffer = resources.createBuffer('low-tree-cover-score', treeScore);
  const treeCoverBuffer = resources.createBuffer('tree-cover-fraction-target', treeCover);
  const treeZonesBuffer = resources.createBuffer('tree-exposure-quadrants', treeZones);
  const exposureThresholdBuffer = resources.createBuffer(
    'tree-exposure-threshold',
    new Float32Array([1 - Number(context.options.treeThreshold) / 100])
  );
  const exposureOutputBuffer = resources.createBuffer('tree-exposure-summary', 4 * 5 * 4);
  const exposureStatusBuffer = resources.createBuffer('tree-exposure-status', 8 * 4);
  const treeScoreView = importBuffer(treeScoreBuffer, 'float32', TARGET_COUNT);
  const treeZonesView = importBuffer(treeZonesBuffer, 'uint32', TARGET_COUNT);
  const exposureThresholdView = importBuffer(exposureThresholdBuffer, 'float32', 1);
  const exposureSummaryView = importBuffer(exposureOutputBuffer, 'float32', 4 * 5);
  const exposureStatusView = importBuffer(exposureStatusBuffer, 'uint32', 8);
  const exposure = addExposureByZoneRecipe(graph, {
    id: 'hyde-park-low-tree-cover-exposure',
    grid: TARGET_GRID,
    hazard: {
      id: 'low-tree-cover-score',
      format: 'float32',
      storage: {kind: 'buffer', values: treeScoreView}
    },
    population: {
      id: 'modeled-target-population',
      format: 'float32',
      storage: {kind: 'buffer', values: targetTotalsView}
    },
    weightMeaning: 'count',
    zones: treeZonesView,
    zoneCount: 4,
    threshold: exposureThresholdView
  });
  addKernelPass(graph, {
    id: 'tree-exposure-summary-export',
    invocationCount: 4,
    bindings: [
      {name: 'total', view: exposure.outputs.totalPopulation, type: 'f32', access: 'read'},
      {name: 'observed', view: exposure.outputs.observedPopulation, type: 'f32', access: 'read'},
      {name: 'exposed', view: exposure.outputs.exposedPopulation, type: 'f32', access: 'read'},
      {name: 'coverage', view: exposure.outputs.coverage, type: 'f32', access: 'read'},
      {name: 'share', view: exposure.outputs.exposedShare, type: 'f32', access: 'read'},
      {name: 'output', view: exposureSummaryView, type: 'f32', access: 'read_write'}
    ],
    body: 'output[outputOffset+index]=total[totalOffset+index];output[outputOffset+4u+index]=observed[observedOffset+index];output[outputOffset+8u+index]=exposed[exposedOffset+index];output[outputOffset+12u+index]=coverage[coverageOffset+index];output[outputOffset+16u+index]=share[shareOffset+index];'
  });
  addKernelPass(graph, {
    id: 'tree-exposure-status-export',
    invocationCount: 8,
    bindings: [
      {
        name: 'thresholdInvalid',
        view: exposure.outputs.thresholdInvalidCount,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'overflow0',
        view: exposure.status.stages[0].status.overflow!,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'overflow1',
        view: exposure.status.stages[1].status.overflow!,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'overflow2',
        view: exposure.status.stages[2].status.overflow!,
        type: 'u32',
        access: 'read'
      },
      {name: 'output', view: exposureStatusView, type: 'u32', access: 'read_write'}
    ],
    body: 'var value=0u;switch(index){case 0u:{value=thresholdInvalid[thresholdInvalidOffset];}case 1u:{value=overflow0[overflow0Offset];}case 2u:{value=overflow1[overflow1Offset];}default:{value=overflow2[overflow2Offset];}}output[outputOffset+index]=value;'
  });
  addKernelPass(graph, {
    id: 'population-status-export',
    invocationCount: 7,
    bindings: [
      {name: 'complete', view: recipe.status.complete, type: 'u32', access: 'read'},
      {name: 'invalidTuples', view: recipe.status.invalidTupleCount, type: 'u32', access: 'read'},
      {name: 'invalidSources', view: recipe.status.invalidSourceCount, type: 'u32', access: 'read'},
      {
        name: 'arithmeticInvalid',
        view: recipe.status.arithmeticInvalidCount,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'coverageCount',
        view: recipe.intermediates.coverage.status.count,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'coverageRequired',
        view: recipe.intermediates.coverage.status.requiredCount,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'coverageOverflow',
        view: recipe.intermediates.coverage.status.overflow,
        type: 'u32',
        access: 'read'
      },
      {name: 'output', view: statusView, type: 'u32', access: 'read_write'}
    ],
    body: 'var value=0u;switch(index){case 0u:{value=complete[completeOffset];}case 1u:{value=invalidTuples[invalidTuplesOffset];}case 2u:{value=invalidSources[invalidSourcesOffset];}case 3u:{value=arithmeticInvalid[arithmeticInvalidOffset];}case 4u:{value=coverageCount[coverageCountOffset];}case 5u:{value=coverageRequired[coverageRequiredOffset];}default:{value=coverageOverflow[coverageOverflowOffset];}}output[outputOffset+index]=value;'
  });
  addKernelPass(graph, {
    id: 'population-pair-status-export',
    invocationCount: 2,
    bindings: [
      {
        name: 'pairOverflow',
        view: recipe.status.stages[2].status.overflow!,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'pairRequired',
        view: recipe.status.stages[2].status.requiredCount!,
        type: 'u32',
        access: 'read'
      },
      {name: 'output', view: pairStatusView, type: 'u32', access: 'read_write'}
    ],
    body: 'output[outputOffset+index]=select(pairOverflow[pairOverflowOffset],pairRequired[pairRequiredOffset],index==1u);'
  });
  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  let generation = 0;
  let ready = false;
  let dirty = true;
  let selectedDisplay = context.options.display;
  let selectedThreshold = context.options.treeThreshold;
  let selectedSupport = context.options.support;
  let latestChartData: {
    generation: number;
    targetTotals: Float32Array;
    treeExposed: Float32Array;
  } | null = null;
  const readoutBuffers: {buffer: Buffer; size: number}[] = [
    {buffer: targetTotalsBuffer, size: TARGET_COUNT * 4},
    {buffer: allocatedTotalsBuffer, size: SOURCE_COUNT * 4},
    {buffer: unallocatedTotalsBuffer, size: SOURCE_COUNT * 4},
    {buffer: eligibleWeightsBuffer, size: SOURCE_COUNT * 4},
    {buffer: transferredFractionsBuffer, size: SOURCE_COUNT * 4},
    {buffer: alignedGroupsBuffer, size: SOURCE_COUNT * 2 * 4},
    {buffer: exposureOutputBuffer, size: 4 * 5 * 4},
    {buffer: sourceValidityBuffer, size: SOURCE_COUNT * 4},
    {buffer: statusBuffer, size: 7 * 4},
    {buffer: pairStatusBuffer, size: 2 * 4},
    {buffer: alignmentStatusBuffer, size: 4 * 4},
    {buffer: exposureStatusBuffer, size: 8 * 4}
  ];
  const publishSupportChoice = () =>
    context.setReadout(
      'supportChoice',
      selectedSupport === 'built-up'
        ? 'Built-up fraction · WorldCover class 50'
        : 'Uniform area · all valid cells'
    );
  const publishTargetChart = () => {
    if (!ready || !latestChartData || latestChartData.generation !== generation) {
      context.setChart('targetCells', null);
      return;
    }
    if (selectedDisplay === 'source-counts') {
      context.setChart('targetCells', {
        kind: 'bars',
        title: 'Observed population by whole tract',
        labels: tractGeoids,
        values: population,
        description:
          'Observed SVI 2022 tract counts, matching the displayed whole source geometries.'
      });
    } else if (selectedDisplay === 'tree-cover') {
      context.setChart('targetCells', {
        kind: 'bars',
        title: `Estimated residents in low mapped tree-cover cells · ${selectedThreshold}%`,
        labels: ['Northwest', 'Northeast', 'Southwest', 'Southeast'],
        values: latestChartData.treeExposed,
        description:
          'GPU exposure outputs weighted by modeled target counts; no health or access interpretation.'
      });
    } else {
      context.setChart('targetCells', {
        kind: 'bars',
        title: `${selectedSupport === 'built-up' ? 'Built-up fraction' : 'Uniform area'} allocation by 500 m cell`,
        labels: Array.from(
          {length: TARGET_COUNT},
          (_, index) => `${Math.floor(index / TARGET_SIZE) + 1}.${(index % TARGET_SIZE) + 1}`
        ),
        values: latestChartData.targetTotals,
        description:
          'The 64 bars are the returned GPU targetTotals; counts are modeled allocations.'
      });
    }
  };
  const clearPublishedResults = () => {
    ready = false;
    latestChartData = null;
    for (const readoutId of [
      'allocatedPopulation',
      'unallocatedPopulation',
      'targetPopulation',
      'massBalance',
      'supportLedger',
      'validSourceCount',
      'alignmentStatus',
      'alignedCounts',
      'coverageStatus',
      'pairStatus',
      'lowTreePopulation',
      'treeCoverage'
    ]) {
      context.setReadout(readoutId, null);
    }
    context.setChart('targetCells', null);
  };
  const createReader = (ticketGeneration: number) =>
    new SummaryReader(resources, 'population-ledger', readoutBuffers, bytes => {
      if (ticketGeneration !== generation) return;
      const floatLength = TARGET_COUNT + SOURCE_COUNT * 6 + 20;
      const values = new Float32Array(bytes, 0, floatLength);
      const uintValues = new Uint32Array(bytes, floatLength * 4);
      const targetTotals = values.slice(0, TARGET_COUNT);
      const allocated = values.slice(TARGET_COUNT, TARGET_COUNT + SOURCE_COUNT);
      const unallocated = values.slice(
        TARGET_COUNT + SOURCE_COUNT,
        TARGET_COUNT + SOURCE_COUNT * 2
      );
      const eligible = values.slice(
        TARGET_COUNT + SOURCE_COUNT * 2,
        TARGET_COUNT + SOURCE_COUNT * 3
      );
      const transferred = values.slice(
        TARGET_COUNT + SOURCE_COUNT * 3,
        TARGET_COUNT + SOURCE_COUNT * 4
      );
      const aligned = values.slice(
        TARGET_COUNT + SOURCE_COUNT * 4,
        TARGET_COUNT + SOURCE_COUNT * 6
      );
      const exposureValues = values.slice(TARGET_COUNT + SOURCE_COUNT * 6);
      const complete = uintValues[12] === 1;
      const validSourceCount = uintValues
        .slice(0, SOURCE_COUNT)
        .reduce((sum, value) => sum + Number(value !== 0), 0);
      const invalidTupleCount = uintValues[13];
      const invalidSourceCount = uintValues[14];
      const arithmeticInvalidCount = uintValues[15];
      const count = uintValues[16];
      const required = uintValues[17];
      const coverageOverflow = uintValues[18];
      const pairOverflow = uintValues[19];
      const pairRequired = uintValues[20];
      const alignmentOffset = SOURCE_COUNT + 7 + 2;
      const exposureStatusOffset = alignmentOffset + 4;
      const alignmentComplete = uintValues[alignmentOffset] === 1;
      const alignmentMissing = uintValues[alignmentOffset + 1];
      const alignmentDuplicates = uintValues[alignmentOffset + 2];
      const alignmentExcluded = uintValues[alignmentOffset + 3];
      const thresholdInvalid = uintValues[exposureStatusOffset];
      const exposureOverflow = uintValues.slice(exposureStatusOffset + 1, exposureStatusOffset + 4);
      const exposureValid =
        exposureValues.slice(0, 12).every(Number.isFinite) &&
        Array.from({length: 4}, (_, zoneIndex) => {
          const totalWeight = exposureValues[zoneIndex];
          const observedWeight = exposureValues[4 + zoneIndex];
          const coverage = exposureValues[12 + zoneIndex];
          const share = exposureValues[16 + zoneIndex];
          return Number.isFinite(coverage) && Number.isFinite(share)
            ? true
            : totalWeight === 0 &&
                observedWeight === 0 &&
                Number.isNaN(coverage) &&
                Number.isNaN(share);
        }).every(Boolean);
      const valid =
        alignmentComplete &&
        alignmentMissing === 0 &&
        alignmentDuplicates === 0 &&
        alignmentExcluded === 0 &&
        complete &&
        count === required &&
        coverageOverflow === 0 &&
        pairOverflow === 0 &&
        invalidTupleCount === 0 &&
        invalidSourceCount === 0 &&
        arithmeticInvalidCount === 0 &&
        thresholdInvalid === 0 &&
        exposureOverflow.every(value => value === 0) &&
        aligned.every(Number.isFinite) &&
        eligible.every(value => Number.isFinite(value) && value >= 0) &&
        transferred.every(value => Number.isFinite(value) && value >= 0 && value <= 1.0001) &&
        exposureValid;
      if (!valid) {
        clearPublishedResults();
        publishSupportChoice();
        context.setReadout(
          'recipeStatus',
          `withheld · complete ${Number(complete)} · invalid tuples ${invalidTupleCount} · invalid sources ${invalidSourceCount} · arithmetic ${arithmeticInvalidCount}`
        );
        context.setReadout('validSourceCount', validSourceCount);
        context.setReadout(
          'alignmentStatus',
          `withheld · missing ${alignmentMissing} · duplicate ${alignmentDuplicates} · excluded ${alignmentExcluded}`
        );
        context.setReadout('coverageStatus', `${count}/${required} · overflow ${coverageOverflow}`);
        context.setReadout('pairStatus', `${pairRequired} required · overflow ${pairOverflow}`);
        context.setStatus(
          'Population support evidence withheld because alignment, coverage, or recipe status failed.'
        );
        context.requestLayers();
        return;
      }
      const sum = (items: Float32Array) => items.reduce((total, value) => total + value, 0);
      const allocatedTotal = sum(allocated);
      const unallocatedTotal = sum(unallocated);
      const targetTotal = sum(targetTotals);
      const sourceTotal = aligned.reduce((total, value) => total + value, 0);
      ready = true;
      context.setReadout('sourcePopulation', formatCount(sourceTotal));
      publishSupportChoice();
      context.setReadout('allocatedPopulation', formatCount(allocatedTotal));
      context.setReadout('unallocatedPopulation', formatCount(unallocatedTotal));
      context.setReadout('targetPopulation', formatCount(targetTotal));
      context.setReadout(
        'massBalance',
        (sourceTotal - allocatedTotal - unallocatedTotal).toFixed(4)
      );
      context.setReadout('validSourceCount', validSourceCount);
      const populationWeightedTransfer =
        population.reduce(
          (sum, _value, index) =>
            sum + (aligned[index * 2] + aligned[index * 2 + 1]) * transferred[index],
          0
        ) / sourceTotal;
      context.setReadout(
        'supportLedger',
        `${eligible.filter(value => value > 0).length}/${SOURCE_COUNT} sources have eligible support · ${Math.round(populationWeightedTransfer * 1000) / 10}% of source residents transferred · zero-support mass retained`
      );
      context.setReadout(
        'alignmentStatus',
        `complete · ${alignmentMissing} missing · ${alignmentDuplicates} duplicate · ${alignmentExcluded} excluded · Cook County namespace 17031`
      );
      context.setReadout(
        'alignedCounts',
        tractGeoids
          .map(
            (geoid, index) =>
              `${geoid}  ${formatCount(aligned[index * 2])} / ${formatCount(aligned[index * 2 + 1])}`
          )
          .join('\n')
      );
      context.setReadout('coverageStatus', `${count}/${required} · overflow ${coverageOverflow}`);
      context.setReadout('pairStatus', `${pairRequired} required · overflow ${pairOverflow}`);
      const quadrantNames = ['Northwest', 'Northeast', 'Southwest', 'Southeast'];
      const treeExposed = exposureValues.slice(8, 12);
      const treeObserved = exposureValues.slice(4, 8);
      const treeTotal = exposureValues.slice(0, 4);
      const treeCoverage = exposureValues.slice(12, 16);
      latestChartData = {
        generation: ticketGeneration,
        targetTotals: targetTotals.slice(),
        treeExposed: treeExposed.slice()
      };
      context.setReadout(
        'lowTreePopulation',
        `${quadrantNames.map((name, index) => `${name}: ${formatCount(treeExposed[index])}`).join(' · ')} residents at or below ${selectedThreshold}% mapped tree cover`
      );
      context.setReadout(
        'treeCoverage',
        `${quadrantNames.map((name, index) => `${name} ${Number.isFinite(treeCoverage[index]) ? `${Math.round(treeCoverage[index] * 100)}%` : 'n/a · zero modeled denominator'}`).join(' · ')} · observed ${formatCount(treeObserved.reduce((sum, value) => sum + value, 0))} / ${formatCount(treeTotal.reduce((sum, value) => sum + value, 0))}`
      );
      context.setReadout(
        'recipeStatus',
        'complete · alignment and allocation finite · no overflow'
      );
      context.setStatus('Population support calculations are complete.');
      context.setLegendExtent('population', [0, 3500]);
      publishTargetChart();
      context.requestLayers();
    });
  let reader = createReader(generation);
  context.setReadout('tractCount', SOURCE_COUNT);
  context.setReadout(
    'sourcePopulation',
    formatCount(population.reduce((total, value) => total + value, 0))
  );
  context.setReadout('recipeStatus', 'waiting for GPU status');
  publishSupportChoice();

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, _frame) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      if (!ready) return [];
      const ink: [number, number, number, number] =
        context.ground() === 'dark' ? [255, 255, 255, 230] : [20, 31, 45, 230];
      if (selectedDisplay === 'source-counts') {
        return [
          new SpatialAnalysisPolygonLayer({
            id: 'observed-source-population',
            analysisFrame: ANALYSIS_FRAME,
            triangles: polygonBuffers.triangles,
            features: polygonBuffers.triangleFeatures,
            vertexCount: polygonBuffers.vertexCount,
            values: sourceValues,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [0, 4000],
            color: [255, 255, 255, 220]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'source-tract-outline',
            analysisFrame: ANALYSIS_FRAME,
            segments,
            instanceCount: polygonFields.outlines.length / 4,
            widthPixels: 1.2,
            color: ink
          })
        ];
      }
      const displayLayers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          id:
            selectedDisplay === 'tree-cover'
              ? 'mapped-tree-cover-fraction'
              : selectedDisplay === 'built-up-fraction'
                ? 'input-built-up-fraction'
                : 'population-target-cells',
          analysisFrame: ANALYSIS_FRAME,
          gridSize:
            selectedDisplay === 'built-up-fraction'
              ? [GRID_SIZE, GRID_SIZE]
              : [TARGET_SIZE, TARGET_SIZE],
          bounds: [-2000, -2000, 2000, 2000],
          rowOrigin: 'north',
          values:
            selectedDisplay === 'tree-cover'
              ? treeCoverBuffer
              : selectedDisplay === 'built-up-fraction'
                ? builtUpFractionBuffer
                : targetTotalsBuffer,
          valueFormat: 'float32',
          colormap: 'cividis',
          valueRange:
            selectedDisplay === 'tree-cover' || selectedDisplay === 'built-up-fraction'
              ? [0, 1]
              : [0, 3500],
          noDataColor: [0, 0, 0, 0],
          color: [255, 255, 255, 210]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'source-tract-outline',
          analysisFrame: ANALYSIS_FRAME,
          segments,
          instanceCount: polygonFields.outlines.length / 4,
          widthPixels: 1.2,
          color: ink
        })
      ];
      if (selectedDisplay === 'tree-cover') {
        displayLayers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'tree-quadrant-boundaries',
            analysisFrame: ANALYSIS_FRAME,
            segments: quadrantBoundaryBuffer,
            instanceCount: 2,
            widthPixels: 1.5,
            color: ink
          })
        );
      }
      return displayLayers;
    },
    setOption(optionId, value) {
      if (
        optionId === 'display' &&
        (value === 'source-counts' ||
          value === 'allocation' ||
          value === 'built-up-fraction' ||
          value === 'tree-cover')
      ) {
        selectedDisplay = value;
        publishTargetChart();
        context.requestLayers();
      }
      if (optionId === 'support' && (value === 'uniform-area' || value === 'built-up')) {
        selectedSupport = value;
        ancillaryBuffer.write(value === 'built-up' ? builtUp : uniform);
        generation++;
        reader.release();
        reader = createReader(generation);
        clearPublishedResults();
        publishSupportChoice();
        context.setReadout('recipeStatus', 'recomputing');
        context.setStatus('Recomputing with the selected allocation support.');
        dirty = true;
        context.requestLayers();
      }
      if (optionId === 'treeThreshold' && (value === '10' || value === '25' || value === '50')) {
        selectedThreshold = value;
        exposureThresholdBuffer.write(new Float32Array([1 - Number(value) / 100]));
        generation++;
        reader.release();
        reader = createReader(generation);
        clearPublishedResults();
        context.setReadout('recipeStatus', 'recomputing tree-cover summary');
        context.setStatus('Recomputing the low mapped tree-cover threshold summary.');
        dirty = true;
        context.requestLayers();
      }
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
