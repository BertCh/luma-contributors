// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUAZPRegions,
  GPUMaxPRegions,
  GPUWardRegions,
  getGPUAZPRegionsParameterValues,
  getGPUMaxPRegionsParameterValues,
  getGPUWardRegionsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

export type CountyRegionalizationOptions = {
  algorithm: 'azp' | 'ward-sequential' | 'ward-mutual-nearest' | 'max-p';
  minimumPopulation: number;
  seed: number;
};

const ROW_COUNT = 21;
const FEATURE_COUNT = 3;
const REGION_COLORS = [
  [0, 114, 178, 230],
  [213, 94, 0, 230],
  [0, 158, 115, 230],
  [204, 121, 167, 230],
  [230, 159, 0, 230],
  [86, 180, 233, 230],
  [51, 51, 51, 230],
  [0, 127, 127, 230],
  [231, 111, 81, 230],
  [38, 70, 83, 230],
  [138, 154, 60, 230],
  [106, 76, 147, 230],
  [196, 69, 105, 230],
  [140, 86, 75, 230],
  [49, 163, 84, 230]
] as const;
const STATUS_LENGTH = {azp: 4, ward: 5, maxP: 6} as const;
type AlgorithmId = CountyRegionalizationOptions['algorithm'];
type AlgorithmBuffers = {labels: Buffer; objective: Buffer; status: Buffer};

/** Contiguous clustering of measured unemployment, poverty and density with a 1990 population floor. */
export async function createCountyRegionalization(
  ctx: SceneContext<CountyRegionalizationOptions>
): Promise<SceneInstance<CountyRegionalizationOptions>> {
  const dataset = ctx.datasets.get('ncovr-new-jersey');
  const frame = {kind: 'azimuthal-equidistant' as const, origin: [-74.55, 40.05] as const};
  const projection = createAnalysisFrameProjection(frame);
  const resources = new SpatialAnalysisResources(ctx.device, 'county-regionalization');
  const geojson = dataset.geojson;
  if (!geojson || geojson.features.length !== ROW_COUNT)
    throw new Error('Expected the complete NCOVR 21-unit geometry');
  const mesh = buildPolygonMesh(geojson, projection.project);
  const polygonBuffers = createPolygonMeshBuffers(resources, mesh, 'counties');
  const sourceColumns = [
    dataset.column<Float32Array>('unemployment1990'),
    dataset.column<Float32Array>('familyPoverty1989'),
    dataset.column<Float32Array>('logDensity1990')
  ];
  const values = new Float32Array(ROW_COUNT * FEATURE_COUNT);
  for (let row = 0; row < ROW_COUNT; row++)
    for (let column = 0; column < FEATURE_COUNT; column++)
      values[row * FEATURE_COUNT + column] = sourceColumns[column][row];
  const population = dataset.column<Float32Array>('population1990');
  const initialLabels = makeConnectedSeedLabels(
    dataset.column<Uint32Array>('queenOffsets'),
    dataset.column<Uint32Array>('queenNeighbors'),
    dataset.column<Uint32Array>('fips')
  );
  const queenOffsets = dataset.column<Uint32Array>('queenOffsets');
  const queenNeighbors = dataset.column<Uint32Array>('queenNeighbors');
  const binaryWeights = new Float32Array(queenNeighbors.length).fill(1);
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'ncovr-county-regionalization'});
  const importedViews = new Map<Buffer, GraphDataView>();
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    const existing = importedViews.get(buffer);
    if (existing) return existing as GraphDataView<Format>;
    const imported = importGraphBuffer(graph, name, buffer, format, length);
    importedViews.set(buffer, imported);
    return imported;
  };
  const weights: GPUSpatialWeightsForScene = {
    spatialContext: projection.spatialContext,
    information: {
      rowCount: ROW_COUNT,
      transform: 'binary',
      symmetric: true,
      provenance: {source: 'NCOVR common-support polygon boundaries', criterion: 'queen'}
    },
    offsets: view(
      'queen-offsets',
      resources.createBuffer('queen-offsets', queenOffsets),
      'uint32',
      ROW_COUNT + 1
    ),
    neighbors: view(
      'queen-neighbors',
      resources.createBuffer('queen-neighbors', queenNeighbors),
      'uint32',
      queenNeighbors.length
    ),
    weights: view(
      'queen-weights',
      resources.createBuffer('queen-weights', binaryWeights),
      'float32',
      binaryWeights.length
    )
  };
  const valuesView = view(
    'regionalization-attributes',
    resources.createBuffer('regionalization-attributes', values),
    'float32',
    values.length
  );
  const populationView = view(
    'population-1990',
    resources.createBuffer('population-1990', population),
    'float32',
    ROW_COUNT
  );
  const thresholdBuffer = resources.createBuffer(
    'minimum-population',
    new Float32Array([ctx.options.minimumPopulation])
  );
  const thresholdView = view('minimum-population', thresholdBuffer, 'float32', 1);
  const initialView = view(
    'azp-initial-labels',
    resources.createBuffer('azp-initial-labels', initialLabels),
    'uint32',
    ROW_COUNT
  );
  const seedBuffers = {
    azp: resources.createBuffer(
      'azp-seed',
      getGPUAZPRegionsParameterValues({seed: ctx.options.seed})
    ),
    ward: resources.createBuffer(
      'ward-seed',
      getGPUWardRegionsParameterValues({seed: ctx.options.seed})
    ),
    maxP: resources.createBuffer(
      'max-p-seed',
      getGPUMaxPRegionsParameterValues({seed: ctx.options.seed})
    )
  };
  const seedViews = {
    azp: view('azp-seed', seedBuffers.azp, 'uint32', 1),
    ward: view('ward-seed', seedBuffers.ward, 'uint32', 1),
    maxP: view('max-p-seed', seedBuffers.maxP, 'uint32', 1)
  };
  const makeOutput = (name: string, statusWords: number): AlgorithmBuffers => ({
    labels: resources.createBuffer(`${name}-labels`, ROW_COUNT * 4),
    objective: resources.createBuffer(`${name}-objective`, 4),
    status: resources.createBuffer(`${name}-status`, statusWords * 4)
  });
  const azp = makeOutput('azp', STATUS_LENGTH.azp);
  const wardSequential = makeOutput('ward-sequential', STATUS_LENGTH.ward);
  const wardMutual = makeOutput('ward-mutual-nearest', STATUS_LENGTH.ward);
  const maxP = makeOutput('max-p', STATUS_LENGTH.maxP);
  const addWard = (
    id: string,
    schedule: 'sequential' | 'mutual-nearest',
    output: AlgorithmBuffers
  ) =>
    graph.add(
      new GPUWardRegions({
        id,
        weights,
        values: valuesView,
        columnCount: FEATURE_COUNT,
        standardize: true,
        targetRegionCount: 6,
        maximumIterations: ROW_COUNT - 6,
        mergeSchedule: schedule,
        parameters: seedViews.ward,
        output: {
          labels: view(`${id}-labels`, output.labels, 'uint32', ROW_COUNT),
          objective: view(`${id}-objective`, output.objective, 'float32', 1),
          status: view(`${id}-status`, output.status, 'uint32', STATUS_LENGTH.ward)
        }
      })
    );
  graph.add(
    new GPUAZPRegions({
      id: 'azp-six-contiguous-seeds',
      weights,
      values: valuesView,
      columnCount: FEATURE_COUNT,
      standardize: true,
      initialLabels: initialView,
      regionCapacity: 6,
      minimumRegionSize: 2,
      maximumIterations: 48,
      parameters: seedViews.azp,
      output: {
        labels: view('azp-labels', azp.labels, 'uint32', ROW_COUNT),
        objective: view('azp-objective', azp.objective, 'float32', 1),
        status: view('azp-status', azp.status, 'uint32', STATUS_LENGTH.azp)
      }
    })
  );
  addWard('ward-sequential', 'sequential', wardSequential);
  addWard('ward-mutual-nearest', 'mutual-nearest', wardMutual);
  graph.add(
    new GPUMaxPRegions({
      id: 'max-p-population-constrained',
      weights,
      values: valuesView,
      columnCount: FEATURE_COUNT,
      standardize: true,
      extensiveValues: populationView,
      threshold: thresholdView,
      startCount: 8,
      maximumGrowthSteps: 128,
      maximumIterations: 24,
      parameters: seedViews.maxP,
      output: {
        labels: view('max-p-labels', maxP.labels, 'uint32', ROW_COUNT),
        objective: view('max-p-objective', maxP.objective, 'float32', 1),
        status: view('max-p-status', maxP.status, 'uint32', STATUS_LENGTH.maxP)
      }
    })
  );

  const outputByAlgorithm: Record<AlgorithmId, AlgorithmBuffers> = {
    azp,
    'ward-sequential': wardSequential,
    'ward-mutual-nearest': wardMutual,
    'max-p': maxP
  };
  const displayLabels = new Map<AlgorithmId, Buffer>();
  for (const algorithm of Object.keys(outputByAlgorithm) as AlgorithmId[]) {
    displayLabels.set(
      algorithm,
      resources.createBuffer(
        `display-${algorithm}-labels`,
        new Uint32Array(ROW_COUNT).fill(0xffffffff)
      )
    );
  }
  const compiled = resources.track(await graph.compileAsync());
  const latestLabels = new Map<AlgorithmId, Uint32Array>();
  const latestDisplayLabels = new Map<AlgorithmId, Uint32Array>();
  const latestStatuses = new Map<AlgorithmId, Uint32Array>();
  const latestObjectives = new Map<AlgorithmId, number>();
  let generation = 0;
  let readGeneration = 0;
  let outputReady = false;
  const selectedBuffers = Object.entries(outputByAlgorithm).flatMap(([key, output]) => [
    {key, kind: 'labels' as const, buffer: output.labels, size: ROW_COUNT * 4},
    {key, kind: 'objective' as const, buffer: output.objective, size: 4},
    {
      key,
      kind: 'status' as const,
      buffer: output.status,
      size: STATUS_LENGTH[key === 'max-p' ? 'maxP' : key.startsWith('ward') ? 'ward' : 'azp'] * 4
    }
  ]);
  let summaryOffsets = new Map<string, number>();
  let totalSummaryBytes = 0;
  for (const item of selectedBuffers) {
    summaryOffsets.set(`${item.key}:${item.kind}`, totalSummaryBytes);
    totalSummaryBytes += item.size;
  }
  const reader = new SummaryReader(
    resources,
    'regionalization-results',
    selectedBuffers.map(item => ({buffer: item.buffer, size: item.size})),
    bytes => {
      if (readGeneration !== generation) return;
      const wordView = new Uint32Array(bytes);
      for (const algorithm of Object.keys(outputByAlgorithm) as AlgorithmId[]) {
        const labelOffset = summaryOffsets.get(`${algorithm}:labels`)! / 4;
        const statusOffset = summaryOffsets.get(`${algorithm}:status`)! / 4;
        latestLabels.set(algorithm, wordView.slice(labelOffset, labelOffset + ROW_COUNT));
        const objectiveOffset = summaryOffsets.get(`${algorithm}:objective`)!;
        latestObjectives.set(algorithm, new Float32Array(bytes, objectiveOffset, 1)[0]);
        const statusLength =
          selectedBuffers.find(item => item.key === algorithm && item.kind === 'status')!.size / 4;
        const status = wordView.slice(statusOffset, statusOffset + statusLength);
        latestStatuses.set(algorithm, status);
        const display = latestLabels.get(algorithm)!.slice();
        if (algorithm === 'max-p') {
          const populationByLabel = new Map<number, number>();
          for (let row = 0; row < ROW_COUNT; row++) {
            const label = display[row];
            if (label !== 0xffffffff)
              populationByLabel.set(label, (populationByLabel.get(label) ?? 0) + population[row]);
          }
          for (let row = 0; row < ROW_COUNT; row++) {
            const label = display[row];
            if (
              label !== 0xffffffff &&
              (status[1] !== 1 ||
                (populationByLabel.get(label) ?? 0) < ctx.options.minimumPopulation)
            )
              display[row] = 0xffffffff;
          }
        }
        displayLabels.get(algorithm)!.write(display);
        latestDisplayLabels.set(algorithm, display);
      }
      outputReady = true;
      updateReadouts();
      ctx.requestLayers();
    }
  );
  const names = geojson.features.map(feature =>
    String(feature.properties?.NAME ?? feature.properties?.name ?? 'County unit')
  );
  const means = [
    dataset.column<Float32Array>('unemployment1990'),
    dataset.column<Float32Array>('familyPoverty1989'),
    dataset.column<Float32Array>('logDensity1990')
  ];
  let dirty = true;
  const updateReadouts = () => {
    if (!outputReady) {
      ctx.setLegendData('regionLabels', []);
      ctx.setReadout('regionCount', 'pending');
      ctx.setReadout('partitionQuality', 'pending');
      ctx.setReadout('partitionCertificate', 'pending GPU label readback');
      ctx.setReadout('minimumStatus', 'pending GPU result');
      return;
    }
    const status = latestStatuses.get(ctx.options.algorithm);
    if (!status) return;
    const labelValues = latestLabels.get(ctx.options.algorithm);
    const visibleLabels = latestDisplayLabels.get(ctx.options.algorithm);
    ctx.setLegendData(
      'regionLabels',
      visibleLabels
        ? [...new Set(Array.from(visibleLabels).filter(label => label !== 0xffffffff))].sort(
            (first, second) => first - second
          )
        : []
    );
    const assignedLabels = labelValues
      ? Array.from(labelValues).filter(label => label !== 0xffffffff)
      : [];
    const regions = new Set(assignedLabels).size;
    const regionSizes = new Map<number, number>();
    const regionPopulation = new Map<number, number>();
    for (let row = 0; row < ROW_COUNT; row++) {
      const label = labelValues?.[row];
      if (label === undefined || label === 0xffffffff) continue;
      regionSizes.set(label, (regionSizes.get(label) ?? 0) + 1);
      regionPopulation.set(label, (regionPopulation.get(label) ?? 0) + population[row]);
    }
    const populations = [...regionPopulation.values()];
    const components = countLabelComponents(
      labelValues ?? new Uint32Array(ROW_COUNT).fill(0xffffffff),
      queenOffsets,
      queenNeighbors
    );
    const objective = latestObjectives.get(ctx.options.algorithm);
    const invalidCount = status[3];
    const complete = assignedLabels.length === ROW_COUNT && invalidCount === 0;
    const targetReached =
      ctx.options.algorithm === 'max-p' ? regions > 0 && regions === status[0] : regions === 6;
    const feasible =
      complete &&
      targetReached &&
      components === regions &&
      (ctx.options.algorithm === 'max-p'
        ? status[1] === 1 && populations.every(value => value >= ctx.options.minimumPopulation)
        : ctx.options.algorithm.startsWith('ward')
          ? status[1] === 1
          : status[1] === 1 && [...regionSizes.values()].every(size => size >= 2));
    ctx.setReadout('regionCount', regions);
    ctx.setReadout('partitionQuality', Number.isFinite(objective) ? objective! : 'not reported');
    ctx.setReadout(
      'partitionCertificate',
      `${feasible ? 'feasible partition' : 'partition constraints not all met'}; ${assignedLabels.length}/${ROW_COUNT} assigned; ${regions} region(s)${ctx.options.algorithm === 'max-p' ? ' reported' : ', target 6'}; ${components} connected component(s); population ${populations.length ? `${Math.min(...populations).toLocaleString()}–${Math.max(...populations).toLocaleString()} people` : 'none'}`
    );
    if (ctx.options.algorithm === 'max-p') {
      ctx.setReadout(
        'minimumStatus',
        `regions ${status[0]}; ${status[1] === 1 ? 'feasible' : 'infeasible'}; infeasible regions ${status[2]}; invalid ${status[3]}; selected start ${status[4]}; improvement moves ${status[5]}; ${populations.every(value => value >= ctx.options.minimumPopulation) ? 'population check passes' : 'population check fails'}`
      );
    } else if (ctx.options.algorithm.startsWith('ward')) {
      ctx.setReadout(
        'minimumStatus',
        `regions ${regions}; ${status[1] === 1 ? 'converged (heuristic stop)' : 'not converged'}; iteration limit ${status[2]}; invalid ${status[3]}; merge rounds ${status[4]}; objective ${objective?.toFixed(4) ?? 'pending'}; ${components} component(s)`
      );
    } else {
      ctx.setReadout(
        'minimumStatus',
        `regions ${regions}; ${status[1] === 1 ? 'local optimum (not global proof)' : 'not converged'}; iteration limit ${status[2]}; invalid ${status[3]}; iterations ${status[0]}; objective ${objective?.toFixed(4) ?? 'pending'}; ${components} component(s)`
      );
    }
  };
  const getLayers = (): Layer[] => {
    return [
      new SpatialAnalysisPolygonLayer({
        id: `ncovr-${ctx.options.algorithm}-regions`,
        analysisFrame: frame,
        triangles: polygonBuffers.triangles,
        features: polygonBuffers.triangleFeatures,
        vertexCount: polygonBuffers.vertexCount,
        values: displayLabels.get(ctx.options.algorithm)!,
        valueFormat: 'uint32',
        noDataValue: 0xffffffff,
        noDataColor: [220, 220, 220, 210],
        colormap: 'category',
        palette: REGION_COLORS,
        opacity: 0.88
      }),
      new SpatialAnalysisSegmentLayer({
        id: `ncovr-${ctx.options.algorithm}-county-lines`,
        analysisFrame: frame,
        segments: polygonBuffers.outline,
        instanceCount: polygonBuffers.outlineCount,
        color: [40, 46, 57, 145],
        widthPixels: 0.65
      })
    ];
  };
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder: CommandEncoder) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        if (!reader.isPending) readGeneration = generation;
        reader.request(commandEncoder);
        dirty = false;
      } else {
        if (!reader.isPending) readGeneration = generation;
        reader.flush(commandEncoder);
      }
    },
    getLayers,
    setOption(id) {
      if (id === 'minimumPopulation')
        thresholdBuffer.write(new Float32Array([ctx.options.minimumPopulation]));
      if (id === 'seed') {
        seedBuffers.azp.write(getGPUAZPRegionsParameterValues({seed: ctx.options.seed}));
        seedBuffers.ward.write(getGPUWardRegionsParameterValues({seed: ctx.options.seed}));
        seedBuffers.maxP.write(getGPUMaxPRegionsParameterValues({seed: ctx.options.seed}));
      }
      if (id === 'algorithm' || id === 'minimumPopulation' || id === 'seed') {
        generation++;
        outputReady = false;
        latestLabels.clear();
        latestDisplayLabels.clear();
        latestStatuses.clear();
        latestObjectives.clear();
        for (const labels of displayLabels.values())
          labels.write(new Uint32Array(ROW_COUNT).fill(0xffffffff));
        dirty = true;
        ctx.requestLayers();
        updateReadouts();
      }
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      let row = -1;
      let nearestDistance = Infinity;
      for (let index = 0; index < mesh.labelPoints.length; index++) {
        const [longitude, latitude] = mesh.labelPoints[index];
        const [candidateX, candidateY] = projection.project(longitude, latitude);
        const distance = (candidateX - x) ** 2 + (candidateY - y) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          row = index;
        }
      }
      if (row < 0 || nearestDistance > 30_000 ** 2) return null;
      const labels = latestLabels.get(ctx.options.algorithm);
      const label = labels?.[row];
      const displayedLabel = latestDisplayLabels.get(ctx.options.algorithm)?.[row];
      const labelDescription =
        label === undefined
          ? 'partition pending'
          : label === 0xffffffff
            ? 'unassigned / no data'
            : displayedLabel === 0xffffffff
              ? `region ${label + 1}; masked by Max-P feasibility status`
              : `region ${label + 1}`;
      return `Nearest county-label match: ${names[row]} · ${ctx.options.algorithm}; ${labelDescription}; 1990 population ${Math.round(population[row]).toLocaleString()}; unemployment ${means[0][row].toFixed(1)}%; family poverty ${means[1][row].toFixed(1)}%; source log density ${means[2][row].toFixed(2)}.`;
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}

function makeConnectedSeedLabels(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  sourceFips: Uint32Array
): Uint32Array {
  // Reproducible source-FIPS groups, certified below against the exact archived queen graph.
  if (sourceFips.length !== ROW_COUNT)
    throw new Error('AZP initialization requires all 21 source FIPS codes');
  const fipsGroups = [
    [34001, 34009, 34011, 34015, 34033],
    [34007, 34005, 34025, 34029],
    [34003, 34017, 34013],
    [34031, 34037, 34041],
    [34019, 34027, 34039, 34035],
    [34021, 34023]
  ];
  const labels = new Uint32Array(ROW_COUNT).fill(0xffffffff);
  for (let region = 0; region < fipsGroups.length; region++) {
    for (const fips of fipsGroups[region]) {
      const row = sourceFips.indexOf(fips);
      if (row < 0 || labels[row] !== 0xffffffff)
        throw new Error('Seed groups must assign each source FIPS exactly once');
      labels[row] = region;
    }
  }
  if (labels.some(label => label === 0xffffffff))
    throw new Error('Seed groups omit a source county unit');
  for (let region = 0; region < fipsGroups.length; region++) {
    const members = new Set(
      Array.from(labels, (label, row) => (label === region ? row : -1)).filter(row => row >= 0)
    );
    if (members.size < 2)
      throw new Error('AZP initialization requires at least two units per region');
    const start = members.values().next().value as number;
    const reached = new Set<number>([start]);
    const queue = [start];
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const row = queue[cursor];
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        const neighbor = neighbors[slot];
        if (members.has(neighbor) && !reached.has(neighbor)) {
          reached.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
    if (reached.size !== members.size)
      throw new Error(`Seed region ${region} is disconnected in queen adjacency`);
  }
  return labels;
}

function countLabelComponents(
  labels: Uint32Array,
  offsets: Uint32Array,
  neighbors: Uint32Array
): number {
  const visited = new Uint8Array(ROW_COUNT);
  let componentCount = 0;
  for (let row = 0; row < ROW_COUNT; row++) {
    if (labels[row] === 0xffffffff || visited[row]) continue;
    componentCount++;
    visited[row] = 1;
    const queue = [row];
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const current = queue[cursor];
      for (let slot = offsets[current]; slot < offsets[current + 1]; slot++) {
        const neighbor = neighbors[slot];
        if (!visited[neighbor] && labels[neighbor] === labels[row]) {
          visited[neighbor] = 1;
          queue.push(neighbor);
        }
      }
    }
  }
  return componentCount;
}

type GPUSpatialWeightsForScene =
  import('@luma-contributors/analysis/gpu-spatial-analysis').GPUSpatialWeights;
