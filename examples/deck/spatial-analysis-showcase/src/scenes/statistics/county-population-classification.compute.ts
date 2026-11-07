// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUClassBreaks,
  GPUColorScale,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues
} from '@luma-contributors/graph/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUClassAssignment} from '@luma-contributors/graph/gpu-spatial-analysis';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {getSchemeColors, packColors} from './choropleth-classes.style';
import type {SceneContext, SceneInstance} from '../scene';

const COUNTY_COUNT = 21;
const CLASS_COUNT = 5;
const COLOR_COUNT = 5;
const FRAME = {kind: 'azimuthal-equidistant' as const, origin: [-74.55, 40.05] as const};
const POPULATION_FORMATTER = new Intl.NumberFormat('en-US', {maximumFractionDigits: 0});

export type CountyPopulationClassificationOptions = {
  representation: 'population-counts' | 'jenks-classes';
  naturalBreaksMode: 'exact' | 'approximate';
};

type ClassificationVariant = {
  mode: CountyPopulationClassificationOptions['naturalBreaksMode'];
  compiled: CompiledGPUCommandGraph<void>;
};

/** Exact and approximate Jenks variants over the 21 measured New Jersey population counts. */
export async function createCountyPopulationClassification(
  ctx: SceneContext<CountyPopulationClassificationOptions>
): Promise<SceneInstance<CountyPopulationClassificationOptions>> {
  const dataset = ctx.datasets.get('ncovr-new-jersey');
  const geojson = dataset.geojson;
  if (!geojson || geojson.features.length !== COUNTY_COUNT) {
    throw new Error('Expected all 21 NCOVR New Jersey county units');
  }

  const sourcePopulation = dataset.column<Float32Array>('population1990');
  const sourceFips = dataset.column<Uint32Array>('fips');
  const populationCounts = Uint32Array.from(sourcePopulation, value => Math.round(value));
  const populationValues = Float32Array.from(populationCounts);
  const resources = new SpatialAnalysisResources(ctx.device, 'county-population-classification');
  const projection = createAnalysisFrameProjection(FRAME);
  const polygonMesh = buildPolygonMesh(geojson, projection.project);
  const polygons = createPolygonMeshBuffers(resources, polygonMesh, 'new-jersey-units');
  const labels = geojson.features.map(feature =>
    String(feature.properties?.NAME ?? feature.properties?.name ?? 'County unit')
  );
  const projectedAnchors = new Float32Array(COUNTY_COUNT * 2);
  for (let row = 0; row < COUNTY_COUNT; row++) {
    const labelPoint = polygonMesh.labelPoints[row]!;
    projectedAnchors.set(projection.project(labelPoint[0], labelPoint[1]), row * 2);
  }
  const populationValuesBuffer = resources.createBuffer(
    'population-1990-float-values',
    populationValues
  );
  const populationCountsBuffer = resources.createBuffer(
    'population-1990-uint-counts',
    populationCounts
  );
  const minimumPopulation = Math.min(...populationCounts);
  const maximumPopulation = Math.max(...populationCounts);
  const numericDomainBuffer = resources.createBuffer(
    'population-numeric-domain',
    new Float32Array([minimumPopulation, maximumPopulation])
  );
  const jenksParameters = resources.createParameterBuffer(
    'jenks-parameters',
    'float32',
    getGPUClassBreaksParameterLength(CLASS_COUNT),
    getGPUClassBreaksParameterValues(
      {method: 'natural-breaks', classCount: CLASS_COUNT},
      CLASS_COUNT
    )
  );
  const numericScaleParameters = resources.createParameterBuffer(
    'numeric-scale-parameters',
    'float32',
    GPU_COLOR_SCALE_PARAMETER_LENGTH,
    getGPUColorScaleParameterValues({
      scale: 'quantize',
      domainCount: 2,
      paletteCount: COLOR_COUNT,
      interpolation: 'step',
      clamp: true,
      noDataColor: 0xffd5d5d5
    })
  );
  const ordinalScaleParameters = resources.createParameterBuffer(
    'ordinal-scale-parameters',
    'float32',
    GPU_COLOR_SCALE_PARAMETER_LENGTH,
    getGPUColorScaleParameterValues({
      scale: 'ordinal',
      domainCount: CLASS_COUNT + 1,
      paletteCount: COLOR_COUNT,
      interpolation: 'step',
      clamp: true,
      noDataColor: 0xffd5d5d5
    })
  );
  const palette = packColors(getSchemeColors('YlOrBr', COLOR_COUNT, 'light'));
  const paletteBuffer = resources.createBuffer('population-palette', palette);
  const breaksBuffer = resources.createBuffer('jenks-breaks', (CLASS_COUNT + 1) * 4);
  const classCountBuffer = resources.createBuffer('jenks-class-count', 4);
  const classIdsBuffer = resources.createBuffer('jenks-class-ids', COUNTY_COUNT * 4);
  const countColorsBuffer = resources.createBuffer('population-count-colors', COUNTY_COUNT * 4);
  const classColorsBuffer = resources.createBuffer('jenks-class-colors', COUNTY_COUNT * 4);
  const generationInputBuffer = resources.createBuffer('request-generation', new Uint32Array([0]));
  const generationOutputBuffer = resources.createBuffer('result-generation', 4);
  ctx.setLegendData('population-counts', {minimumPopulation, maximumPopulation});
  const results: ClassificationVariant[] = [];

  const buildVariant = async (
    mode: CountyPopulationClassificationOptions['naturalBreaksMode']
  ): Promise<ClassificationVariant> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `new-jersey-jenks-${mode}`
    });
    const importView = createGraphBufferImporter(graph);
    const view = <Format extends 'float32' | 'uint32'>(
      _name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importView(buffer, format, length);
    const populationValueView = view(
      'population-values',
      populationValuesBuffer,
      'float32',
      COUNTY_COUNT
    );
    const populationCountView = view(
      'population-counts',
      populationCountsBuffer,
      'uint32',
      COUNTY_COUNT
    );
    const breaksView = view('jenks-breaks', breaksBuffer, 'float32', CLASS_COUNT + 1);
    const classCountView = view('jenks-class-count', classCountBuffer, 'uint32', 1);
    const classIdsView = view('jenks-class-ids', classIdsBuffer, 'uint32', COUNTY_COUNT);
    const paletteView = view('palette', paletteBuffer, 'uint32', COLOR_COUNT);
    graph.add(
      new GPUClassBreaks({
        id: `jenks-${mode}`,
        values: populationValueView,
        parameters: jenksParameters.importToGraph(graph),
        maximumClassCount: CLASS_COUNT,
        methods: ['natural-breaks'],
        naturalBreaksBinCount: 128,
        naturalBreaksMode: mode,
        output: {breaks: breaksView, classCount: classCountView}
      })
    );
    graph.add(
      new GPUClassAssignment({
        id: `assign-jenks-${mode}`,
        values: populationValueView,
        breaks: breaksView,
        classCount: classCountView,
        output: classIdsView
      })
    );
    graph.add(
      new GPUColorScale({
        id: `color-population-${mode}`,
        values: populationCountView,
        integerValues: 'numeric',
        domain: view('population-domain', numericDomainBuffer, 'float32', 2),
        palette: paletteView,
        parameters: numericScaleParameters.importToGraph(graph),
        maximumDomainCount: 2,
        maximumPaletteCount: COLOR_COUNT,
        output: {colors: view('population-colors', countColorsBuffer, 'uint32', COUNTY_COUNT)}
      })
    );
    graph.add(
      new GPUColorScale({
        id: `color-jenks-classes-${mode}`,
        values: classIdsView,
        integerValues: 'ordinal',
        domain: breaksView,
        domainCount: classCountView,
        palette: paletteView,
        parameters: ordinalScaleParameters.importToGraph(graph),
        maximumDomainCount: CLASS_COUNT + 1,
        maximumPaletteCount: COLOR_COUNT,
        output: {colors: view('jenks-colors', classColorsBuffer, 'uint32', COUNTY_COUNT)}
      })
    );
    addKernelPass(graph, {
      id: `snapshot-new-jersey-jenks-generation-${mode}`,
      invocationCount: 1,
      bindings: [
        {
          name: 'generation',
          view: view('generation-input', generationInputBuffer, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'snapshot',
          view: view('generation-output', generationOutputBuffer, 'uint32', 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: 'snapshot[snapshotOffset] = generation[generationOffset];'
    });
    return {mode, compiled: resources.track(await graph.compileAsync())};
  };

  results.push(await buildVariant('exact'), await buildVariant('approximate'));
  const reader = new SummaryReader(
    resources,
    'new-jersey-jenks-summary',
    [
      {buffer: breaksBuffer, size: (CLASS_COUNT + 1) * 4},
      {buffer: classCountBuffer, size: 4},
      {buffer: classIdsBuffer, size: COUNTY_COUNT * 4},
      {buffer: countColorsBuffer, size: COUNTY_COUNT * 4},
      {buffer: classColorsBuffer, size: COUNTY_COUNT * 4},
      {buffer: generationOutputBuffer, size: 4}
    ],
    bytes => {
      const returnedGeneration = new Uint32Array(
        bytes,
        (CLASS_COUNT + 1) * 4 + 4 + COUNTY_COUNT * 12,
        1
      )[0]!;
      if (returnedGeneration !== generation) return;
      const breaks = new Float32Array(bytes, 0, CLASS_COUNT + 1);
      const classCount = new Uint32Array(bytes, (CLASS_COUNT + 1) * 4, 1)[0]!;
      const classIds = new Uint32Array(bytes, (CLASS_COUNT + 1) * 4 + 4, COUNTY_COUNT);
      const countColors = new Uint32Array(
        bytes,
        (CLASS_COUNT + 1) * 4 + 4 + COUNTY_COUNT * 4,
        COUNTY_COUNT
      );
      const classColors = new Uint32Array(
        bytes,
        (CLASS_COUNT + 1) * 4 + 4 + COUNTY_COUNT * 8,
        COUNTY_COUNT
      );
      latestClassIds = classIds.slice();
      const frequencies = Array.from({length: CLASS_COUNT}, () => 0);
      for (const classId of classIds) if (classId < classCount) frequencies[classId]++;
      const innerBreaks = Array.from(breaks.slice(1, classCount), value =>
        POPULATION_FORMATTER.format(value)
      );
      ctx.setReadout(
        'jenksSummary',
        `${activeMode} Fisher–Jenks · ${classCount} classes · population edges ${innerBreaks.join(' | ')} people`
      );
      ctx.setReadout(
        'assignmentSummary',
        `${classIds.length} GPU class assignments · counts ${frequencies.join(' / ')}`
      );
      const colorWord = (value: number) => `0x${value.toString(16).padStart(8, '0')}`;
      ctx.setReadout(
        'assignmentEvidence',
        labels
          .map(
            (label, row) =>
              `${sourceFips[row]} ${label}: ${POPULATION_FORMATTER.format(populationCounts[row]!)} people · class ${classIds[row]! + 1} · numeric ${colorWord(countColors[row]!)} · ordinal ${colorWord(classColors[row]!)}`
          )
          .join(' | ')
      );
      ctx.setChart('classCounts', {
        kind: 'bars',
        title: 'New Jersey units assigned to each Jenks class',
        labels: frequencies.map((_frequency, index) => `Class ${index + 1}`),
        values: frequencies,
        yLabel: 'County units',
        description: 'Counts of the GPU-returned class IDs for the active 21-unit Jenks run.'
      });
      ctx.setStatus(`${activeMode} Jenks completed for ${classIds.length} New Jersey units.`);
      ctx.setLegendData('population-classes', {
        classCount,
        breaks: Array.from(breaks.slice(1, classCount)),
        counts: frequencies
      });
      ctx.refreshLegends();
      ctx.requestLayers();
      ctx.refreshTooltip();
    }
  );
  let activeMode = ctx.options.naturalBreaksMode;
  let generation = 0;
  let graphDirty = true;
  let latestClassIds: Uint32Array | null = null;
  const getActiveVariant = () => results.find(result => result.mode === activeMode)!;

  ctx.setReadout(
    'sourceSummary',
    `${COUNTY_COUNT} historical county units · 1990 population ${POPULATION_FORMATTER.format(minimumPopulation)}–${POPULATION_FORMATTER.format(maximumPopulation)} people`
  );
  ctx.setReadout('jenksSummary', 'GPU Jenks edges pending.');
  ctx.setReadout('assignmentSummary', 'GPU class assignments pending.');
  ctx.setReadout(
    'assignmentEvidence',
    'Per-unit GPU class IDs and both packed color outputs pending.'
  );
  ctx.setReadout(
    'colorSummary',
    'Count view uses numeric uint32 values; class view uses ordinal uint32 IDs.'
  );
  ctx.setChart('classCounts', null);

  return {
    getCompiledGraphs: () => [getActiveVariant().compiled],
    encode(commandEncoder: CommandEncoder) {
      if (graphDirty) {
        generation++;
        generationInputBuffer.write(Uint32Array.of(generation));
        getActiveVariant().compiled.encode(commandEncoder, {parameters: undefined});
        reader.request(commandEncoder);
        graphDirty = false;
      } else {
        reader.flush(commandEncoder);
      }
    },
    setOption(id, _value, options) {
      if (id === 'naturalBreaksMode') activeMode = options.naturalBreaksMode;
      generation++;
      generationInputBuffer.write(Uint32Array.of(generation));
      graphDirty = true;
      latestClassIds = null;
      ctx.setReadout('jenksSummary', 'GPU Jenks edges pending.');
      ctx.setReadout('assignmentSummary', 'GPU class assignments pending.');
      ctx.setReadout(
        'assignmentEvidence',
        'Per-unit GPU class IDs and both packed color outputs pending.'
      );
      ctx.setChart('classCounts', null);
      ctx.setLegendData('population-classes', null);
      ctx.setStatus(`${activeMode} Jenks is running on 21 units.`);
      ctx.refreshLegends();
      ctx.requestLayers();
      ctx.refreshTooltip();
    },
    getLayers(): Layer[] {
      if (!latestClassIds) return [];
      const classView = ctx.options.representation === 'jenks-classes';
      return [
        new SpatialAnalysisPolygonLayer({
          id: 'new-jersey-population-classification-fill',
          analysisFrame: FRAME,
          triangles: polygons.triangles,
          features: polygons.triangleFeatures,
          vertexCount: polygons.vertexCount,
          values: classView ? classColorsBuffer : countColorsBuffer,
          valueFormat: 'uint32',
          colormap: 'rgba',
          opacity: 0.95
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'new-jersey-population-classification-boundaries',
          analysisFrame: FRAME,
          segments: polygons.outline,
          instanceCount: polygons.outlineCount,
          color: [57, 63, 69, 170],
          widthPixels: 0.7
        })
      ];
    },
    getTooltip(event) {
      if (!event.coordinate || !latestClassIds) return null;
      const [projectedEasting, projectedNorthing] = projection.project(
        event.coordinate[0],
        event.coordinate[1]
      );
      let nearestRow = -1;
      let nearestDistance = Infinity;
      for (let row = 0; row < COUNTY_COUNT; row++) {
        const distance = Math.hypot(
          projectedAnchors[row * 2]! - projectedEasting,
          projectedAnchors[row * 2 + 1]! - projectedNorthing
        );
        if (distance < nearestDistance) {
          nearestRow = row;
          nearestDistance = distance;
        }
      }
      if (nearestRow < 0 || nearestDistance > 80_000) return null;
      const classId = latestClassIds[nearestRow]!;
      const representation = ctx.options.representation;
      const display =
        representation === 'population-counts'
          ? `${POPULATION_FORMATTER.format(populationCounts[nearestRow]!)} people (numeric uint32)`
          : `Class ${classId + 1} (ordinal uint32)`;
      return `${labels[nearestRow]} · FIPS ${sourceFips[nearestRow]} · ${display} · nearest label point ${Math.round(nearestDistance / 1000)} km`;
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
