// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUMultipleTestingMethod,
  GPUParameterBuffer,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {getGPUCalendarBucketsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {
  addSpaceTimeHotSpotsRecipe,
  getSpaceTimeStepIndex,
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  type GPUSpaceTimeStep
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogParameterSpec
} from '../../analysis/catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from '../../analysis/catalog-binding-types';
import {
  appendCatalogRow,
  makeCatalogTable,
  type CatalogFeatureCollection,
  type CatalogCoordinateSystem
} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {finiteSpacetimeValue} from '../../analysis/catalog-spacetime-common';
import {parseKeplerIsoTimestamp} from '../../profile/iso-timestamp';
import {makeLocalFrame} from '../../analysis/catalog-local-frame';
import {
  makeInputSpec,
  makeOutputSpec,
  makeNumberParameter,
  makeStringParameter
} from '../../analysis/catalog-platform3-b-common';

const BINDING_ID = 'curated-emerging-hot-spots';
/** Host materialization bounds; a 65,536-bin cube bounds statistics readback and publication. */
export const EMERGING_HOT_SPOTS_LIMITS = {
  events: 100000,
  cells: 4096,
  slices: 256,
  bins: 65536
} as const;
const PARAMETERS: readonly CatalogParameterSpec[] = [
  COORDINATE_SYSTEM_PARAMETER,
  makeStringParameter('timeProperty', 'Event time column', 'observed'),
  {
    name: 'timeUnit',
    label: 'Numeric epoch unit (ISO strings use their offset or UTC)',
    kind: 'enum',
    options: ['milliseconds', 'seconds'],
    defaultValue: 'milliseconds'
  },
  makeNumberParameter('cellSize', 'Cell size', 1000, 1e-6, 1e9, 'crs'),
  {
    name: 'timeStep',
    label: 'Calendar time step (UTC)',
    kind: 'enum',
    options: ['year', 'quarter', 'month', 'week', 'day', 'hour'],
    defaultValue: 'month'
  },
  makeNumberParameter(
    'timeInterval',
    'Calendar steps per slice',
    1,
    1,
    10000,
    undefined,
    'integer'
  ),
  makeNumberParameter(
    'sliceCount',
    'Slice count (0 covers the observed range)',
    0,
    0,
    EMERGING_HOT_SPOTS_LIMITS.slices,
    undefined,
    'integer'
  ),
  makeNumberParameter('radius', 'Spatial radius in cells', 1.5, 0, 4, 'cells'),
  makeNumberParameter(
    'temporalWindow',
    'Previous slices in each neighbourhood',
    1,
    0,
    255,
    'slices',
    'integer'
  ),
  makeNumberParameter('confidenceLevel', 'Bin confidence level', 0.95, 0.5, 0.9999),
  {
    name: 'multipleTesting',
    label: 'Bin multiple testing correction',
    kind: 'enum',
    options: ['benjamini-hochberg', 'bonferroni', 'none'],
    defaultValue: 'benjamini-hochberg'
  }
];

/** Input conversion only: analysis and event counting remain in the published recipe. */
export function prepareEmergingHotSpots(
  collection: CatalogFeatureCollection,
  overrides: Readonly<Record<string, CatalogParameterValue>>
) {
  const values = Object.fromEntries(
    PARAMETERS.map(parameter => [parameter.name, parameter.defaultValue!])
  );
  Object.assign(values, overrides);
  for (const parameter of PARAMETERS) {
    const value = values[parameter.name];
    if (parameter.kind === 'number' || parameter.kind === 'integer') {
      validateFiniteNumber(
        BINDING_ID,
        parameter.name,
        value as number,
        parameter.minimum,
        parameter.maximum
      );
      if (parameter.kind === 'integer')
        validateRequirement(
          BINDING_ID,
          Number.isInteger(value),
          `${parameter.name} must be an integer`
        );
    } else if (parameter.options)
      validateRequirement(
        BINDING_ID,
        parameter.options.includes(String(value)),
        `${parameter.name} must be one of ${parameter.options.join(', ')}`
      );
    else
      validateRequirement(
        BINDING_ID,
        typeof value === 'string',
        `${parameter.name} must be a string`
      );
  }
  const settings = {
    radius: Number(values['radius']),
    temporalWindow: Number(values['temporalWindow']),
    confidenceLevel: Number(values['confidenceLevel'])
  };
  getGPUEmergingHotSpotsParameterValues({
    ...settings,
    multipleTesting: values['multipleTesting'] as GPUMultipleTestingMethod
  });
  validateCapacity(
    BINDING_ID,
    'points.length',
    collection.features.length,
    1,
    EMERGING_HOT_SPOTS_LIMITS.events
  );
  const coordinates = collection.features.map((feature, index) => {
    const coordinates = feature.geometry?.coordinates;
    validateRequirement(
      BINDING_ID,
      feature.geometry?.type === 'Point' &&
        Array.isArray(coordinates) &&
        coordinates.length >= 2 &&
        coordinates.slice(0, 2).every(value => typeof value === 'number' && Number.isFinite(value)),
      `points[${index}] must have one finite Point`
    );
    return coordinates as [number, number];
  });
  const frame = makeLocalFrame(values['coordinateSystem'] as CatalogCoordinateSystem, coordinates);
  const positions = coordinates.map(position => frame.forward(position));
  validateRequirement(
    BINDING_ID,
    positions.every(position => position.every(Number.isFinite)),
    'points must project to finite coordinates'
  );
  const cellSize = Number(values['cellSize']);
  let minimumX = Infinity,
    minimumY = Infinity,
    maximumX = -Infinity,
    maximumY = -Infinity;
  for (const [horizontal, vertical] of positions) {
    minimumX = Math.min(minimumX, horizontal);
    minimumY = Math.min(minimumY, vertical);
    maximumX = Math.max(maximumX, horizontal);
    maximumY = Math.max(maximumY, vertical);
  }
  const originX = Math.floor(minimumX / cellSize) * cellSize;
  const originY = Math.floor(minimumY / cellSize) * cellSize;
  const width = Math.floor((maximumX - originX) / cellSize) + 1;
  const height = Math.floor((maximumY - originY) / cellSize) + 1;
  const cellCount = width * height;
  validateCapacity(BINDING_ID, 'cellCount', cellCount, 1, EMERGING_HOT_SPOTS_LIMITS.cells);
  const packedPositions = Float32Array.from(
    positions.flatMap(([horizontal, vertical]) => [horizontal - originX, vertical - originY])
  );
  // Upper cell edges are exclusive in the recipe. Reject Float32 rounding onto an outer edge.
  validateRequirement(
    BINDING_ID,
    packedPositions.every(
      (value, index) =>
        Number.isFinite(value) && value >= 0 && value < (index % 2 ? height : width) * cellSize
    ),
    'cellSize must retain every event inside the Float32 lattice bounds'
  );
  const readMilliseconds = (value: unknown, property: string): number => {
    const milliseconds =
      typeof value === 'number'
        ? value * (values['timeUnit'] === 'seconds' ? 1000 : 1)
        : typeof value === 'string'
          ? parseKeplerIsoTimestamp(value)
          : value instanceof Date
            ? value.getTime()
            : NaN;
    validateRequirement(
      BINDING_ID,
      Number.isSafeInteger(milliseconds) && Number.isFinite(new Date(milliseconds).getTime()),
      `${property} must be an integer epoch millisecond or a valid ISO timestamp (missing timezone means UTC)`
    );
    return milliseconds;
  };
  const milliseconds = collection.features.map((feature, index) =>
    readMilliseconds(
      feature.properties?.[String(values['timeProperty'])],
      `points[${index}].${values['timeProperty']}`
    )
  );
  let earliest = Infinity,
    latest = -Infinity;
  for (const time of milliseconds) {
    earliest = Math.min(earliest, time);
    latest = Math.max(latest, time);
  }
  const timeStep = values['timeStep'] as GPUSpaceTimeStep;
  const fields = (milliseconds: number) => {
    const date = new Date(milliseconds);
    return {
      year: date.getUTCFullYear(),
      month: date.getUTCMonth() + 1,
      day: date.getUTCDate(),
      hour: date.getUTCHours()
    };
  };
  const origin = fields(earliest);
  // Calendar origins always start on the selected step boundary (weeks anchor to the selected day).
  if (timeStep === 'year') origin.month = 1;
  if (timeStep === 'quarter') origin.month = Math.floor((origin.month - 1) / 3) * 3 + 1;
  if (['year', 'quarter', 'month'].includes(timeStep)) origin.day = 1;
  if (timeStep !== 'hour') origin.hour = 0;
  const interval = Number(values['timeInterval']);
  const stepsPerUnit = timeStep === 'week' ? 7 : 1;
  const sliceCount =
    Number(values['sliceCount']) ||
    Math.floor(
      (getSpaceTimeStepIndex(timeStep, fields(latest)) - getSpaceTimeStepIndex(timeStep, origin)) /
        (interval * stepsPerUnit)
    ) + 1;
  validateCapacity(BINDING_ID, 'sliceCount', sliceCount, 1, EMERGING_HOT_SPOTS_LIMITS.slices);
  validateCapacity(
    BINDING_ID,
    'cellCount * sliceCount',
    cellCount * sliceCount,
    1,
    EMERGING_HOT_SPOTS_LIMITS.bins
  );
  const timestamps = new Uint32Array(milliseconds.length * 2);
  milliseconds.forEach((time, index) => {
    const words = BigInt(time);
    timestamps[2 * index] = Number(BigInt.asUintN(32, words));
    timestamps[2 * index + 1] = Number(BigInt.asUintN(32, words >> 32n));
  });
  return {
    values,
    settings,
    frame,
    originX,
    originY,
    cellSize,
    width,
    height,
    cellCount,
    positions: packedPositions,
    timestamps,
    slices: {step: timeStep, origin, count: sliceCount, interval}
  };
}

export const CURATED_EMERGING_HOT_SPOTS_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Emerging hot spots',
  domain: 'analysis',
  summary:
    'Count timestamped events into a calendar cube and classify each cell by hot/cold persistence and Mann–Kendall trend.',
  capabilityIds: ['spatial-autocorrelation-and-inference'],
  contributorNames: ['GPUCalendarBuckets', 'GPUPeriodIndex', 'GPUEmergingHotSpots'],
  gpuContributorNames: ['GPUCalendarBuckets', 'GPUPeriodIndex', 'GPUEmergingHotSpots'],
  engine: 'gpu',
  inputs: [
    makeInputSpec(
      'points',
      'Timestamped events',
      ['point'],
      EMERGING_HOT_SPOTS_LIMITS.events,
      EMERGING_HOT_SPOTS_LIMITS.events
    )
  ],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec(
      'cells',
      'Emerging hot spot cells',
      'polygon',
      ['cell_index'],
      EMERGING_HOT_SPOTS_LIMITS.cells
    ),
    makeOutputSpec(
      'cube',
      'Space-time bins',
      'none',
      ['cell_index', 'slice_index'],
      EMERGING_HOT_SPOTS_LIMITS.bins
    )
  ],
  cleanup:
    'GPU only; the catalog session destroys graphs and storage after readback, and the parameter buffer is destroyed in finally. Plain tables publish through native Sources.',
  declaredBehavior: [
    'Each row is one event; values are not summed. Requires one finite Point and a time per row. ISO timestamps use their offset, or UTC when omitted, matching profiling; impossible dates are rejected. Numeric epochs use timeUnit. Integer milliseconds are packed as signed Int64 words, including pre-1970 times.',
    'EPSG:4326 is projected to the catalog local azimuthal equidistant frame in metres; planar coordinates keep their declared units. The lattice aligns to cellSize multiples in that frame and includes empty cells. Cell identity is row-major within this run.',
    'UTC calendar steps form a monotone time axis. The first slice starts at the earliest event step; sliceCount 0 covers the range. A shorter slice count can exclude later events; skippedEventCount and a diagnostic report exclusions.',
    'The recipe uses binary lattice-disc neighbours including self and the current and previous temporalWindow slices. Benjamini-Hochberg corrects over valid space-time bins by default. Category codes are the library codes 0–17; UNDETERMINED is preserved.',
    'Trend tests use the recipe defaults (at least ten slices, no autocorrelation correction); overlapping temporal windows can invalidate independence. Inference must use the full population. At most 100000 events, 4096 cells, 256 slices and 65536 bins.'
  ],
  limitation:
    'GPU only. Counts of events, not sums or rates; UTC only. Local projection, Float32 statistics and normal-approximation Gi* p-values; trend inference is uncorrected for serial correlation and multiple cells. No sampled significance preview.',
  async runGpu(context, inputs, values) {
    const prepared = prepareEmergingHotSpots(inputs['points'], values);
    const {cellCount, slices, width, height, cellSize, originX, originY, frame} = prepared;
    const binCount = cellCount * slices.count;
    const parameters = new GPUParameterBuffer(context.device, {
      id: BINDING_ID,
      format: 'float32',
      length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
    });
    try {
      return await withCatalogGpuSession(context.device, BINDING_ID, async session => {
        const cube = session.output('cube', 'uint32', binCount);
        const giZScores = session.output('gi-z', 'float32', binCount);
        const pValues = session.output('p', 'float32', binCount);
        const adjustedPValues = session.output('adjusted-p', 'float32', binCount);
        const category = session.output('category', 'uint32', cellCount);
        const trendZ = session.output('trend-z', 'float32', cellCount);
        const trendP = session.output('trend-p', 'float32', cellCount);
        const trendS = session.output('trend-s', 'sint32', cellCount);
        const hotSliceCount = session.output('hot-count', 'uint32', cellCount);
        const coldSliceCount = session.output('cold-count', 'uint32', cellCount);
        const lastValidSlice = session.output('last-valid', 'uint32', cellCount);
        const skippedEventCount = session.output('skipped', 'uint32', 1);
        const recipe = addSpaceTimeHotSpotsRecipe(session.graph, {
          timestamps: session.input('timestamps', prepared.timestamps, 'uint32x2'),
          calendarParameters: session.input(
            'calendar',
            getGPUCalendarBucketsParameterValues(0),
            'sint32'
          ),
          slices,
          cells: {
            kind: 'lattice',
            positions: session.input('positions', prepared.positions, 'float32x2'),
            width,
            height,
            bounds: [0, 0, width * cellSize, height * cellSize],
            maximumRadius: 4
          },
          parameters,
          settings: prepared.settings,
          multipleTesting: prepared.values['multipleTesting'] as GPUMultipleTestingMethod,
          outputs: {
            cube: cube.view,
            giZScores: giZScores.view,
            pValues: pValues.view,
            adjustedPValues: adjustedPValues.view,
            category: category.view,
            trendZ: trendZ.view,
            trendP: trendP.view,
            trendS: trendS.view,
            hotSliceCount: hotSliceCount.view,
            coldSliceCount: coldSliceCount.view,
            lastValidSlice: lastValidSlice.view,
            status: {skippedEventCount: skippedEventCount.view}
          }
        });
        session.run();
        const [
          counts,
          scores,
          rawProbabilities,
          adjustedProbabilities,
          categories,
          trendScores,
          trendProbabilities,
          trendWords,
          hotCounts,
          coldCounts,
          lastSlices,
          skipped
        ] = await Promise.all([
          session.readUint32(cube),
          session.readFloat32(giZScores),
          session.readFloat32(pValues),
          session.readFloat32(adjustedPValues),
          session.readUint32(category),
          session.readFloat32(trendZ),
          session.readFloat32(trendP),
          session.readUint32(trendS),
          session.readUint32(hotSliceCount),
          session.readUint32(coldSliceCount),
          session.readUint32(lastValidSlice),
          session.readScalar(skippedEventCount)
        ]);
        const trendStatistics = new Int32Array(
          trendWords.buffer,
          trendWords.byteOffset,
          trendWords.length
        );
        const cells = makeCatalogTable(
          'cells',
          'Emerging hot spot cells',
          {
            cell_index: 'int32',
            category: 'int32',
            category_name: 'string',
            trend_z: 'float64',
            trend_p: 'float64',
            trend_s: 'int32',
            hot_slice_count: 'int32',
            cold_slice_count: 'int32',
            last_valid_slice: 'int32',
            geometry: 'string'
          },
          'geometry'
        );
        const bins = makeCatalogTable('cube', 'Space-time bins', {
          cell_index: 'int32',
          slice_index: 'int32',
          event_count: 'int32',
          gi_z: 'float64',
          p_value: 'float64',
          adjusted_p_value: 'float64'
        });
        const categoryNames = Object.fromEntries(
          Object.entries(GPU_EMERGING_HOT_SPOTS_CATEGORIES).map(([name, code]) => [code, name])
        );
        for (let cell = 0; cell < cellCount; cell++) {
          const horizontal = originX + (cell % width) * cellSize;
          const vertical = originY + Math.floor(cell / width) * cellSize;
          const ring = [
            [horizontal, vertical],
            [horizontal + cellSize, vertical],
            [horizontal + cellSize, vertical + cellSize],
            [horizontal, vertical + cellSize],
            [horizontal, vertical]
          ].map(([horizontal, vertical]) => frame.inverse(horizontal, vertical));
          appendCatalogRow(cells, {
            cell_index: cell,
            category: categories[cell],
            category_name: categoryNames[categories[cell]],
            trend_z: finiteSpacetimeValue(trendScores[cell]),
            trend_p: finiteSpacetimeValue(trendProbabilities[cell]),
            trend_s: trendStatistics[cell],
            hot_slice_count: hotCounts[cell],
            cold_slice_count: coldCounts[cell],
            last_valid_slice: lastSlices[cell] === 0xffffffff ? null : lastSlices[cell],
            geometry: JSON.stringify({type: 'Polygon', coordinates: [ring]})
          });
          for (let slice = 0; slice < slices.count; slice++) {
            const bin = cell * slices.count + slice;
            appendCatalogRow(bins, {
              cell_index: cell,
              slice_index: slice,
              event_count: counts[bin],
              gi_z: finiteSpacetimeValue(scores[bin]),
              p_value: finiteSpacetimeValue(rawProbabilities[bin]),
              adjusted_p_value: finiteSpacetimeValue(adjustedProbabilities[bin])
            });
          }
        }
        return {
          tables: [cells, bins],
          diagnostics: skipped
            ? [
                {
                  code: 'slice-range',
                  severity: 'warning',
                  count: skipped,
                  message: 'Events outside the selected calendar range were excluded.'
                }
              ]
            : [],
          dispatchedContributors: ['GPUCalendarBuckets', 'GPUPeriodIndex', 'GPUEmergingHotSpots'],
          summary: {
            eventCount: prepared.timestamps.length / 2,
            cellCount,
            sliceCount: slices.count,
            gridWidth: width,
            gridHeight: height,
            cellSize,
            skippedEventCount: skipped,
            countedEventCount: counts.reduce((total, count) => total + count, 0),
            multipleTesting: recipe.inference.multipleTesting,
            significanceLevel: recipe.inference.significanceLevel!,
            coordinateSystem: String(prepared.values['coordinateSystem']),
            cellOriginX: originX,
            cellOriginY: originY,
            radius: prepared.settings.radius,
            temporalWindow: prepared.settings.temporalWindow,
            testedRowsDefinition: recipe.inference.testedRowsDefinition!,
            timeStep: slices.step,
            timeInterval: slices.interval,
            timeOrigin: JSON.stringify(slices.origin),
            trendAutocorrelationCorrection: 'none'
          }
        };
      });
    } finally {
      parameters.destroy();
    }
  }
};
