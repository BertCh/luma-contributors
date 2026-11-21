// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  GPU_PLANAR_SPATIAL_CONTEXT,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT
} from '@luma-contributors/analysis/analysis-core';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPUAnalysisSpacing,
  GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
  decodeGPUAnalysisSpacingStatistics,
  GPUAnalysisContiguityStatistics,
  decodeGPUAnalysisContiguityStatistics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  guessGPUAnalysisMeasurementLevel,
  type GPUAnalysisColumnProfile
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {CatalogGpuSession, withCatalogGpuSession} from '../analysis/catalog-gpu-session';
import {getKeplerDatasetProfileUndeclared} from './dataset-profile';
import type {KeplerDatasetProfile} from './dataset-profile-types';
import type {KeplerDatasetProfileStageRunner} from './dataset-profile-stages';

/** A borrowed packed slice. The host keeps it alive until the job's finished promise settles. */
export type KeplerDatasetProfileResidentSlice = {
  buffer: Buffer;
  length: number;
  byteOffset?: number;
};

export type KeplerDatasetProfileGpuOptions = {
  device: Device;
  rowCount: number;
  /** Normalized float32 values (NaN for null), or uint32 dictionary codes (0xffffffff for null). */
  columns: readonly (KeplerDatasetProfileResidentSlice & {name: string; categoryCount?: number})[];
  /** Locations can outnumber table rows, for example MultiPoint features. */
  positions?: KeplerDatasetProfileResidentSlice;
  positionsCoverage?: 'all-locations' | 'feature-anchors';
  positionMask?: KeplerDatasetProfileResidentSlice;
  polygons?: {
    positions: KeplerDatasetProfileResidentSlice;
    ringOffsets: KeplerDatasetProfileResidentSlice;
    polygonOffsets: KeplerDatasetProfileResidentSlice;
    featureOffsets: KeplerDatasetProfileResidentSlice;
    neighborCapacity: number;
    pairCapacity: number;
    snapTolerance?: number;
    coordinateExtent?: number;
  };
};

const OPERATION = 'KeplerDatasetProfileGpu';
const MAXIMUM_ROW_COUNT = 2 ** 24;
const MAXIMUM_COLUMN_COUNT = 16;
const REGISTER_COUNT = 4096;

/** Drain every started mapping before session disposal, even when one read fails synchronously. */
async function readProfileBuffers<const Values extends readonly unknown[]>(
  reads: {[Index in keyof Values]: () => Promise<Values[Index]>}
): Promise<Values> {
  const results = await Promise.allSettled(reads.map(read => Promise.resolve().then(read)));
  for (const result of results) if (result.status === 'rejected') throw result.reason;
  return results.map(
    result => (result as PromiseFulfilledResult<unknown>).value
  ) as unknown as Values;
}

function validateSlice(
  name: string,
  slice: KeplerDatasetProfileResidentSlice,
  components = 1
): void {
  validateCapacity(OPERATION, `${name}.length`, slice.length, 0, MAXIMUM_ROW_COUNT);
  const byteOffset = slice.byteOffset ?? 0;
  validateCapacity(OPERATION, `${name}.byteOffset`, byteOffset, 0, slice.buffer.byteLength);
  validateRequirement(OPERATION, byteOffset % 4 === 0, `${name}.byteOffset must be divisible by 4`);
  validateRequirement(
    OPERATION,
    byteOffset + slice.length * components * 4 <= slice.buffer.byteLength,
    `${name} byte range must fit buffer.byteLength`
  );
}

function importResident<Format extends 'float32' | 'uint32' | 'float32x2'>(
  session: CatalogGpuSession,
  name: string,
  slice: KeplerDatasetProfileResidentSlice,
  format: Format
): GraphDataView<Format> {
  return session.graph.createDataView(
    session.graph.importBuffer(
      {
        id: name,
        byteLength: slice.buffer.byteLength,
        usage: slice.buffer.usage
      },
      slice.buffer
    ),
    {format, length: slice.length, byteOffset: slice.byteOffset ?? 0}
  );
}

/** Adapts the existing GPU contributors; only scalar summaries are read back to the host. */
export function createKeplerDatasetProfileGpuRunner(
  options: KeplerDatasetProfileGpuOptions
): KeplerDatasetProfileStageRunner {
  validateCapacity(OPERATION, 'rowCount', options.rowCount, 0, MAXIMUM_ROW_COUNT);
  validateCapacity(OPERATION, 'columns.length', options.columns.length, 0, MAXIMUM_COLUMN_COUNT);
  const names = new Set<string>();
  for (const column of options.columns) {
    validateRequirement(
      OPERATION,
      column.name.length > 0 && !names.has(column.name),
      'columns.name must be nonempty and unique'
    );
    names.add(column.name);
    validateSlice(`columns.${column.name}`, column);
    validateRequirement(
      OPERATION,
      column.length === options.rowCount,
      'columns.length must equal rowCount'
    );
    if (column.categoryCount !== undefined)
      validateCapacity(OPERATION, 'categoryCount', column.categoryCount, 1, MAXIMUM_ROW_COUNT);
  }
  validateRequirement(
    OPERATION,
    options.positionsCoverage === undefined ||
      options.positionsCoverage === 'all-locations' ||
      options.positionsCoverage === 'feature-anchors',
    'positionsCoverage must be all-locations or feature-anchors'
  );
  if (options.positions) validateSlice('positions', options.positions, 2);
  if (options.positionMask) {
    validateSlice('positionMask', options.positionMask);
    validateRequirement(
      OPERATION,
      options.positionMask.length === options.positions?.length,
      'positionMask.length must equal positions.length'
    );
  }
  if (options.polygons) {
    const polygons = options.polygons;
    validateSlice('polygons.positions', polygons.positions, 2);
    for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const)
      validateSlice(`polygons.${name}`, polygons[name]);
    validateRequirement(
      OPERATION,
      polygons.featureOffsets.length === options.rowCount + 1,
      'polygons.featureOffsets.length must equal rowCount + 1'
    );
    validateCapacity(OPERATION, 'neighborCapacity', polygons.neighborCapacity, 1);
    validateCapacity(OPERATION, 'pairCapacity', polygons.pairCapacity, 1);
  }
  const validateSource = (profile: KeplerDatasetProfile) => {
    validateRequirement(
      OPERATION,
      profile.rowCount === options.rowCount,
      'profile.rowCount must equal resident rowCount'
    );
    validateRequirement(
      OPERATION,
      options.columns.every(column =>
        profile.columns.some(metadata => metadata.name === column.name)
      ),
      'columns.name must exist in profile.columns'
    );
  };
  if (options.rowCount === 0) return {};
  const runner: KeplerDatasetProfileStageRunner = {};
  if (options.columns.length)
    runner.columns = async ({profile, signal}) => {
      signal.throwIfAborted();
      validateSource(profile);
      return withCatalogGpuSession(options.device, 'dataset-profile-columns', async session => {
        const statistics = session.output(
          'statistics',
          'float32',
          options.columns.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT
        );
        const counts = session.output('counts', 'uint32', options.columns.length * 2);
        const registers = session.output(
          'registers',
          'uint32',
          options.columns.length * REGISTER_COUNT
        );
        session.graph.add(
          new GPUColumnProfile({
            id: 'dataset-profile-columns',
            columns: options.columns.map(column =>
              column.categoryCount === undefined
                ? {values: importResident(session, column.name, column, 'float32')}
                : {
                    kind: 'category',
                    categoryCount: column.categoryCount,
                    values: importResident(session, column.name, column, 'uint32')
                  }
            ),
            histogramBinCount: 1,
            hyperLogLogPrecision: 12,
            output: {
              statistics: statistics.view,
              counts: counts.view,
              hyperLogLogRegisters: registers.view
            }
          })
        );
        session.run(signal);
        const [values, countValues, registerValues] = await readProfileBuffers([
          () => session.readFloat32(statistics),
          () => session.readUint32(counts),
          () => session.readUint32(registers)
        ]);
        signal.throwIfAborted();
        const originals = getKeplerDatasetProfileUndeclared(profile).columns;
        const refined = new Map<string, GPUAnalysisColumnProfile>();
        options.columns.forEach((input, index) => {
          const metadata = originals.find(column => column.name === input.name)!;
          const statistic = (field: keyof typeof GPU_COLUMN_PROFILE_STATISTIC) =>
            values[
              index * GPU_COLUMN_PROFILE_STATISTIC_COUNT + GPU_COLUMN_PROFILE_STATISTIC[field]
            ];
          validateRequirement(
            OPERATION,
            statistic('overflowCount') === 0,
            `columns.${input.name} category overflowCount must be 0`
          );
          const count = countValues[index * 2];
          let occupied = 0;
          for (const register of registerValues.subarray(
            index * REGISTER_COUNT,
            (index + 1) * REGISTER_COUNT
          ))
            if (register > 0) occupied++;
          // Occupied registers give a guaranteed lower bound; HLL's estimate is not a confidence bound.
          const column: GPUAnalysisColumnProfile = {
            ...metadata,
            rowCount: options.rowCount,
            nullCount: countValues[index * 2 + 1],
            distinctCount: Math.min(
              count,
              Math.max(occupied, Math.round(statistic('distinctEstimate')))
            ),
            distinctCountIsEstimate: count > 0,
            distinctCountLowerBound: occupied,
            distinctCountUpperBound: count
          };
          for (const key of [
            'minimum',
            'maximum',
            'mean',
            'standardDeviation',
            'integral',
            'sampleSize',
            'measurement',
            'measurementSource',
            'measurementConfidence',
            'measurementReason'
          ] as const)
            delete column[key];
          if (input.categoryCount === undefined) {
            for (const key of ['minimum', 'maximum', 'mean', 'standardDeviation'] as const)
              if (Number.isFinite(statistic(key))) column[key] = statistic(key);
            if (metadata.type === 'integer') column.integral = true;
          }
          const guess = guessGPUAnalysisMeasurementLevel(column);
          if (guess)
            Object.assign(column, {
              measurement: guess.level,
              measurementSource: 'guess',
              measurementReason: guess.reason,
              measurementConfidence: guess.confidence
            });
          refined.set(input.name, column);
        });
        return originals.map(column => refined.get(column.name) ?? column);
      });
    };
  if (options.positions)
    runner.spacing = async ({profile, signal}) => {
      signal.throwIfAborted();
      validateSource(profile);
      validateRequirement(
        OPERATION,
        Boolean(profile.coordinates),
        'profile.coordinates must identify a coordinate model'
      );
      return withCatalogGpuSession(options.device, 'dataset-profile-spacing', async session => {
        const summary = session.output('summary', 'uint32', GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
        const positions = options.positions!;
        session.graph.add(
          new GPUAnalysisSpacing({
            id: 'dataset-profile-spacing',
            positions: importResident(session, 'positions', positions, 'float32x2'),
            mask: options.positionMask
              ? importResident(session, 'mask', options.positionMask, 'uint32')
              : undefined,
            gridSize: positions.length > 10_000 ? [256, 256] : [16, 16],
            spatialContext:
              profile.coordinates!.model === 'geographic'
                ? GPU_GREAT_CIRCLE_SPATIAL_CONTEXT
                : GPU_PLANAR_SPATIAL_CONTEXT,
            output: {summary: summary.view}
          })
        );
        session.run(signal);
        const words = await session.readUint32(summary);
        signal.throwIfAborted();
        const spacing = decodeGPUAnalysisSpacingStatistics(words);
        return spacing
          ? {...spacing, locationCoverage: options.positionsCoverage ?? 'feature-anchors'}
          : undefined;
      });
    };
  if (options.polygons)
    runner.contiguity = async ({profile, signal}) => {
      signal.throwIfAborted();
      validateSource(profile);
      return withCatalogGpuSession(options.device, 'dataset-profile-contiguity', async session => {
        const polygons = options.polygons!;
        const offsets = session.output('offsets', 'uint32', options.rowCount + 1);
        const neighbors = session.output('neighbors', 'uint32', polygons.neighborCapacity);
        const weights = session.output('weights', 'float32', polygons.neighborCapacity);
        const overflow = session.output('overflow', 'uint32', 1);
        const requiredCount = session.output('required-count', 'uint32', 1);
        const candidateOverflow = session.output('candidate-overflow', 'uint32', 1);
        const cardinality = session.output('cardinality', 'uint32', options.rowCount);
        const counts = session.output('counts', 'uint32', 6);
        const median = session.output('median', 'float32', 1);
        const validity = session.output('validity', 'uint32', 1);
        session.graph.add(
          new GPUAnalysisContiguityStatistics({
            id: 'dataset-profile-contiguity',
            criterion: 'queen',
            pairCapacity: polygons.pairCapacity,
            snapTolerance: polygons.snapTolerance,
            coordinateExtent: polygons.coordinateExtent,
            polygons: {
              kind: 'polygons',
              positions: importResident(session, 'positions', polygons.positions, 'float32x2'),
              ringOffsets: importResident(session, 'rings', polygons.ringOffsets, 'uint32'),
              polygonOffsets: importResident(
                session,
                'polygons',
                polygons.polygonOffsets,
                'uint32'
              ),
              featureOffsets: importResident(session, 'features', polygons.featureOffsets, 'uint32')
            },
            output: {
              weights: {
                offsets: offsets.view,
                neighbors: neighbors.view,
                weights: weights.view,
                status: {
                  overflow: overflow.view,
                  requiredCount: requiredCount.view,
                  candidateOverflow: candidateOverflow.view
                }
              },
              cardinality: cardinality.view,
              counts: counts.view,
              median: median.view,
              validity: validity.view
            }
          })
        );
        session.run(signal);
        const [countValues, medianValues, validityValues, overflowValues, candidateValues] =
          await readProfileBuffers([
            () => session.readUint32(counts),
            () => session.readFloat32(median),
            () => session.readUint32(validity),
            () => session.readUint32(overflow),
            () => session.readUint32(candidateOverflow)
          ]);
        signal.throwIfAborted();
        const result = decodeGPUAnalysisContiguityStatistics({
          counts: countValues,
          median: medianValues,
          validity: validityValues,
          weights: {status: {overflow: overflowValues, candidateOverflow: candidateValues}}
        });
        validateRequirement(
          OPERATION,
          result.exact,
          'contiguity overflow and candidateOverflow must be 0 and validity must be 1; no exact summary was published'
        );
        if (!result.exact) return undefined;
        return {
          minimum: result.minimum,
          median: result.median,
          maximum: result.maximum,
          islandCount: result.islands
        };
      });
    };
  return runner;
}
