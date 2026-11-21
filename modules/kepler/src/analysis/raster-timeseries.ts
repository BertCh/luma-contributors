// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  GPURasterRollingMean,
  GPURasterWhittakerSmoother,
  GPURasterHANTS,
  GPURasterHarmonicRegression,
  GPURasterBFASTMonitor,
  GPURasterCOLDMonitor,
  GPURasterMedoidComposite,
  GPURasterGeometricMedianComposite,
  GPURasterBestAvailablePixel,
  GPURasterPhenologyMetrics,
  GPURasterStreamingAccumulator,
  GPURasterClimateAnomaly,
  GPURasterStandardizedIndexCalibration,
  GPURasterStandardizedIndex,
  GPURasterTemporalResample,
  GPURasterLaggedCorrelation,
  GPURasterSnapshotCovariance,
  GPURasterEOFProjection,
  GPURasterClimateRunAccumulator,
  getGPURasterStreamingAccumulatorParameterValues,
  decodeGPURasterStreamingAccumulatorStatus,
  resolveGPURasterStandardizedIndexZeroAtom,
  getGPURasterWhittakerSmootherParameterValues,
  getGPURasterHANTSParameterValues,
  getGPURasterHarmonicRegressionParameterValues,
  getGPURasterBFASTMonitorParameterValues,
  getGPURasterCOLDMonitorParameterValues,
  getGPURasterGeometricMedianCompositeParameterValues,
  getGPURasterBestAvailablePixelParameterValues,
  getGPURasterPhenologyMetricsParameterValues,
  getGPURasterStandardizedIndexCalibrationParameterValues,
  getGPURasterStandardizedIndexParameterValues,
  getGPURasterTemporalResampleParameterValues,
  getGPURasterLaggedCorrelationParameterValues,
  getGPURasterSnapshotCovarianceParameterValues,
  getGPURasterSnapshotCovarianceMatrix,
  getGPURasterSnapshotEOFDecomposition,
  getGPURasterCentredFractionalYears,
  getGPURasterClimateRunAccumulatorParameterValues,
  getGPURasterFixedPointValues
} from '@luma-contributors/analysis/raster-analysis';
import {withCatalogGpuSession, type CatalogGpuOutput} from './catalog-gpu-session';
import {
  getRasterTimeSeriesBinding,
  type RasterTimeSeriesOperation,
  type RasterTimeSeriesSettings
} from './catalog-raster-timeseries-bindings';

export {
  RASTER_TIMESERIES_BINDINGS,
  getRasterTimeSeriesBinding,
  isRasterTimeSeriesGlobalOutput
} from './catalog-raster-timeseries-bindings';
export type {
  RasterTimeSeriesOperation,
  RasterTimeSeriesSettings
} from './catalog-raster-timeseries-bindings';

export type RasterTimeStack = {
  cellCount: number;
  sliceCount: number;
  bandCount: number;
  /** Cell-major, time-major, band-interleaved. Invalid observations must be NaN. */
  values: Float32Array;
  /** Strictly increasing Unix milliseconds. */
  times: readonly number[];
};
export type RasterTimeSeriesColumn = {
  values: Float32Array | Float64Array | Uint32Array;
  components: number;
  scope: 'cell' | 'global';
};
export type RasterTimeSeriesResult = {
  /** Count-loss certificates in cell-major slot order (cell * slotCount + slot).
   * This reports observation-count saturation only, not moment quantization or numerical loss.
   * Raw words are also available in columns.streamingStatus; affected published statistics are NaN.
   * Equal codes share frozen decoded records; malformed certificate lengths reject the result.
   */
  streamingCountStatus?: readonly {complete: boolean; overflow: boolean}[];
  columns: Record<string, RasterTimeSeriesColumn>;
  contributors: readonly string[];
  readbackBytes: number;
  settings: RasterTimeSeriesSettings;
};

export const RASTER_TIME_STACK_LIMITS = {
  cells: 65536,
  slices: 256,
  bands: 16,
  values: 1048576
} as const;

export function validateRasterTimeStack(stack: RasterTimeStack): void {
  for (const [name, value, maximum] of [
    ['cellCount', stack.cellCount, RASTER_TIME_STACK_LIMITS.cells],
    ['sliceCount', stack.sliceCount, RASTER_TIME_STACK_LIMITS.slices],
    ['bandCount', stack.bandCount, RASTER_TIME_STACK_LIMITS.bands]
  ] as const) {
    if (!Number.isInteger(value) || value < 1 || value > maximum)
      throw new Error(`Raster time stack ${name} must be 1–${maximum}`);
  }
  const length = stack.cellCount * stack.sliceCount * stack.bandCount;
  if (length > RASTER_TIME_STACK_LIMITS.values || stack.values.length !== length)
    throw new Error('Raster time stack values length/capacity mismatch (maximum 1048576)');
  if (
    stack.times.length !== stack.sliceCount ||
    stack.times.some(
      (time, index) =>
        !Number.isSafeInteger(time) ||
        Math.abs(time) > 8640000000000000 ||
        (index > 0 && time <= stack.times[index - 1])
    )
  )
    throw new Error('Raster time stack needs strictly increasing Unix milliseconds, one per slice');
}

/** Match the contributor's float32 subtraction, with a fallback only for an empty/zero baseline. */
function getCalibrationMagnitude(values: Float32Array, location = 0): number {
  let magnitude = 0;
  for (const value of values)
    if (Number.isFinite(value))
      magnitude = Math.max(magnitude, Math.abs(Math.fround(value - Math.fround(location))));
  if (!Number.isFinite(magnitude) || magnitude > 1e15)
    throw new Error('Shifted calibration magnitude exceeds 1e15; rescale the values/location');
  return magnitude || 1;
}

/** Owns each finite graph execution through readback; all GPU buffers are released, also on failure.
 * Rolling mean accepts windowLength in [1,128] slices and optional activeSliceCount in [0,sliceCount].
 * Short chunks keep the allocated cell-major sliceCount pitch; inactive outputs are NaN.
 * Calibration/anomaly operations default to min(12,sliceCount) baseline slices; explicit counts are validated.
 */
export async function runRasterTimeSeries(
  device: Device,
  stack: RasterTimeStack,
  operation: RasterTimeSeriesOperation,
  settings: RasterTimeSeriesSettings = {}
): Promise<RasterTimeSeriesResult> {
  validateRasterTimeStack(stack);
  const binding = getRasterTimeSeriesBinding(operation, settings);
  const resolved: RasterTimeSeriesSettings = {
    ...binding.defaults,
    ...settings,
    ...('calibrationSliceCount' in binding.defaults &&
    settings['calibrationSliceCount'] === undefined
      ? {calibrationSliceCount: Math.min(12, stack.sliceCount)}
      : {})
  };
  const number = (
    name: string,
    fallback: number,
    minimum = -1e15,
    maximum = 1e15,
    integer = false
  ): number => {
    const value = resolved[name] ?? fallback;
    if (
      typeof value !== 'number' ||
      !Number.isFinite(value) ||
      value < minimum ||
      value > maximum ||
      (integer && !Number.isInteger(value))
    )
      throw new Error(
        `${binding.title}: ${name} must be ${integer ? 'an integer' : 'finite'} in [${minimum}, ${maximum}]`
      );
    return value;
  };
  const count = (name: string, fallback: number, minimum: number, maximum: number) =>
    number(name, fallback, minimum, maximum, true);
  const {cellCount, sliceCount, bandCount} = stack;
  const multiBand = ['cold', 'medoid', 'geometric-median', 'best-pixel'].includes(operation);
  if (!multiBand && bandCount !== 1) throw new Error(`${binding.title} needs one band per date`);
  if (['rolling-mean', 'lagged-correlation', 'climate-runs'].includes(operation)) {
    const spacing = stack.times[1] - stack.times[0];
    if (stack.times.some((time, index) => index > 0 && time - stack.times[index - 1] !== spacing))
      throw new Error(`${binding.title} requires equal acquisition spacing; resample first`);
    if (operation === 'climate-runs' && spacing !== 86400000)
      throw new Error('Climate runs requires daily acquisitions');
  }
  const scalarCount = cellCount * sliceCount;
  const rollingActiveSliceCount =
    operation === 'rolling-mean'
      ? count('activeSliceCount', sliceCount, 0, sliceCount)
      : sliceCount;
  let maximumMagnitude = 0;
  for (let index = 0; index < stack.values.length; index++) {
    if (operation === 'rolling-mean' && index % sliceCount >= rollingActiveSliceCount) continue;
    const value = stack.values[index];
    if (Number.isFinite(value)) maximumMagnitude = Math.max(maximumMagnitude, Math.abs(value));
  }
  maximumMagnitude ||= 1;
  if (maximumMagnitude > 1e15)
    throw new Error('Raster time stack magnitude exceeds 1e15; rescale the values');
  return withCatalogGpuSession(device, 'raster-timeseries', async session => {
    const columns: Record<string, RasterTimeSeriesColumn> = {};
    const pending: {
      name: string;
      output: CatalogGpuOutput<GPUVectorFormat>;
      components: number;
      scope: 'cell' | 'global';
      integer: boolean;
    }[] = [];
    let readbackBytes = 0;
    let streamingSlotCount = 0;
    let streamingSlots: Uint32Array | undefined;
    let allocatedOutputBytes = 0;
    const allocateOutput = <Format extends GPUVectorFormat>(
      name: string,
      format: Format,
      length: number
    ) => {
      allocatedOutputBytes += length * 4;
      if (allocatedOutputBytes > 64 * 1024 * 1024)
        throw new Error(
          'Raster time-series outputs and state exceed the 64 MiB allocation limit; use a smaller window'
        );
      return session.output(name, format, length);
    };
    const parameters = (values: Float32Array) => session.input('parameters', values, 'float32');
    const output = (
      name: string,
      components = 1,
      integer = false,
      scope: 'cell' | 'global' = 'cell',
      rows = cellCount
    ) => {
      const allocated = allocateOutput(name, integer ? 'uint32' : 'float32', rows * components);
      pending.push({name, output: allocated, components, scope, integer});
      return allocated.view;
    };
    // Explicit formats on the two allocators preserve the contributor prop types.
    const floats = (name: string, components = 1) =>
      output(name, components) as ReturnType<typeof session.output<'float32'>>['view'];
    const integers = (name: string, components = 1) =>
      output(name, components, true) as ReturnType<typeof session.output<'uint32'>>['view'];
    const scratchFloats = (name: string, length: number) =>
      allocateOutput(name, 'float32', length).view;
    const scratchIntegers = (name: string, length: number) =>
      allocateOutput(name, 'uint32', length).view;
    const values = session.input('values', stack.values, 'float32');
    const maskValues = Uint32Array.from({length: scalarCount}, (_, observation) => {
      for (let band = 0; band < bandCount; band++)
        if (!Number.isFinite(stack.values[observation * bandCount + band])) return 0;
      return 1;
    });
    const mask = session.input('mask', maskValues, 'uint32');
    const words = new Uint32Array(sliceCount * 2);
    stack.times.forEach((time, index) => {
      const unsigned = BigInt.asUintN(64, BigInt(time));
      words[index * 2] = Number(unsigned & 0xffffffffn);
      words[index * 2 + 1] = Number(unsigned >> 32n);
    });
    const dateAxis = {
      acquisitionTimes: session.input('times', words, 'uint32x2'),
      temporalContext: {
        unit: 'millisecond',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    } as const;
    const common = {cellCount, sliceCount, values, mask};
    const harmonicCount = () => count('harmonicCount', 1, 1, 3);
    const streamParameters = () =>
      getGPURasterStreamingAccumulatorParameterValues(
        {
          reset: true,
          activeSliceCount: rollingActiveSliceCount,
          maximumValueMagnitude: maximumMagnitude,
          maximumStreamSliceCount: sliceCount
        },
        sliceCount
      );
    const smooth = () => {
      const smoothed = floats('smoothed', sliceCount);
      session.graph.add(
        new GPURasterWhittakerSmoother({
          ...common,
          dateAxis,
          differenceOrder: count('differenceOrder', 2, 1, 3),
          reweightPassCount: count('reweightPassCount', 3, 0, 8),
          parameters: parameters(
            getGPURasterWhittakerSmootherParameterValues({
              smoothingParameter: number('smoothingParameter', 10, 0, 1e7)
            })
          ),
          output: {smoothed, status: integers('smoothingStatus')}
        })
      );
      return smoothed;
    };
    switch (operation) {
      case 'rolling-mean': {
        const windowLength = count('windowLength', 3, 1, 128);
        session.graph.add(
          new GPURasterRollingMean({
            cellCount,
            chunkSliceCount: sliceCount,
            windowLength,
            values,
            mask,
            parameters: parameters(streamParameters()),
            ring: scratchFloats('ring', cellCount * windowLength),
            seenCount: scratchIntegers('seen', cellCount),
            output: floats('rollingMean', sliceCount)
          })
        );
        break;
      }
      case 'whittaker':
        smooth();
        break;
      case 'hants': {
        const harmonics = harmonicCount();
        session.graph.add(
          new GPURasterHANTS({
            ...common,
            dateAxis,
            harmonicCount: harmonics,
            rejectionSide: 'both',
            parameters: parameters(
              getGPURasterHANTSParameterValues({
                period: number('period', 1, 1e-6),
                fitErrorTolerance: number('fitErrorTolerance', 1, 0)
              })
            ),
            output: {
              smoothed: floats('smoothed', sliceCount),
              coefficients: floats('coefficients', 1 + 2 * harmonics),
              status: integers('status')
            }
          })
        );
        break;
      }
      case 'harmonic-regression': {
        const harmonics = harmonicCount();
        session.graph.add(
          new GPURasterHarmonicRegression({
            ...common,
            dateAxis,
            harmonicCount: harmonics,
            parameters: parameters(
              getGPURasterHarmonicRegressionParameterValues({period: number('period', 1, 1e-6)})
            ),
            output: {
              fitted: floats('fitted', sliceCount),
              coefficients: floats('coefficients', 2 + 2 * harmonics),
              summary: integers('summary', 2),
              rmse: floats('rmse')
            }
          })
        );
        break;
      }
      case 'bfast': {
        const harmonics = harmonicCount();
        session.graph.add(
          new GPURasterBFASTMonitor({
            ...common,
            dateAxis,
            harmonicCount: harmonics,
            historySliceCount: count('historySliceCount', 12, 1, sliceCount - 1),
            parameters: parameters(
              getGPURasterBFASTMonitorParameterValues({
                period: number('period', 1, 1e-6),
                bandwidth: number('bandwidth', 0.25, 1e-6, 1),
                criticalValue: number('criticalValue', 2, 0)
              })
            ),
            output: {
              scores: floats('scores', 2),
              coefficients: floats('coefficients', 2 + 2 * harmonics),
              cellSummary: integers('cellSummary', 3)
            }
          })
        );
        break;
      }
      case 'cold': {
        const segments = count('maximumSegmentCount', 4, 1, 32);
        session.graph.add(
          new GPURasterCOLDMonitor({
            ...common,
            bandCount,
            dateAxis,
            minimumObservations: count('minimumObservations', 12, 4, 256),
            consecutiveCount: count('consecutiveCount', 6, 2, 16),
            maximumSegmentCount: segments,
            parameters: parameters(
              getGPURasterCOLDMonitorParameterValues({
                period: number('period', 1, 1e-6),
                changeThreshold: number('changeThreshold', 15, 0),
                outlierThreshold: number('outlierThreshold', 100, 0),
                minimumRMSE: Array(bandCount).fill(number('minimumRMSE', 1, 1e-6))
              })
            ),
            output: {
              segmentStatistics: floats('segmentStatistics', segments * bandCount * 2),
              segments: integers('segments', segments * 4),
              coefficients: floats('coefficients', segments * bandCount * 8),
              cellSummary: integers('cellSummary', 3)
            }
          })
        );
        break;
      }
      case 'medoid':
        session.graph.add(
          new GPURasterMedoidComposite({
            ...common,
            bandCount,
            output: {
              composite: floats('composite', bandCount),
              sliceIndex: integers('sliceIndex'),
              status: integers('status')
            }
          })
        );
        break;
      case 'geometric-median':
        session.graph.add(
          new GPURasterGeometricMedianComposite({
            ...common,
            bandCount,
            iterationCount: count('iterationCount', 16, 1, 64),
            parameters: parameters(
              getGPURasterGeometricMedianCompositeParameterValues({
                distanceEpsilon: number('distanceEpsilon', 1e-5, 1e-10)
              })
            ),
            output: {
              composite: floats('composite', bandCount),
              status: integers('status'),
              lastStep: floats('lastStep')
            }
          })
        );
        break;
      case 'best-pixel': {
        const attributes = new Float32Array(sliceCount * 2);
        stack.times.forEach((time, index) => {
          const date = new Date(time);
          attributes[index * 2] = (time - Date.UTC(date.getUTCFullYear(), 0, 1)) / 86400000 + 1;
          attributes[index * 2 + 1] = date.getUTCFullYear();
        });
        session.graph.add(
          new GPURasterBestAvailablePixel({
            ...common,
            bandCount,
            sliceAttributes: session.input('attributes', attributes, 'float32'),
            parameters: parameters(
              getGPURasterBestAvailablePixelParameterValues({
                targetDayOfYear: number('targetDayOfYear', 180, 1, 366),
                dayOfYearSigma: number('dayOfYearSigma', 45, 1e-6),
                dayOfYearWeight: 1,
                targetYear: number('targetYear', 2020),
                maximumYearDistance: number('maximumYearDistance', 10, 1e-6),
                yearWeight: 1,
                minimumCloudDistance: 0,
                maximumCloudDistance: 1,
                cloudWeight: 0
              })
            ),
            output: {
              composite: floats('composite', bandCount),
              bestSlice: integers('bestSlice'),
              bestScore: floats('bestScore')
            }
          })
        );
        break;
      }
      case 'phenology': {
        const smoothed = smooth();
        session.graph.add(
          new GPURasterPhenologyMetrics({
            cellCount,
            sliceCount,
            smoothed,
            dateAxis,
            maximumSeasonCount: 1,
            parameters: parameters(
              getGPURasterPhenologyMetricsParameterValues({
                startFraction: number('startFraction', 0.25, 0.00001, 0.99999),
                endFraction: number('endFraction', 0.15, 0.00001, 0.99999),
                minimumAmplitude: number('minimumAmplitude', 0, 0)
              })
            ),
            output: {
              metrics: floats('metrics', 10),
              flags: integers('flags'),
              seasonCount: integers('seasonCount'),
              status: integers('status')
            }
          })
        );
        break;
      }
      case 'streaming':
      case 'climate-anomaly':
      case 'standardized-calibration':
      case 'standardized-index': {
        const slotCount = count('slotCount', 1, 1, 12);
        if (slotCount !== 1 && slotCount !== 12)
          throw new Error('slotCount must be 1 or 12 (UTC months)');
        const calibrationCount =
          operation === 'streaming'
            ? sliceCount
            : count('calibrationSliceCount', Math.min(12, sliceCount), 2, sliceCount);
        const slots = Uint32Array.from(stack.times, time =>
          slotCount === 1 ? 0 : new Date(time).getUTCMonth()
        );
        const baseline = new Float32Array(cellCount * calibrationCount);
        for (let cell = 0; cell < cellCount; cell++)
          baseline.set(
            stack.values.subarray(cell * sliceCount, cell * sliceCount + calibrationCount),
            cell * calibrationCount
          );
        const baselineValues = session.input('baseline', baseline, 'float32');
        const slotIndices = session.input('slots', slots, 'uint32');
        const baselineSlots = session.input(
          'baseline-slots',
          slots.slice(0, calibrationCount),
          'uint32'
        );
        if (operation === 'streaming' || operation === 'climate-anomaly') {
          const packed = getGPURasterStreamingAccumulatorParameterValues(
            {
              reset: true,
              maximumValueMagnitude: getCalibrationMagnitude(baseline),
              maximumStreamSliceCount: calibrationCount
            },
            calibrationCount
          );
          const accumulatorParameters = parameters(packed);
          streamingSlotCount = slotCount;
          streamingSlots = slots;
          const state = {
            status: integers('streamingStatus', slotCount),
            counts: integers('counts', slotCount),
            sums: scratchIntegers('sums', cellCount * slotCount * 3),
            moments: scratchIntegers('moments', cellCount * slotCount * 3)
          };
          session.graph.add(
            new GPURasterStreamingAccumulator({
              cellCount,
              chunkSliceCount: calibrationCount,
              slotCount,
              values: baselineValues,
              slotIndices: baselineSlots,
              parameters: accumulatorParameters,
              state
            })
          );
          session.graph.add(
            new GPURasterClimateAnomaly({
              cellCount,
              slotCount,
              chunkSliceCount: sliceCount,
              state,
              accumulatorParameters,
              values,
              slotIndices,
              standardized: number('standardized', 1, 0, 1) === 1,
              output: {
                climatologyMean: floats('mean', slotCount),
                standardDeviation: floats('standardDeviation', slotCount),
                ...(operation === 'climate-anomaly' ? {anomaly: floats('anomaly', sliceCount)} : {})
              }
            })
          );
        } else {
          const distribution = resolved['distribution'] ?? 'gamma';
          if (distribution !== 'gamma' && distribution !== 'fisk')
            throw new Error('distribution must be gamma or fisk');
          const zeroAtom = resolveGPURasterStandardizedIndexZeroAtom(
            `raster-timeseries-${operation}`,
            {
              zeroAtom:
                resolved.zeroAtom === undefined ? distribution === 'gamma' : resolved.zeroAtom
            }
          );
          const fit = floats('fit', slotCount * 3);
          const location = number('location', 0);
          session.graph.add(
            new GPURasterStandardizedIndexCalibration({
              cellCount,
              chunkSliceCount: calibrationCount,
              slotCount,
              distribution,
              zeroAtom,
              values: baselineValues,
              slotIndices: baselineSlots,
              parameters: parameters(
                getGPURasterStandardizedIndexCalibrationParameterValues(
                  {
                    reset: true,
                    maximumValueMagnitude: getCalibrationMagnitude(baseline, location),
                    location,
                    maximumStreamSliceCount: calibrationCount
                  },
                  calibrationCount
                )
              ),
              state: {
                counts: integers('counts', slotCount * 3),
                moments: scratchIntegers('calibration-moments', cellCount * slotCount * 9)
              },
              fit
            })
          );
          if (operation === 'standardized-index')
            session.graph.add(
              new GPURasterStandardizedIndex({
                cellCount,
                chunkSliceCount: sliceCount,
                slotCount,
                distribution,
                zeroAtom,
                values,
                slotIndices,
                fit,
                parameters: parameters(
                  getGPURasterStandardizedIndexParameterValues({location}, sliceCount)
                ),
                output: {index: floats('index', sliceCount), status: integers('status', sliceCount)}
              })
            );
        }
        break;
      }
      case 'resample': {
        const years = getGPURasterCentredFractionalYears(
          BigInt64Array.from(stack.times, BigInt),
          'millisecond'
        );
        const gridCount = count('gridCount', 24, 2, 256);
        session.graph.add(
          new GPURasterTemporalResample({
            ...common,
            gridCount,
            gridEnd: years[sliceCount - 1],
            dateAxis,
            parameters: parameters(
              getGPURasterTemporalResampleParameterValues({
                gridStart: years[0],
                gridStep: (years[sliceCount - 1] - years[0]) / (gridCount - 1),
                maximumGap: number('maximumGap', 1, 0)
              })
            ),
            output: {values: floats('resampled', gridCount), status: integers('status', gridCount)}
          })
        );
        break;
      }
      case 'lagged-correlation': {
        const reference = resolved['referenceIndex'];
        const lags = resolved['lags'];
        if (
          !Array.isArray(reference) ||
          reference.length !== sliceCount ||
          reference.some(value => !Number.isFinite(value))
        )
          throw new Error('referenceIndex needs one finite value per acquisition');
        if (
          !Array.isArray(lags) ||
          lags.length < 1 ||
          lags.length > 64 ||
          lags.some(value => !Number.isInteger(value) || Math.abs(value) >= sliceCount)
        )
          throw new Error('lags needs 1–64 integer sample offsets smaller than sliceCount');
        session.graph.add(
          new GPURasterLaggedCorrelation({
            ...common,
            referenceIndex: session.input('reference', Float32Array.from(reference), 'float32'),
            lags,
            parameters: parameters(
              getGPURasterLaggedCorrelationParameterValues({
                minimumPairCount: count('minimumPairCount', 3, 2, sliceCount)
              })
            ),
            output: {
              correlations: floats('correlations', lags.length),
              status: integers('status', lags.length),
              pairCount: integers('pairCount', lags.length)
            }
          })
        );
        break;
      }
      case 'snapshot-covariance':
      case 'eof': {
        const packed = getGPURasterSnapshotCovarianceParameterValues(
          {maximumValueMagnitude: maximumMagnitude},
          cellCount
        );
        const covarianceParameters = parameters(packed);
        const covarianceWords = session.output(
          'covariance-words',
          'uint32',
          ((sliceCount * (sliceCount + 1)) / 2) * 3
        );
        const keptCellCount = session.output('kept', 'uint32', 1);
        session.graph.add(
          new GPURasterSnapshotCovariance({
            ...common,
            parameters: covarianceParameters,
            output: {covarianceWords: covarianceWords.view, keptCellCount: keptCellCount.view}
          })
        );
        session.run();
        const words = await session.readUint32(covarianceWords);
        const kept = await session.readUint32(keptCellCount);
        readbackBytes += words.byteLength + kept.byteLength;
        const matrix = getGPURasterSnapshotCovarianceMatrix(words, sliceCount, packed[0]);
        columns['covariance'] = {values: matrix, components: sliceCount, scope: 'global'};
        columns['keptCellCount'] = {values: kept, components: 1, scope: 'global'};
        if (operation === 'eof') {
          const modeCount = count('modeCount', 2, 1, Math.min(32, sliceCount));
          const decomposition = getGPURasterSnapshotEOFDecomposition(matrix, sliceCount, modeCount);
          columns['explainedVarianceRatio'] = {
            values: decomposition.explainedVarianceRatio,
            components: modeCount,
            scope: 'global'
          };
          // A separate graph avoids resubmitting the covariance graph during projection.
          const projection = await withCatalogGpuSession(
            device,
            'raster-eof-projection',
            async projectionSession => {
              const eofs = projectionSession.output('eofs', 'float32', cellCount * modeCount);
              const status = projectionSession.output('status', 'uint32', cellCount);
              projectionSession.graph.add(
                new GPURasterEOFProjection({
                  cellCount,
                  sliceCount,
                  modeCount,
                  values: projectionSession.input('values', stack.values, 'float32'),
                  mask: projectionSession.input('mask', maskValues, 'uint32'),
                  parameters: projectionSession.input('parameters', packed, 'float32'),
                  temporalModes: projectionSession.input(
                    'temporalModes',
                    decomposition.temporalModes,
                    'float32'
                  ),
                  singularValues: projectionSession.input(
                    'singularValues',
                    decomposition.singularValues,
                    'float32'
                  ),
                  output: {eofs: eofs.view, status: status.view}
                })
              );
              projectionSession.run();
              return {
                eofs: await projectionSession.readFloat32(eofs),
                status: await projectionSession.readUint32(status)
              };
            }
          );
          columns['eofs'] = {values: projection.eofs, components: modeCount, scope: 'cell'};
          columns['status'] = {values: projection.status, components: 1, scope: 'cell'};
          readbackBytes += projection.eofs.byteLength + projection.status.byteLength;
        }
        return {columns, contributors: binding.contributors, readbackBytes, settings: resolved};
      }
      case 'climate-runs': {
        const threshold = number('threshold', 0);
        const packed = getGPURasterClimateRunAccumulatorParameterValues(
          {
            reset: true,
            threshold,
            maximumIncrement: maximumMagnitude + Math.abs(threshold),
            maximumStreamSliceCount: sliceCount
          },
          sliceCount
        );
        const degreeDays = session.output('degree-days', 'uint32', cellCount * 3);
        session.graph.add(
          new GPURasterClimateRunAccumulator({
            cellCount,
            chunkSliceCount: sliceCount,
            values,
            mask,
            parameters: parameters(packed),
            condition: 'above',
            minimumSpellLength: count('minimumSpellLength', 3, 1, 256),
            state: {runs: integers('runs', 6), degreeDays: degreeDays.view}
          })
        );
        session.run();
        const words = await session.readUint32(degreeDays);
        columns['degreeDays'] = {
          values: getGPURasterFixedPointValues(words, packed[3]),
          components: 1,
          scope: 'cell'
        };
        readbackBytes += words.byteLength;
        break;
      }
    }
    if (operation !== 'climate-runs') session.run();
    for (const entry of pending) {
      const values = entry.integer
        ? await session.readUint32(entry.output)
        : await session.readFloat32(entry.output);
      readbackBytes += values.byteLength;
      columns[entry.name] = {values, components: entry.components, scope: entry.scope};
    }
    const streamingWords = columns['streamingStatus']?.values;
    let streamingCountStatus: RasterTimeSeriesResult['streamingCountStatus'];
    if (streamingSlotCount > 0) {
      const expectedCount = cellCount * streamingSlotCount;
      validateRequirement(
        `raster-timeseries-${operation}`,
        streamingWords instanceof Uint32Array && streamingWords.length === expectedCount,
        `streamingStatus must contain exactly ${expectedCount} uint32 words; received ${streamingWords?.length ?? 0}`
      );
      // Decode each valid code once per result. Frozen records can safely be shared by all slots.
      const decodedStatuses = [
        Object.freeze(decodeGPURasterStreamingAccumulatorStatus(Uint32Array.of(0))),
        Object.freeze(decodeGPURasterStreamingAccumulatorStatus(Uint32Array.of(1)))
      ] as const;
      const statuses = new Array<{complete: boolean; overflow: boolean}>(expectedCount);
      // Forward the optional certificate to supporting anomaly producers. Keep host suppression
      // authoritative as well, using the submitted slot table rather than caller-owned dates.
      for (let entry = 0; entry < expectedCount; entry++) {
        const word = streamingWords[entry];
        const status =
          decodedStatuses[word] ??
          decodeGPURasterStreamingAccumulatorStatus(streamingWords.subarray(entry, entry + 1));
        statuses[entry] = status;
        if (status.complete) continue;
        for (const name of ['mean', 'standardDeviation']) {
          if (columns[name]) columns[name].values[entry] = NaN;
        }
        const anomaly = columns['anomaly'];
        if (anomaly) {
          const cell = Math.floor(entry / streamingSlotCount);
          const slot = entry % streamingSlotCount;
          for (let slice = 0; slice < sliceCount; slice++) {
            if (streamingSlots![slice] === slot) anomaly.values[cell * sliceCount + slice] = NaN;
          }
        }
      }
      streamingCountStatus = statuses;
    }
    return {
      columns,
      contributors: binding.contributors,
      readbackBytes,
      settings: resolved,
      ...(streamingCountStatus ? {streamingCountStatus} : {})
    };
  });
}
