// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURasterLaggedCorrelation} from '../../../src/raster-analysis/raster-time-series/gpu-raster-lagged-correlation';
import {GPURasterMedoidComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite';
import {GPURasterPhenologyMetrics} from '../../../src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics';
import {GPURasterRollingMean} from '../../../src/raster-analysis/raster-time-series/gpu-raster-rolling-mean';
import {GPURasterStandardizedIndex} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {GPURasterStandardizedIndexCalibration} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {GPURasterStreamingAccumulator} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {GPURasterTemporalResample} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {GPURasterWhittakerSmoother} from '../../../src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother';
import {GPURasterTemporal} from '../../../src/raster-analysis/remote-sensing/gpu-raster-temporal';
import {GPURasterSeasonalBreak} from '../../../src/raster-analysis/remote-sensing/gpu-raster-seasonal-break';
import {GPUChangeDetection} from '../../../src/raster-analysis/change-detection/gpu-change-detection';
import type {GPURasterIrregularDateAxis} from '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis';

it('exposes omitted optional props and nested fields with their original packed types for every converted class', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `float-${serial++}`, 'float32', length);
    const integers = (length: number) =>
      createTransientView(graph, `integer-${serial++}`, 'uint32', length);
    const values = floats(5);
    const parameters = floats(7);
    const dateAxis: GPURasterIrregularDateAxis = {
      acquisitionTimes: createTransientView(graph, 'dates', 'uint32x2', 5),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    };
    const lagged = new GPURasterLaggedCorrelation({
      cellCount: 1,
      sliceCount: 5,
      values,
      referenceIndex: floats(5),
      parameters,
      lags: [0],
      output: {correlations: floats(1), status: integers(1)}
    });
    expectTypeOf(lagged.props.mask).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    expectTypeOf(lagged.props.output.pairCount).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    const medoid = new GPURasterMedoidComposite({
      cellCount: 1,
      sliceCount: 5,
      bandCount: 1,
      values,
      output: {composite: floats(1), sliceIndex: integers(1), status: integers(1)}
    });
    expectTypeOf(medoid.props.mask).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    const phenology = new GPURasterPhenologyMetrics({
      cellCount: 1,
      sliceCount: 5,
      maximumSeasonCount: 1,
      smoothed: values,
      parameters,
      output: {
        metrics: floats(10),
        flags: integers(1),
        seasonCount: integers(1),
        status: integers(1)
      }
    });
    expectTypeOf(phenology.props.dateAxis).toEqualTypeOf<GPURasterIrregularDateAxis | undefined>();
    const rolling = new GPURasterRollingMean({
      cellCount: 1,
      chunkSliceCount: 5,
      windowLength: 3,
      values,
      parameters,
      ring: floats(3),
      seenCount: integers(1),
      output: floats(5)
    });
    expectTypeOf(rolling.props.mask).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    const standardized = new GPURasterStandardizedIndex({
      cellCount: 1,
      chunkSliceCount: 5,
      distribution: 'fisk',
      values,
      parameters,
      fit: floats(3),
      output: {index: floats(5), status: integers(5)}
    });
    expectTypeOf(standardized.props.mask).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    expectTypeOf(standardized.props.zeroInflated).toEqualTypeOf<boolean | undefined>();
    const calibration = new GPURasterStandardizedIndexCalibration({
      cellCount: 1,
      chunkSliceCount: 5,
      distribution: 'fisk',
      values,
      parameters,
      state: {counts: integers(3), moments: integers(9)},
      fit: floats(3)
    });
    expectTypeOf(calibration.props.mask).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    expectTypeOf(calibration.props.slotIndices).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    const streaming = new GPURasterStreamingAccumulator({
      cellCount: 1,
      chunkSliceCount: 5,
      values,
      parameters,
      state: {counts: integers(1), sums: integers(3)}
    });
    expectTypeOf(streaming.props.state.moments).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf(streaming.props.state.extrema).toEqualTypeOf<
      GraphDataView<'float32'> | undefined
    >();
    const resample = new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 5,
      gridCount: 5,
      values,
      parameters,
      dateAxis,
      output: {values: floats(5), status: integers(5)}
    });
    expectTypeOf(resample.props.mask).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    const smoother = new GPURasterWhittakerSmoother({
      cellCount: 1,
      sliceCount: 5,
      values,
      parameters,
      output: {smoothed: floats(5), status: integers(1)}
    });
    expectTypeOf(smoother.props.dateAxis).toEqualTypeOf<GPURasterIrregularDateAxis | undefined>();
    expectTypeOf(smoother.props.qualityWeights).toEqualTypeOf<
      GraphDataView<'float32'> | undefined
    >();
    const temporal = new GPURasterTemporal({
      cellCount: 1,
      sliceCount: 5,
      values,
      operation: 'mean',
      output: {
        values: floats(1),
        counts: integers(1),
        chosenSlice: integers(1),
        status: integers(1)
      }
    });
    expectTypeOf(temporal.props.dateAxis).toEqualTypeOf<GPURasterIrregularDateAxis | undefined>();
    const seasonal = new GPURasterSeasonalBreak({
      cellCount: 1,
      sliceCount: 12,
      values: floats(12),
      period: 4,
      minimumSegmentLength: 3,
      output: {
        coefficients: floats(5),
        components: floats(36),
        breakIndex: integers(1),
        improvement: floats(1),
        status: integers(1)
      }
    });
    expectTypeOf(seasonal.props.output.diagnostics).toEqualTypeOf<
      GraphDataView<'float32'> | undefined
    >();
    const change = new GPUChangeDetection({
      cellCount: 1,
      sliceCount: 5,
      slices: values,
      parameters,
      output: {difference: floats(1)}
    });
    expectTypeOf(change.props.output.tPValue).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
    expectTypeOf(change.props.dateAxis).toEqualTypeOf<GPURasterIrregularDateAxis | undefined>();
    expect([
      lagged.props.mask,
      medoid.props.mask,
      phenology.props.dateAxis,
      rolling.props.mask,
      standardized.props.zeroInflated,
      calibration.props.slotIndices,
      streaming.props.state.moments,
      resample.props.mask,
      smoother.props.dateAxis,
      temporal.props.dateAxis,
      seasonal.props.output.diagnostics,
      change.props.output.tPValue
    ]).toEqual(Array(12).fill(undefined));

    const vector = <Format extends 'float32' | 'uint32'>(view: GraphDataView<Format>) =>
      new GraphVectorView({
        id: `vector-${serial++}`,
        name: 'vector',
        format: view.format,
        length: view.length,
        valueLength: view.length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: [view]
      });
    const chunked = new GPURasterWhittakerSmoother({
      cellCount: 1,
      sliceCount: 5,
      values: vector(values),
      parameters,
      output: {smoothed: vector(floats(5)), status: vector(integers(1))}
    });
    expectTypeOf(chunked.props.mask).toEqualTypeOf<
      GraphVectorView<'uint32'> | GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf(chunked.props.output.smoothed).toEqualTypeOf<
      GraphVectorView<'float32'> | GraphDataView<'float32'>
    >();
    expectTypeOf(chunked.props.dateAxis).toEqualTypeOf<GPURasterIrregularDateAxis | undefined>();
    const chunkedLagged = new GPURasterLaggedCorrelation({
      cellCount: 1,
      sliceCount: 5,
      values: vector(values),
      referenceIndex: floats(5),
      parameters,
      lags: [0],
      output: {correlations: vector(floats(1)), status: vector(integers(1))}
    });
    expectTypeOf(chunkedLagged.props.output.pairCount).toEqualTypeOf<
      GraphVectorView<'uint32'> | GraphDataView<'uint32'> | undefined
    >();
  } finally {
    device.destroy();
  }
});
