// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expectTypeOf, it} from 'vitest';
import {
  GPUCalendarBuckets,
  type GPUCalendarBucketsOutput,
  type GPUCalendarBucketsProps
} from '../../../src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets';
import {
  GPUTimeWordsRelative,
  type GPUTimeWordsRelativeProps
} from '../../../src/dataframe-analysis/time-words-relative/gpu-time-words-relative';
import {
  GPUPeriodIndex,
  type GPUPeriodIndexOutput
} from '../../../src/dataframe-analysis/period-index/gpu-period-index';
import {
  GPUCellPeriodKeys,
  type GPUCellPeriodKeysOutput
} from '../../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {
  GPUClassAssignment,
  type GPUClassAssignmentProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {
  GPUFittingCoordinateEncoder,
  type GPUFittingCoordinateEncoderOutput
} from '../../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {
  GPUForestInference,
  type GPUForestInferenceOutput
} from '../../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

it('keeps exported props and output aliases packed by default', () => {
  expectTypeOf<GPUCalendarBucketsOutput['year']>().toEqualTypeOf<
    GraphDataView<'sint32'> | undefined
  >();
  expectTypeOf<GPUCalendarBucketsProps['timestamps']>().toEqualTypeOf<GraphDataView<'uint32x2'>>();
  expectTypeOf<GPUTimeWordsRelativeProps['mask']>().toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
  expectTypeOf<GPUTimeWordsRelativeProps['output']>().toEqualTypeOf<GraphDataView<'float32'>>();
  expectTypeOf<GPUPeriodIndexOutput['periods']>().toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf<GPUCellPeriodKeysOutput['keys']>().toEqualTypeOf<GraphDataView<'uint32x2'>>();
  expectTypeOf<GPUClassAssignmentProps['output']>().toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf<GPUFittingCoordinateEncoderOutput['features']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
  expectTypeOf<GPUForestInferenceOutput['predictions']>().toEqualTypeOf<GraphDataView<'float32'>>();
});

it('preserves inferred packed and chunked ports, including omitted optional fields', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const packed = <Format extends 'float32' | 'uint32' | 'uint32x2' | 'sint32'>(
      format: Format,
      length = 1
    ) => createTransientView(graph, `view-${serial++}`, format, length);
    const chunked = <Format extends 'float32' | 'uint32' | 'uint32x2' | 'sint32'>(
      format: Format,
      length = 1
    ) => createVectorView(`chunks-${serial++}`, format, [packed(format, length)]);
    const temporalContext = {
      unit: 'millisecond',
      epoch: 'unix',
      encoding: 'int64-words',
      axis: {kind: 'irregular'}
    } as const;
    const calendarPacked = new GPUCalendarBuckets({
      timestamps: packed('uint32x2'),
      parameters: packed('sint32', 2),
      output: {year: packed('sint32')}
    });
    const calendarChunked = new GPUCalendarBuckets({
      timestamps: chunked('uint32x2'),
      parameters: packed('sint32', 2),
      output: {year: chunked('sint32')}
    });
    expectTypeOf(calendarPacked.props.timestamps).toExtend<GraphDataView<'uint32x2'>>();
    expectTypeOf(calendarPacked.props.output.year).toExtend<GraphDataView<'sint32'>>();
    expectTypeOf(calendarPacked.props.output.month).toExtend<GraphDataView<'uint32'> | undefined>();
    expectTypeOf(calendarChunked.props.timestamps).toExtend<GraphVectorView<'uint32x2'>>();
    expectTypeOf(calendarChunked.props.output.year).toExtend<GraphVectorView<'sint32'>>();
    expectTypeOf(calendarChunked.props.output.month).toExtend<
      GraphVectorView<'uint32'> | undefined
    >();
    const relativePacked = new GPUTimeWordsRelative({
      timestamps: packed('uint32x2'),
      temporalContext,
      origin: 0n,
      output: packed('float32')
    });
    const relativeChunked = new GPUTimeWordsRelative({
      timestamps: chunked('uint32x2'),
      temporalContext,
      origin: 0n,
      output: chunked('float32')
    });
    expectTypeOf(relativePacked.props.output).toExtend<GraphDataView<'float32'>>();
    expectTypeOf(relativeChunked.props.output).toExtend<GraphVectorView<'float32'>>();
    const periods = {kind: 'fixed', width: 1n, origin: 0n, count: 1} as const;
    const periodPacked = new GPUPeriodIndex({
      timestamps: packed('uint32x2'),
      temporalContext,
      periods,
      output: {periods: packed('uint32')}
    });
    const periodChunked = new GPUPeriodIndex({
      timestamps: chunked('uint32x2'),
      temporalContext,
      periods,
      output: {periods: chunked('uint32')}
    });
    expectTypeOf(periodPacked.props.output.periods).toExtend<GraphDataView<'uint32'>>();
    expectTypeOf(periodChunked.props.output.periods).toExtend<GraphVectorView<'uint32'>>();
    const keysPacked = new GPUCellPeriodKeys({
      cells: packed('uint32x2'),
      periods: packed('uint32'),
      periodCount: 1,
      fill: 'zeros',
      layout: 'zero-period-bit-identity',
      output: {keys: packed('uint32x2')}
    });
    const keysChunked = new GPUCellPeriodKeys({
      cells: chunked('uint32x2'),
      periods: chunked('uint32'),
      periodCount: 1,
      fill: 'zeros',
      layout: 'zero-period-bit-identity',
      output: {keys: chunked('uint32x2')}
    });
    expectTypeOf(keysPacked.props.output.keys).toExtend<GraphDataView<'uint32x2'>>();
    expectTypeOf(keysChunked.props.output.keys).toExtend<GraphVectorView<'uint32x2'>>();
    const classesPacked = new GPUClassAssignment({
      values: packed('float32'),
      breaks: packed('float32', 2),
      classCount: packed('uint32'),
      output: packed('uint32')
    });
    const classesChunked = new GPUClassAssignment({
      values: chunked('float32'),
      breaks: packed('float32', 2),
      classCount: packed('uint32'),
      output: chunked('uint32')
    });
    expectTypeOf(classesPacked.props.output).toExtend<GraphDataView<'uint32'>>();
    expectTypeOf(classesChunked.props.output).toExtend<GraphVectorView<'uint32'>>();
    expectTypeOf(classesPacked.classifiedValues.classes).toExtend<GraphDataView<'uint32'>>();
    expectTypeOf(classesChunked.classifiedValues.classes).toExtend<GraphVectorView<'uint32'>>();
    const encoding = {
      encoding: 'fourier',
      rowCount: 1,
      dimensions: 2,
      frequencyCount: 1,
      frequencies: packed('float32', 2)
    } as const;
    const encoderPacked = new GPUFittingCoordinateEncoder({
      ...encoding,
      coordinates: packed('float32', 2),
      output: {features: packed('float32', 2), validity: packed('uint32')}
    });
    const encoderChunked = new GPUFittingCoordinateEncoder({
      ...encoding,
      coordinates: chunked('float32', 2),
      output: {features: chunked('float32', 2), validity: chunked('uint32')}
    });
    expectTypeOf(encoderPacked.props.output.features).toExtend<GraphDataView<'float32'>>();
    expectTypeOf(encoderChunked.props.output.features).toExtend<GraphVectorView<'float32'>>();
    const forest = {
      rowCount: 1,
      model: {
        featureCount: 1,
        treeCount: 1,
        nodeCount: 1,
        outputCount: 1,
        maximumDepth: 0,
        task: 'regression',
        transform: 'identity'
      },
      nodes: packed('uint32', 6),
      trees: packed('uint32', 4),
      leafValues: packed('float32'),
      parameters: packed('float32', 2)
    } as const;
    const forestPacked = new GPUForestInference({
      ...forest,
      features: packed('float32'),
      output: {predictions: packed('float32'), validity: packed('uint32')}
    });
    const forestChunked = new GPUForestInference({
      ...forest,
      features: chunked('float32'),
      output: {predictions: chunked('float32'), validity: chunked('uint32')}
    });
    expectTypeOf(forestPacked.props.output.predictions).toExtend<GraphDataView<'float32'>>();
    expectTypeOf(forestPacked.props.output.scores).toExtend<GraphDataView<'float32'> | undefined>();
    expectTypeOf(forestChunked.props.output.predictions).toExtend<GraphVectorView<'float32'>>();
    expectTypeOf(forestChunked.props.output.scores).toExtend<
      GraphVectorView<'float32'> | undefined
    >();
    const rasterChunked = new GPUForestInference({
      ...forest,
      rasterStack: {
        bands: [
          {id: 'band', format: 'float32', storage: {kind: 'buffer', values: packed('float32')}}
        ],
        grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'}
      },
      output: {predictions: chunked('float32'), validity: chunked('uint32')}
    });
    expectTypeOf(rasterChunked.props.output.predictions).toExtend<GraphVectorView<'float32'>>();
  } finally {
    device.destroy();
  }
});
