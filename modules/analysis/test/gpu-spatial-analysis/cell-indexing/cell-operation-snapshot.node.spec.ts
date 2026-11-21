// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUPointToCell} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell';
import {GPUCellMeasures} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell';
import {CPUPointToCellOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {CPUCellMeasuresOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {CPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';
import {GraphDataView, type GPUCPUVector, type GraphVectorView} from '@luma.gl/graph/gpu-core';

/** Reads the buffer ID behind a port view, which these specs always build as a data view. */
function getPortBufferId(view: GraphDataView | GraphVectorView | undefined): string | undefined {
  if (view === undefined) {
    return undefined;
  }
  if (!(view instanceof GraphDataView)) {
    throw new Error('Expected the port to be a GraphDataView');
  }
  return view.buffer.id;
}
function createCPUVector(
  format: 'float32x2' | 'uint32x2',
  values: Float32Array | Uint32Array
): GPUCPUVector {
  const length = values.length / 2;
  return {format, length, chunks: [{values, length}], pack: () => values};
}
import {GPUPointToCellOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation';
import {GPUCellMeasuresOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {GPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

it('operation closures keep their original cell configuration and output selections (DGGS-115)', async () => {
  const pointProps = {
    family: 'quadbin' as const,
    resolution: 2,
    positions: 'positions',
    rowCount: 1,
    output: {cells: 'cells', validity: 'validity' as string | undefined}
  };
  const point = new GPUPointToCellOperation(pointProps);
  const cpuPoint = new CPUPointToCellOperation(pointProps);
  pointProps.resolution = 99;
  pointProps.output.validity = undefined;
  const pointContributor = constructNetworkOperationContributor(point, {
    positions: {format: 'float32x2', length: 1}
  }) as GPUPointToCell;
  expect(getPortBufferId(pointContributor.props.output.validity)).toBe('validity');
  expect(getPortBufferId(pointContributor.props.output.cells)).toBe('cells');
  const pointResult = await cpuPoint.execute({
    inputs: {positions: createCPUVector('float32x2', Float32Array.of(0, 0))}
  });
  expect(Object.keys(pointResult).sort()).toEqual(['cells', 'validity']);
  // Zoom 2: equator tile (2,2), Morton 12, Quadbin header/footer gives 0x482cffffffffffff.
  expect(Array.from(pointResult['cells'] as Uint32Array)).toEqual([0xffffffff, 0x482cffff]);
  expect(Array.from(pointResult['validity'] as Uint32Array)).toEqual([1]);
  const measuresProps = {
    family: 'h3' as const,
    cells: 'cells',
    rowCount: 1,
    maximumEdgeCount: 6,
    output: {areas: 'areas' as string | undefined, edgeCounts: 'edgeCounts'}
  };
  const measures = new GPUCellMeasuresOperation(measuresProps);
  const cpuMeasures = new CPUCellMeasuresOperation(measuresProps);
  measuresProps.maximumEdgeCount = 99;
  measuresProps.output.areas = undefined;
  const measuresContributor = constructNetworkOperationContributor(measures, {
    cells: {format: 'uint32x2', length: 1}
  }) as GPUCellMeasures;
  expect(getPortBufferId(measuresContributor.props.output.areas)).toBe('areas');
  expect(getPortBufferId(measuresContributor.props.output.edgeCounts)).toBe('edgeCounts');
  const measuresResult = await cpuMeasures.execute({
    inputs: {cells: createCPUVector('uint32x2', Uint32Array.of(0, 0))}
  });
  expect(Object.keys(measuresResult).sort()).toEqual(['areas', 'edgeCounts']);
  expect(Array.from(measuresResult['areas'] as Float32Array)).toEqual([0]);
  expect(Array.from(measuresResult['edgeCounts'] as Uint32Array)).toEqual([0]);
  const healpixProps = {
    order: 0,
    scheme: 'uniq' as const,
    positions: 'positions',
    rowCount: 1,
    output: {cells: 'cells', validity: 'validity' as string | undefined}
  };
  const healpix = new GPUHealpixPointToCellOperation(healpixProps);
  const cpuHealpix = new CPUHealpixPointToCellOperation(healpixProps);
  healpixProps.order = 99;
  healpixProps.output.validity = undefined;
  const healpixContributor = constructNetworkOperationContributor(healpix, {
    positions: {format: 'float32x2', length: 1}
  }) as GPUHealpixPointToCell;
  expect(getPortBufferId(healpixContributor.props.output.validity)).toBe('validity');
  expect(getPortBufferId(healpixContributor.props.output.cells)).toBe('cells');
  const healpixResult = await cpuHealpix.execute({
    inputs: {positions: createCPUVector('float32x2', Float32Array.of(45, 45))}
  });
  expect(Object.keys(healpixResult).sort()).toEqual(['cells', 'validity']);
  expect(Array.from(healpixResult['cells'] as Uint32Array)).toEqual([4, 0]);
  expect(Array.from(healpixResult['validity'] as Uint32Array)).toEqual([1]);
});

it('DGN-13 captures longitude policy in both named forms', async () => {
  const props = {
    family: 'quadkey' as const,
    resolution: 5,
    longitudePolicy: 'wrap' as 'wrap' | 'family-default',
    positions: 'positions',
    rowCount: 1,
    output: {cells: 'cells'}
  };
  const gpu = new GPUPointToCellOperation(props);
  const cpu = new CPUPointToCellOperation(props);
  props.longitudePolicy = 'family-default';
  const contributor = constructNetworkOperationContributor(gpu, {
    positions: {format: 'float32x2', length: 1}
  }) as GPUPointToCell;
  expect(contributor.props.longitudePolicy).toBe('wrap');
  const rows = await cpu.execute({
    inputs: {positions: createCPUVector('float32x2', Float32Array.of(210, 0))}
  });
  // Longitude -150: tile (2,16), interleaved path 516 at zoom 5.
  expect(rows['cells']).toEqual(Uint32Array.of(516, 0x14000000));
});
