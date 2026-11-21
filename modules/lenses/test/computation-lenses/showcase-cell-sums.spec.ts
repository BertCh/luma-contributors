// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellAggregation, GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, test} from 'vitest';
import {
  GPUColorScale,
  getGPUColorScaleParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  addShowcaseCellTableColumnsNode,
  addShowcaseCertifiedSumCountsNode
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/cells/cell-pyramid-sums';

test('DGGS-102: actual showcase sum selection masks saturation, preserves counts and clears on rerun', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const output = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format.endsWith('x2') ? 2 : 1;
    const buffer = device.createBuffer({
      id: name,
      byteLength: length * components * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const view = graph.createDataView(
      graph.importBuffer({id: name, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format, length}
    );
    return {buffer, view};
  };
  const positions = output('positions', 'float32x2', 2);
  positions.buffer.write(Float32Array.of(10, 10, 10, 10));
  const values = output('values', 'float32', 2);
  const keys = output('keys', 'uint32x2', 2);
  const cells = output('cells', 'uint32x2', 4);
  const counts = output('counts', 'uint32', 4);
  const sums = output('sums', 'uint32x2', 4);
  const sumValues = output('sum-values', 'float32', 4);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const candidateOverflow = output('candidate-overflow', 'uint32', 1);
  const sumOverflow = output('sum-overflow', 'uint32', 1);
  const selectedSums = output('selected-sums', 'float32', 4);
  const sumMask = output('sum-mask', 'uint32', 4);
  const selectedCounts = output('selected-counts', 'float32', 4);
  const certifiedSumCounts = output('certified-sum-counts', 'uint32', 4);
  const countMask = output('count-mask', 'uint32', 4);
  const status = {
    count: count.view,
    overflow: overflow.view,
    candidateOverflow: candidateOverflow.view,
    sumOverflow: sumOverflow.view
  };
  graph.add(
    new GPUPointToCell({
      family: 'quadbin',
      resolution: 6,
      positions: positions.view,
      output: {cells: keys.view}
    })
  );
  graph.add(
    new GPUCellAggregation({
      family: 'quadbin',
      resolution: 6,
      cells: keys.view,
      values: values.view,
      output: {
        cells: cells.view,
        counts: counts.view,
        sums: sums.view,
        sumValues: sumValues.view,
        status
      }
    })
  );
  addShowcaseCellTableColumnsNode(
    graph,
    'sum-selection',
    {counts: counts.view, sumValues: sumValues.view, status},
    {values: selectedSums.view, mask: sumMask.view}
  );
  addShowcaseCellTableColumnsNode(
    graph,
    'count-selection',
    {counts: counts.view, status},
    {values: selectedCounts.view, mask: countMask.view}
  );
  addShowcaseCertifiedSumCountsNode(
    graph,
    'sum-style-counts',
    counts.view,
    status,
    certifiedSumCounts.view
  );
  const colors = output('colors', 'uint32', 4);
  const domain = output('domain', 'float32', 2);
  domain.buffer.write(Float32Array.of(0, 10));
  const palette = output('palette', 'uint32', 2);
  palette.buffer.write(Uint32Array.of(0xff0000ff, 0xff0000ff));
  const parameterValues = getGPUColorScaleParameterValues({
    scale: 'linear',
    domainCount: 2,
    paletteCount: 2,
    noDataColor: 0
  });
  const parameters = output('color-parameters', 'float32', parameterValues.length);
  parameters.buffer.write(parameterValues);
  graph.add(
    new GPUColorScale({
      values: selectedSums.view,
      mask: sumMask.view,
      domain: domain.view,
      palette: palette.view,
      parameters: parameters.view,
      maximumDomainCount: 2,
      maximumPaletteCount: 2,
      output: {colors: colors.view}
    })
  );
  const compiled = await graph.compileAsync();
  const readWords = async (buffer: Buffer) => {
    const bytes = await buffer.readAsync();
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  };
  const readFloats = async (buffer: Buffer) => {
    const bytes = await buffer.readAsync();
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  };
  try {
    // 2+4=6 exactly; 2^47 exceeds a row's fixed-point headroom at scale 65536.
    // Four output slots and only one occupied cell keep all capacity words zero.
    for (const value of [2, 2 ** 47, 2]) {
      // Poison every destination before each run. A selector that skips inactive tails
      // must fail even on the first submission, and after a saturated submission.
      selectedSums.buffer.write(Float32Array.of(91, 92, 93, 94));
      selectedCounts.buffer.write(Float32Array.of(81, 82, 83, 84));
      sumMask.buffer.write(Uint32Array.of(7, 7, 7, 7));
      countMask.buffer.write(Uint32Array.of(7, 7, 7, 7));
      certifiedSumCounts.buffer.write(Uint32Array.of(9, 9, 9, 9));
      colors.buffer.write(Uint32Array.of(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff));
      values.buffer.write(Float32Array.of(value, 4));
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const saturated = value > 2;
      expect(Array.from(await readWords(overflow.buffer))).toEqual([0]);
      expect(Array.from(await readWords(candidateOverflow.buffer))).toEqual([0]);
      expect(Array.from(await readWords(sumOverflow.buffer))).toEqual([Number(saturated)]);
      expect(Array.from(await readWords(sumMask.buffer))).toEqual([saturated ? 0 : 1, 0, 0, 0]);
      expect(Array.from(await readFloats(selectedSums.buffer))).toEqual([
        saturated ? 0 : 6,
        0,
        0,
        0
      ]);
      expect(Array.from(await readWords(colors.buffer))).toEqual([
        saturated ? 0 : 0xff0000ff,
        0,
        0,
        0
      ]);
      expect(Array.from(await readWords(certifiedSumCounts.buffer))).toEqual([
        saturated ? 0 : 2,
        0,
        0,
        0
      ]);
      expect(Array.from(await readWords(countMask.buffer))).toEqual([1, 0, 0, 0]);
      expect(Array.from(await readFloats(selectedCounts.buffer))).toEqual([2, 0, 0, 0]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
