// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {
  AuditEmergingStage,
  AuditEmergingTrendOutputs
} from './audit-emerging-hot-spots-fixture';

/** Extracts the compensation code and adapts the interpreter's missing ldexp builtin. */
export function getInterpretedEmergingCompensationWGSL(source: string): string {
  const start = source.indexOf('fn getAdditionRoundingError(');
  const end = source.indexOf('fn getEmergingCoefficientScale(', start);
  if (start < 0 || end <= start) throw new Error('Gi* compensation declarations are missing');
  // The integer error helper scales by powers in [-126, 104], so exp2 is normal and exact.
  return `fn probeLdexp(value: f32, power: i32) -> f32 {
  return value * exp2(f32(power));
}
${source.slice(start, end).replaceAll('ldexp(', 'probeLdexp(')}`;
}

/** Executes short trend/classification kernels; never an expectation oracle. */
export function executeAuditEmergingStage(
  fixture: AuditEmergingStage,
  options: {roundCorrectionMeanDown?: boolean} = {}
): AuditEmergingTrendOutputs {
  if (
    fixture.sliceCount >= 64 ||
    (fixture.correction !== 'none' && !options.roundCorrectionMeanDown)
  ) {
    throw new Error(
      'The serial interpreter cannot execute cooperative barriers or the corrected Sen heap reliably'
    );
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const cells = fixture.scores.length / fixture.sliceCount;
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `emerging-stage-${serial++}`, format, length);
  const parameters = view('float32', 8);
  const output = {
    giZScores: view('float32', fixture.scores.length),
    trendZ: view('float32', cells),
    trendP: view('float32', cells),
    trendS: view('sint32', cells),
    category: view('uint32', cells),
    hotSliceCount: view('uint32', cells),
    coldSliceCount: view('uint32', cells),
    lastValidSlice: view('uint32', cells)
  };
  const memory = new Map<GraphDataView['buffer'], ArrayBuffer>();
  const getMemory = (field: GraphDataView) => {
    let bytes = memory.get(field.buffer);
    if (!bytes) {
      bytes = new ArrayBuffer(field.buffer.byteLength);
      new Uint32Array(bytes).fill(0x7f7f7f7f);
      memory.set(field.buffer, bytes);
    }
    return bytes;
  };
  new Float32Array(getMemory(parameters)).set(fixture.parameters);
  new Float32Array(getMemory(output.giZScores)).set(fixture.scores);
  try {
    const nodes = new GPUEmergingHotSpots({
      values: view('float32', fixture.scores.length),
      gridWidth: cells,
      gridHeight: 1,
      sliceCount: fixture.sliceCount,
      multipleTesting: 'none',
      finalSlice: fixture.finalSlice,
      trendAutocorrelationCorrection: fixture.correction,
      trendCorrectionMaximumLag: fixture.maximumLag || undefined,
      trendCorrectionCriticalZ: fixture.correctionCriticalZ,
      parameters,
      output
    })
      .getCommandNodes(graph)
      .filter(node => node.id.endsWith('-slice-counts') || node.id.endsWith('-mann-kendall'));
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id)!;
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const bytes = getMemory(binding.view);
          const range = getViewBindingRange(binding.view);
          return [
            index,
            binding.type === 'f32'
              ? new Float32Array(bytes, range.offset, range.size / 4)
              : binding.type === 'i32'
                ? new Int32Array(bytes, range.offset, range.size / 4)
                : new Uint32Array(bytes, range.offset, range.size / 4)
          ];
        })
      );
      // The interpreter needs an explicit three-axis workgroup attribute and omits phony reads.
      // WgslExec incorrectly infers an unsuffixed `var statistic = 0` as unsigned. WGSL
      // concretizes it as i32; the explicit suffix preserves the specified shader semantics.
      let source = record.source
        .replace('var statistic = 0;', 'var statistic = 0i;')
        .replace(/^\s*_\s*=[^;]*;/gm, '')
        .replace(/@workgroup_size\((\d+)\)/g, '@workgroup_size($1, 1, 1)');
      // A permitted one-ulp division error can leave identical nonzero centered residuals for
      // a constant signal on Metal. Inject that error only in this focused regression, whose
      // pairwise slopes are all -1 (the interpreter handles that heap, unlike general fixtures).
      if (options.roundCorrectionMeanDown) {
        source = source.replace(
          'seriesMean /= lengthValue;',
          'seriesMean = (seriesMean / lengthValue) * 0.99999994;'
        );
      }
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: {...record.constants}}
      );
    }
    const result = {} as AuditEmergingTrendOutputs;
    for (const port of [
      'trendZ',
      'trendP',
      'trendS',
      'category',
      'hotSliceCount',
      'coldSliceCount',
      'lastValidSlice'
    ] as const) {
      const field = output[port];
      const bytes = getMemory(field);
      result[port] = Array.from(
        field.format === 'float32'
          ? new Float32Array(bytes, field.byteOffset, field.length)
          : field.format === 'sint32'
            ? new Int32Array(bytes, field.byteOffset, field.length)
            : new Uint32Array(bytes, field.byteOffset, field.length)
      );
    }
    return result;
  } finally {
    device.destroy();
  }
}
