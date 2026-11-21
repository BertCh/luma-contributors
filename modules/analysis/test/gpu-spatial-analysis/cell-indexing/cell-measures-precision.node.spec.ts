// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addCellChunkNodes} from '../cell-topology/gpu-cell-chunks.test-utils';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUCellMeasures,
  type GPUCellMeasuresProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {computeCellMeasuresOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {
  GPUCellMeasuresOperation,
  type CellMeasuresOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from '../../../src/testing/index';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

for (const family of ['h3', 's2', 'quadbin'] as const) {
  it(`${family} opts into published integer-controlled fp64 and retains the default float32 path`, () => {
    const device = createNullWebGPUDevice();
    try {
      const getSource = (precision?: GPUCellMeasuresProps['precision']) => {
        const graph = new GPUCommandGraph(device);
        const cells = createTransientView(graph, 'cells', 'uint32x2', 257);
        const output = {
          areas: createTransientView(graph, 'areas', 'float32', cells.length),
          perimeters: createTransientView(graph, 'perimeters', 'float32', cells.length),
          edgeLengths: createTransientView(graph, 'edges', 'float32', cells.length * 10),
          edgeCounts: createTransientView(graph, 'counts', 'uint32', cells.length),
          pentagons: createTransientView(graph, 'pentagons', 'uint32', cells.length)
        };
        const props: GPUCellMeasuresProps = {family, cells, maximumEdgeCount: 10, output};
        const contributor = new GPUCellMeasures({...props, precision});
        const [kernel] = getGPUCommandNodeKernelSources(addCellChunkNodes(graph, contributor));
        return kernel!.source;
      };
      const ordinary = getSource();
      expect(ordinary).toBe(getSource('float32'));
      expect(ordinary).not.toContain('fn sum_fp64(');
      const extended = getSource('double-single');
      for (const legacyPrecision of ['fp64', 'extended']) {
        expect(() => getSource(legacyPrecision as never)).toThrow(/precision.*double-single/);
      }
      expect(extended).toContain('fn fp64_two_sum_integer_bits(');
      expect(extended).not.toContain('fp64arithmetic.SPLIT');
      expect(extended).toContain('fn cellMeasuresInteger64(');
      // Parse the assembled shader, including the published module and all domain helpers.
      expect(() => new WgslReflect(extended)).not.toThrow();
      expect(() => getSource('double' as GPUCellMeasuresProps['precision'])).toThrow(
        /cell-measures precision.*float32.*double-single/
      );
    } finally {
      device.destroy();
    }
  });
}

it('operation lowering forwards precision, and the CPU oracle keeps binary64 for either choice', () => {
  const props: CellMeasuresOperationProps = {
    family: 'h3',
    precision: 'double-single',
    rowCount: 1,
    cells: 'cells',
    output: {areas: 'areas'}
  };
  // Independent h3-js cellArea('8ebedac6ecdb89f', 'm2') = 7.1095521951539355,
  // rounded once to float32 for the output layout: 7.109552383422852.
  for (const precision of [undefined, 'float32', 'double-single'] as const) {
    expect(() => new GPUCellMeasuresOperation({...props, precision})).not.toThrow();
    const result = computeCellMeasuresOnCPU(Uint32Array.of(0x6ecdb89f, 0x08ebedac), {
      family: 'h3',
      precision
    });
    expect(Number.isFinite(result.areas[0])).toBe(true);
    expect(result.areas[0]).toBe(7.109552383422852);
  }
  const contributor = constructNetworkOperationContributor(new GPUCellMeasuresOperation(props), {
    cells: {format: 'uint32x2', length: 1}
  }) as GPUCellMeasures;
  expect(contributor.props.precision).toBe('double-single');
  const cells = Uint32Array.of(0xffffffff, 0x8009fff);
  expect(() =>
    computeCellMeasuresOnCPU(cells, {family: 'h3', precision: 'double' as 'double-single'})
  ).toThrow(/cell-measures precision/);
});

it('the unit-sphere S2 compatibility API uses the same fp64 geometry and validates precision', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      cells: createTransientView(graph, 'cells', 'uint32x2', 1),
      output: {
        measures: createTransientView(graph, 'measures', 'float32x2', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    };
    for (const precision of ['double-single'] as const) {
      const selected = new GPUS2CellMeasures({...props, id: `precision-${precision}`, precision});
      expect(
        getGPUCommandNodeKernelSources(addCellChunkNodes(graph, selected))[0]!.source
      ).toContain('fn fp64_two_sum_integer_bits(');
    }
    const contributor = new GPUS2CellMeasures({...props, precision: 'double-single'});
    const [kernel] = getGPUCommandNodeKernelSources(addCellChunkNodes(graph, contributor));
    expect(kernel!.source).toContain('fn fp64_two_sum_integer_bits(');
    expect(() => new WgslReflect(kernel!.source)).not.toThrow();
    expect(() => new GPUS2CellMeasures({...props, precision: 'double' as 'double-single'})).toThrow(
      /s2-cell-measures precision/
    );
  } finally {
    device.destroy();
  }
});
