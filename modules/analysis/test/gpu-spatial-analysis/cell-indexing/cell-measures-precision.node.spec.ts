// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
      const getSource = (precision?: GPUCellMeasuresProps['precision']) => {
        const contributor = new GPUCellMeasures({...props, precision});
        const [kernel] = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        return kernel!.source;
      };
      const ordinary = getSource();
      expect(ordinary).toBe(getSource('float32'));
      expect(ordinary).not.toContain('fn sum_fp64(');
      const extended = getSource('fp64');
      expect(extended).toContain('fn fp64_two_sum_integer_bits(');
      expect(extended).not.toContain('fp64arithmetic.SPLIT');
      expect(extended).toContain('fn cellMeasuresInteger64(');
      // Parse the assembled shader, including the published module and all domain helpers.
      expect(() => new WgslReflect(extended)).not.toThrow();
      expect(() => getSource('double' as GPUCellMeasuresProps['precision'])).toThrow(
        /cell-measures precision must be 'float32' or 'fp64'/
      );
    } finally {
      device.destroy();
    }
  });
}

it('operation lowering forwards precision, and the CPU oracle keeps binary64 for either choice', () => {
  const props: CellMeasuresOperationProps = {
    family: 'h3',
    precision: 'fp64',
    rowCount: 1,
    cells: 'cells',
    output: {areas: 'areas'}
  };
  const contributor = constructNetworkOperationContributor(new GPUCellMeasuresOperation(props), {
    cells: {format: 'uint32x2', length: 1}
  }) as GPUCellMeasures;
  expect(contributor.props.precision).toBe('fp64');
  const cells = Uint32Array.of(0xffffffff, 0x8009fff);
  expect(computeCellMeasuresOnCPU(cells, {family: 'h3', precision: 'fp64'})).toEqual(
    computeCellMeasuresOnCPU(cells, {family: 'h3', precision: 'float32'})
  );
  expect(() =>
    computeCellMeasuresOnCPU(cells, {family: 'h3', precision: 'double' as 'fp64'})
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
    const contributor = new GPUS2CellMeasures({...props, precision: 'fp64'});
    const [kernel] = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernel!.source).toContain('fn fp64_two_sum_integer_bits(');
    expect(() => new WgslReflect(kernel!.source)).not.toThrow();
    expect(() => new GPUS2CellMeasures({...props, precision: 'double' as 'fp64'})).toThrow(
      /s2-cell-measures precision/
    );
  } finally {
    device.destroy();
  }
});
