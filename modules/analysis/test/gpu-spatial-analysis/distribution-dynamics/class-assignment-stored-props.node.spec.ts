// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUClassAssignment,
  type GPUClassAssignmentProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

// Backs the stored-option example of docs/api/gpu-spatial-analysis/operators-dynamics.md.
it('keeps a satisfies-checked stored closure and rejects unknown keys', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const values = createTransientView(graph, 'values', 'float32', 1);
    const breaks = createTransientView(graph, 'breaks', 'float32', 3);
    const classCount = createTransientView(graph, 'class-count', 'uint32', 1);
    // The edge results (value 1 on edges [0, 1, 2]: left 1, right 0) execute in the GPU spec
    // class-assignment-closure-w3.spec.ts; here the stored option object is checked as written.
    for (const intervalClosure of ['left', 'right'] as const) {
      const output = createTransientView(graph, `classes-${intervalClosure}`, 'uint32', 1);
      const props = {
        id: `assign-${intervalClosure}`,
        values,
        breaks,
        classCount,
        intervalClosure,
        output
      } satisfies GPUClassAssignmentProps;
      const contributor = new GPUClassAssignment(props);
      expect(contributor.props.intervalClosure).toBe(intervalClosure);
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
    }
    expect(
      () =>
        new GPUClassAssignment({
          values,
          breaks,
          classCount,
          intervalClosure: 'center' as 'left',
          output: createTransientView(graph, 'classes-center', 'uint32', 1)
        })
    ).toThrow("intervalClosure must be 'left' or 'right'");
    const widened: GPUClassAssignmentProps = {
      values,
      breaks,
      classCount,
      output: createTransientView(graph, 'classes-typo', 'uint32', 1)
    };
    // A misspelled key behind a widened type is rejected at run time with the accepted keys.
    const misspelled = {...widened, intervalClosue: 'right'};
    expect(() => new GPUClassAssignment(misspelled as GPUClassAssignmentProps)).toThrow(
      'class-assignment unknown prop intervalClosue; accepted keys: id, values, breaks, classCount, mask, intervalClosure, output'
    );
  } finally {
    device.destroy();
  }
});
