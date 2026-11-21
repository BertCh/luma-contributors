// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from './spatial-weights-harness';
import {
  createLagAuditCase,
  LAG_AUDIT_EXPECTED,
  type LagAuditOptions
} from './audit-weights-a-lag-fixture';

it('whole/chunked cooperative lag matches dense NumPy literals for scalar/vector float/count inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scenario of Object.keys(LAG_AUDIT_EXPECTED) as LagAuditOptions['scenario'][])
    for (const chunked of [false, true])
      for (const normalize of [false, true])
        for (const columnCount of [1, 2] as const)
          for (const valueFormat of ['float32', 'uint32'] as const) {
            const rig = new WeightsRig(device);
            const backing = rig.output('float32', 8 * columnCount + 2);
            backing.buffer.write(new Float32Array(8 * columnCount + 2).fill(-999));
            const fixture = createLagAuditCase(
              {
                input: (data, format) => rig.input(data, format, data.length),
                output: length =>
                  rig.graph.createDataView(backing.view.buffer, {
                    format: 'float32',
                    byteOffset: 4,
                    length
                  })
              },
              {scenario, chunked, normalize, columnCount, valueFormat}
            );
            try {
              rig.run(fixture.contributor);
              const actual = await readFloat32(backing.buffer, 8 * columnCount + 2);
              expect(actual[0]).toBe(-999);
              expect(actual[actual.length - 1]).toBe(-999);
              for (const [row, expected] of fixture.expected.entries()) {
                expect(Number.isFinite(actual[row + 1])).toBe(true);
                // Integer sums are exact; one f32 division at magnitude <=41 needs <=2e-6.
                expect(Math.abs(actual[row + 1] - expected)).toBeLessThanOrEqual(2e-6);
              }
            } finally {
              rig.destroy();
            }
          }
});

it('whole/chunked cooperative lag refuses outer or local loss and restores every component on recovery', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true])
    for (const lossKind of ['overflow', 'candidateOverflow'] as const)
      for (const lossLocation of chunked ? (['outer', 'chunk'] as const) : (['outer'] as const)) {
        const rig = new WeightsRig(device);
        const output = rig.output('float32', 16);
        const fixture = createLagAuditCase(
          {
            input: (data, format) => rig.input(data, format, data.length),
            output: () => output.view
          },
          {
            scenario: 'masked',
            chunked,
            normalize: true,
            columnCount: 2,
            valueFormat: 'uint32',
            lossKind,
            lossLocation
          }
        );
        rig.graph.add(fixture.contributor);
        const compiled = rig.graph.compile();
        try {
          for (const loss of [1, 0]) {
            rig.bufferOf(fixture.loss!).write(Uint32Array.of(loss));
            submitGraph(device, compiled, undefined);
            const actual = await readFloat32(output.buffer, 16);
            for (const [row, value] of actual.entries()) {
              if (loss) expect(value).toBeNaN();
              else {
                expect(Number.isFinite(value)).toBe(true);
                expect(Math.abs(value - fixture.expected[row])).toBeLessThanOrEqual(2e-6);
              }
            }
          }
        } finally {
          compiled.destroy();
          rig.destroy();
        }
      }
});
