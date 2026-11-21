// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPURipleyEnvelopeOperation} from '../../../src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation';
import {
  GPURipleyEnvelopeOperation,
  type RipleyEnvelopeOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {constructSamplingContributor} from '../../gpu-spatial-analysis/point-process/validation-sampling-operation-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const Operation of [CPURipleyEnvelopeOperation, GPURipleyEnvelopeOperation]) {
  for (const inherited of [false, true]) {
    it(`W5-13 ${Operation.name} snapshots rowCount once, inherited=${inherited}`, async () => {
      let reads = 0;
      let outputReads = 0;
      let rowCount = 2;
      let outputName = 'statistics';
      const output: RipleyEnvelopeOperationProps['output'] = {};
      Object.defineProperty(output, 'statistics', {
        get: () => {
          outputReads++;
          return outputName;
        }
      });
      const props: RipleyEnvelopeOperationProps = {
        positions: 'positions',
        rowCount: 2,
        parameters: 'parameters',
        gridSize: [2, 2],
        radiusCount: 1,
        simulationCount: 2,
        output
      };
      const descriptor = {
        get: () => {
          reads++;
          return rowCount;
        },
        configurable: true,
        enumerable: false
      };
      if (inherited) {
        Reflect.deleteProperty(props, 'rowCount');
        Object.setPrototypeOf(props, Object.defineProperty({}, 'rowCount', descriptor));
      } else Object.defineProperty(props, 'rowCount', descriptor);
      const operation = new Operation(props);
      expect(reads).toBe(1);
      expect(outputReads).toBe(1);
      rowCount = 3;
      outputName = 'replacement';
      props.gridSize = [4, 4];
      if (operation instanceof GPURipleyEnvelopeOperation) {
        expect(() =>
          constructSamplingContributor(operation, {
            positions: {format: 'float32x2', length: 2},
            parameters: {format: 'float32', length: 8}
          })
        ).not.toThrow();
        expect(operation.outputs['statistics'].value).toBe('statistics');
      } else {
        const device = createNullWebGPUDevice();
        try {
          const result = await runOperationGraph(device, {
            operations: [operation],
            outputs: ['statistics'],
            inputs: {
              positions: {format: 'float32x2', values: Float32Array.of(0.25, 0.25, 0.75, 0.75)},
              parameters: {
                format: 'float32',
                values: getGPURipleyParameterValues({
                  bounds: [0, 0, 1, 1],
                  maximumDistance: 1,
                  seed: 0
                })
              }
            }
          });
          expect(result['statistics'].length).toBe(6);
          expect(result['statistics'][0]).toBe(2);
        } finally {
          device.destroy();
        }
      }
      expect(reads).toBe(1);
      expect(outputReads).toBe(1);
    });
  }
}

for (const Operation of [CPURipleyEnvelopeOperation, GPURipleyEnvelopeOperation]) {
  it(`S3 ${Operation.name} captures nested context before caller reuse`, async () => {
    const spatialContext: NonNullable<RipleyEnvelopeOperationProps['spatialContext']> = {
      coordinateSpace: 'planar',
      metric: 'native',
      frame: {id: 'original-plane', kind: 'crs'}
    };
    const operation = new Operation({
      positions: 'positions',
      parameters: 'parameters',
      rowCount: 2,
      gridSize: [2, 2],
      radiusCount: 1,
      simulationCount: 2,
      spatialContext,
      output: {statistics: 'statistics'}
    });
    spatialContext.frame!.id = '';
    if (operation instanceof GPURipleyEnvelopeOperation) {
      expect(() =>
        constructSamplingContributor(operation, {
          positions: {format: 'float32x2', length: 2},
          parameters: {format: 'float32', length: 8}
        })
      ).not.toThrow();
    } else {
      const device = createNullWebGPUDevice();
      try {
        const result = await runOperationGraph(device, {
          operations: [operation],
          outputs: ['statistics'],
          inputs: {
            positions: {format: 'float32x2', values: Float32Array.of(0.25, 0.25, 0.75, 0.75)},
            parameters: {
              format: 'float32',
              values: getGPURipleyParameterValues({
                bounds: [0, 0, 1, 1],
                maximumDistance: 1,
                seed: 0
              })
            }
          }
        });
        expect(result['statistics'].length).toBe(6);
        expect(result['statistics'][0]).toBe(2);
      } finally {
        device.destroy();
      }
    }
  });
}
