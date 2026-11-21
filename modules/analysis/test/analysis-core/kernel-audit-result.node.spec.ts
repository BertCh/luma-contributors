// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {auditKernelCase, getKernelAuditResultFailures} from './kernel-audit/kernel-audit-runner';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
import {DATAFRAME_CASES} from './kernel-audit/cases-dataframe';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('rejects one declared node without a shader module in family and registry checks (CORE-114)', () => {
  const result = auditKernelCase({
    exportName: 'GPUFixture',
    name: 'one-node-no-shader',
    build: ({graph}) => {
      graph.add({
        id: 'no-shader',
        type: 'compute',
        resources: [],
        compile: () => ({encode: () => {}})
      });
    }
  });
  expect(result.nodeCount).toBe(1);
  expect(result.moduleCount).toBe(0);
  expect(getKernelAuditResultFailures(result)).toEqual([
    {location: 'compile', check: 'no shader module', detail: 'recorded 0 modules'}
  ]);
});

it('accepts a documented refusal and a shader-producing control (CORE-114)', () => {
  // A valid refusal has already passed the runner's documented-limit/message checks.
  expect(
    getKernelAuditResultFailures({findings: [], nodeCount: 0, moduleCount: 0, rejected: true})
  ).toEqual([]);
  const result = auditKernelCase({
    exportName: 'GPUFixture',
    name: 'shader-control',
    build: ({graph, view}) => {
      const output = view('uint32', 1);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'shader-control',
          operation: 'GPUFixture',
          invocationCount: 1,
          bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
          body: 'output[outputOffset] = 7u;'
        })
      );
    }
  });
  expect(result.nodeCount).toBe(1);
  expect(result.moduleCount).toBeGreaterThan(0);
  expect(getKernelAuditResultFailures(result)).toEqual([]);
});

it('runs discovered family callbacks and rejects a zero-module success (B8)', () => {
  const callbacks = new Map<string, () => void>();
  registerKernelAudit(
    'registration-regression',
    [
      {
        exportName: 'GPUFittingGradientSums',
        name: 'zero-module fixture',
        build: ({graph}) => {
          graph.add({
            id: 'empty-compute',
            type: 'compute',
            resources: [],
            compile: () => ({encode: () => {}})
          });
        }
      },
      {
        exportName: 'GPUFittingGradientSums',
        name: 'shader fixture',
        build: ({graph, view}) => {
          graph.add(
            createWGSLKernelNode(graph, {
              id: 'registered-control',
              operation: 'GPUFittingGradientSums',
              invocationCount: 1,
              bindings: [
                {name: 'output', view: view('uint32', 1), type: 'u32', access: 'read_write'}
              ],
              body: 'output[outputOffset] = 9u;'
            })
          );
        }
      },
      {
        exportName: 'GPUColumnProfile',
        name: 'row metadata one above uint32 addressing',
        // Exercise the real constructor/structured refusal and its observed construction phase.
        build: DATAFRAME_CASES.find(
          candidate => candidate.name === 'row metadata one above uint32 addressing'
        )!.build
      },
      {
        exportName: 'GPUFixture',
        name: 'not publicly discovered',
        build: () => {
          throw new Error('must not run');
        }
      }
    ],
    {
      describe: (_name, define) => define(),
      it: (name, run) => {
        callbacks.set(name, run);
      }
    }
  );
  expect([...callbacks.keys()]).toEqual([
    'GPUFittingGradientSums zero-module fixture',
    'GPUFittingGradientSums shader fixture',
    'GPUColumnProfile row metadata one above uint32 addressing'
  ]);
  expect(callbacks.get('GPUFittingGradientSums zero-module fixture')).toThrow(/no shader module/);
  expect(callbacks.get('GPUFittingGradientSums shader fixture')).not.toThrow();
  expect(callbacks.get('GPUColumnProfile row metadata one above uint32 addressing')).not.toThrow();
});
