// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUBufferSelection,
  type GPUBufferSelectionProps
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

let propsCount = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUBufferSelectionProps> = {}
): GPUBufferSelectionProps {
  const prefix = `props-${propsCount++}-`;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${prefix}${name}`, format, length);
  return {
    points: {kind: 'points', positions: view('points', 'float32x2', 10)},
    features: {
      kind: 'segments',
      starts: view('starts', 'float32x2', 3),
      ends: view('ends', 'float32x2', 3)
    },
    radius: view('distance', 'float32', 1),
    candidateCapacity: 16,
    output: {outputMask: view('mask', 'uint32', 10)},
    ...overrides
  };
}

it('GPUBufferSelection builds nodes for a valid configuration', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-node'});
  const selection = new GPUBufferSelection(createProps(graph));
  expect(selection.id).toBe('buffer-selection');
  const ids = selection.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('buffer-selection-mask');
  expect(ids.some(id => id.startsWith('buffer-selection-join'))).toBe(true);
  expect(ids).toContain('buffer-selection-publish');
  const matches = selection.getMatches(graph);
  expect(matches.ids.length).toBe(10);
  for (const view of [
    matches.status.count,
    matches.status.requiredCount,
    matches.status.overflow,
    matches.status.candidateOverflow
  ]) {
    expect(view?.format).toBe('uint32');
    expect(view?.length).toBe(1);
  }
  expect(ids).toContain('buffer-selection-publish-candidate');
});

it('GPUBufferSelection builds one mask node per chunk and a publish node with output', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-chunks'});
  const view = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const props = createProps(graph, {
    points: {
      kind: 'points',
      positions: createVectorView('points', 'float32x2', [
        view('points-0', 'float32x2', 6),
        view('points-1', 'float32x2', 4)
      ])
    },
    output: {
      outputMask: createVectorView('mask', 'uint32', [
        view('mask-0', 'uint32', 6),
        view('mask-1', 'uint32', 4)
      ]),
      matches: {
        ids: view('out-ids', 'uint32', 4),
        status: {
          count: view('out-count', 'uint32', 1),
          overflow: view('out-overflow', 'uint32', 1)
        }
      }
    }
  });
  const selection = new GPUBufferSelection(props);
  const ids = selection.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('buffer-selection-mask-chunk-0');
  expect(ids).toContain('buffer-selection-mask-chunk-1');
  expect(ids).toContain('buffer-selection-publish');
  const matches = selection.getMatches(graph);
  expect(matches.ids).toBe(props.output.matches!.ids);
  expect(matches.status.count).toBe(props.output.matches!.status.count);
  expect(matches.status.overflow).toBe(props.output.matches!.status.overflow);
  expect(matches.status.candidateOverflow.length).toBe(1);
});

it('GPUBufferSelection validates outputs and the distance view', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-validation'});
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const emptyOutput = new GPUBufferSelection(
    createProps(graph, {id: 'implicit-matches', output: {}})
  );
  const matches = emptyOutput.getMatches(graph);
  expect(matches.ids.length).toBe(10);
  expect(matches.status.count.length).toBe(1);
  expect(matches.status.overflow.length).toBe(1);
  expect(matches.status.candidateOverflow.length).toBe(1);
  expect(emptyOutput.getCommandNodes(graph).map(node => node.id)).toContain(
    'implicit-matches-publish'
  );
  expect(
    () => new GPUBufferSelection(createProps(graph, {radius: view('bad-distance', 'float32', 2)}))
  ).toThrow(/radius needs one float32 row/);
  expect(
    () =>
      new GPUBufferSelection(
        createProps(graph, {output: {outputMask: view('short-mask', 'uint32', 9)}})
      )
  ).toThrow(/outputMask must match the source topology/);
  expect(
    () =>
      new GPUBufferSelection(
        createProps(graph, {
          output: {
            matches: {
              ids: view('empty-ids', 'uint32', 0),
              status: {
                count: view('count', 'uint32', 1),
                overflow: view('overflow', 'uint32', 1)
              }
            }
          }
        })
      )
  ).toThrow(/output.matches.ids needs at least one row/);
  const invalidProps = createProps(graph, {candidateCapacity: 0});
  const scratch = vi.spyOn(graph, 'createTransientBuffer');
  const nodes = vi.spyOn(graph, 'add');
  expect(() => new GPUBufferSelection(invalidProps)).toThrow(/candidateCapacity/);
  expect(scratch).not.toHaveBeenCalled();
  expect(nodes).not.toHaveBeenCalled();
});

it('GPUBufferSelection resolves planar and great-circle contexts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-context'});
  expect(new GPUBufferSelection(createProps(graph)).spatialContext.coordinateSpace).toBe('planar');
  const greatCircle = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const geodesic = new GPUBufferSelection(
    createProps(new GPUCommandGraph(createNullWebGPUDevice()), {spatialContext: greatCircle})
  );
  expect(geodesic.spatialContext.metric).toBe('great-circle');
  // The context reaches the join, which runs the great-circle traversal.
  const joinGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const ids = new GPUBufferSelection(createProps(joinGraph, {spatialContext: greatCircle}))
    .getCommandNodes(joinGraph)
    .map(node => node.id);
  expect(ids).toContain('buffer-selection-join-traverse');
  expect(ids).not.toContain('buffer-selection-join-probe');
  const kilometerGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const kilometerSelection = new GPUBufferSelection(
    createProps(kilometerGraph, {
      spatialContext: {...greatCircle, units: 'kilometers'},
      candidateCapacity: 3
    })
  );
  expect(kilometerSelection.spatialContext.units).toBe('kilometers');
  const kilometerNodeIds = kilometerSelection.getCommandNodes(kilometerGraph).map(node => node.id);
  expect(kilometerNodeIds).toContain('buffer-selection-join-traverse');
  expect(kilometerNodeIds).not.toContain('buffer-selection-join-probe');
  expect(
    () =>
      new GPUBufferSelection(
        createProps(new GPUCommandGraph(createNullWebGPUDevice()), {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        })
      )
  ).toThrow(/great-circle/);
});

it('distance measurements reject Lambert frames and accept AEQD meters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const lambert = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'local-frame', kind: 'local-lambert-equal-area'}
  } as const;
  const create = (spatialContext: GPUSpatialContext) =>
    new GPUBufferSelection(createProps(graph, {spatialContext}));
  expect(() => create(lambert)).toThrow(/measures distance/);
  expect(() =>
    create({...lambert, frame: {...lambert.frame, kind: 'local-azimuthal-equidistant'}})
  ).not.toThrow();
  device.destroy();
});
