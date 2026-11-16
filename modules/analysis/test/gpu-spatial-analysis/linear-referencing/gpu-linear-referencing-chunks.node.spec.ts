// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPULinearReferencing,
  type GPULinearReferencingProps
} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing';
import {expectValidQueryShaderLiterals} from './gpu-linear-referencing-chunks.test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

type ChunkedProps = GPULinearReferencingProps<
  GraphVectorView<'uint32'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'float32x2'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'sint32'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'float32x2'>
>;

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
function createProps(graph: GPUCommandGraph, lengths: number[], spherical = false): ChunkedProps {
  const rows = <Format extends 'float32x2' | 'float32' | 'uint32' | 'sint32'>(
    name: string,
    format: Format
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length + 64);
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: ((index % 3) + 1) * 4
        });
      })
    );
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 2),
    queryPositions: rows('queries', 'float32x2'),
    radius: createTransientView(graph, 'radius', 'float32', 1),
    candidateCapacity: 64,
    spatialContext: spherical
      ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      : undefined,
    output: {
      lineIndices: rows('line-indices', 'uint32'),
      segmentIndices: rows('segment-indices', 'uint32'),
      fractions: rows('fractions', 'float32'),
      footPoints: rows('foot-points', 'float32x2'),
      distances: rows('distances', 'float32'),
      measures: rows('measures', 'float32'),
      normalizedMeasures: rows('normalized', 'float32'),
      sides: rows('sides', 'sint32'),
      signedOffsets: rows('signed-offsets', 'float32'),
      status: {
        candidateOverflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCandidateCount: createTransientView(graph, 'required', 'uint32', 1)
      }
    }
  };
}

it('validates matching topology and all aligned writable ranges including scalar status', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  expect(
    () =>
      new GPULinearReferencing({
        ...props,
        output: {
          ...props.output,
          measures: createTransientView(graph, 'wrong-topology', 'float32', 6)
        }
      })
  ).toThrow(/topology/);
  expect(
    () =>
      new GPULinearReferencing({
        ...props,
        output: {
          ...props.output,
          status: {
            candidateOverflow: props.output.status.candidateOverflow,
            requiredCandidateCount: props.output.status.candidateOverflow
          }
        }
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('prechecks candidate and shared spherical scratch before graph mutation', () => {
  for (const spherical of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, [1, 3, 2], spherical);
    if (spherical) props.leafCapacity = 8_388_608;
    else props.candidateCapacity = 20_000_000;
    const contributor = new GPULinearReferencing(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      spherical ? /nodeData needs/ : /candidatePairsAndPositions needs/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  }
});

for (const spherical of [false, true]) {
  it(`accepts 30 million queries and shares prefix/index stages (spherical=${spherical})`, () => {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPULinearReferencing(
      createProps(graph, Array(10).fill(3_000_000), spherical)
    );
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.filter(node => node.id === 'linear-referencing-vertex-measures')).toHaveLength(1);
    expect(nodes.filter(node => node.id.endsWith('-project'))).toHaveLength(10);
    if (spherical)
      expect(nodes.filter(node => node.id === 'linear-referencing-segment-table')).toHaveLength(1);
    else expect(nodes.filter(node => node.id === 'linear-referencing-segments')).toHaveLength(1);
  });
  it(`keeps query shader/pipeline counts fixed and dispatches each chunk without padding (spherical=${spherical})`, () => {
    const pipelines = new Set<string>();
    for (const count of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const lengths = Array.from({length: count}, (_, index) => [1, 257, 513][index % 3]);
      const contributor = new GPULinearReferencing(createProps(graph, lengths, spherical));
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(() => graph.add(nodes)).not.toThrow();
      const queryNodes = nodes.filter(
        node => node.id.endsWith('-project') || /-output(?:-\d+)?$/.test(node.id)
      );
      expect(
        queryNodes
          .filter(node => node.id.endsWith('-project'))
          .map(node => node.workload?.maximumWorkgroupCount)
      ).toEqual(lengths.map(length => Math.ceil(length / 256)));
      for (const key of getGPUCommandNodePipelineKeys(queryNodes)) pipelines.add(key);
      for (const {source, constants} of getGPUCommandNodeKernelSources(queryNodes)) {
        expect(constants).toEqual({});
        expect(new WgslReflect(source).storage.length).toBeLessThanOrEqual(8);
      }
      // Include the shared geometry/index kernels, not just the per-chunk projection/scatter.
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
        expectValidQueryShaderLiterals(source);
        const identifiers =
          source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
    }
    expect(pipelines.size).toBe(3);
  });
}

it('rejects sphere radii that overflow or vanish in shader float32 at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2], true);
  for (const sphereRadius of [1e100, Number.MIN_VALUE])
    expect(
      () =>
        new GPULinearReferencing({
          ...props,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            sphereRadius
          }
        })
    ).toThrow(/sphereRadius.*finite/);
});
