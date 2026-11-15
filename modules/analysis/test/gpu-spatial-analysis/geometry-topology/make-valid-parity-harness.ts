// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';

/** Squares from the review: reversed low-bit IDs followed by one uint32 outlier. */
export function createMakeValidParityScene(groupBits: number) {
  const smallGroupCount = 2 ** groupBits;
  const featureCount = smallGroupCount + 1;
  const segmentCount = 4 * featureCount;
  // Unwritten segment and face rows, and a non-power-of-two leaf count, stay in the probe.
  const segmentCapacity = segmentCount + 7;
  const faceCapacity = featureCount + 1;
  const sourceIds = new Uint32Array(faceCapacity).fill(0xffffffff);
  const endpoints = new Float32Array(4 * segmentCapacity).fill(Number.NaN);
  const sourceRings = new Uint32Array(segmentCapacity).fill(0xffffffff);
  const probes = new Float32Array(4 * faceCapacity);
  for (let feature = 0; feature < featureCount; feature++) {
    let group = 0xfffffffe;
    if (feature < smallGroupCount) {
      group = 0;
      for (let bit = 0; bit < groupBits; bit++) {
        group = (group << 1) | ((feature >> bit) & 1);
      }
    }
    sourceIds[feature] = group;
    const left = 3 * feature;
    endpoints.set(
      [left, 0, left + 1, 0, left + 1, 0, left + 1, 1, left + 1, 1, left, 1, left, 1, left, 0],
      16 * feature
    );
    sourceRings.fill(feature, 4 * feature, 4 * feature + 4);
    probes.set([left, 0, left + 1, 0], 4 * feature);
  }
  return {
    featureCount,
    segmentCount,
    segmentCapacity,
    faceCapacity,
    sourceIds,
    endpoints,
    sourceRings,
    probes
  };
}

/**
 * Runs the production parity preparation, sorts, BVH and face kernel on already-noded squares.
 * Only the fixture inputs bypass noding/ring assembly. A second copy of the recorded face body
 * counts actual node visits and exact-crossing candidates in its existing output binding, so
 * the probe keeps the eight-binding limit and cannot mistake elapsed GPU time for work.
 */
export function createMakeValidParityProbe(graph: GPUCommandGraph, groupBits: number) {
  recordWGSLKernelNodes(graph);
  const scene = createMakeValidParityScene(groupBits);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `parity-view-${serial++}`, format, length);
  const scalar = () => view('uint32', 1);
  const status = () => ({
    count: scalar(),
    requiredCount: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar()
  });
  const segments = scene.segmentCapacity;
  const directed = 2 * segments;
  const faces = scene.faceCapacity;
  const contributor = new GPUMakeValid({
    id: 'parity-probe',
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', scene.segmentCount),
      ringOffsets: view('uint32', scene.featureCount + 1),
      polygonOffsets: view('uint32', scene.featureCount + 1),
      featureOffsets: view('uint32', scene.featureCount + 1)
    },
    intersectionCapacity: 1,
    precision: {vertexTolerance: 1e-5, storage: 'float32', predicates: 'exact-or-uncertain'},
    output: {
      nodedSegments: {
        endpoints: view('float32x4', segments),
        sourceFeatureIds: view('uint32', segments),
        sourceRingIds: view('uint32', segments),
        sourceEdgeIds: view('uint32', segments),
        sourceStartParameters: view('float32', segments),
        sourceEndParameters: view('float32', segments),
        status: status()
      },
      directedEdges: {
        endpoints: view('float32x4', directed),
        sourceSegmentIds: view('uint32', directed),
        sourceFeatureIds: view('uint32', directed),
        twinIds: view('uint32', directed),
        ringIds: view('uint32', directed),
        flags: view('uint32', directed),
        status: status()
      },
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 5 * faces),
        featureOffsets: view('uint32', faces + 1),
        polygonOffsets: view('uint32', faces + 1),
        ringOffsets: view('uint32', faces + 1),
        sourceIds: view('uint32', faces)
      },
      diagnostics: {
        edgeClasses: view('uint32', segments),
        closedRingCount: scalar(),
        openChainCount: scalar(),
        cutEdgeCount: scalar(),
        dangleCount: scalar(),
        unusedEdgeCount: scalar()
      },
      status: status()
    }
  });
  const commands = contributor.getCommandNodes(graph);
  const preparation = getWGSLKernelNodeRecord(graph, 'parity-probe-parity-segments')!;
  const parity = getWGSLKernelNodeRecord(graph, 'parity-probe-face-parity')!;
  const binding = (record: typeof parity, name: string) =>
    record.bindings.find(value => value.name === name)!.view;
  const inputs: {view: GraphDataView; values: Float32Array | Uint32Array}[] = [
    {view: binding(preparation, 'count'), values: Uint32Array.of(scene.segmentCount)},
    {view: binding(preparation, 'endpoints'), values: scene.endpoints},
    {view: binding(preparation, 'nodedRings'), values: scene.sourceRings},
    {
      view: binding(preparation, 'ringSourceIds'),
      values: scene.sourceIds.slice(0, scene.featureCount)
    },
    {view: binding(parity, 'faceSourceIds'), values: scene.sourceIds},
    {view: binding(parity, 'probes'), values: scene.probes}
  ];
  const diagnostics = view('uint32', 3 * faces);
  const visitSite = 'let minimum = vec2f(nodeMinima[';
  const candidateSite = 'let crossing = exactProbeCrossing(';
  const publishSite = 'keep[keepOffset + index] = keepFace;';
  for (const site of [visitSite, candidateSite, publishSite]) {
    if (parity.body.split(site).length !== 2)
      throw new Error(`parity probe needs exactly one ${site}`);
  }
  const measured = createWGSLKernelNode(graph, {
    id: 'parity-probe-measured',
    operation: 'test',
    invocationCount: parity.invocationCount,
    bindings: parity.bindings.map(value =>
      value.name === 'keep' ? {...value, view: diagnostics} : value
    ),
    declarations: parity.declarations,
    body: `var visitCount = 0u; var candidateCount = 0u;\n${parity.body}`
      .replace(visitSite, `visitCount++;\n${visitSite}`)
      .replace(candidateSite, `candidateCount++;\n${candidateSite}`)
      .replace(
        publishSite,
        `keep[keepOffset + 3u * index] = visitCount;
  keep[keepOffset + 3u * index + 1u] = candidateCount;
  keep[keepOffset + 3u * index + 2u] = keepFace;`
      )
  });
  const nodes = commands.filter(
    node => node.id.startsWith('parity-probe-parity-') || node.id === parity.id
  );
  nodes.push(measured);
  return {scene, nodes, inputs, diagnostics, keep: binding(parity, 'keep')};
}
