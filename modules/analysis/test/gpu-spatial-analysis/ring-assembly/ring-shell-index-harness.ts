// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

/** Bit-reversed disjoint annuli and one distant shell from the scale review. */
export function createRingShellIndexScene(annulusBits: number, transpose = false) {
  const annulusCount = 2 ** annulusBits;
  const ringCount = 2 * annulusCount + 1;
  const ringCapacity = ringCount + 2;
  const bounds = new Float32Array(4 * ringCapacity).fill(Number.NaN);
  const positions = new Float32Array(10 * ringCapacity).fill(Number.NaN);
  const areas = new Float32Array(ringCapacity).fill(Number.NaN);
  const isHole = new Uint32Array(ringCapacity).fill(1);
  const ringOffsets = Uint32Array.from(
    {length: ringCapacity + 1},
    (_, ring) => 5 * Math.min(ring, ringCount)
  );
  const addRing = (
    ring: number,
    minimumX: number,
    minimumY: number,
    maximumX: number,
    maximumY: number,
    hole: boolean
  ) => {
    const vertices = hole
      ? [
          minimumX,
          minimumY,
          minimumX,
          maximumY,
          maximumX,
          maximumY,
          maximumX,
          minimumY,
          minimumX,
          minimumY
        ]
      : [
          minimumX,
          minimumY,
          maximumX,
          minimumY,
          maximumX,
          maximumY,
          minimumX,
          maximumY,
          minimumX,
          minimumY
        ];
    if (transpose) {
      // Rotate into the negative y half-plane: both coordinate-key words and signs matter.
      for (let vertex = 0; vertex < vertices.length; vertex += 2) {
        const first = vertices[vertex];
        vertices[vertex] = vertices[vertex + 1];
        vertices[vertex + 1] = -first;
      }
      bounds.set([minimumY, -maximumX, maximumY, -minimumX], 4 * ring);
    } else {
      bounds.set([minimumX, minimumY, maximumX, maximumY], 4 * ring);
    }
    positions.set(vertices, 10 * ring);
    areas[ring] = (hole ? -1 : 1) * (maximumX - minimumX) * (maximumY - minimumY);
    isHole[ring] = hole ? 1 : 0;
  };
  for (let annulus = 0; annulus < annulusCount; annulus++) {
    let coordinate = 0;
    for (let bit = 0; bit < annulusBits; bit++)
      coordinate = (coordinate << 1) | ((annulus >> bit) & 1);
    const left = 3 * coordinate;
    addRing(2 * annulus, left, 0, left + 2, 2, false);
    addRing(2 * annulus + 1, left + 0.5, 0.5, left + 1.5, 1.5, true);
  }
  addRing(2 * annulusCount, 1e9, 0, 1e9 + 128, 128, false);
  return {annulusCount, ringCount, ringCapacity, bounds, positions, areas, isHole, ringOffsets};
}

/**
 * Runs production shell-index preparation and the recorded production shell-assignment shader.
 * Ring metadata bypasses tracing/statistics, keeping the probe about the index and traversal.
 * Instrumentation writes visits, candidates and the selected original ring ID through the
 * existing output binding, retaining the portable eight-storage-binding limit.
 */
export function createRingShellIndexProbe(
  graph: GPUCommandGraph,
  annulusBits: number,
  transpose = false
) {
  recordWGSLKernelNodes(graph);
  const scene = createRingShellIndexScene(annulusBits, transpose);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `shell-probe-view-${serial++}`, format, length);
  const contributor = new GPUSegmentRingAssembly({
    id: 'shell-probe',
    vertexTolerance: 0.01,
    endpoints: view('float32x4', 4 * scene.ringCount),
    groups: view('uint32', 4 * scene.ringCount),
    output: {
      positions: view('float32x2', 5 * scene.ringCapacity),
      ringOffsets: view('uint32', scene.ringCapacity + 1),
      ringShells: view('uint32', scene.ringCapacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  });
  const commands = contributor.getCommandNodes(graph);
  const preparation = getWGSLKernelNodeRecord(graph, 'shell-probe-shell-index-records')!;
  const traversal = getWGSLKernelNodeRecord(graph, 'shell-probe-ring-shells')!;
  const binding = (record: typeof traversal, name: string) =>
    record.bindings.find(value => value.name === name)!.view;
  const inputs: {view: GraphDataView; values: Float32Array | Uint32Array}[] = [
    {view: binding(traversal, 'ringBounds'), values: scene.bounds},
    {view: binding(preparation, 'areas'), values: scene.areas},
    {view: binding(preparation, 'isHole'), values: scene.isHole},
    {view: binding(preparation, 'count'), values: Uint32Array.of(scene.ringCount)},
    {view: binding(preparation, 'groups'), values: new Uint32Array(scene.ringCapacity)},
    {view: binding(traversal, 'positions'), values: scene.positions},
    {view: binding(traversal, 'ringOffsets'), values: scene.ringOffsets}
  ];
  const diagnostics = view('uint32', 3 * scene.ringCapacity);
  const visitSite = 'while (node != 4294967295u) {';
  const candidateSite = 'node = hit.y;';
  const publishSite = 'shells[shellsOffset + index] = best;';
  if (traversal.declarations!.split(visitSite).length !== 2)
    throw new Error('shell probe needs one traversal loop');
  for (const site of [candidateSite, publishSite]) {
    if (traversal.body.split(site).length !== 2)
      throw new Error(`shell probe needs exactly one ${site}`);
  }
  const body = `visitCount = 0u; var candidateCount = 0u;
  shells[shellsOffset + 3u * index] = 0u;
  shells[shellsOffset + 3u * index + 1u] = 0u;
${traversal.body}`
    .replace(candidateSite, `${candidateSite}\n  candidateCount++;`)
    .replace(
      publishSite,
      `${publishSite}
  shells[shellsOffset + 3u * index] = visitCount;
  shells[shellsOffset + 3u * index + 1u] = candidateCount;`
    )
    .replaceAll('shells[shellsOffset + index]', 'shells[shellsOffset + 3u * index + 2u]');
  const measured = createWGSLKernelNode(graph, {
    id: 'shell-probe-measured',
    operation: 'test',
    invocationCount: traversal.invocationCount,
    bindings: traversal.bindings.map(value =>
      value.name === 'shells' ? {...value, view: diagnostics} : value
    ),
    declarations: `var<private> visitCount: u32;\n${traversal.declarations!.replace(visitSite, `${visitSite}\n    visitCount++;`)}`,
    body
  });
  const nodes = commands.filter(node => node.id.startsWith('shell-probe-shell-index-'));
  nodes.push(measured);
  return {
    scene,
    nodes,
    inputs,
    diagnostics,
    index: {
      minima: binding(traversal, 'nodeMinima'),
      maxima: binding(traversal, 'nodeMaxima'),
      leaves: binding(traversal, 'leafIds')
    }
  };
}
