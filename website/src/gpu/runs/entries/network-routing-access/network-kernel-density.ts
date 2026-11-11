// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkKernelDensity: event intensity along roads, from shortest-path distances. Every
// intersection is a query; roads are colored by the triangular-kernel intensity of the events
// (dots) within the network bandwidth.

import {GPUNetworkKernelDensity} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, uploadSearch} from './shared/access';
import {createLegend} from '../../text';
import {addMatrixFootprint, addZoneSegments} from './shared/display';
import {chooseSticky} from '../../pick';
import {
  createNodeIndex,
  createRoads,
  getRoadContext,
  getRoadValues,
  gatherPositions,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-kernel-density.parameters';

const EVENT_COUNT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Network kernel density of the events (dots), per intersection.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 8);
    // Each intersection represents half the length of its roads.
    const quadratureLengths = new Float32Array(network.nodeCount);
    for (let node = 0; node < network.nodeCount; node++) {
      for (let slot = network.offsets[node]; slot < network.offsets[node + 1]; slot++) {
        quadratureLengths[node] += network.lengths[slot] / 2;
      }
    }
    const pick = (clustered: boolean) =>
      Uint32Array.from({length: EVENT_COUNT}, (_, index) =>
        clustered
          ? network.findNearestNode(
              world.width * (0.25 + 0.5 * (index % 2)) + (random() - 0.5) * 0.2,
              0.5 + (random() - 0.5) * 0.4
            )
          : network.findNearestNode(random() * world.width, random())
      );
    const uniform = pick(false);
    const clustered = pick(true);
    const defaultClustering =
      parameters.find(parameter => parameter.id === 'clustering')?.default ?? 0;
    const lastIndex = EVENT_COUNT - 1;
    return {
      network,
      quadratureLengths,
      uniform,
      clustered,
      events: new Uint32Array(EVENT_COUNT),
      index: createNodeIndex(network),
      // The last event follows the pointer; it starts at the node the clustering slider gives it.
      defaultLast:
        lastIndex / EVENT_COUNT < defaultClustering ? clustered[lastIndex] : uniform[lastIndex],
      dragNode: -1
    };
  },
  build(builder, inputs, values) {
    const {network} = inputs;
    const search = uploadSearch(builder, network);
    const queries = Uint32Array.from({length: network.nodeCount}, (_, index) => index);
    // Events change with the clustering slider, so their nodes are rewritten each frame.
    const eventNodes = builder.allocate('event-nodes', 'uint32', EVENT_COUNT);
    const distances = addCostMatrix(builder, search, queries, network.nodeCount, eventNodes);
    const intensity = builder.allocate('intensity', 'float32', network.nodeCount);
    const kernelMass = builder.allocate('kernel-mass', 'float32', EVENT_COUNT);
    const status = builder.allocate('status', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkKernelDensity({
        distances: distances.view,
        eventCount: EVENT_COUNT,
        quadratureLengths: builder.upload('quadrature-lengths', 'float32', inputs.quadratureLengths)
          .view,
        bandwidth: values.bandwidth,
        output: {intensity: intensity.view, kernelMass: kernelMass.view, status: status.view}
      })
    );
    // The dragged event's own kernel: 1 at the event falling to 0 at the bandwidth along the roads,
    // read on the GPU from the distance column of the last event.
    const draggedEvent = builder.upload('dragged-event', 'uint32', Uint32Array.of(EVENT_COUNT - 1));
    const bandwidthBuffer = builder.upload(
      'bandwidth',
      'float32',
      Float32Array.of(values.bandwidth)
    );
    const footprint = addMatrixFootprint(builder, {
      id: 'event',
      costs: distances,
      nodeCount: network.nodeCount,
      columnCount: EVENT_COUNT,
      mode: 'column',
      selector: draggedEvent,
      limit: bandwidthBuffer,
      floor: 0
    });
    const roads = uploadRoads(builder, network);
    const reach = addZoneSegments(builder, {
      id: 'event',
      nodes: roads.nodes,
      edges: roads.edges,
      values: footprint
    });
    const events = builder.upload(
      'events',
      'float32x2',
      gatherPositions(network.nodes, inputs.uniform)
    );
    return {
      outputs: {
        ...roads,
        intensity,
        reach,
        events
      },
      update: current => {
        const mix = Uint32Array.from(inputs.uniform, (uniformNode, index) =>
          index / EVENT_COUNT < current.clustering ? inputs.clustered[index] : uniformNode
        );
        if (inputs.dragNode >= 0) mix[EVENT_COUNT - 1] = inputs.dragNode;
        eventNodes.write(mix);
        events.write(gatherPositions(network.nodes, mix));
        search.costLimit.write(Float32Array.of(4));
      }
    };
  },
  rebuildOn: ['bandwidth'],
  // The pointer carries one event along the roads; the intensity re-forms around it.
  interaction: {
    kind: 'drag',
    hint: 'drag an event',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.network.nodes[inputs.defaultLast * 2],
        inputs.network.nodes[inputs.defaultLast * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const previous = inputs.dragNode >= 0 ? inputs.dragNode : inputs.defaultLast;
      const node = chooseSticky(inputs.index, pointer.x, pointer.y, previous);
      const changed = node !== inputs.dragNode;
      inputs.dragNode = node;
      const intensity = peek(outputs.intensity);
      let peak = 0;
      if (intensity)
        for (const value of intensity) if (Number.isFinite(value)) peak = Math.max(peak, value);
      const at: [number, number] = [
        inputs.network.nodes[node * 2],
        inputs.network.nodes[node * 2 + 1]
      ];
      return {
        rerun: changed,
        glyphs: [{shape: 'crosshair', at, size: 12}],
        readout: `event at node ${node} · bandwidth ${values.bandwidth.toFixed(2)} · peak ${peak.toFixed(1)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    // The footprint of the dragged event: a halo along the roads within the bandwidth of it.
    {layer: 'lines', segments: outputs.reach, radius: 6, color: [...theme.ink, 0.35]},
    getRoadValues(outputs, outputs.intensity, {radius: 3, range: [0, 9]}),
    {layer: 'dots', positions: outputs.events, radius: 4.5, color: [...theme.ink, 1]},
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        entries: [
          {label: 'roads within the bandwidth of the dragged event', swatch: [...theme.ink]},
          {label: 'density on the roads (output)', swatch: [...theme.ramp[4]]}
        ]
      })
    }
  ]
});
