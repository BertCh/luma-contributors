// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUKernelFlowSmoothing,
  GPUMapMatchedEdgeVolumes,
  GPURadiationFlowModel,
  getGPUKernelFlowSmoothingParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {COORDINATE_SYSTEM_PARAMETER, type CatalogOperationBinding} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  getLineParts,
  getPointPositions,
  makeCatalogGeometryText,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {
  makeInputSpec,
  makeOutputSpec,
  makeNumberParameter,
  makeStringParameter
} from './catalog-platform3-b-common';
import {
  getNumber,
  getString,
  getSourceRow,
  makeBinding,
  makeEnum,
  makeResult,
  readNumbers,
  readPositions
} from './catalog-network-categorical-common';

const MAXIMUM_ZONES = 128; // Reference-tier cubic work and a bounded 16,384-row arc publication.

export function prepareFlowMatrix(
  zones: CatalogFeatureCollection,
  flows: CatalogFeatureCollection,
  originProperty: string,
  destinationProperty: string,
  weightProperty: string
) {
  const zoneIndices = new Map<string, number>();
  zones.features.forEach((feature, index) => {
    const identity = String(feature.id ?? index);
    if (zoneIndices.has(identity)) throw new Error('Zone identities must be unique');
    zoneIndices.set(identity, index);
  });
  const weights = readNumbers(flows, weightProperty, true);
  const matrix = new Float32Array(zones.features.length ** 2);
  const occupied = new Set<number>();
  flows.features.forEach((feature, index) => {
    const origin = zoneIndices.get(String(feature.properties?.[originProperty]));
    const destination = zoneIndices.get(String(feature.properties?.[destinationProperty]));
    if (origin === undefined || destination === undefined)
      throw new Error(
        'Flow endpoint must name a zone feature id (or its row index when id is absent)'
      );
    const slot = origin * zones.features.length + destination;
    if (occupied.has(slot))
      throw new Error('Flow pairs must be unique; aggregate repeated OD pairs before smoothing');
    occupied.add(slot);
    matrix[slot] = weights[index];
  });
  return matrix;
}

export const FLOW_BINDINGS: readonly CatalogOperationBinding[] = [
  ...(['kernel-flow-smoothing', 'radiation-flow-model'] as const).map(id => {
    const smoothing = id === 'kernel-flow-smoothing';
    const title = smoothing ? 'Kernel flow smoothing' : 'Radiation flow model';
    const contributor = smoothing ? 'GPUKernelFlowSmoothing' : 'GPURadiationFlowModel';
    return makeBinding({
      id,
      title,
      domain: 'network',
      summary: `${title} between identified zones, published as origin-destination arcs.`,
      contributorNames: [contributor],
      inputs: [
        makeInputSpec('zones', 'Zone centroids', ['point'], MAXIMUM_ZONES, MAXIMUM_ZONES),
        ...(smoothing
          ? [
              makeInputSpec(
                'flows',
                'Unique OD flows',
                ['line'],
                MAXIMUM_ZONES ** 2,
                MAXIMUM_ZONES ** 2 * 2
              )
            ]
          : [])
      ],
      parameters: [
        COORDINATE_SYSTEM_PARAMETER,
        makeStringParameter(
          'populationProperty',
          smoothing ? 'Location weight property' : 'Population property',
          'population'
        ),
        ...(smoothing
          ? [
              makeStringParameter('originProperty', 'Origin zone property', 'origin'),
              makeStringParameter(
                'destinationProperty',
                'Destination zone property',
                'destination'
              ),
              makeStringParameter('weightProperty', 'Flow weight property', 'weight'),
              makeNumberParameter('neighborhoodSize', 'Weighted neighborhood size', 2, 0, 1e9),
              makeNumberParameter('bandwidthRatio', 'Bandwidth ratio', 0.5, 1e-6, 100)
            ]
          : [
              makeStringParameter('outflowProperty', 'Outflow property', 'outflow'),
              makeEnum('normalization', 'Radiation normalization', ['none', 'finite-size'])
            ])
      ],
      outputs: [
        makeOutputSpec(
          id,
          title,
          'line',
          ['origin_index', 'destination_index', 'origin_id', 'destination_id'],
          MAXIMUM_ZONES ** 2
        )
      ],
      declaredBehavior: [
        'Zone ids must be unique; absent ids use source row indices. Every zone must have one finite Point or singleton MultiPoint centroid and finite nonnegative weights.',
        'Distances are native planar units or great-circle meters for EPSG:4326. Outputs use the original endpoint coordinates, with straight geographic arc geometry.',
        'Missing OD pairs are zero. Repeated OD pairs are rejected. Smoothing is a weighted mean and does not conserve total flow. Radiation uses strictly nearer intervening populations and a zero diagonal.',
        'Invalid values fail before GPU allocation; no result is published on failure.'
      ],
      limitation:
        'At most 128 zones (reference-tier cubic contributors). Dense OD publication includes zero flows; no random radiation draw or automatic zone assignment.',
      runGpu: async (context, inputs, values) => {
        const zones = inputs['zones'];
        if (zones.features.length < 2) throw new Error('At least two zones are required');
        const geographic = values['coordinateSystem'] === 'EPSG:4326';
        const positions = readPositions(zones, geographic);
        const populations = readNumbers(zones, getString(values, 'populationProperty'), true);
        const matrix = smoothing
          ? prepareFlowMatrix(
              zones,
              inputs['flows'],
              getString(values, 'originProperty'),
              getString(values, 'destinationProperty'),
              getString(values, 'weightProperty')
            )
          : undefined;
        if (
          !smoothing &&
          new Set(zones.features.map((feature, index) => String(feature.id ?? index))).size !==
            zones.features.length
        )
          throw new Error('Zone identities must be unique');
        const outflows = smoothing
          ? undefined
          : readNumbers(zones, getString(values, 'outflowProperty'), true);
        return withCatalogGpuSession(context.device, `catalog-${id}`, async session => {
          const output = session.output('flows', 'float32', zones.features.length ** 2);
          const radii = session.output('radii', 'float32', zones.features.length);
          const spatialContext = geographic
            ? {
                coordinateSpace: 'longitude-latitude' as const,
                metric: 'great-circle' as const,
                units: 'meters' as const
              }
            : {coordinateSpace: 'planar' as const, metric: 'native' as const};
          const positionView = session.input('positions', positions, 'float32x2');
          const populationView = session.input('populations', populations, 'float32');
          if (smoothing)
            session.graph.add(
              new GPUKernelFlowSmoothing({
                spatialContext,
                positions: positionView,
                locationWeights: populationView,
                flows: {kind: 'matrix', matrix: session.input('matrix', matrix!, 'float32')},
                parameters: session.input(
                  'parameters',
                  getGPUKernelFlowSmoothingParameterValues({
                    neighborhoodSize: getNumber(values, 'neighborhoodSize'),
                    bandwidthRatio: getNumber(values, 'bandwidthRatio')
                  }),
                  'float32'
                ),
                output: {smoothed: output.view, radii: radii.view}
              })
            );
          else
            session.graph.add(
              new GPURadiationFlowModel({
                spatialContext,
                positions: positionView,
                populations: populationView,
                outflows: session.input('outflows', outflows!, 'float32'),
                normalization: values['normalization'] as 'none' | 'finite-size',
                output: {flows: output.view}
              })
            );
          session.run();
          const flows = await session.readFloat32(output);
          const distances = smoothing ? await session.readFloat32(radii) : undefined;
          return makeResult(
            id,
            title,
            {
              origin_index: 'int32',
              destination_index: 'int32',
              origin_id: 'string',
              destination_id: 'string',
              flow: 'float64',
              origin_radius: 'float64',
              destination_radius: 'float64',
              geometry: 'string'
            },
            Array.from(flows, (flow, slot) => {
              const origin = Math.floor(slot / zones.features.length);
              const destination = slot % zones.features.length;
              return {
                origin_index: origin,
                destination_index: destination,
                origin_id: String(zones.features[origin].id ?? origin),
                destination_id: String(zones.features[destination].id ?? destination),
                flow,
                origin_radius: distances?.[origin] ?? null,
                destination_radius: distances?.[destination] ?? null,
                geometry: makeCatalogGeometryText('LineString', [
                  getPointPositions(zones.features[origin].geometry)[0],
                  getPointPositions(zones.features[destination].geometry)[0]
                ])
              };
            })
          );
        });
      }
    });
  }),
  makeBinding({
    id: 'map-matched-edge-volumes',
    title: 'Map-matched edge volumes',
    domain: 'network',
    summary:
      'Count traversals from already matched edge sequences, retaining repeated traversals and breaks.',
    contributorNames: ['GPUMapMatchedEdgeVolumes'],
    inputs: [
      makeInputSpec('edges', 'Matched directed network edges', ['line'], 4096, 32768),
      makeInputSpec('tracks', 'Matched tracks with edge sequences', ['line'], 1024, 32768)
    ],
    parameters: [
      makeStringParameter('edgesProperty', 'Traversed edge list property', 'traversedEdges'),
      makeNumberParameter(
        'traversedEdgeCapacity',
        'Edges per track capacity',
        256,
        1,
        4096,
        undefined,
        'integer'
      )
    ],
    outputs: [
      makeOutputSpec(
        'map-matched-edge-volumes',
        'Matched edge volumes',
        'line',
        ['source_index', 'source_id'],
        4096
      )
    ],
    declaredBehavior: [
      'Input edges are directed and identified by their source row index. Track properties contain arrays of those indices; null or 0xffffffff separates sub-tracks.',
      'Every entry counts as one traversal, including loops; these are not counts of GPS fixes. Unknown edge ids and sequences longer than the declared capacity fail before dispatch.',
      'Every source edge is published, including zero-volume edges, with its original geometry and identity.'
    ],
    limitation:
      'Consumes already map-matched tracks; does not infer routes from line geometry. Turning movement tables are not exposed.',
    runGpu: async (context, inputs, values) => {
      const edges = inputs['edges'];
      const tracks = inputs['tracks'];
      if (
        edges.features.some(
          feature =>
            getLineParts(feature.geometry).length !== 1 ||
            getLineParts(feature.geometry)[0].length < 2
        )
      )
        throw new Error('Each edge must be one line with at least two vertices');
      const capacity = getNumber(values, 'traversedEdgeCapacity');
      const sequences = prepareTraversedEdges(
        tracks,
        getString(values, 'edgesProperty'),
        edges.features.length,
        capacity
      );
      return withCatalogGpuSession(context.device, 'catalog-edge-volumes', async session => {
        const volumes = session.output('volumes', 'uint32', edges.features.length);
        const truncated = session.output('truncated', 'uint32', 1);
        session.graph.add(
          new GPUMapMatchedEdgeVolumes({
            traversedEdges: {
              edges: session.input('edges', sequences.edges, 'uint32'),
              counts: session.input('counts', sequences.counts, 'uint32'),
              completeness: 'known-complete'
            },
            trackCount: tracks.features.length,
            traversedEdgeCapacity: capacity,
            edgeCount: edges.features.length,
            output: {edgeVolumes: volumes.view, status: {candidateOverflow: truncated.view}}
          })
        );
        session.run();
        const output = await session.readUint32(volumes);
        return makeResult(
          'map-matched-edge-volumes',
          'Matched edge volumes',
          {source_index: 'int32', source_id: 'string', geometry: 'string', volume: 'int32'},
          edges.features.map((_, index) => ({
            ...getSourceRow(edges, index),
            volume: output[index]
          })),
          {
            candidateOverflow: await session.readScalar(truncated),
            trackCount: tracks.features.length
          }
        );
      });
    }
  })
];

export function prepareTraversedEdges(
  tracks: CatalogFeatureCollection,
  property: string,
  edgeCount: number,
  capacity: number
) {
  const edges = new Uint32Array(tracks.features.length * capacity).fill(0xffffffff);
  const counts = new Uint32Array(tracks.features.length);
  tracks.features.forEach((feature, index) => {
    const sequence = feature.properties?.[property];
    if (!Array.isArray(sequence) || sequence.length > capacity)
      throw new Error('Traversed edge list is missing or exceeds edges per track capacity');
    counts[index] = sequence.length;
    sequence.forEach((edge, slot) => {
      if (
        edge !== null &&
        edge !== 0xffffffff &&
        (!Number.isInteger(edge) || edge < 0 || edge >= edgeCount)
      )
        throw new Error('Traversed edge index is outside the source edge table');
      edges[index * capacity + slot] = edge === null ? 0xffffffff : edge;
    });
  });
  return {edges, counts};
}
