// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUBufferSurface,
  GPUConvexStraightSkeleton,
  GPUMaximumInscribedCircle,
  GPUMinimumClearance,
  GPULineNoding,
  GPUOffsetCurve,
  GPUOffsetCurveComponents,
  GPUOffsetExpansion,
  GPUVectorStraightSkeleton,
  getGPUOffsetCurveRowsPerVertex
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {loadFootprints, type Footprint, type FootprintKey} from './footprint-shape-lab-data';

export type FootprintShapeOptions = {
  footprint: FootprintKey;
  question: 'wavefront' | 'circle' | 'clearance' | 'move';
  offsetDistance: number;
  bufferDirection: 'inward' | 'outward';
  curveSide: 'left' | 'right';
  join: 'round' | 'mitre' | 'bevel';
  quadSegments: number;
  mitreLimit: number;
  budget: 'quick' | 'resolved';
  tolerance: number;
};

type Point = [number, number];
type ProvenanceMarker = {position: Point; ring: number; polygon: number; localIndex: number};
type LabResult = {
  paths: Point[][];
  secondaryPaths?: Point[][];
  points: Point[];
  status: string;
  detail: string;
  area?: number;
};
type Variant = {
  key: string;
  footprint: Footprint;
  graph: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  reader?: SummaryReader;
  buffers: Buffer[];
  byteSizes: number[];
  result: LabResult;
  generation: number;
};

const FLOAT_FORMATS = ['float32', 'float32x2', 'float32x4'] as const;
const UINT_FORMATS = ['uint32', 'uint32x2', 'uint32x4'] as const;
type FloatFormat = (typeof FLOAT_FORMATS)[number];
type UintFormat = (typeof UINT_FORMATS)[number];

const REFERENCE_VALUES = {
  courtyard: {circleRadius: 18.3556437407, minimumClearance: 0.1519976729},
  convex: {
    circleRadius: 28.6538888484,
    minimumClearance: 56.7707254306,
    bufferArea: {
      round: {outward: 6875.38742554, inward: 4411.04924439},
      mitre: {outward: 6889.44752088, inward: 4411.04924439},
      bevel: {outward: 6857.44275212, inward: 4411.04924439}
    }
  }
} as const;

function formatReferenceCheck(value: number, reference: number, tolerance: number): string {
  const difference = Math.abs(value - reference);
  return `GEOS reference ${reference.toFixed(3)} · Δ ${difference.toFixed(3)} / ${tolerance.toFixed(3)} m${difference <= tolerance ? ' · within tolerance' : ' · outside tolerance'}`;
}

function isWithinReferenceTolerance(value: number, reference: number, tolerance: number): boolean {
  return Math.abs(value - reference) <= tolerance;
}

/** Each question compiles only its own bounded graph. Results are decoded from GPU buffers. */
export async function createFootprintShapeLab(
  ctx: SceneContext<FootprintShapeOptions>
): Promise<SceneInstance<FootprintShapeOptions>> {
  const dataset = ctx.datasets.get('chicago-buildings');
  const footprints = loadFootprints(dataset);
  const resources = new SpatialAnalysisResources(ctx.device, 'footprint-shape-lab-input');
  const footprintBuffers = new Map<
    FootprintKey,
    {positions: Buffer; rings: Buffer; polygons: Buffer; features: Buffer}
  >();
  for (const footprint of Object.values(footprints)) {
    footprintBuffers.set(footprint.key, {
      positions: resources.createBuffer(`${footprint.key}-positions`, footprint.positions),
      rings: resources.createBuffer(`${footprint.key}-rings`, footprint.ringOffsets),
      polygons: resources.createBuffer(`${footprint.key}-polygons`, footprint.polygonOffsets),
      features: resources.createBuffer(`${footprint.key}-features`, footprint.featureOffsets)
    });
  }
  const cache = new Map<string, Variant>();
  let active: Variant | null = null;
  let cameraFootprint: FootprintKey = 'courtyard';
  let generation = 0;
  let destroyed = false;
  let needsEncode = false;
  let layers: ReturnType<SceneInstance<FootprintShapeOptions>['getLayers']> = [];
  const keyFor = (options: Readonly<FootprintShapeOptions>) =>
    `${options.footprint}:${options.question}:${options.offsetDistance}:${options.bufferDirection}:${options.curveSide}:${options.join}:${options.quadSegments}:${options.mitreLimit}:${options.budget}:${options.tolerance}`;
  const setPending = () => {
    layers = [];
    ctx.setReadout('shapeStatus', 'Compiling selected GPU question…');
    ctx.setReadout('shapeEvidence', 'Previous geometry cleared');
    ctx.setReadout('shapeSource', '');
    ctx.requestLayers();
  };

  const buildVariant = async (
    options: Readonly<FootprintShapeOptions>,
    identity: number
  ): Promise<Variant> => {
    const footprint = footprints[options.footprint];
    const key = keyFor(options);
    const resourceId = `${key.replaceAll(':', '-')}-${identity}`;
    const variantResources = new SpatialAnalysisResources(
      ctx.device,
      `footprint-shape-${resourceId}`
    );
    try {
      const graph = new GPUCommandGraph<void>(ctx.device, {
        id: `footprint-shape-${resourceId}`
      });
      const importBuffer = createGraphBufferImporter(graph);
      const source = footprintBuffers.get(footprint.key)!;
      const positions = importBuffer(source.positions, 'float32x2', footprint.vertexCount);
      const rings = importBuffer(source.rings, 'uint32', footprint.ringOffsets.length);
      const polygons = importBuffer(source.polygons, 'uint32', footprint.polygonOffsets.length);
      const features = importBuffer(source.features, 'uint32', footprint.featureOffsets.length);
      const inputPolygon = {
        positions,
        ringOffsets: rings,
        polygonOffsets: polygons,
        featureOffsets: features
      };
      const ownedBuffers: Buffer[] = [];
      const byteSizes: number[] = [];
      const output = <Format extends FloatFormat | UintFormat>(
        name: string,
        format: Format,
        length: number,
        bytesPerRow: number
      ): GraphDataView<Format> => {
        const buffer = variantResources.createBuffer(name, length * bytesPerRow);
        ownedBuffers.push(buffer);
        byteSizes.push(buffer.byteLength);
        return importBuffer(buffer, format, length);
      };
      const vertexOwners = output('vertex-to-polygon', 'uint32', footprint.vertexCount, 4);
      const vertexRingOwners = output('vertex-to-ring', 'uint32', footprint.vertexCount, 4);
      const ringOwners = output('ring-to-polygon', 'uint32', footprint.ringCount, 4);
      const localIndex = output('vertex-local-index', 'uint32', footprint.vertexCount, 4);
      const vertexRingLocalIndex = output(
        'vertex-ring-local-index',
        'uint32',
        footprint.vertexCount,
        4
      );
      const ringLocalIndex = output('ring-local-index', 'uint32', footprint.ringCount, 4);
      graph.add(
        new GPUOffsetExpansion({
          id: 'footprint-vertex-to-ring',
          offsets: rings,
          output: {
            owners: vertexRingOwners,
            localIndex: vertexRingLocalIndex,
            status: {
              count: output('ring-vertex-count', 'uint32', 1, 4),
              overflow: output('ring-vertex-overflow', 'uint32', 1, 4),
              requiredCount: output('ring-vertex-required', 'uint32', 1, 4)
            }
          }
        })
      );
      graph.add(
        new GPUOffsetExpansion({
          id: 'footprint-ring-to-polygon',
          offsets: polygons,
          output: {
            owners: ringOwners,
            localIndex: ringLocalIndex,
            status: {
              count: output('ring-count', 'uint32', 1, 4),
              overflow: output('ring-overflow', 'uint32', 1, 4),
              requiredCount: output('ring-required', 'uint32', 1, 4)
            }
          }
        })
      );
      graph.add(
        new GPUOffsetExpansion({
          id: 'footprint-vertex-to-polygon',
          offsets: rings,
          ownerMap: ringOwners,
          output: {
            owners: vertexOwners,
            localIndex,
            status: {
              count: output('vertex-count', 'uint32', 1, 4),
              overflow: output('vertex-overflow', 'uint32', 1, 4),
              requiredCount: output('vertex-required', 'uint32', 1, 4)
            }
          }
        })
      );
      let result: LabResult = {paths: [], points: [], status: 'GPU work pending', detail: ''};
      if (options.question === 'wavefront') {
        const maximumVertices = options.footprint === 'convex' ? 16 : 32;
        const maximumEvents = options.budget === 'quick' ? 2 : 32;
        const segmentCapacity = 2 * maximumVertices + 2 * maximumEvents;
        const segments = output('wavefront-arcs', 'float32x4', segmentCapacity, 16);
        const sourceEdges = output('wavefront-edge-ids', 'uint32x2', segmentCapacity, 8);
        const offsetPositions = output(
          'wavefront-offset',
          'float32x2',
          maximumVertices + maximumEvents * 2,
          8
        );
        const links = output('wavefront-links', 'uint32x2', maximumVertices + maximumEvents * 2, 8);
        const status = output('wavefront-status', 'uint32x4', 1, 16);
        if (options.footprint === 'convex') {
          graph.add(
            new GPUConvexStraightSkeleton({
              spatialContext: footprint.projection.spatialContext,
              positions,
              ringOffsets: rings,
              maximumVertices,
              maximumIterations: options.budget === 'quick' ? 1 : 16,
              segmentCapacity,
              offsetDistance: options.offsetDistance,
              eventTolerance: 1e-5,
              output: {
                segments,
                sourceEdges,
                offsetPositions: output('convex-offset', 'float32x2', 16, 8),
                status
              }
            })
          );
        } else {
          graph.add(
            new GPUVectorStraightSkeleton({
              ...inputPolygon,
              maximumVertices,
              maximumEvents,
              segmentCapacity,
              offsetDistance: options.offsetDistance,
              eventTolerance: 1e-5,
              output: {segments, sourceEdges, offsetPositions, offsetLinks: links, status}
            })
          );
        }
        result = {
          paths: [],
          points: [],
          status: 'Wavefront requested',
          detail:
            'Arc count, edge provenance, offset contour and budget status are read from GPU output.'
        };
      } else if (options.question === 'circle') {
        const circles = output('circle-center-radius-error', 'float32x4', 1, 16);
        const witnesses = output('circle-witness', 'float32x2', 1, 8);
        const status = output('circle-search-status', 'uint32', 1, 4);
        graph.add(
          new GPUMaximumInscribedCircle({
            ...inputPolygon,
            spatialContext: footprint.projection.spatialContext,
            tolerance: options.tolerance,
            cellCapacity: options.budget === 'quick' ? 32 : 512,
            maximumIterations: options.budget === 'quick' ? 32 : 2048,
            output: {circles, witnesses, status}
          })
        );
        result = {
          paths: [],
          points: [],
          status: 'Bounded circle search pending',
          detail: 'Center, radius, witness and remaining geometric search error.'
        };
      } else if (options.question === 'clearance') {
        const clearances = output('minimum-displacement', 'float32', 1, 4);
        const lines = output('clearance-witness', 'float32x4', 1, 16);
        const vertexIds = output('fragile-vertex-id', 'uint32', 1, 4);
        graph.add(
          new GPUMinimumClearance({
            spatialContext: footprint.projection.spatialContext,
            geometry: {kind: 'polygons', ...inputPolygon},
            output: {clearances, lines, vertexIds}
          })
        );
        result = {
          paths: [],
          points: [],
          status: 'Clearance pending',
          detail: 'Smallest vertex displacement to invalidity with its nearest witness.'
        };
      } else {
        const signedDistance =
          options.footprint === 'convex'
            ? options.bufferDirection === 'inward'
              ? -options.offsetDistance
              : options.offsetDistance
            : options.curveSide === 'right'
              ? -options.offsetDistance
              : options.offsetDistance;
        const parametersBuffer = variantResources.createBuffer(
          'offset-parameters',
          Float32Array.of(signedDistance, options.mitreLimit, 0, 0)
        );
        const parameters = importBuffer(parametersBuffer, 'float32', 4);
        ownedBuffers.push(parametersBuffer);
        byteSizes.push(parametersBuffer.byteLength);
        if (options.footprint === 'convex') {
          const surface = createPolygonSurface(
            output,
            positions,
            rings,
            polygons,
            features,
            parameters,
            footprint,
            options
          );
          graph.add(
            new GPUBufferSurface({
              spatialContext: footprint.projection.spatialContext,
              geometry: {
                kind: 'polygons',
                positions,
                ringOffsets: rings,
                polygonOffsets: polygons,
                featureOffsets: features
              },
              parameters,
              joinStyle: options.join,
              quadSegments: options.quadSegments,
              vertexTolerance: 0.05,
              capacity: {intersections: 512, nodedSegments: 2048},
              output: surface
            })
          );
          result = {
            paths: [],
            points: [],
            status: 'Polygon buffer pending',
            detail:
              'Topology-resolved polygon offset surface; signed distance follows polygon-buffer semantics.'
          };
        } else {
          const linePath = footprint.ringPaths[0].slice(0, -1);
          const linePositions = new Float32Array(linePath.length * 2);
          for (let index = 0; index < linePath.length; index++) {
            const point = footprint.projection.project(linePath[index][0], linePath[index][1]);
            linePositions.set(point, index * 2);
          }
          const lineBuffer = variantResources.createBuffer(
            'selected-boundary-chain',
            linePositions
          );
          const lineOffsets = variantResources.createBuffer(
            'boundary-chain-offsets',
            Uint32Array.of(0, linePath.length)
          );
          ownedBuffers.push(lineBuffer, lineOffsets);
          byteSizes.push(lineBuffer.byteLength, lineOffsets.byteLength);
          const lineView = importBuffer(lineBuffer, 'float32x2', linePath.length);
          const lineOffsetView = importBuffer(lineOffsets, 'uint32', 2);
          const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(
            options.join,
            options.quadSegments,
            'intersection'
          );
          const rawPositions = output(
            'raw-offset-curve',
            'float32x2',
            linePath.length * rowsPerVertex,
            8
          );
          const rawOffsetsBuffer = variantResources.createBuffer(
            'raw-line-offsets',
            Uint32Array.of(0, linePath.length * rowsPerVertex)
          );
          ownedBuffers.push(rawOffsetsBuffer);
          byteSizes.push(rawOffsetsBuffer.byteLength);
          const rawOffsets = importBuffer(rawOffsetsBuffer, 'uint32', 2);
          // This line is the selected source boundary chain; the result is an offset curve, not a path.
          graph.add(
            new GPUOffsetCurve({
              id: 'footprint-raw-offset-curve',
              geometryType: 'lines',
              spatialContext: footprint.projection.spatialContext,
              positions: lineView,
              lineOffsets: lineOffsetView,
              joinStyle: options.join,
              quadSegments: options.quadSegments,
              parameters,
              output: {
                positions: rawPositions,
                counts: output('offset-row-counts', 'uint32', linePath.length, 4)
              }
            })
          );
          const nodingIntersectionCapacity = 128;
          // GPULineNoding reserves at most input rows + eight vertices per intersection event.
          // Padded raw offset rows still occupy atomic segment rows; 64 truncated the 203-row case.
          const nodingSegmentCapacity = rawPositions.length + nodingIntersectionCapacity * 8;
          // Opt-in contract diagnostics duplicate raw-curve noding only during acceptance runs.
          if (new URLSearchParams(window.location.search).has('geometryDiagnostics')) {
            graph.add(
              new GPULineNoding({
                id: 'footprint-raw-curve-noding-diagnostic',
                lines: {kind: 'lines', positions: rawPositions, lineOffsets: rawOffsets},
                intersectionCapacity: nodingIntersectionCapacity,
                output: {
                  endpoints: output(
                    'diagnostic-noded-endpoints',
                    'float32x4',
                    nodingSegmentCapacity,
                    16
                  ),
                  sourceFeatureIds: output(
                    'diagnostic-noded-features',
                    'uint32',
                    nodingSegmentCapacity,
                    4
                  ),
                  sourceRingIds: output(
                    'diagnostic-noded-rings',
                    'uint32',
                    nodingSegmentCapacity,
                    4
                  ),
                  sourceEdgeIds: output(
                    'diagnostic-noded-edges',
                    'uint32',
                    nodingSegmentCapacity,
                    4
                  ),
                  sourceStartParameters: output(
                    'diagnostic-noded-starts',
                    'float32',
                    nodingSegmentCapacity,
                    4
                  ),
                  sourceEndParameters: output(
                    'diagnostic-noded-ends',
                    'float32',
                    nodingSegmentCapacity,
                    4
                  ),
                  status: {
                    count: output('diagnostic-noding-count', 'uint32', 1, 4),
                    requiredCount: output('diagnostic-noding-required', 'uint32', 1, 4),
                    overflow: output('diagnostic-noding-overflow', 'uint32', 1, 4),
                    candidateOverflow: output('diagnostic-noding-candidate', 'uint32', 1, 4),
                    uncertainCount: output('diagnostic-noding-uncertain', 'uint32', 1, 4)
                  }
                }
              })
            );
          }
          const componentPositions = output(
            'selected-component-points',
            'float32x2',
            linePath.length * rowsPerVertex * 2,
            8
          );
          const componentEdges = output(
            'selected-source-edge-ids',
            'uint32',
            linePath.length * rowsPerVertex * 2,
            4
          );
          const components = output('selected-components', 'uint32x4', 64, 16);
          const componentStatus = output('component-status', 'uint32x4', 2, 16);
          graph.add(
            new GPUOffsetCurveComponents({
              id: 'footprint-offset-curve-components',
              original: {positions: lineView, lineOffsets: lineOffsetView},
              raw: {positions: rawPositions, lineOffsets: rawOffsets},
              rowsPerVertex,
              distance: signedDistance,
              tolerance: Math.max(0.02, Math.abs(options.offsetDistance) * 0.01),
              segmentCapacity: nodingSegmentCapacity,
              intersectionCapacity: nodingIntersectionCapacity,
              output: {
                positions: componentPositions,
                sourceEdges: componentEdges,
                components,
                status: componentStatus
              }
            })
          );
          result = {
            paths: [],
            points: [],
            status: 'Selected curve components pending',
            detail:
              'Raw curve is resolved into connected selected components with original edge IDs.'
          };
        }
      }
      const compiled = variantResources.track(await graph.compileAsync());
      const variant: Variant = {
        key,
        footprint,
        graph: compiled,
        resources: variantResources,
        buffers: ownedBuffers,
        byteSizes,
        result,
        generation: identity
      };
      variant.reader = new SummaryReader(
        variantResources,
        `read-${identity}`,
        ownedBuffers.map((buffer, index) => ({buffer, size: byteSizes[index]})),
        bytes => {
          if (destroyed || active !== variant || variant.generation !== generation) return;
          if (options.question === 'circle') {
            const circleIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('circle-center-radius-error')
            );
            const circle = new Float32Array(
              bytes.slice(byteSizes.slice(0, circleIndex).reduce((sum, value) => sum + value, 0))
            );
            const witnessIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('circle-witness')
            );
            const witness = new Float32Array(
              bytes.slice(byteSizes.slice(0, witnessIndex).reduce((sum, value) => sum + value, 0))
            );
            const statusIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('circle-search-status')
            );
            const status = new Uint32Array(
              bytes.slice(byteSizes.slice(0, statusIndex).reduce((sum, value) => sum + value, 0))
            )[0];
            const finiteCircle =
              circle.slice(0, 4).every(Number.isFinite) && circle[2] >= 0 && circle[3] >= 0;
            const finiteWitness = witness.slice(0, 2).every(Number.isFinite);
            if ((status === 0 || status === 1) && finiteCircle && finiteWitness) {
              const center = footprint.projection.unproject(circle[0], circle[1]);
              const radius = Math.max(circle[2], 0);
              const ring: Point[] = [];
              for (let index = 0; index <= 64; index++) {
                const angle = (index * Math.PI * 2) / 64;
                ring.push(
                  footprint.projection.unproject(
                    circle[0] + Math.cos(angle) * radius,
                    circle[1] + Math.sin(angle) * radius
                  )
                );
              }
              variant.result = {
                paths: [ring],
                points: [center, footprint.projection.unproject(witness[0], witness[1])],
                status:
                  status === 0
                    ? `Search tolerance met · GEOS reference ${isWithinReferenceTolerance(radius, REFERENCE_VALUES[options.footprint].circleRadius, Math.max(options.tolerance, 0.25)) ? 'within tolerance' : 'outside tolerance'}`
                    : 'Budget exhausted · bounded answer shown',
                detail: `Radius ${radius.toFixed(2)} m · remaining geometric search error ${circle[3].toFixed(3)} m · ${status === 0 ? formatReferenceCheck(radius, REFERENCE_VALUES[options.footprint].circleRadius, Math.max(options.tolerance, 0.25)) : 'reference acceptance withheld because the search budget was exhausted'} · f32 rounding is not certified.`
              };
            } else
              variant.result = {
                paths: [],
                points: [],
                status: status === 2 ? 'Empty geometry' : 'Non-finite input',
                detail: 'No circle geometry is drawn for this status.'
              };
          } else if (options.question === 'clearance') {
            const lineIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('clearance-witness')
            );
            const clearanceIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('minimum-displacement')
            );
            const idIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('fragile-vertex-id')
            );
            const getBufferByteOffset = (index: number) =>
              byteSizes.slice(0, index).reduce((sum, value) => sum + value, 0);
            const clearance = new Float32Array(
              bytes.slice(
                getBufferByteOffset(clearanceIndex),
                getBufferByteOffset(clearanceIndex) + 4
              )
            )[0];
            const witness = new Float32Array(
              bytes.slice(getBufferByteOffset(lineIndex), getBufferByteOffset(lineIndex) + 16)
            );
            const vertexId = new Uint32Array(
              bytes.slice(getBufferByteOffset(idIndex), getBufferByteOffset(idIndex) + 4)
            )[0];
            if (
              vertexId < footprint.vertexCount &&
              Number.isFinite(clearance) &&
              clearance >= 0 &&
              witness.every(Number.isFinite)
            )
              variant.result = {
                paths: [
                  [
                    footprint.projection.unproject(witness[0], witness[1]),
                    footprint.projection.unproject(witness[2], witness[3])
                  ]
                ],
                points: [
                  footprint.projection.unproject(witness[0], witness[1]),
                  footprint.projection.unproject(witness[2], witness[3])
                ],
                status: `Minimum displacement ${clearance.toFixed(3)} m · GEOS reference ${isWithinReferenceTolerance(clearance, REFERENCE_VALUES[options.footprint].minimumClearance, 0.05) ? 'within tolerance' : 'outside tolerance'}`,
                detail: `Fragile stored vertex ${vertexId} to its nearest non-adjacent segment or vertex · ${formatReferenceCheck(clearance, REFERENCE_VALUES[options.footprint].minimumClearance, 0.05)}.`
              };
            else
              variant.result = {
                paths: [],
                points: [],
                status: 'No finite clearance witness',
                detail: ''
              };
          } else if (options.question === 'wavefront') {
            const index = ownedBuffers.findIndex(buffer => buffer.id.includes('wavefront-status'));
            const statusByteOffset = byteSizes
              .slice(0, index)
              .reduce((sum, value) => sum + value, 0);
            const status = new Uint32Array(bytes.slice(statusByteOffset, statusByteOffset + 16));
            const arcsIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('wavefront-arcs')
            );
            const arcsAt = byteSizes.slice(0, arcsIndex).reduce((sum, value) => sum + value, 0);
            const arcs = new Float32Array(bytes.slice(arcsAt, arcsAt + byteSizes[arcsIndex]));
            const offsetName =
              options.footprint === 'convex' ? 'convex-offset' : 'wavefront-offset';
            const offsetIndex = ownedBuffers.findIndex(buffer => buffer.id.includes(offsetName));
            const offsetCapacity = byteSizes[offsetIndex] / 8;
            const offsetAt = byteSizes.slice(0, offsetIndex).reduce((sum, value) => sum + value, 0);
            let wavefrontComplete =
              status[3] === 0 &&
              status[0] === status[1] &&
              status[0] <= arcs.length / 4 &&
              status[2] <= offsetCapacity;
            if (wavefrontComplete) {
              wavefrontComplete = arcs.slice(0, status[0] * 4).every(Number.isFinite);
            }
            const paths: Point[][] = [];
            for (let arc = 0; wavefrontComplete && arc < status[0]; arc++)
              paths.push([
                footprint.projection.unproject(arcs[arc * 4], arcs[arc * 4 + 1]),
                footprint.projection.unproject(arcs[arc * 4 + 2], arcs[arc * 4 + 3])
              ]);
            const offsetValues = new Float32Array(
              bytes.slice(offsetAt, offsetAt + byteSizes[offsetIndex])
            );
            const offsetPaths: Point[][] = [];
            let convexOffsetArea: number | undefined;
            if (wavefrontComplete && options.footprint === 'convex') {
              wavefrontComplete = offsetValues.slice(0, status[2] * 2).every(Number.isFinite);
              const offsetPath: Point[] = [];
              for (let index = 0; index < status[2]; index++)
                offsetPath.push(
                  footprint.projection.unproject(
                    offsetValues[index * 2],
                    offsetValues[index * 2 + 1]
                  )
                );
              if (offsetPath.length > 2) offsetPaths.push([...offsetPath, offsetPath[0]]);
              if (wavefrontComplete) {
                let doubledArea = 0;
                for (let vertex = 0; vertex < status[2]; vertex++) {
                  const nextVertex = (vertex + 1) % status[2];
                  doubledArea +=
                    offsetValues[vertex * 2] * offsetValues[nextVertex * 2 + 1] -
                    offsetValues[nextVertex * 2] * offsetValues[vertex * 2 + 1];
                }
                convexOffsetArea = Math.abs(doubledArea / 2);
              }
            } else if (wavefrontComplete) {
              const linksIndex = ownedBuffers.findIndex(buffer =>
                buffer.id.includes('wavefront-links')
              );
              const linksAt = byteSizes.slice(0, linksIndex).reduce((sum, value) => sum + value, 0);
              const links = new Uint32Array(bytes.slice(linksAt, linksAt + byteSizes[linksIndex]));
              // Vector wavefront snapshots retain fixed slots after events; activeCount is not
              // a packed-prefix length. Only links identify active slots and their complete loops.
              const activeSlots = new Set<number>();
              for (let slot = 0; slot < offsetCapacity; slot++) {
                const previous = links[slot * 2];
                const next = links[slot * 2 + 1];
                if (previous !== 0xffffffff || next !== 0xffffffff) activeSlots.add(slot);
              }
              wavefrontComplete = activeSlots.size === status[2];
              for (const slot of activeSlots) {
                const previous = links[slot * 2];
                const next = links[slot * 2 + 1];
                if (
                  !activeSlots.has(previous) ||
                  !activeSlots.has(next) ||
                  links[previous * 2 + 1] !== slot ||
                  links[next * 2] !== slot ||
                  !Number.isFinite(offsetValues[slot * 2]) ||
                  !Number.isFinite(offsetValues[slot * 2 + 1])
                ) {
                  wavefrontComplete = false;
                }
              }
              const used = new Set<number>();
              for (const startSlot of activeSlots) {
                if (!wavefrontComplete || used.has(startSlot)) continue;
                const offsetPath: Point[] = [];
                let current = startSlot;
                while (!used.has(current) && used.size < activeSlots.size) {
                  used.add(current);
                  offsetPath.push(
                    footprint.projection.unproject(
                      offsetValues[current * 2],
                      offsetValues[current * 2 + 1]
                    )
                  );
                  current = links[current * 2 + 1];
                }
                if (current !== startSlot || offsetPath.length < 3) {
                  wavefrontComplete = false;
                  break;
                }
                offsetPaths.push([...offsetPath, offsetPath[0]]);
              }
              wavefrontComplete &&= used.size === activeSlots.size;
            }
            if (!wavefrontComplete) {
              paths.length = 0;
              offsetPaths.length = 0;
              convexOffsetArea = undefined;
            }
            const edgeIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('wavefront-edge-ids')
            );
            const edgeAt = byteSizes.slice(0, edgeIndex).reduce((sum, value) => sum + value, 0);
            const edgeIds = new Uint32Array(bytes.slice(edgeAt, edgeAt + byteSizes[edgeIndex]));
            variant.result = {
              paths,
              secondaryPaths: offsetPaths,
              points: [],
              status: !wavefrontComplete
                ? `Withheld · written/required arcs ${status[0]}/${status[1]} · offset vertices ${status[2]}/${offsetCapacity} · flags ${status[3]}`
                : `${status[0]} arcs · ${status[2]} active offset vertices`,
              detail: `Required arcs ${status[1]}; first source-edge pair ${edgeIds[0] ?? 0xffffffff} / ${edgeIds[1] ?? 0xffffffff}. Requested inward offset ${options.offsetDistance} m.${convexOffsetArea === undefined ? '' : ` Convex offset contour area ${convexOffsetArea.toFixed(3)} m².`}${!wavefrontComplete && status[3] === 0 ? ' Readback topology is incomplete or invalid.' : ''}`
            };
          } else {
            const outputPositions = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('selected-component-points')
            );
            const topology = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('selected-components')
            );
            const statusIndex = ownedBuffers.findIndex(buffer =>
              buffer.id.includes('component-status')
            );
            if (outputPositions >= 0) {
              const data = new Float32Array(bytes);
              const indexes = new Uint32Array(bytes);
              const first =
                byteSizes.slice(0, outputPositions).reduce((sum, value) => sum + value, 0) / 4;
              const topologyAt =
                byteSizes.slice(0, topology).reduce((sum, value) => sum + value, 0) / 4;
              const statusAt =
                byteSizes.slice(0, statusIndex).reduce((sum, value) => sum + value, 0) / 4;
              const componentStatus = indexes.slice(statusAt, statusAt + 8);
              const componentCount = componentStatus[0];
              const componentCapacity = byteSizes[topology] / 16;
              const componentVertexCapacity = byteSizes[outputPositions] / 8;
              let componentRangesValid =
                componentCount <= componentCapacity &&
                componentStatus[1] <= componentVertexCapacity;
              for (
                let component = 0;
                component < componentCount && componentRangesValid;
                component++
              ) {
                const row = topologyAt + component * 4;
                const firstVertex = indexes[row];
                const vertexCount = indexes[row + 1];
                componentRangesValid =
                  firstVertex <= componentStatus[1] &&
                  vertexCount <= componentStatus[1] - firstVertex &&
                  vertexCount <= componentVertexCapacity - firstVertex;
                for (let point = 0; point < vertexCount && componentRangesValid; point++)
                  componentRangesValid =
                    Number.isFinite(data[first + (firstVertex + point) * 2]) &&
                    Number.isFinite(data[first + (firstVertex + point) * 2 + 1]);
              }
              const componentComplete =
                componentStatus[4] === 0 &&
                componentStatus[5] === 0 &&
                componentStatus[0] === componentStatus[2] &&
                componentStatus[1] === componentStatus[3] &&
                componentRangesValid;
              const paths: Point[][] = [];
              for (
                let component = 0;
                componentComplete && component < componentCount;
                component++
              ) {
                const row = topologyAt + component * 4;
                const vertex = indexes[row];
                const count = indexes[row + 1];
                const path: Point[] = [];
                for (let point = 0; point < count; point++)
                  path.push(
                    footprint.projection.unproject(
                      data[first + (vertex + point) * 2],
                      data[first + (vertex + point) * 2 + 1]
                    )
                  );
                if (path.length > 1) paths.push(path);
              }
              const rawIndex = ownedBuffers.findIndex(buffer =>
                buffer.id.includes('raw-offset-curve')
              );
              const rawAt = byteSizes.slice(0, rawIndex).reduce((sum, value) => sum + value, 0) / 4;
              const countIndex = ownedBuffers.findIndex(buffer =>
                buffer.id.includes('offset-row-counts')
              );
              const countAt =
                byteSizes.slice(0, countIndex).reduce((sum, value) => sum + value, 0) / 4;
              const rawCounts = indexes.slice(countAt, countAt + footprint.ringOffsets[1] - 1);
              const rawPath: Point[] = [];
              const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(
                options.join,
                options.quadSegments
              );
              const rawCapacity = byteSizes[rawIndex] / 8;
              const rawComplete =
                rawCounts.length === footprint.ringOffsets[1] - 1 &&
                rawCounts.every(count => count <= rowsPerVertex) &&
                rawCounts.reduce((sum, count) => sum + count, 0) <= rawCapacity;
              const rawUsedRows = rawCounts.reduce((sum, count) => sum + count, 0);
              const rawFloats = new Float32Array(
                bytes.slice(rawAt * 4, rawAt * 4 + rawUsedRows * 8)
              );
              const rawOutputComplete = rawComplete && rawFloats.every(Number.isFinite);
              for (let vertex = 0; rawOutputComplete && vertex < rawCounts.length; vertex++) {
                for (let row = 0; row < rawCounts[vertex]; row++) {
                  const rawRow = vertex * rowsPerVertex + row;
                  const point: Point = [data[rawAt + rawRow * 2], data[rawAt + rawRow * 2 + 1]];
                  if (
                    rawPath.length === 0 ||
                    rawPath[rawPath.length - 1][0] !== point[0] ||
                    rawPath[rawPath.length - 1][1] !== point[1]
                  )
                    rawPath.push(footprint.projection.unproject(point[0], point[1]));
                }
              }
              const sourceEdgeIndex = ownedBuffers.findIndex(buffer =>
                buffer.id.includes('selected-source-edge-ids')
              );
              const sourceEdgeAt =
                byteSizes.slice(0, sourceEdgeIndex).reduce((sum, value) => sum + value, 0) / 4;
              const sourceEdge = indexes[sourceEdgeAt];
              const diagnosticValues = [
                'count',
                'required',
                'overflow',
                'candidate',
                'uncertain'
              ].map(name => {
                const bufferIndex = ownedBuffers.findIndex(buffer =>
                  buffer.id.includes(`diagnostic-noding-${name}`)
                );
                if (bufferIndex < 0) return null;
                const byteOffset = byteSizes
                  .slice(0, bufferIndex)
                  .reduce((sum, value) => sum + value, 0);
                return new Uint32Array(bytes, byteOffset, 1)[0];
              });
              const diagnosticDetail =
                diagnosticValues[0] === null
                  ? ''
                  : ` Raw noding diagnostic: written ${diagnosticValues[0]}, required ${diagnosticValues[1]}, overflow ${diagnosticValues[2]}, candidateOverflow ${diagnosticValues[3]}, uncertain ${diagnosticValues[4]}.`;
              variant.result = {
                paths,
                secondaryPaths: rawOutputComplete && rawPath.length > 1 ? [rawPath] : [],
                points: [],
                status: !componentComplete
                  ? `Withheld · written/required components ${componentStatus[0]}/${componentStatus[2]}, vertices ${componentStatus[1]}/${componentStatus[3]} · overflow/noding ${componentStatus[4]}/${componentStatus[5]}`
                  : `${componentStatus[0]} written / ${componentStatus[2]} required components · ${componentStatus[1]} / ${componentStatus[3]} vertices`,
                detail: `Signed distance ${options.curveSide === 'left' ? '+' : '-'}${options.offsetDistance} m (${options.curveSide} of source direction). Raw curve ${rawOutputComplete ? `${rawUsedRows} rows` : 'withheld because row counts or coordinates exceeded capacity'}; first selected source-edge ID ${sourceEdge ?? 0xffffffff}.${diagnosticDetail}`
              };
            } else if (options.footprint === 'convex') {
              const findOffset = (name: string) =>
                byteSizes
                  .slice(
                    0,
                    ownedBuffers.findIndex(buffer => buffer.id.includes(name))
                  )
                  .reduce((sum, value) => sum + value, 0);
              const readUint = (name: string, count: number) => {
                return new Uint32Array(bytes.slice(findOffset(name), findOffset(name) + count * 4));
              };
              const featureCount = readUint('buffer-count', 1)[0];
              const featureOverflow = readUint('buffer-overflow', 1)[0];
              const requiredRingCount = readUint('buffer-required-count', 1)[0];
              const candidateOverflow = readUint('buffer-candidate-overflow', 1)[0];
              const uncertainCount = readUint('buffer-uncertain', 1)[0];
              const boundaryCount = readUint('buffer-boundary-count', 1)[0];
              const boundaryOverflow = readUint('buffer-boundary-overflow', 1)[0];
              const boundaryRequired = readUint('buffer-boundary-required', 1)[0];
              const boundaryCandidateOverflow = readUint(
                'buffer-boundary-candidate-overflow',
                1
              )[0];
              const complete =
                featureOverflow === 0 &&
                candidateOverflow === 0 &&
                uncertainCount === 0 &&
                boundaryOverflow === 0 &&
                boundaryCandidateOverflow === 0 &&
                requiredRingCount <= 32 &&
                boundaryCount === boundaryRequired &&
                featureCount <= 32 &&
                boundaryCount <= Math.max(128, footprint.vertexCount * 24);
              const featureOffsets =
                complete && featureCount <= 32
                  ? readUint('buffer-feature-offsets', featureCount + 1)
                  : new Uint32Array();
              const polygonCount = featureOffsets[featureCount] ?? 0;
              const polygonOffsets = complete
                ? readUint('buffer-polygon-offsets', polygonCount + 1)
                : new Uint32Array();
              const ringCount = polygonOffsets[polygonCount] ?? 0;
              const ringOffsets = complete
                ? readUint('buffer-ring-offsets', ringCount + 1)
                : new Uint32Array();
              const geometryComplete =
                complete &&
                featureCount <= 32 &&
                polygonCount <= 32 &&
                ringCount <= 32 &&
                ringCount === requiredRingCount &&
                ringOffsets[ringCount] <= Math.max(128, footprint.vertexCount * 24);
              const positionsAt = findOffset('buffer-positions');
              const positionValues = new Float32Array(
                bytes.slice(positionsAt, positionsAt + ringOffsets[ringCount] * 8)
              );
              const paths: Point[][] = [];
              let signedArea = 0;
              for (let ring = 0; geometryComplete && ring < ringCount; ring++) {
                const first = ringOffsets[ring];
                const last = ringOffsets[ring + 1];
                const path: Point[] = [];
                let ringAreaTwice = 0;
                for (let row = first; row < last; row++) {
                  const next = row + 1 === last ? first : row + 1;
                  const positionX = positionValues[row * 2];
                  const positionY = positionValues[row * 2 + 1];
                  ringAreaTwice +=
                    positionX * positionValues[next * 2 + 1] - positionValues[next * 2] * positionY;
                  path.push(footprint.projection.unproject(positionX, positionY));
                }
                signedArea += ringAreaTwice / 2;
                if (path.length > 2) paths.push([...path, path[0]]);
              }
              const area = Math.abs(signedArea);
              const hasBufferReference =
                options.offsetDistance === 4 &&
                (options.join !== 'round' || options.quadSegments === 8) &&
                (options.join !== 'mitre' || options.mitreLimit === 5);
              const bufferReferenceDetail = hasBufferReference
                ? formatReferenceCheck(
                    area,
                    REFERENCE_VALUES.convex.bufferArea[options.join][options.bufferDirection],
                    15
                  ).replace(' m', ' m²')
                : 'GEOS comparison is defined for the 4 m reference settings only';
              variant.result = {
                paths: geometryComplete ? paths : [],
                points: [],
                area: geometryComplete ? area : undefined,
                status: !geometryComplete
                  ? `Withheld · buffer incomplete; features ${featureCount}, rings required ${requiredRingCount}/32, boundary ${boundaryCount}/${boundaryRequired}, candidate ${candidateOverflow}/${boundaryCandidateOverflow}, uncertain ${uncertainCount}`
                  : `${polygonCount} GPU buffer polygon(s) · ${ringCount} rings${hasBufferReference ? ` · GEOS reference ${isWithinReferenceTolerance(area, REFERENCE_VALUES.convex.bufferArea[options.join][options.bufferDirection], 15) ? 'within tolerance' : 'outside tolerance'}` : ''}`,
                detail: geometryComplete
                  ? `Topology-resolved offset surface area ${area.toFixed(1)} m² · ${bufferReferenceDetail} · join ${options.join} · ${options.quadSegments} quarter-circle segments · mitre limit ${options.mitreLimit}.`
                  : `Buffer status overflow ${featureOverflow}, candidate overflow ${candidateOverflow}, uncertain ${uncertainCount}; required rings ${requiredRingCount}; boundary rows ${boundaryCount}/${boundaryRequired}, overflow ${boundaryOverflow}, candidate overflow ${boundaryCandidateOverflow}. Area is withheld.`
              };
            }
          }
          const offsetOf = (name: string) => {
            const index = ownedBuffers.findIndex(buffer => buffer.id.includes(name));
            return byteSizes.slice(0, index).reduce((sum, value) => sum + value, 0);
          };
          const readProvenance = (name: string, count: number) =>
            new Uint32Array(bytes.slice(offsetOf(name), offsetOf(name) + count * 4));
          const vertexRing = readProvenance('vertex-to-ring', footprint.vertexCount);
          const vertexPolygon = readProvenance('vertex-to-polygon', footprint.vertexCount);
          const vertexLocalIndex = readProvenance('vertex-ring-local-index', footprint.vertexCount);
          const ringPolygon = readProvenance('ring-to-polygon', footprint.ringCount);
          const ringLocalIndex = readProvenance('ring-local-index', footprint.ringCount);
          const polygonVertexRows = readProvenance('vertex-count', 1)[0];
          const polygonVertexRequired = readProvenance('vertex-required', 1)[0];
          const polygonVertexOverflow = readProvenance('vertex-overflow', 1)[0];
          const vertexRows = readProvenance('ring-vertex-count', 1)[0];
          const vertexRequired = readProvenance('ring-vertex-required', 1)[0];
          const vertexOverflow = readProvenance('ring-vertex-overflow', 1)[0];
          const ringRows = readProvenance('ring-count', 1)[0];
          const ringRequired = readProvenance('ring-required', 1)[0];
          const ringOverflow = readProvenance('ring-overflow', 1)[0];
          const provenanceComplete =
            polygonVertexOverflow === 0 &&
            vertexOverflow === 0 &&
            ringOverflow === 0 &&
            polygonVertexRows === footprint.vertexCount &&
            vertexRows === footprint.vertexCount &&
            ringRows === footprint.ringCount &&
            polygonVertexRequired === footprint.vertexCount &&
            vertexRequired === footprint.vertexCount &&
            ringRequired === footprint.ringCount;
          const provenancePoints: ProvenanceMarker[] = Array.from(
            {length: provenanceComplete ? footprint.vertexCount : 0},
            (_, index) => ({
              position: footprint.projection.unproject(
                footprint.positions[index * 2],
                footprint.positions[index * 2 + 1]
              ),
              ring: vertexRing[index],
              polygon: vertexPolygon[index],
              localIndex: vertexLocalIndex[index]
            })
          );
          const footprintPaths = footprint.ringPaths.map(path => path.slice());
          const drawnPaths = [...footprintPaths, ...variant.result.paths];
          layers = [
            new PathLayer({
              id: `footprint-source-${identity}`,
              data: drawnPaths.slice(0, footprintPaths.length),
              getPath: item => item,
              getColor: [45, 56, 53, 225],
              getWidth: 2.2,
              widthUnits: 'pixels',
              pickable: false
            }),
            new PathLayer({
              id: `footprint-output-${identity}`,
              data: variant.result.paths,
              getPath: item => item,
              getColor: options.question === 'circle' ? [46, 125, 151, 240] : [190, 91, 62, 240],
              getWidth: options.question === 'circle' ? 2 : 3,
              widthUnits: 'pixels',
              pickable: false
            }),
            new PathLayer({
              id: `footprint-secondary-${identity}`,
              data: variant.result.secondaryPaths ?? [],
              getPath: item => item,
              getColor: options.question === 'move' ? [194, 133, 92, 190] : [46, 125, 151, 240],
              getWidth: 2.5,
              widthUnits: 'pixels',
              pickable: false
            }),
            new ScatterplotLayer({
              id: `footprint-witness-${identity}`,
              data: variant.result.points,
              getPosition: item => item,
              getRadius: 5,
              radiusUnits: 'pixels',
              getFillColor: [43, 120, 151, 255],
              stroked: true,
              getLineColor: [255, 255, 255, 255],
              lineWidthMinPixels: 1,
              pickable: false
            }),
            new ScatterplotLayer({
              id: `footprint-provenance-${identity}`,
              data: provenancePoints,
              getPosition: item => item.position,
              getRadius: item => 2 + Math.min(item.localIndex, 8) * 0.2,
              radiusUnits: 'pixels',
              getFillColor: item => (item.ring === 0 ? [45, 56, 53, 190] : [46, 125, 151, 220]),
              stroked: true,
              getLineColor: [250, 247, 238, 230],
              lineWidthMinPixels: 0.5,
              pickable: false
            })
          ];
          ctx.setReadout('shapeStatus', variant.result.status);
          ctx.setReadout('shapeEvidence', variant.result.detail);
          ctx.setReadout(
            'shapeSource',
            `Building ${footprint.sourceId} · ${footprint.polygonCount} polygon · ${footprint.ringCount} rings · ${footprint.vertexCount} stored / ${footprint.uniqueVertexCount} unique positions` +
              ` · vertices→rings ${vertexRows}/${footprint.vertexCount} (required ${vertexRequired}, overflow ${vertexOverflow}); rings→polygons ${ringRows}/${footprint.ringCount} (required ${ringRequired}, overflow ${ringOverflow}); composed vertices→polygons ${polygonVertexRows}/${footprint.vertexCount} (required ${polygonVertexRequired}, overflow ${polygonVertexOverflow})` +
              (provenanceComplete
                ? `; vertex 0 → ring ${vertexRing[0]}, local ${vertexLocalIndex[0]}, polygon ${vertexPolygon[0]}; ring 1 → polygon ${ringPolygon[1] ?? 0xffffffff}, local ${ringLocalIndex[1] ?? 0xffffffff}`
                : '; ownership sample withheld because topology expansion was incomplete')
          );
          ctx.requestLayers();
        }
      );
      return variant;
    } catch (error) {
      variantResources.destroy();
      throw error;
    }
  };

  const activate = async (options: Readonly<FootprintShapeOptions>) => {
    const identity = ++generation;
    const optionSnapshot = {...options};
    const key = keyFor(optionSnapshot);
    active = null;
    needsEncode = false;
    setPending();
    let variant = cache.get(key);
    if (!variant) {
      try {
        variant = await buildVariant(optionSnapshot, identity);
      } catch (error) {
        if (destroyed || identity !== generation) return;
        const message = `GPU graph could not compile: ${error instanceof Error ? error.message : String(error)}`;
        ctx.setStatus(message);
        ctx.setReadout('shapeStatus', message);
        ctx.setReadout('shapeEvidence', 'No geometry is drawn for a failed graph.');
        ctx.setReadout('shapeSource', '');
        return;
      }
      if (destroyed || identity !== generation) {
        variant.resources.destroy();
        return;
      }
      cache.set(key, variant);
      while (cache.size > 8) {
        const oldest = cache.keys().next().value;
        if (!oldest || oldest === key) break;
        const stale = cache.get(oldest);
        stale?.reader?.release();
        stale?.resources.destroy();
        cache.delete(oldest);
      }
    }
    active = variant;
    ctx.setStatus('');
    active.generation = identity;
    active.reader?.markStale();
    needsEncode = true;
    if (cameraFootprint !== options.footprint) {
      cameraFootprint = options.footprint;
      const selectedFootprint = footprints[options.footprint];
      ctx.flyTo(
        {
          longitude: selectedFootprint.longitude,
          latitude: selectedFootprint.latitude,
          zoom: 19,
          pitch: 0,
          bearing: 0
        },
        {transitionMs: 700}
      );
    }
    ctx.requestLayers();
  };

  await activate(ctx.options);
  return {
    getCompiledGraphs: () => [...cache.values()].map(variant => variant.graph),
    encode(commandEncoder) {
      if (!active) return;
      if (needsEncode) {
        active.graph.encode(commandEncoder, {parameters: undefined});
        needsEncode = false;
      }
      active.reader?.flush(commandEncoder);
    },
    getLayers: () => layers,
    setOption(id, _value, state) {
      if (
        id === 'footprint' ||
        id === 'question' ||
        id === 'offsetDistance' ||
        id === 'bufferDirection' ||
        id === 'curveSide' ||
        id === 'join' ||
        id === 'quadSegments' ||
        id === 'mitreLimit' ||
        id === 'budget' ||
        id === 'tolerance'
      )
        void activate(state);
    },
    destroy() {
      destroyed = true;
      generation++;
      for (const variant of cache.values()) {
        variant.reader?.release();
        variant.resources.destroy();
      }
      cache.clear();
      resources.destroy();
    }
  };
}

function createPolygonSurface(
  output: <Format extends FloatFormat | UintFormat>(
    name: string,
    format: Format,
    length: number,
    bytesPerRow: number
  ) => GraphDataView<Format>,
  positions: GraphDataView<'float32x2'>,
  rings: GraphDataView<'uint32'>,
  polygons: GraphDataView<'uint32'>,
  features: GraphDataView<'uint32'>,
  _parameters: GraphDataView<'float32'>,
  footprint: Footprint,
  _options: Readonly<FootprintShapeOptions>
) {
  const vertexCapacity = Math.max(128, footprint.vertexCount * 24);
  const ringCapacity = 32;
  const polygonCapacity = 32;
  return {
    geometry: {
      kind: 'polygons' as const,
      positions: output('buffer-positions', 'float32x2', vertexCapacity, 8),
      ringOffsets: output('buffer-ring-offsets', 'uint32', ringCapacity + 1, 4),
      polygonOffsets: output('buffer-polygon-offsets', 'uint32', polygonCapacity + 1, 4),
      featureOffsets: output('buffer-feature-offsets', 'uint32', polygonCapacity + 1, 4),
      sourceIds: output('buffer-source-ids', 'uint32', polygonCapacity, 4)
    },
    status: {
      count: output('buffer-count', 'uint32', 1, 4),
      overflow: output('buffer-overflow', 'uint32', 1, 4),
      requiredCount: output('buffer-required-count', 'uint32', 1, 4),
      candidateOverflow: output('buffer-candidate-overflow', 'uint32', 1, 4),
      uncertainCount: output('buffer-uncertain', 'uint32', 1, 4)
    },
    boundary: {
      endpoints: output('buffer-boundary-endpoints', 'float32x4', vertexCapacity, 16),
      operandIds: output('buffer-boundary-operands', 'uint32', vertexCapacity, 4),
      sourceFeatureIds: output('buffer-boundary-feature-ids', 'uint32', vertexCapacity, 4),
      status: {
        count: output('buffer-boundary-count', 'uint32', 1, 4),
        overflow: output('buffer-boundary-overflow', 'uint32', 1, 4),
        requiredCount: output('buffer-boundary-required', 'uint32', 1, 4),
        candidateOverflow: output('buffer-boundary-candidate-overflow', 'uint32', 1, 4)
      }
    }
  };
}
