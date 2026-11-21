// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGeographicDistribution,
  GPULabelPoint,
  GPUMaximumInscribedCircle,
  GPUMinimumBounds,
  GPUMinimumClearance,
  getGPUGeographicDistributionParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogGeometryText,
  getLineParts,
  getPointPositions,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeature,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {packCatalogPolygons} from './catalog-gpu-pack';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  closeRing,
  collectPositions,
  getFeatureVertices,
  getSourceColumns,
  isPointInPolygonRings,
  makeCirclePositions,
  projectPolygonParts,
  readPropertyNumber,
  readPropertyText,
  type PlanarPoint
} from './catalog-platform3-a-shared';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

function makePolygonText(frame: LocalFrame, ring: readonly PlanarPoint[]): string {
  const positions = closeRing(ring.map(point => frame.inverse(point[0], point[1])));
  return makeCatalogGeometryText('Polygon', [positions]);
}

/**
 * GPUMinimumBounds with one group per feature over all of its vertices, in the metric local frame
 * of the CPU reference. Rectangle width and length are the sorted sides; the angle is the direction
 * of the defining hull edge, which differs from the CPU value only for ties (squares).
 */
export async function runGpuMinimumBounds(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  shape: 'rectangle' | 'circle',
  coordinateSystem: CatalogCoordinateSystem
): Promise<CatalogRunOutput> {
  const frame = makeLocalFrame(coordinateSystem, collectPositions(features), {shift: 'center'});
  const vertices: number[] = [];
  const labels: number[] = [];
  features.features.forEach((feature, row) => {
    for (const position of getFeatureVertices(feature)) {
      const [x, y] = frame.forward(position);
      vertices.push(x, y);
      labels.push(row);
    }
  });
  if (!labels.length) throw new CatalogGpuUnsupportedError('No finite vertices');
  const groupCount = features.features.length;
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-minimum-bounds',
    async session => {
      const corners = session.output('corners', 'float32x2', groupCount * 4);
      const sizes = session.output('sizes', 'float32x4', groupCount);
      const circles = session.output('circles', 'float32x3', groupCount);
      const hullSizes = session.output('hull-sizes', 'uint32', groupCount);
      const overflow = session.output('overflow', 'uint32', 1);
      session.graph.add(
        new GPUMinimumBounds({
          positions: session.input('positions', Float32Array.from(vertices), 'float32x2'),
          labels: session.input('labels', Uint32Array.from(labels), 'uint32'),
          groupCount,
          maximumHullVertices: 4096,
          output:
            shape === 'rectangle'
              ? {
                  rectangleCorners: corners.view,
                  rectangleSizes: sizes.view,
                  hullSizes: hullSizes.view,
                  status: {overflow: overflow.view}
                }
              : {
                  circles: circles.view,
                  hullSizes: hullSizes.view,
                  status: {overflow: overflow.view}
                }
        })
      );
      session.run();
      return {
        corners: await session.readFloat32(corners),
        sizes: await session.readFloat32(sizes),
        circles: await session.readFloat32(circles),
        hullSizes: await session.readUint32(hullSizes),
        overflow: await session.readScalar(overflow)
      };
    }
  );
  // A hull above the cap yields NaN rows; the CPU engine handles those.
  if (result.overflow) throw new CatalogGpuUnsupportedError('A convex hull exceeded the GPU cap');
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'minimum-bounds',
    'Minimum bounds',
    {
      bounds_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      shape: 'string',
      area: 'float64',
      width: 'float64',
      length: 'float64',
      angle_degrees: 'float64',
      radius: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  features.features.forEach((feature, row) => {
    const base = {bounds_id: `${row}`, ...getSourceColumns(feature, row), shape};
    if (shape === 'rectangle') {
      if (result.hullSizes[row] < 3 || !(result.sizes[4 * row + 3] > 0)) {
        addCatalogDiagnostic(
          diagnostics,
          'degenerate-feature',
          'warning',
          'Features with fewer than three non-collinear vertices produce no bounds row.'
        );
        return;
      }
      const sizeWidth = result.sizes[4 * row];
      const sizeHeight = result.sizes[4 * row + 1];
      const ring: PlanarPoint[] = [0, 1, 2, 3].map(corner => [
        result.corners[2 * (4 * row + corner)],
        result.corners[2 * (4 * row + corner) + 1]
      ]);
      appendCatalogRow(table, {
        ...base,
        area: result.sizes[4 * row + 3],
        width: Math.min(sizeWidth, sizeHeight),
        length: Math.max(sizeWidth, sizeHeight),
        angle_degrees: (result.sizes[4 * row + 2] * 180) / Math.PI,
        geometry: makePolygonText(frame, ring)
      });
    } else {
      const radius = result.circles[3 * row + 2];
      if (!(radius > 0)) {
        addCatalogDiagnostic(
          diagnostics,
          'degenerate-feature',
          'warning',
          'Features with a single distinct vertex produce no bounds row.'
        );
        return;
      }
      appendCatalogRow(table, {
        ...base,
        area: Math.PI * radius ** 2,
        radius,
        geometry: makePolygonText(
          frame,
          makeCirclePositions(result.circles[3 * row], result.circles[3 * row + 1], radius, 64)
        )
      });
    }
  });
  if (shape === 'circle')
    addCatalogDiagnostic(
      diagnostics,
      'inscribed-polygon',
      'info',
      'The circle is written as a 64-gon inscribed in the true circle; radius and area are exact.'
    );
  return {
    tables: [table],
    diagnostics,
    summary: {boundsCount: table.rowCount},
    dispatchedContributors: ['GPUMinimumBounds']
  };
}

function getPolygonFeatureRows(
  polygons: CatalogFeatureCollection,
  frame: LocalFrame
): {feature: CatalogFeature; row: number; parts: PlanarPoint[][][]}[] {
  const rows: {feature: CatalogFeature; row: number; parts: PlanarPoint[][][]}[] = [];
  polygons.features.forEach((feature, row) => {
    const parts = projectPolygonParts(feature, frame);
    if (parts.length) rows.push({feature, row, parts});
  });
  return rows;
}

/**
 * GPULabelPoint or GPUMaximumInscribedCircle (the pole of inaccessibility) over a feature at a time
 * with the even-odd rule. The CPU reference picks the best part of a MultiPolygon; the GPU searches
 * the feature at once, so the result agrees to the search precision for the dominant part. Centers
 * agree to the search precision (the CPU stops at `precision` of the shell extent, the label search
 * at about 1e-5 of the bounds, the circle search at `tolerance`); the clearance agrees closely.
 */
export async function runGpuLabelPoints(
  context: CatalogGpuContext,
  polygons: CatalogFeatureCollection,
  asCircle: boolean,
  precisionFraction: number,
  coordinateSystem: CatalogCoordinateSystem
): Promise<CatalogRunOutput> {
  const frame = makeLocalFrame(coordinateSystem, collectPositions(polygons), {shift: 'center'});
  const rows = getPolygonFeatureRows(polygons, frame);
  if (!rows.length) throw new CatalogGpuUnsupportedError('No polygon feature');
  const packed = packCatalogPolygons(
    rows.map(entry => entry.feature),
    {project: position => frame.forward(position), closing: 'drop'}
  );
  let extent = 0;
  for (const value of packed.positions) extent = Math.max(extent, Math.abs(value));
  const tolerance = Math.max(extent * precisionFraction, 1e-3);
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-label-point',
    async session => {
      const points = session.output('points', 'float32x2', rows.length);
      const distances = session.output('distances', 'float32', rows.length);
      const circles = session.output('circles', 'float32x4', rows.length);
      const witnesses = session.output('witnesses', 'float32x2', rows.length);
      const status = session.output('status', 'uint32', rows.length);
      const geometry = {
        positions: session.input('positions', packed.positions, 'float32x2'),
        featureOffsets: session.input('feature-offsets', packed.featureOffsets, 'uint32'),
        polygonOffsets: session.input('polygon-offsets', packed.polygonOffsets, 'uint32'),
        ringOffsets: session.input('ring-offsets', packed.ringOffsets, 'uint32')
      };
      session.graph.add(
        asCircle
          ? new GPUMaximumInscribedCircle({
              ...geometry,
              tolerance,
              output: {circles: circles.view, witnesses: witnesses.view, status: status.view}
            })
          : new GPULabelPoint({
              ...geometry,
              output: {points: points.view, publishedCenterClearance: distances.view}
            })
      );
      session.run();
      return asCircle
        ? {
            centers: await session.readFloat32(circles),
            stride: 4,
            clearances: undefined,
            status: await session.readUint32(status)
          }
        : {
            centers: await session.readFloat32(points),
            stride: 2,
            clearances: await session.readFloat32(distances),
            status: undefined
          };
    }
  );
  const table = makeCatalogTable(
    asCircle ? 'inscribed-circles' : 'label-points',
    asCircle ? 'Maximum inscribed circles' : 'Label points',
    {
      label_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      clearance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  rows.forEach(({feature, row, parts}, index) => {
    const x = result.centers[result.stride * index];
    const y = result.centers[result.stride * index + 1];
    const clearance = asCircle ? result.centers[4 * index + 2] : result.clearances![index];
    // The part holding the center is the reported part (the CPU reports the best part).
    let partIndex = parts.findIndex(rings => isPointInPolygonRings(x, y, rings));
    if (partIndex < 0) partIndex = 0;
    appendCatalogRow(table, {
      label_id: `${row}:${partIndex}`,
      ...getSourceColumns(feature, row),
      part_index: partIndex,
      clearance,
      geometry: asCircle
        ? makePolygonText(frame, makeCirclePositions(x, y, Math.max(clearance, 0), 64))
        : makeCatalogGeometryText('Point', frame.inverse(x, y))
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (
    polygons.features.some(
      feature => (feature.geometry as {type?: string})?.type === 'MultiPolygon'
    )
  )
    addCatalogDiagnostic(
      diagnostics,
      'largest-part',
      'info',
      'MultiPolygon features report the part with the largest clearance.'
    );
  if (asCircle && result.status?.some(value => value === 1))
    addCatalogDiagnostic(
      diagnostics,
      'search-budget-exhausted',
      'warning',
      'Some maximum inscribed circles stopped at their cell or iteration budget; their radius is a lower bound.'
    );
  return {
    tables: [table],
    diagnostics,
    summary: {outputCount: table.rowCount},
    dispatchedContributors: [asCircle ? 'GPUMaximumInscribedCircle' : 'GPULabelPoint']
  };
}

const MAXIMUM_CLEARANCE_VERTICES = 5000;

type ClearanceRow = {
  feature: CatalogFeature;
  sourceIndex: number;
  rings: PlanarPoint[][];
  isPolygon: boolean;
};

/**
 * GPUMinimumClearance. Polygon features run as polygons and single-part line features as lines;
 * vertex counts, skipped and degenerate statuses follow the CPU reference. The clearance is the
 * GPU's float32 value (relative error about 1e-6 in the shifted metric frame); the realizing line
 * may pick a different vertex or segment when two vertex-segment pairs tie.
 */
export async function runGpuMinimumClearance(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): Promise<CatalogRunOutput> {
  const frame = makeLocalFrame(coordinateSystem, collectPositions(features), {shift: 'center'});
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'minimum-clearance',
    'Minimum clearance',
    {
      source_index: 'int32',
      source_id: 'string',
      clearance: 'float64',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  const rows: (ClearanceRow | null)[] = [];
  const measured: ClearanceRow[] = [];
  features.features.forEach((feature, sourceIndex) => {
    const type = (feature.geometry as {type?: string} | null)?.type ?? '';
    const isPolygon = type.includes('Polygon');
    const rings: PlanarPoint[][] = [];
    if (isPolygon)
      for (const polygon of projectPolygonParts(feature, frame)) rings.push(...polygon);
    else {
      const parts = getLineParts(feature.geometry).filter(part => part.length >= 3);
      // A multi-part line measures across its parts; the line layout has one linestring per row.
      if (parts.length > 1) throw new CatalogGpuUnsupportedError('Multi-part line features');
      for (const part of parts) rings.push(part.map(position => frame.forward(position)));
    }
    const row = {feature, sourceIndex, rings, isPolygon};
    rows.push(row);
    const vertexCount = rings.reduce((sum, ring) => sum + ring.length, 0);
    if (vertexCount >= 3 && vertexCount <= MAXIMUM_CLEARANCE_VERTICES) measured.push(row);
  });
  const clearances = new Map<number, {clearance: number; line: number[]}>();
  for (const isPolygon of [true, false]) {
    const group = measured.filter(row => row.isPolygon === isPolygon);
    if (!group.length) continue;
    const positions: number[] = [];
    const offsets = [0];
    const polygonOffsets = [0];
    const featureOffsets = [0];
    for (const row of group) {
      for (const ring of row.rings) {
        for (const [x, y] of ring) positions.push(x, y);
        offsets.push(positions.length / 2);
        if (!isPolygon) featureOffsets.push(offsets.length - 1);
      }
      if (isPolygon) {
        polygonOffsets.push(offsets.length - 1);
        featureOffsets.push(polygonOffsets.length - 1);
      }
    }
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-minimum-clearance',
      async session => {
        const clearanceOutput = session.output('clearances', 'float32', group.length);
        const lineOutput = session.output('lines', 'float32x4', group.length);
        session.graph.add(
          new GPUMinimumClearance({
            geometry: isPolygon
              ? {
                  kind: 'polygons',
                  positions: session.input('positions', Float32Array.from(positions), 'float32x2'),
                  featureOffsets: session.input(
                    'feature-offsets',
                    Uint32Array.from(featureOffsets),
                    'uint32'
                  ),
                  polygonOffsets: session.input(
                    'polygon-offsets',
                    Uint32Array.from(polygonOffsets),
                    'uint32'
                  ),
                  ringOffsets: session.input('ring-offsets', Uint32Array.from(offsets), 'uint32')
                }
              : {
                  kind: 'lines',
                  positions: session.input('positions', Float32Array.from(positions), 'float32x2'),
                  lineOffsets: session.input('line-offsets', Uint32Array.from(offsets), 'uint32')
                },
            output: {clearances: clearanceOutput.view, lines: lineOutput.view}
          })
        );
        session.run();
        return {
          clearances: await session.readFloat32(clearanceOutput),
          lines: await session.readFloat32(lineOutput)
        };
      }
    );
    group.forEach((row, index) =>
      clearances.set(row.sourceIndex, {
        clearance: result.clearances[index],
        line: Array.from(result.lines.slice(4 * index, 4 * index + 4))
      })
    );
  }
  rows.forEach((row, sourceIndex) => {
    const base = getSourceColumns(features.features[sourceIndex], sourceIndex);
    const vertexCount = row!.rings.reduce((sum, ring) => sum + ring.length, 0);
    if (vertexCount < 3) {
      appendCatalogRow(table, {...base, status: 'degenerate'});
      return;
    }
    if (vertexCount > MAXIMUM_CLEARANCE_VERTICES) {
      appendCatalogRow(table, {...base, status: 'skipped-too-many-vertices'});
      addCatalogDiagnostic(
        diagnostics,
        'clearance-skipped',
        'warning',
        `Features over ${MAXIMUM_CLEARANCE_VERTICES} vertices are not measured.`
      );
      return;
    }
    const found = clearances.get(sourceIndex)!;
    if (!Number.isFinite(found.clearance)) {
      appendCatalogRow(table, {...base, status: 'degenerate'});
      return;
    }
    appendCatalogRow(table, {
      ...base,
      clearance: found.clearance,
      status: 'ok',
      geometry: makeCatalogGeometryText('LineString', [
        frame.inverse(found.line[0], found.line[1]),
        frame.inverse(found.line[2], found.line[3])
      ])
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'vertex-to-segment',
    'info',
    'Clearance is the smallest distance from a vertex to a segment that does not own it, as in the JTS definition restricted to vertex-segment pairs.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {measuredCount: table.columns['status'].filter(status => status === 'ok').length},
    dispatchedContributors: ['GPUMinimumClearance']
  };
}

const DISTRIBUTION_VERTEX_COUNT = 64;

/**
 * GPUGeographicDistribution in the shifted metric frame: per-group count, mean and median centers,
 * standard distance and the standard deviational ellipse (plain standard-deviation convention, like
 * the CPU reference). Float32 sums give about 1e-6 relative error; the Weiszfeld median stops at the
 * contributor tolerance of 1e-3 frame units (the CPU iterates to 1e-9), and the ellipse ring starts
 * on the contributor's short axis, so its first vertex differs from the CPU ring.
 */
export async function runGpuGeographicDistribution(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  parameters: {
    groupProperty: string;
    weightProperty: string;
    standardDeviations: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): Promise<CatalogRunOutput> {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(features), {
    shift: 'center'
  });
  const groups = new Map<string, {x: number; y: number; weight: number}[]>();
  let skippedCount = 0;
  features.features.forEach(feature => {
    const weight = parameters.weightProperty
      ? readPropertyNumber(feature, parameters.weightProperty)
      : 1;
    if (weight === null || weight < 0) {
      skippedCount++;
      return;
    }
    let representatives = getPointPositions(feature.geometry);
    if (!representatives.length) {
      const vertices = getFeatureVertices(feature);
      if (vertices.length)
        representatives = [
          [
            vertices.reduce((sum, vertex) => sum + vertex[0], 0) / vertices.length,
            vertices.reduce((sum, vertex) => sum + vertex[1], 0) / vertices.length
          ]
        ];
    }
    const key = readPropertyText(feature, parameters.groupProperty);
    for (const position of representatives) {
      const [x, y] = frame.forward(position);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push({x, y, weight});
    }
  });
  if (skippedCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-weight',
      'warning',
      'Features with a missing, non-numeric or negative weight are skipped.',
      skippedCount
    );
  const ordered = [...groups.entries()].sort(([first], [second]) =>
    first < second ? -1 : first > second ? 1 : 0
  );
  const rowCount = ordered.reduce((sum, [, points]) => sum + points.length, 0);
  if (!rowCount) throw new CatalogGpuUnsupportedError('No points');
  const positions = new Float32Array(rowCount * 2);
  const weights = new Float32Array(rowCount);
  const groupIds = new Uint32Array(rowCount);
  let cursor = 0;
  ordered.forEach(([, points], group) => {
    for (const point of points) {
      positions[2 * cursor] = point.x;
      positions[2 * cursor + 1] = point.y;
      weights[cursor] = point.weight;
      groupIds[cursor++] = group;
    }
  });
  const groupCount = ordered.length;
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-geographic-distribution',
    async session => {
      const weightSums = session.output('weight-sums', 'float32', groupCount);
      const meanCenters = session.output('mean-centers', 'float32x2', groupCount);
      const medianCenters = session.output('median-centers', 'float32x2', groupCount);
      const standardDistances = session.output('standard-distances', 'float32', groupCount);
      const ellipses = session.output('ellipses', 'float32', groupCount * 3);
      const ellipseVertices = session.output(
        'ellipse-vertices',
        'float32x2',
        groupCount * DISTRIBUTION_VERTEX_COUNT
      );
      session.graph.add(
        new GPUGeographicDistribution({
          positions: session.input('positions', positions, 'float32x2'),
          weights: session.input('weights', weights, 'float32'),
          groupIds: session.input('group-ids', groupIds, 'uint32'),
          groupCount,
          maximumMedianIterations: 256,
          polygonVertexCount: DISTRIBUTION_VERTEX_COUNT,
          parameters: session.input(
            'parameters',
            getGPUGeographicDistributionParameterValues({
              standardDeviations: parameters.standardDeviations,
              ellipseConvention: 'standard'
            }),
            'float32'
          ),
          output: {
            weightSums: weightSums.view,
            meanCenters: meanCenters.view,
            medianCenters: medianCenters.view,
            standardDistances: standardDistances.view,
            ellipses: ellipses.view,
            ellipseVertices: ellipseVertices.view
          }
        })
      );
      session.run();
      return {
        weightSums: await session.readFloat32(weightSums),
        meanCenters: await session.readFloat32(meanCenters),
        medianCenters: await session.readFloat32(medianCenters),
        standardDistances: await session.readFloat32(standardDistances),
        ellipses: await session.readFloat32(ellipses),
        ellipseVertices: await session.readFloat32(ellipseVertices)
      };
    }
  );
  const ellipseTable = makeCatalogTable(
    'distribution-ellipses',
    'Standard deviational ellipses',
    {
      group: 'string',
      count: 'int32',
      total_weight: 'float64',
      standard_distance: 'float64',
      major_radius: 'float64',
      minor_radius: 'float64',
      rotation_degrees: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const centerTable = makeCatalogTable(
    'distribution-centers',
    'Mean and median centers',
    {center_id: 'string', group: 'string', kind: 'string', count: 'int32', geometry: 'string'},
    'geometry'
  );
  ordered.forEach(([group, points], index) => {
    const totalWeight = result.weightSums[index];
    if (!(totalWeight > 0)) {
      addCatalogDiagnostic(
        diagnostics,
        'zero-weight-group',
        'warning',
        'Groups with zero total weight are skipped.'
      );
      return;
    }
    const sigmaX = result.ellipses[3 * index + 1];
    const sigmaY = result.ellipses[3 * index + 2];
    const hasEllipse = points.length >= 3 && sigmaX > 0 && sigmaY > 0;
    // The long axis points at angle + 90 degrees; wrap to (-90, 90].
    let rotation = ((result.ellipses[3 * index] + Math.PI / 2) * 180) / Math.PI;
    if (rotation > 90) rotation -= 180;
    const ring: PlanarPoint[] = [];
    if (hasEllipse)
      for (let vertex = 0; vertex < DISTRIBUTION_VERTEX_COUNT; vertex++)
        ring.push([
          result.ellipseVertices[2 * (index * DISTRIBUTION_VERTEX_COUNT + vertex)],
          result.ellipseVertices[2 * (index * DISTRIBUTION_VERTEX_COUNT + vertex) + 1]
        ]);
    appendCatalogRow(ellipseTable, {
      group,
      count: points.length,
      total_weight: totalWeight,
      standard_distance: result.standardDistances[index] / parameters.standardDeviations,
      major_radius: hasEllipse ? sigmaY : null,
      minor_radius: hasEllipse ? sigmaX : null,
      rotation_degrees: hasEllipse ? rotation : null,
      geometry: hasEllipse ? makePolygonText(frame, ring) : null
    });
    appendCatalogRow(centerTable, {
      center_id: `${group}:mean`,
      group,
      kind: 'mean',
      count: points.length,
      geometry: makeCatalogGeometryText(
        'Point',
        frame.inverse(result.meanCenters[2 * index], result.meanCenters[2 * index + 1])
      )
    });
    appendCatalogRow(centerTable, {
      center_id: `${group}:median`,
      group,
      kind: 'median',
      count: points.length,
      geometry: makeCatalogGeometryText(
        'Point',
        frame.inverse(result.medianCenters[2 * index], result.medianCenters[2 * index + 1])
      )
    });
    if (!hasEllipse)
      addCatalogDiagnostic(
        diagnostics,
        'ellipse-undefined',
        'warning',
        'Groups with fewer than three points or no spread have no ellipse polygon.'
      );
  });
  return {
    tables: [ellipseTable, centerTable],
    diagnostics,
    summary: {groupCount: ellipseTable.rowCount},
    dispatchedContributors: ['GPUGeographicDistribution']
  };
}
