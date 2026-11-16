// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {
  directedEdgeToBoundary,
  getDirectedEdgeDestination,
  isValidCell,
  originToDirectedEdges
} from 'h3-js';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {
  CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
  CURATED_CELL_FLOW_TOPOLOGY_BINDING
} from '../../src/curated/bindings/curated-cell-flow-topology-binding';
import {
  createCellFlowTopologySample,
  createSecondCellFlowTopologySample
} from '../../src/curated/samples/cell-flow-topology';
import {profileKeplerColumnTable} from '../../src/profile/dataset-profile';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-cell-flow-topology';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {getLineParts, validateCatalogCoordinateSystem} from '../../src/analysis/catalog-geojson';
import {getWorkflowFeatures, makeWorkflowData} from '../../src/workflows/workflow-data';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';

// Same domains and tolerance as the existing network-expansion.spec.ts cell-flow recipe fixture.
// The second table changes row order, weights, column names, key spelling and rejection cases.
for (const second of [false, true]) {
  it(`cell-flow recipe parity and native publication tables (${second ? 'mapped freight' : 'pentagon sample'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = second ? createSecondCellFlowTopologySample() : createCellFlowTopologySample();
    const columns = second
      ? {origin: 'from_hex', destination: 'to_hex', flow: 'tonnes'}
      : {origin: 'origin', destination: 'destination', flow: 'flow'};
    const resolution = second ? 2 : 1;
    const profile = profileKeplerColumnTable(input.table, {
      datasetId: input.table.id,
      label: input.table.label,
      inferPointColumns: false,
      declarations: {measurement: {[columns.flow]: 'amount'}}
    });
    let mapping = createKeplerCuratedAnalysisMapping(CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS, [
      profile
    ]);
    for (const [role, column] of Object.entries(columns))
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'flows', role, columns: [column]},
        CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
        [profile]
      );
    const request = buildKeplerCuratedAnalysisRequest(
      CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
      mapping,
      {parameterOverrides: {resolution}}
    );
    const result = await runCatalogOperationAsync(
      CURATED_CELL_FLOW_TOPOLOGY_BINDING,
      request.parameters,
      {flows: input},
      {device}
    );
    const mismatches: string[] = [];
    const check = (condition: boolean, message: string) => {
      if (!condition) mismatches.push(message);
    };
    check(result.engine === 'gpu', `engine ${result.engine}`);
    check(
      result.dispatchedContributors.includes('GPUH3CellIdentities'),
      'GPUH3CellIdentities was not dispatched'
    );
    check(result.summary['recipe'] === 'addCellFlowTopologyRecipe', 'wrong recipe provenance');
    check(request.inputDatasetIds['flows'] === input.table.id, 'mapped dataset identity');
    const pairs = result.tables.find(table => table.id === 'pairs')!;
    const cells = result.tables.find(table => table.id === 'cells')!;
    const edges = result.tables.find(table => table.id === 'edges')!;
    check(pairs.rowCount === input.table.rowCount, 'original OD row count');
    check(
      pairs.geometryColumn === undefined && cells.geometryColumn === undefined,
      'table input was given artificial geometry'
    );
    check(edges.geometryColumn === 'geometry', 'native geometry column');
    const downstream = makeWorkflowData(edges, {identityColumn: 'directed_edge'});
    const downstreamProfile = profileKeplerWorkbenchData(downstream);
    check(
      downstreamProfile.coordinates?.model === 'geographic' &&
        downstreamProfile.coordinates.source === 'declared',
      'declared geographic output profile'
    );
    try {
      validateCatalogCoordinateSystem('cell-flow-downstream', 'EPSG:4326', {
        edges: getWorkflowFeatures(downstream)
      });
    } catch (error) {
      mismatches.push(`downstream ingestion: ${String(error)}`);
    }
    const canonical = (value: unknown): string | null => {
      if (typeof value !== 'string') return null;
      try {
        const key = BigInt(value.toLowerCase().startsWith('0x') ? value : `0x${value}`).toString(
          16
        );
        return isValidCell(key) && Number((BigInt(`0x${key}`) >> 52n) & 15n) === resolution
          ? key
          : null;
      } catch {
        return null;
      }
    };
    const outgoing = new Map<string, number>();
    const incoming = new Map<string, number>();
    const neighboring = new Map<string, number>();
    let invalidCount = 0;
    for (let row = 0; row < input.table.rowCount; row++) {
      const origin = canonical(input.table.columns[columns.origin][row]);
      const destination = canonical(input.table.columns[columns.destination][row]);
      const rawValue = input.table.columns[columns.flow][row];
      const value = typeof rawValue === 'number' ? Math.fround(rawValue) : Number.NaN;
      const valid = origin !== null && destination !== null && Number.isFinite(value) && value >= 0;
      check(
        pairs.columns['validity'][row] === valid,
        `pair ${row} validity ${pairs.columns['validity'][row]} expected ${valid}`
      );
      check(pairs.columns['source_index'][row] === row, `pair ${row} source row`);
      check(
        pairs.columns['source_id'][row] === input.table.columns[input.identityColumn!][row],
        `pair ${row} source identity`
      );
      check(
        pairs.columns['origin'][row] === input.table.columns[columns.origin][row],
        `pair ${row} origin text`
      );
      check(
        pairs.columns['destination'][row] === input.table.columns[columns.destination][row],
        `pair ${row} destination text`
      );
      if (valid) {
        check(
          cells.columns['cell_id'][Number(pairs.columns['origin_index'][row])] === origin,
          `pair ${row} exact origin row mapping`
        );
        check(
          cells.columns['cell_id'][Number(pairs.columns['destination_index'][row])] === destination,
          `pair ${row} exact destination row mapping`
        );
      }
      if (!valid || !origin || !destination) {
        invalidCount++;
        continue;
      }
      outgoing.set(origin, (outgoing.get(origin) ?? 0) + value);
      incoming.set(destination, (incoming.get(destination) ?? 0) + value);
      if (
        originToDirectedEdges(origin).some(edge => getDirectedEdgeDestination(edge) === destination)
      ) {
        const key = `${origin}:${destination}`;
        neighboring.set(key, (neighboring.get(key) ?? 0) + value);
      }
    }
    check(
      result.summary['invalidCount'] === invalidCount,
      `invalidCount ${result.summary['invalidCount']} expected ${invalidCount}`
    );
    let expectedEdgeCount = 0;
    const cellKeys = cells.columns['cell_id'].map(String);
    for (let row = 0; row < cells.rowCount; row++) {
      const key = cellKeys[row];
      check(cells.columns['cell_index'][row] === row, `cell ${row} index`);
      check(
        cells.columns['outgoing'][row] === (outgoing.get(key) ?? 0),
        `cell ${key} outgoing ${cells.columns['outgoing'][row]} expected ${outgoing.get(key) ?? 0}`
      );
      check(cells.columns['incoming'][row] === (incoming.get(key) ?? 0), `cell ${key} incoming`);
      check(cells.columns['validity'][row] === isValidCell(key), `cell ${key} validity`);
      if (isValidCell(key)) expectedEdgeCount += originToDirectedEdges(key).length;
    }
    check(
      edges.rowCount === expectedEdgeCount,
      `edge count ${edges.rowCount} expected ${expectedEdgeCount}`
    );
    const seenEdges = new Set<string>();
    for (let row = 0; row < edges.rowCount; row++) {
      const edge = String(edges.columns['directed_edge'][row]);
      const origin = String(edges.columns['origin'][row]);
      const destination = String(edges.columns['destination'][row]);
      check(!seenEdges.has(edge), `duplicate directed edge ${edge}`);
      seenEdges.add(edge);
      check(originToDirectedEdges(origin).includes(edge), `edge ${edge} origin`);
      check(getDirectedEdgeDestination(edge) === destination, `edge ${edge} neighbor key`);
      const expectedIndex = cellKeys.indexOf(destination);
      check(
        edges.columns['destination_index'][row] === (expectedIndex < 0 ? null : expectedIndex),
        `edge ${edge} neighbor row`
      );
      check(
        edges.columns['flow'][row] === (neighboring.get(`${origin}:${destination}`) ?? 0),
        `edge ${edge} flow ${edges.columns['flow'][row]}`
      );
      const geometry = JSON.parse(String(edges.columns['geometry'][row]));
      const parts = getLineParts(geometry);
      for (const part of parts) {
        check(part.length >= 2, `edge ${edge} nonempty line part`);
        for (const [longitude, latitude] of part)
          check(
            Math.abs(longitude) <= 180 && Math.abs(latitude) <= 90,
            `edge ${edge} geographic bounds`
          );
        for (let vertex = 1; vertex < part.length; vertex++)
          check(
            Math.abs(part[vertex][0] - part[vertex - 1][0]) <= 180,
            `edge ${edge} antimeridian continuity`
          );
      }
      const expectedBoundary = directedEdgeToBoundary(edge);
      check(
        (geometry.type === 'LineString' || geometry.type === 'MultiLineString') &&
          parts.length >= 1,
        `edge ${edge} publication geometry`
      );
      check(
        edges.columns['boundary_count'][row] === expectedBoundary.length,
        `edge ${edge} distortion vertex count`
      );
      expectedBoundary.forEach(([latitude, longitude], vertex) => {
        // Seam splitting adds vertices; every original recipe vertex must still be present.
        const actual = parts
          .flat()
          .find(
            position =>
              Math.abs(((position[0] - longitude + 540) % 360) - 180) < 0.0003 &&
              Math.abs(position[1] - latitude) < 0.0003
          );
        check(
          actual !== undefined,
          `edge ${edge} vertex ${vertex}: ${JSON.stringify(actual)} expected [${longitude},${latitude}]`
        );
      });
    }
    check(
      result.summary['outgoingTotal'] === (second ? 25 : 27),
      'recipe fixture outgoing conservation'
    );
    check(
      result.summary['incomingTotal'] === (second ? 25 : 27),
      'recipe fixture incoming conservation'
    );
    check(
      result.summary['neighboringFlowTotal'] === (second ? 15 : 9),
      'recipe fixture neighbor sum'
    );
    check(
      Boolean(
        planKeplerResultPresentation(edges, CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR.outputs[2])
      ),
      'flow presentation plan'
    );
    if (mismatches.length)
      throw new Error(
        `Cell-flow ${second ? 'mapped freight' : 'pentagon'} recipe parity:\n${mismatches.join('\n')}`
      );
  }, 120000);
}

it('cell-flow reports all rejected rows without publishing a placeholder cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createCellFlowTopologySample();
  const result = await runCatalogOperationAsync(
    CURATED_CELL_FLOW_TOPOLOGY_BINDING,
    {
      originProperty: 'origin',
      destinationProperty: 'destination',
      flowProperty: 'flow',
      resolution: 1
    },
    {
      flows: {
        ...input,
        table: {
          ...input.table,
          columns: {
            ...input.table.columns,
            origin: Array(6).fill(null),
            destination: Array(6).fill('not-h3')
          }
        }
      }
    },
    {device}
  );
  if (
    result.engine !== 'gpu' ||
    result.summary['invalidCount'] !== 6 ||
    result.tables.map(table => table.rowCount).join(',') !== '6,0,0'
  )
    throw new Error(`Cell-flow all-invalid GPU status/publication: ${JSON.stringify(result)}`);
}, 120000);
