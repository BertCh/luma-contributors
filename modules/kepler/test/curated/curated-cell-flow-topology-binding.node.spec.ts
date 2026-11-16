// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  directedEdgeToBoundary,
  getDirectedEdgeDestination,
  isValidCell,
  originToDirectedEdges
} from 'h3-js';
import {
  CELL_FLOW_TOPOLOGY_LIMITS,
  CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
  CURATED_CELL_FLOW_TOPOLOGY_BINDING,
  prepareCellFlowTopology,
  publishCellFlowTopology,
  type CellFlowTopologyReadback,
  type PreparedCellFlowTopology
} from '../../src/curated/bindings/curated-cell-flow-topology-binding';
import {
  CELL_FLOW_TOPOLOGY_SAMPLE_LICENSE,
  createCellFlowTopologySample,
  createSecondCellFlowTopologySample
} from '../../src/curated/samples/cell-flow-topology';
import {CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-cell-flow-topology';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {profileKeplerColumnTable} from '../../src/profile/dataset-profile';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {
  getCatalogBinding,
  normalizeCatalogParameters,
  runCatalogOperation,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogTableInput} from '../../src/analysis/catalog-data-input';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';
import {
  getWorkflowFeatures,
  hashWorkflowData,
  makeWorkflowData
} from '../../src/workflows/workflow-data';
import {getLineParts, validateCatalogCoordinateSystem} from '../../src/analysis/catalog-geojson';

const PARAMETERS = {
  originProperty: 'origin',
  destinationProperty: 'destination',
  flowProperty: 'flow',
  resolution: 1
};

/** Independent h3-js readback fixture for the publication adapter, never a production fallback. */
function makeReadback(prepared: PreparedCellFlowTopology): CellFlowTopologyReadback {
  const cellCount = prepared.cells.length / 2;
  const result: CellFlowTopologyReadback = {
    outgoingTotals: new Float32Array(cellCount),
    incomingTotals: new Float32Array(cellCount),
    directedEdges: new Uint32Array(cellCount * 12),
    neighboringCells: new Uint32Array(cellCount * 12),
    edgeFlows: new Float32Array(cellCount * 6),
    boundaries: new Float32Array(cellCount * 36),
    boundaryCounts: new Uint32Array(cellCount * 6),
    validity: new Uint32Array(prepared.values.length),
    invalidCount: 0
  };
  const writeKey = (array: Uint32Array, offset: number, key: string) => {
    const integer = BigInt(`0x${key}`);
    array[offset] = Number(integer & 0xffffffffn);
    array[offset + 1] = Number(integer >> 32n);
  };
  prepared.cellKeys.forEach((key, cell) => {
    if (!isValidCell(key)) return;
    for (const edge of originToDirectedEdges(key)) {
      const direction = Number((BigInt(`0x${edge}`) >> 56n) & 7n);
      const slot = cell * 6 + direction - 1;
      writeKey(result.directedEdges, slot * 2, edge);
      writeKey(result.neighboringCells, slot * 2, getDirectedEdgeDestination(edge));
      const boundary = directedEdgeToBoundary(edge);
      result.boundaryCounts[slot] = boundary.length;
      boundary.forEach(([latitude, longitude], vertex) => {
        result.boundaries[slot * 6 + vertex * 2] = longitude;
        result.boundaries[slot * 6 + vertex * 2 + 1] = latitude;
      });
    }
  });
  for (let row = 0; row < prepared.values.length; row++) {
    const origin = prepared.origins[row];
    const destination = prepared.destinations[row];
    const value = prepared.values[row];
    if (
      !prepared.cellKeys[origin] ||
      !prepared.cellKeys[destination] ||
      !isValidCell(prepared.cellKeys[origin]) ||
      !isValidCell(prepared.cellKeys[destination]) ||
      !Number.isFinite(value) ||
      value < 0
    ) {
      result.invalidCount++;
      continue;
    }
    result.validity[row] = 1;
    result.outgoingTotals[origin] += value;
    result.incomingTotals[destination] += value;
    for (const edge of originToDirectedEdges(prepared.cellKeys[origin])) {
      if (getDirectedEdgeDestination(edge) === prepared.cellKeys[destination]) {
        const direction = Number((BigInt(`0x${edge}`) >> 56n) & 7n);
        result.edgeFlows[origin * 6 + direction - 1] += value;
      }
    }
  }
  return result;
}

function makeManyCells(count: number): CatalogTableInput {
  const input = createCellFlowTopologySample();
  return {
    ...input,
    table: {
      ...input.table,
      rowCount: count,
      columns: {
        origin: Array.from({length: count}, (_, index) =>
          (0x81083ffffffffffn + BigInt(index)).toString(16)
        ),
        destination: Array(count).fill('81083ffffffffff'),
        flow: Array(count).fill(1)
      },
      columnTypes: {origin: 'string', destination: 'string', flow: 'float64'}
    },
    identityColumn: undefined
  };
}

describe('curated cell flow topology', () => {
  it('decodes exact 64-bit keys and stable pair indices without feature geometry', () => {
    const input = createCellFlowTopologySample();
    const prepared = prepareCellFlowTopology({flows: input}, PARAMETERS);
    expect(prepared.input).toBe(input);
    expect(prepared.cellKeys).toEqual(['81083ffffffffff', '8108bffffffffff', '81e6bffffffffff']);
    expect([...prepared.origins]).toEqual([0, 0, 1, 0, 0, 0xffffffff]);
    expect([...prepared.destinations]).toEqual([1, 1, 0, 2, 0, 1]);
    for (let cell = 0; cell < prepared.cellKeys.length; cell++)
      expect((BigInt(prepared.cells[cell * 2 + 1]) << 32n) | BigInt(prepared.cells[cell * 2])).toBe(
        BigInt(`0x${prepared.cellKeys[cell]}`)
      );
    expect(BigInt(Number(BigInt(`0x${prepared.cellKeys[0]}`)))).not.toBe(
      BigInt(`0x${prepared.cellKeys[0]}`)
    );
    expect(input.table.geometryColumn).toBeUndefined();
  });

  it('maps an independent geometry-free dataset through real column profiling and role choices', () => {
    const input = createSecondCellFlowTopologySample();
    const profile = profileKeplerColumnTable(input.table, {
      datasetId: 'freight',
      label: input.table.label,
      inferPointColumns: false,
      declarations: {measurement: {tonnes: 'amount'}}
    });
    expect(profile.kind).toBe('attributes');
    let mapping = createKeplerCuratedAnalysisMapping(CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS, [
      profile
    ]);
    for (const [role, column] of [
      ['origin', 'from_hex'],
      ['destination', 'to_hex'],
      ['flow', 'tonnes']
    ])
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'flows', role, columns: [column]},
        CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
        [profile]
      );
    const request = buildKeplerCuratedAnalysisRequest(
      CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS,
      mapping,
      {parameterOverrides: {resolution: 2}}
    );
    expect(request.inputDatasetIds).toEqual({flows: 'freight'});
    const prepared = prepareCellFlowTopology({flows: input}, request.parameters);
    expect(prepared.cellKeys.slice(0, 3)).toEqual([
      '82e697fffffffff',
      '827eb7fffffffff',
      '827e87fffffffff'
    ]);
    expect(prepared.origins[1]).toBe(prepared.origins[2]);
    expect(prepared.origins[9]).toBe(0xffffffff);
    const output = publishCellFlowTopology(prepared, makeReadback(prepared));
    expect(output.summary).toMatchObject({
      outgoingTotal: 25,
      incomingTotal: 25,
      neighboringFlowTotal: 15,
      invalidCount: 5,
      invalidCellCount: 1,
      resolution: 2
    });
    expect(output.tables[0].columns['source_id'].slice(1, 3)).toEqual(['repeat', 'repeat']);
    expect(output.tables[0].columns['origin'][2]).toBe(input.table.columns['from_hex'][2]);
    expect(output.summary['antimeridianBoundaryCount']).toBeGreaterThan(0);
    for (const text of output.tables[2].columns['geometry']) {
      for (const coordinates of getLineParts(JSON.parse(String(text))))
        for (let vertex = 1; vertex < coordinates.length; vertex++)
          expect(Math.abs(coordinates[vertex][0] - coordinates[vertex - 1][0])).toBeLessThanOrEqual(
            180
          );
    }
  });

  it('round-trips second-sample edges through geographic workflow profiling and catalog ingestion', () => {
    const prepared = prepareCellFlowTopology(
      {flows: createSecondCellFlowTopologySample()},
      {
        originProperty: 'from_hex',
        destinationProperty: 'to_hex',
        flowProperty: 'tonnes',
        resolution: 2
      }
    );
    const edges = publishCellFlowTopology(prepared, makeReadback(prepared)).tables[2];
    const data = makeWorkflowData(edges, {identityColumn: 'directed_edge'});
    const profile = profileKeplerWorkbenchData(data);
    expect(profile.coordinates).toMatchObject({model: 'geographic', source: 'declared'});
    expect(hashWorkflowData(data)).not.toBe(
      hashWorkflowData(
        makeWorkflowData(
          {
            ...edges,
            coordinateSystem: 'planar'
          },
          {identityColumn: 'directed_edge'}
        )
      )
    );
    expect(hashWorkflowData(data)).not.toBe(
      hashWorkflowData(
        makeWorkflowData(
          {
            ...edges,
            coordinateSystem: undefined
          },
          {identityColumn: 'directed_edge'}
        )
      )
    );
    expect(data.kind).toBe('line');
    const features = getWorkflowFeatures(data);
    expect(() =>
      validateCatalogCoordinateSystem('downstream', 'EPSG:4326', {features})
    ).not.toThrow();
    expect(features.features.some(feature => feature.geometry?.type === 'MultiLineString')).toBe(
      true
    );
    expect(edges.rowCount).toBe(18); // All three valid hexagons retain six directed-edge rows.
    expect(new Set(edges.columns['directed_edge']).size).toBe(edges.rowCount);
    expect(edges.columns['flow'].reduce<number>((total, value) => total + Number(value), 0)).toBe(
      15
    );
    for (const [row, feature] of features.features.entries()) {
      expect(feature.id).toBe(edges.columns['directed_edge'][row]);
      for (const part of getLineParts(feature.geometry)) {
        expect(part.length).toBeGreaterThanOrEqual(2);
        for (const [longitude, latitude] of part) {
          expect(Math.abs(longitude)).toBeLessThanOrEqual(180);
          expect(Math.abs(latitude)).toBeLessThanOrEqual(90);
        }
        for (let vertex = 1; vertex < part.length; vertex++)
          expect(Math.abs(part[vertex][0] - part[vertex - 1][0])).toBeLessThanOrEqual(180);
      }
      if (feature.geometry?.type === 'MultiLineString') {
        const parts = getLineParts(feature.geometry);
        for (let part = 1; part < parts.length; part++) {
          const previous = parts[part - 1].at(-1)!;
          const next = parts[part][0];
          expect(Math.abs(previous[0])).toBe(180);
          expect(next[0]).toBe(-previous[0]);
          expect(next[1]).toBe(previous[1]);
        }
      }
    }
    const measures = runCatalogOperation(
      getCatalogBinding('geometry-measures')!,
      {coordinateSystem: 'EPSG:4326'},
      {features}
    );
    expect(measures.tables[0].rowCount).toBe(edges.rowCount);
    expect(measures.tables[0].columns['source_id']).toEqual(edges.columns['directed_edge']);
    expect(
      measures.tables[0].columns['length'].every(
        value => typeof value === 'number' && Number.isFinite(value) && value > 0
      )
    ).toBe(true);
  });

  it('matches the existing recipe fixture totals, repeated edges, self and nonneighbor conservation', () => {
    const prepared = prepareCellFlowTopology({flows: createCellFlowTopologySample()}, PARAMETERS);
    const output = publishCellFlowTopology(prepared, makeReadback(prepared));
    const [pairs, cells, edges] = output.tables;
    expect(cells.columns['outgoing']).toEqual([23, 4, 0]);
    expect(cells.columns['incoming']).toEqual([15, 5, 7]);
    expect(output.summary).toMatchObject({
      outgoingTotal: 27,
      incomingTotal: 27,
      neighboringFlowTotal: 9,
      invalidCount: 1,
      validPairCount: 5,
      flowUnit: 'input flow units',
      boundaryCoordinates: 'EPSG:4326 degrees'
    });
    expect(pairs.rowCount).toBe(6);
    expect(pairs.columns['validity']).toEqual([true, true, true, true, true, false]);
    expect(pairs.columns['source_index']).toEqual([0, 1, 2, 3, 4, 5]);
    expect(pairs.columns['status'][5]).toBe('rejected');
    expect(pairs.columns['flow'][5]).toBe(17);
    expect(edges.rowCount).toBe(17); // pentagon five directions, two hexagons six each
    expect(edges.columns['flow'].filter(value => Number(value) > 0).sort()).toEqual([4, 5]);
    expect(edges.columns['destination_index']).toContain(null);
    expect(output.diagnostics[0]).toMatchObject({code: 'rejected-flow', count: 1});
    expect(pairs.geometryColumn).toBeUndefined();
    expect(cells.geometryColumn).toBeUndefined();
    expect(edges.geometryColumn).toBe('geometry');
    expect(JSON.parse(String(edges.columns['geometry'][0])).type).toBe('LineString');
  });

  it('has a complete descriptor, field mappings, units, presentation and license reference', () => {
    expect(
      validateKeplerOperationDescriptor(
        CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR,
        CURATED_CELL_FLOW_TOPOLOGY_BINDING
      )
    ).toEqual([]);
    const prepared = prepareCellFlowTopology({flows: createCellFlowTopologySample()}, PARAMETERS);
    const output = publishCellFlowTopology(prepared, makeReadback(prepared));
    const presentation = planKeplerResultPresentation(
      output.tables[2],
      CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR.outputs[2]
    );
    expect(presentation).toBeDefined();
    expect(CELL_FLOW_TOPOLOGY_SAMPLE_LICENSE.license).toBe('MIT');
    expect(CELL_FLOW_TOPOLOGY_SAMPLE_LICENSE.reference).toContain(
      'THIRD-PARTY-DATA.md#curated-cell-flow-topology'
    );
    expect(CURATED_CELL_FLOW_TOPOLOGY_BINDING.engine).toBe('gpu');
    expect(CURATED_CELL_FLOW_TOPOLOGY_BINDING.runData).toBeUndefined();
  });

  it('does not offer numeric identifiers for text H3 roles', () => {
    const input = createCellFlowTopologySample();
    const profile = profileKeplerColumnTable(
      {
        ...input.table,
        columns: {...input.table.columns, origin: [1, 2, 3, 4, 5, 6]},
        columnTypes: {...input.table.columnTypes, origin: 'float64'}
      },
      {datasetId: 'bad', label: 'Bad'}
    );
    const mapping = createKeplerCuratedAnalysisMapping(CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS, [
      profile
    ]);
    expect(
      mapping.slots[0].roles
        .find(role => role.role === 'origin')
        ?.candidates.map(candidate => candidate.name)
    ).not.toContain('origin');
  });

  it('rejects missing or wrong-type mapped columns and feature inputs before allocation', () => {
    const input = createCellFlowTopologySample();
    expect(() =>
      prepareCellFlowTopology({flows: input}, {...PARAMETERS, originProperty: 'flow'})
    ).toThrow(/curated-cell-flow-topology originProperty/);
    expect(() =>
      prepareCellFlowTopology({flows: input}, {...PARAMETERS, flowProperty: 'absent'})
    ).toThrow(/flowProperty/);
    expect(() =>
      prepareCellFlowTopology({flows: {type: 'FeatureCollection', features: []}}, PARAMETERS)
    ).toThrow(/flows must be a geometry-free table/);
    expect(() => prepareCellFlowTopology({}, PARAMETERS)).toThrow(/flows/);
    expect(() => prepareCellFlowTopology({flows: input}, {...PARAMETERS, resolution: 16})).toThrow(
      /resolution.*\[0, 15\]/
    );
  });

  it('enforces row, unique-cell and work capacities before GPU allocation', () => {
    expect(prepareCellFlowTopology({flows: makeManyCells(1000)}, PARAMETERS).cellKeys.length).toBe(
      1000
    );
    expect(() =>
      prepareCellFlowTopology(
        {flows: makeManyCells(CELL_FLOW_TOPOLOGY_LIMITS.pairs + 1)},
        PARAMETERS
      )
    ).toThrow(/table.rowCount/);
    expect(() =>
      prepareCellFlowTopology(
        {flows: makeManyCells(CELL_FLOW_TOPOLOGY_LIMITS.cells + 1)},
        PARAMETERS
      )
    ).toThrow(/cellCount/);
    expect(() => prepareCellFlowTopology({flows: makeManyCells(1001)}, PARAMETERS)).toThrow(
      /cellCount \* pairCount/
    );
    expect(() => prepareCellFlowTopology({flows: makeManyCells(0)}, PARAMETERS)).toThrow(
      /flows.rowCount/
    );
  });

  it('retains malformed and missing rows, and never publishes an internal placeholder cell', () => {
    const input = createCellFlowTopologySample();
    const bad: CatalogTableInput = {
      ...input,
      table: {
        ...input.table,
        columns: {
          ...input.table.columns,
          origin: Array(6).fill(null),
          destination: Array(6).fill('not-h3')
        }
      }
    };
    const prepared = prepareCellFlowTopology({flows: bad}, PARAMETERS);
    expect(prepared.cellKeys).toEqual([]);
    expect([...prepared.cells]).toEqual([0, 0]);
    const output = publishCellFlowTopology(prepared, makeReadback(prepared));
    expect(output.tables.map(table => table.rowCount)).toEqual([6, 0, 0]);
    expect(output.summary['invalidCount']).toBe(6);
  });

  it('does not turn negative underflow, missing, or overflowing flow values into accepted zero', () => {
    const input = createCellFlowTopologySample();
    const prepared = prepareCellFlowTopology(
      {
        flows: {
          ...input,
          table: {
            ...input.table,
            columns: {
              ...input.table.columns,
              flow: [-1e-50, null, Number.NaN, Number.POSITIVE_INFINITY, 1e40, 0]
            }
          }
        }
      },
      PARAMETERS
    );
    const output = publishCellFlowTopology(prepared, makeReadback(prepared));
    expect(output.summary['validPairCount']).toBe(0); // final zero has a missing origin
  });

  it('rejects malformed readback, inconsistent validity and nonfinite aggregate outputs', () => {
    const prepared = prepareCellFlowTopology({flows: createCellFlowTopologySample()}, PARAMETERS);
    const result = makeReadback(prepared);
    expect(() => publishCellFlowTopology(prepared, {...result, invalidCount: 0})).toThrow(
      /status.invalidCount/
    );
    expect(() =>
      publishCellFlowTopology(prepared, {...result, validity: new Uint32Array(1)})
    ).toThrow(/validity.length/);
    result.outgoingTotals[0] = Number.POSITIVE_INFINITY;
    expect(() => publishCellFlowTopology(prepared, result)).toThrow(/must be finite/);
    const malformedBoundary = makeReadback(prepared);
    malformedBoundary.boundaryCounts[1] = 4;
    expect(() => publishCellFlowTopology(prepared, malformedBoundary)).toThrow(/boundaryCounts/);
  });

  it('requires a device and normalizes only declared parameters', async () => {
    expect(normalizeCatalogParameters(CURATED_CELL_FLOW_TOPOLOGY_BINDING, PARAMETERS)).toEqual(
      PARAMETERS
    );
    await expect(
      runCatalogOperationAsync(CURATED_CELL_FLOW_TOPOLOGY_BINDING, PARAMETERS, {
        flows: createCellFlowTopologySample()
      })
    ).rejects.toThrow();
  });
});
