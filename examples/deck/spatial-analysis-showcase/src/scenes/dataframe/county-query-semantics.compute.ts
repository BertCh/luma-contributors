// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDataFrame, column, literal, not} from '@luma.gl/graph/gpu-dataframe';
import type {GPUDataFrameQueryParameters} from '@luma.gl/graph/gpu-dataframe';
import {GPURecordBatch, GPUTable} from '@luma.gl/graph/gpu-tables';
import type {PolygonMesh} from '../../cartography/polygon-mesh';
import {createFeatureLocator, getPolygonLayout, buildPolygonMesh} from '../statistics/b5-geometry';
import {loadPolygonLayout} from '../geometry/b3-common';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {SpatialAnalysisResources, formatCount} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {LocalMetricProjection} from '../../engine/projection';
import {fetchJson} from '../../data/loaders';
import type {LoadedDataset} from '../../data/catalog';
import type {SceneContext, SceneInstance} from '../scene';

const COUNTY_DATASET = 'us-counties';
const ELECTION_DATASET = 'us-elections';
const DISPLAY_ROWS = 8;
const STATE_NAMES: Record<number, string> = {
  1: 'Alabama',
  2: 'Alaska',
  4: 'Arizona',
  5: 'Arkansas',
  6: 'California',
  8: 'Colorado',
  9: 'Connecticut',
  10: 'Delaware',
  11: 'District of Columbia',
  12: 'Florida',
  13: 'Georgia',
  15: 'Hawaii',
  16: 'Idaho',
  17: 'Illinois',
  18: 'Indiana',
  19: 'Iowa',
  20: 'Kansas',
  21: 'Kentucky',
  22: 'Louisiana',
  23: 'Maine',
  24: 'Maryland',
  25: 'Massachusetts',
  26: 'Michigan',
  27: 'Minnesota',
  28: 'Mississippi',
  29: 'Missouri',
  30: 'Montana',
  31: 'Nebraska',
  32: 'Nevada',
  33: 'New Hampshire',
  34: 'New Jersey',
  35: 'New Mexico',
  36: 'New York',
  37: 'North Carolina',
  38: 'North Dakota',
  39: 'Ohio',
  40: 'Oklahoma',
  41: 'Oregon',
  42: 'Pennsylvania',
  44: 'Rhode Island',
  45: 'South Carolina',
  46: 'South Dakota',
  47: 'Tennessee',
  48: 'Texas',
  49: 'Utah',
  50: 'Vermont',
  51: 'Virginia',
  53: 'Washington',
  54: 'West Virginia',
  55: 'Wisconsin',
  56: 'Wyoming'
};

type CountyColumns = {
  fips: 'uint32';
  stateFips: 'uint32';
  share2016: 'float32';
  share2020: 'float32';
  share2024: 'float32';
};
type ElectionRows = {fips: 'uint32'; share2024: 'float32'};
export type CountyQuerySemanticsOptions = {
  join: 'inner' | 'left' | 'semi' | 'anti' | 'lookup';
  sortDirection: 'ascending' | 'descending';
  nulls: 'first' | 'last';
  nanMode: 'source-null' | 'valid-nan';
  showOutline: boolean;
};

type SortVariant = {
  representation: 'source-null' | 'valid-nan';
  direction: 'ascending' | 'descending';
  nulls: 'first' | 'last';
  nans: 'first' | 'last';
  global: ReturnType<ReturnType<GPUDataFrame<CountyColumns>['sortByGlobal']>['compile']>;
  batches: ReturnType<ReturnType<GPUDataFrame<CountyColumns>['sortBy']>['compile']>;
};
type DiagnosticSortVariant = SortVariant & {globalPreview: Buffer};

function getPhysicalBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

function makeVector<Format extends 'uint32' | 'float32'>(
  name: string,
  format: Format,
  buffer: Buffer,
  rowCount: number,
  byteOffset = 0
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name,
    format,
    data: [new GPUData<Format>({buffer, format, length: rowCount, byteOffset, ownsBuffer: false})],
    ownsData: false
  });
}

function makeCountyBatch(
  buffers: Record<keyof CountyColumns, Buffer>,
  start: number,
  count: number
): GPURecordBatch<CountyColumns> {
  const gpuData = {
    fips: new GPUData<'uint32'>({
      buffer: buffers.fips,
      format: 'uint32',
      length: count,
      byteOffset: start * 4,
      ownsBuffer: false
    }),
    stateFips: new GPUData<'uint32'>({
      buffer: buffers.stateFips,
      format: 'uint32',
      length: count,
      byteOffset: start * 4,
      ownsBuffer: false
    }),
    share2016: new GPUData<'float32'>({
      buffer: buffers.share2016,
      format: 'float32',
      length: count,
      byteOffset: start * 4,
      ownsBuffer: false
    }),
    share2020: new GPUData<'float32'>({
      buffer: buffers.share2020,
      format: 'float32',
      length: count,
      byteOffset: start * 4,
      ownsBuffer: false
    }),
    share2024: new GPUData<'float32'>({
      buffer: buffers.share2024,
      format: 'float32',
      length: count,
      byteOffset: start * 4,
      ownsBuffer: false
    })
  };
  const fields = [
    {name: 'fips', format: 'uint32' as const},
    {name: 'stateFips', format: 'uint32' as const},
    {name: 'share2016', format: 'float32' as const, nullable: true},
    {name: 'share2020', format: 'float32' as const, nullable: true},
    {name: 'share2024', format: 'float32' as const, nullable: true}
  ];
  return new GPURecordBatch<CountyColumns>({
    gpuData,
    fields,
    numRows: count,
    sourceInfo: {sourceBatchIndex: 0, sourceRowIndexOffset: start, sourceRowCount: count}
  });
}

function makeSingleColumnFrame(
  resources: SpatialAnalysisResources,
  keys: Uint32Array,
  values: Float32Array
): GPUDataFrame<ElectionRows> {
  const keyBuffer = resources.createBuffer('observed-election-fips', keys);
  const valueBuffer = resources.createBuffer('observed-election-share', values);
  const table = new GPUTable<ElectionRows>({
    vectors: {
      fips: makeVector('fips', 'uint32', keyBuffer, keys.length),
      share2024: makeVector('share2024', 'float32', valueBuffer, values.length)
    }
  });
  return new GPUDataFrame<ElectionRows>({table, ownership: 'owned'});
}

function countyLabel(
  names: {name: string[]; state: string[]},
  fips: Uint32Array,
  row: number
): string {
  return `${names.name[row] ?? `FIPS ${fips[row]}`}, ${names.state[row] ?? 'state unavailable'}`;
}

function makeCountyMesh(counties: LoadedDataset, projection: LocalMetricProjection): PolygonMesh {
  const sourceLayout = loadPolygonLayout(counties);
  const layout = getPolygonLayout(counties);
  const geometry = buildPolygonMesh(layout);
  const triangles = new Float32Array(geometry.positions.length);
  const outlineSegments = new Float32Array(geometry.outline.length);
  for (let index = 0; index < geometry.positions.length; index += 2) {
    triangles.set(
      projection.project(geometry.positions[index], geometry.positions[index + 1]),
      index
    );
  }
  for (let index = 0; index < geometry.outline.length; index += 4) {
    outlineSegments.set(
      [
        ...projection.project(geometry.outline[index], geometry.outline[index + 1]),
        ...projection.project(geometry.outline[index + 2], geometry.outline[index + 3])
      ],
      index
    );
  }
  const outlineFeatures: number[] = [];
  let part = 0;
  for (let feature = 0; feature < layout.featureCount; feature++) {
    while (part < layout.partFeature.length && layout.partFeature[part] === feature) {
      for (
        let ring = layout.polygonRingOffsets[part];
        ring < layout.polygonRingOffsets[part + 1];
        ring++
      ) {
        let vertexCount = layout.ringOffsets[ring + 1] - layout.ringOffsets[ring];
        const firstVertex = layout.ringOffsets[ring];
        const lastVertex = firstVertex + vertexCount - 1;
        if (
          vertexCount > 1 &&
          layout.vertices[firstVertex * 2] === layout.vertices[lastVertex * 2] &&
          layout.vertices[firstVertex * 2 + 1] === layout.vertices[lastVertex * 2 + 1]
        )
          vertexCount--;
        if (vertexCount >= 3)
          for (let edge = 0; edge < vertexCount; edge++) outlineFeatures.push(feature);
      }
      part++;
    }
  }
  return {
    featureCount: sourceLayout.featureCount,
    triangles,
    triangleFeatures: geometry.featureRows,
    outlineSegments,
    outlineFeatures: Uint32Array.from(outlineFeatures),
    labelPoints: [],
    areas: new Float64Array(layout.featureCount)
  };
}

/** Interactive audit of local county coverage, nulls, joins and row ordering. */
export async function createCountyQuerySemantics(
  ctx: SceneContext<CountyQuerySemanticsOptions>
): Promise<SceneInstance<CountyQuerySemanticsOptions>> {
  const {device} = ctx;
  const counties = ctx.datasets.get(COUNTY_DATASET);
  const elections = ctx.datasets.get(ELECTION_DATASET);
  const rowCount = counties.count;
  const fips = counties.column<Uint32Array>('fips');
  const stateFipsSource = counties.column<Uint8Array>('stateFips');
  const stateFips = Uint32Array.from(stateFipsSource);
  const share2016 = elections.column<Float32Array>('demTwoParty2016');
  const share2020 = elections.column<Float32Array>('demTwoParty2020');
  const share2024 = elections.column<Float32Array>('demTwoParty2024');
  const share2016Validity = Uint32Array.from(share2016, value => (Number.isFinite(value) ? 1 : 0));
  const share2020Validity = Uint32Array.from(share2020, value => (Number.isFinite(value) ? 1 : 0));
  const share2024Validity = Uint32Array.from(share2024, value => (Number.isFinite(value) ? 1 : 0));
  const resources = new SpatialAnalysisResources(device, 'county-query-semantics');
  const origin = counties.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const names = await fetchJson<{name: string[]; state: string[]}>(
    counties.fileUrl('names.json'),
    ctx.signal
  );
  const layout = loadPolygonLayout(counties);
  const countyGeometry = getPolygonLayout(counties);
  const countyLocator = createFeatureLocator(countyGeometry);
  const mesh = makeCountyMesh(counties, projection);
  const polygonBuffers = createPolygonMeshBuffers(resources, mesh, 'county-query-map');
  const projectCounty = getLocalProjector(origin);
  const outlineCounts = new Uint32Array(rowCount);
  for (const featureRow of mesh.outlineFeatures) outlineCounts[featureRow]++;
  const selectionBuffer = resources.createBuffer(
    'selected-county-outline',
    new Float32Array(Math.max(1, ...outlineCounts) * 4 * 4)
  );
  let selectedCountySegments = 0;

  // Upload original float32 payloads, including NaNs; validity sidecars define analytical nulls.
  const gpuShares = {share2016, share2020, share2024};
  const sourceBuffers = {
    fips: resources.createBuffer('county-fips', fips),
    stateFips: resources.createBuffer('county-state-fips', stateFips),
    share2016: resources.createBuffer('county-share-2016', gpuShares.share2016),
    share2020: resources.createBuffer('county-share-2020', gpuShares.share2020),
    share2024: resources.createBuffer('county-share-2024', gpuShares.share2024),
    valid2016: resources.createBuffer('validity-2016', share2016Validity),
    valid2020: resources.createBuffer('validity-2020', share2020Validity),
    valid2024: resources.createBuffer('validity-2024', share2024Validity)
  };
  const sourceBatch = makeCountyBatch(sourceBuffers, 0, rowCount);
  const frame = new GPUDataFrame<CountyColumns>({
    table: new GPUTable<CountyColumns>({batches: [sourceBatch]}),
    validity: {
      share2016: makeVector('share2016', 'uint32', sourceBuffers.valid2016, rowCount),
      share2020: makeVector('share2020', 'uint32', sourceBuffers.valid2020, rowCount),
      share2024: makeVector('share2024', 'uint32', sourceBuffers.valid2024, rowCount)
    },
    ownership: 'owned'
  });
  const stateBatchStarts = [0];
  for (let row = 1; row < rowCount; row++) {
    if (stateFips[row] !== stateFips[row - 1]) stateBatchStarts.push(row);
  }
  // These are application-created contiguous slices of one source batch, not upstream batches.
  // sourceBatchIndex therefore stays 0 while sourceRowIndexOffset keeps each slice's source rows.
  const stateBatches = stateBatchStarts.map((start, index) =>
    makeCountyBatch(sourceBuffers, start, (stateBatchStarts[index + 1] ?? rowCount) - start)
  );
  const stateFrame = new GPUDataFrame<CountyColumns>({
    table: new GPUTable<CountyColumns>({batches: stateBatches}),
    validity: {
      share2016: new GPUVector({
        type: 'data',
        name: 'share2016',
        format: 'uint32',
        data: stateBatchStarts.map(
          (start, index) =>
            new GPUData<'uint32'>({
              buffer: sourceBuffers.valid2016,
              format: 'uint32',
              length: (stateBatchStarts[index + 1] ?? rowCount) - start,
              byteOffset: start * 4,
              ownsBuffer: false
            })
        ),
        ownsData: false
      }),
      share2020: new GPUVector({
        type: 'data',
        name: 'share2020',
        format: 'uint32',
        data: stateBatchStarts.map(
          (start, index) =>
            new GPUData<'uint32'>({
              buffer: sourceBuffers.valid2020,
              format: 'uint32',
              length: (stateBatchStarts[index + 1] ?? rowCount) - start,
              byteOffset: start * 4,
              ownsBuffer: false
            })
        ),
        ownsData: false
      }),
      share2024: new GPUVector({
        type: 'data',
        name: 'share2024',
        format: 'uint32',
        data: stateBatchStarts.map(
          (start, index) =>
            new GPUData<'uint32'>({
              buffer: sourceBuffers.valid2024,
              format: 'uint32',
              length: (stateBatchStarts[index + 1] ?? rowCount) - start,
              byteOffset: start * 4,
              ownsBuffer: false
            })
        ),
        ownsData: false
      })
    },
    ownership: 'owned'
  });
  // Diagnostic view: the same archived float32 NaN payloads are marked valid so the sorter
  // applies its independent NaN placement option. The source-null frame remains untouched.
  const allValidBuffer = resources.createBuffer(
    'validity-2024-diagnostic-all-valid',
    new Uint32Array(rowCount).fill(1)
  );
  const diagnosticFrame = new GPUDataFrame<CountyColumns>({
    table: frame.table,
    validity: {
      ...frame.validity,
      share2024: makeVector('share2024', 'uint32', allValidBuffer, rowCount)
    },
    ownership: 'borrowed'
  });
  const diagnosticStateFrame = new GPUDataFrame<CountyColumns>({
    table: stateFrame.table,
    validity: {
      ...stateFrame.validity,
      share2024: new GPUVector({
        type: 'data',
        name: 'share2024',
        format: 'uint32',
        data: stateBatchStarts.map(
          (start, index) =>
            new GPUData<'uint32'>({
              buffer: allValidBuffer,
              format: 'uint32',
              length: (stateBatchStarts[index + 1] ?? rowCount) - start,
              byteOffset: start * 4,
              ownsBuffer: false
            })
        ),
        ownsData: false
      })
    },
    ownership: 'borrowed'
  });

  const observedKeys: number[] = [];
  const observedShares: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (share2024Validity[row]) {
      observedKeys.push(fips[row]);
      observedShares.push(gpuShares.share2024[row]);
    }
  }
  const observedFrame = makeSingleColumnFrame(
    resources,
    Uint32Array.from(observedKeys),
    Float32Array.from(observedShares)
  );
  const joinTypes = ['inner', 'left', 'semi', 'anti'] as const;
  const joinQueries = joinTypes.map(joinType =>
    frame[
      joinType === 'inner'
        ? 'innerJoin'
        : joinType === 'left'
          ? 'leftJoin'
          : joinType === 'semi'
            ? 'semiJoin'
            : 'antiJoin'
    ](observedFrame, {
      leftOn: 'fips',
      rightOn: 'fips',
      capacity: rowCount,
      indexCapacity: 8192,
      maxProbeCount: 16
    }).compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: `county-${joinType}-join`})
    )
  );
  const lookupQuery = frame
    .lookup(observedFrame, {
      leftOn: 'fips',
      rightOn: 'fips',
      indexCapacity: 8192,
      maxProbeCount: 16
    })
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-observed-lookup'})
    );
  const nullQuery = frame
    .aggregate({counties: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-null-count'}));
  const valid2016Query = frame
    .filter(column('share2016').isValid())
    .aggregate({valid: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-valid-2016'}));
  const valid2020Query = frame
    .filter(column('share2020').isValid())
    .aggregate({valid: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-valid-2020'}));
  const validQuery = frame
    .filter(column('share2024').isValid())
    .aggregate({valid: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-valid-filter'}));
  const nullFilterQuery = frame
    .filter(column('share2024').isNull())
    .aggregate({missing: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-null-filter'}));
  const changeQuery = frame
    .withColumn(
      'gpuChange2020to2024',
      column('share2024').subtract(column('share2020')).multiply(literal(100))
    )
    .filter(column('gpuChange2020to2024').isValid())
    .aggregate({meanChange: {mean: 'gpuChange2020to2024'}, counties: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-share-change'}));
  const groupQuery = frame
    .filter(column('share2024').isValid())
    .groupBy('stateFips', {groupCount: 57})
    .aggregate({counties: 'count', meanShare: {mean: 'share2024'}})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-state-group'}));

  // Source shares are fractions, not percentages. Retain the county-aligned validity sidecar so
  // complements, margins, null-aware NOT, and tie predicates use the original 3,109 rows.
  const expressionQuery = frame
    .withColumn('republicanShare2024', column('share2024').negate().add(literal(1)))
    .withColumn(
      'normalizedMargin2024',
      column('share2024').subtract(literal(0.5)).divide(literal(0.5))
    )
    .filter(column('share2024').isValid())
    .aggregate({
      validRows: 'count',
      meanDemocraticShare: {mean: 'share2024'},
      meanRepublicanShare: {mean: 'republicanShare2024'},
      meanNormalizedMargin: {mean: 'normalizedMargin2024'}
    })
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-share-expressions'})
    );
  const notAboveHalfQuery = frame
    .filter(not(column('share2024').greaterThan(literal(0.5))))
    .aggregate({notAboveHalf: 'count'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-not-above-half'})
    );
  const notAboveHalfOrUnknownQuery = frame
    .filter(
      column('share2024')
        .isNull()
        .or(not(column('share2024').greaterThan(literal(0.5))))
    )
    .aggregate({notAboveHalfOrUnknown: 'count'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
        id: 'county-not-above-half-or-unknown'
      })
    );
  const tieQuery = frame
    .filter(column('share2024').equal(literal(0.5)))
    .aggregate({ties: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-share-ties'}));
  const nonTieQuery = frame
    .filter(column('share2024').notEqual(literal(0.5)))
    .aggregate({nonTies: 'count'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-share-non-ties'})
    );
  const expressionQueries = [
    expressionQuery,
    notAboveHalfQuery,
    notAboveHalfOrUnknownQuery,
    tieQuery,
    nonTieQuery
  ];

  const sortVariants: SortVariant[] = [];
  for (const direction of ['ascending', 'descending'] as const) {
    for (const nulls of ['first', 'last'] as const) {
      const options = {direction, nulls, nans: 'last' as const};
      sortVariants.push({
        representation: 'source-null',
        direction,
        nulls,
        nans: 'last',
        global: frame.sortByGlobal('share2024', options).compile(
          new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
            id: `county-global-${direction}-${nulls}`
          })
        ),
        batches: stateFrame.sortBy('share2024', options).compile(
          new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
            id: `county-state-batches-${direction}-${nulls}`
          })
        )
      });
    }
  }
  const diagnosticSortVariants: DiagnosticSortVariant[] = [];
  for (const direction of ['ascending', 'descending'] as const) {
    for (const nans of ['first', 'last'] as const) {
      const options = {direction, nulls: 'last' as const, nans};
      diagnosticSortVariants.push({
        representation: 'valid-nan',
        direction,
        nulls: nans,
        nans,
        globalPreview: resources.createBuffer(
          `valid-nan-order-preview-${direction}-${nans}`,
          DISPLAY_ROWS * 2 * 4
        ),
        global: diagnosticFrame.sortByGlobal('share2024', options).compile(
          new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
            id: `county-global-valid-nan-${direction}-${nans}`
          })
        ),
        batches: diagnosticStateFrame.sortBy('share2024', options).compile(
          new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
            id: `county-state-batches-valid-nan-${direction}-${nans}`
          })
        )
      });
    }
  }
  const diagnosticValidQuery = diagnosticFrame
    .filter(column('share2024').isValid())
    .aggregate({valid: 'count'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-valid-nan-diagnostic'})
    );
  const diagnosticNullQuery = diagnosticFrame
    .filter(column('share2024').isNull())
    .aggregate({missing: 'count'})
    .compile(
      new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'county-null-nan-diagnostic'})
    );
  const allQueries = [
    ...joinQueries,
    lookupQuery,
    nullQuery,
    valid2016Query,
    valid2020Query,
    validQuery,
    nullFilterQuery,
    changeQuery,
    groupQuery,
    ...expressionQueries,
    ...sortVariants.flatMap(variant => [variant.global, variant.batches]),
    ...diagnosticSortVariants.flatMap(variant => [variant.global, variant.batches]),
    diagnosticValidQuery,
    diagnosticNullQuery
  ];
  const summaryBuffers: Array<{buffer: Buffer; size: number}> = [];
  function addSummary(data: GPUData, size: number): void {
    summaryBuffers.push({buffer: getPhysicalBuffer(data), size});
  }
  joinQueries.forEach(query => {
    query.requiredCounts.data.forEach(data => addSummary(data, 4));
    query.overflows.data.forEach(data => addSummary(data, 4));
    addSummary(query.indexStatistics.data[0], 24);
    query.lookupStatistics.data.forEach(data => addSummary(data, 16));
    addSummary(query.contractViolation.data[0], 4);
  });
  addSummary(lookupQuery.lookupStatistics.data[0], 16);
  addSummary(nullQuery.table.gpuVectors.counties.data[0], 4);
  addSummary(valid2016Query.table.gpuVectors.valid.data[0], 4);
  addSummary(valid2020Query.table.gpuVectors.valid.data[0], 4);
  addSummary(validQuery.table.gpuVectors.valid.data[0], 4);
  addSummary(nullFilterQuery.table.gpuVectors.missing.data[0], 4);
  addSummary(changeQuery.table.gpuVectors.counties.data[0], 4);
  addSummary(changeQuery.table.gpuVectors.meanChange.data[0], 4);
  addSummary(groupQuery.table.gpuVectors.counties.data[0], 57 * 4);
  addSummary(groupQuery.table.gpuVectors.meanShare.data[0], 57 * 4);
  addSummary(expressionQuery.table.gpuVectors.validRows.data[0], 4);
  addSummary(expressionQuery.table.gpuVectors.meanDemocraticShare.data[0], 4);
  addSummary(expressionQuery.table.gpuVectors.meanRepublicanShare.data[0], 4);
  addSummary(expressionQuery.table.gpuVectors.meanNormalizedMargin.data[0], 4);
  addSummary(notAboveHalfQuery.table.gpuVectors.notAboveHalf.data[0], 4);
  addSummary(notAboveHalfOrUnknownQuery.table.gpuVectors.notAboveHalfOrUnknown.data[0], 4);
  addSummary(tieQuery.table.gpuVectors.ties.data[0], 4);
  addSummary(nonTieQuery.table.gpuVectors.nonTies.data[0], 4);
  sortVariants.forEach(variant => {
    addSummary(variant.global.globalRowIndices.data[0], DISPLAY_ROWS * 4);
    variant.batches.rowIndices.data
      .slice(0, 8)
      .forEach(data => addSummary(data, Math.min(1, data.length) * 4));
  });
  diagnosticSortVariants.forEach(variant => {
    summaryBuffers.push({buffer: variant.globalPreview, size: DISPLAY_ROWS * 2 * 4});
    variant.batches.rowIndices.data
      .slice(0, 8)
      .forEach(data => addSummary(data, Math.min(1, data.length) * 4));
  });
  addSummary(diagnosticValidQuery.table.gpuVectors.valid.data[0], 4);
  addSummary(diagnosticNullQuery.table.gpuVectors.missing.data[0], 4);
  const summaryReader = new SummaryReader(
    resources,
    'county-query-summary',
    summaryBuffers,
    bytes => {
      if (destroyed || readbackSignature !== latestSignature) return;
      const view = new DataView(bytes);
      let offset = 0;
      let allJoinOverflow = false;
      let joinDescriptions = '';
      for (let joinIndex = 0; joinIndex < joinQueries.length; joinIndex++) {
        let rows = 0;
        let overflow = false;
        for (
          let batchIndex = 0;
          batchIndex < joinQueries[joinIndex].requiredCounts.data.length;
          batchIndex++
        ) {
          rows += view.getUint32(offset, true);
          offset += 4;
        }
        for (
          let batchIndex = 0;
          batchIndex < joinQueries[joinIndex].overflows.data.length;
          batchIndex++
        ) {
          overflow ||= view.getUint32(offset, true) !== 0;
          offset += 4;
        }
        const statsOffset = offset;
        const duplicateKeys = view.getUint32(statsOffset + 4, true);
        const indexOverflow = view.getUint32(statsOffset + 8, true);
        offset += 24;
        let probes = 0;
        let maximumProbe = 0;
        for (
          let batchIndex = 0;
          batchIndex < joinQueries[joinIndex].lookupStatistics.data.length;
          batchIndex++
        ) {
          probes += view.getUint32(offset + 8, true);
          maximumProbe = Math.max(maximumProbe, view.getUint32(offset + 12, true));
          offset += 16;
        }
        const contractViolation = view.getUint32(offset, true) !== 0;
        offset += 4;
        allJoinOverflow ||=
          overflow || indexOverflow !== 0 || contractViolation || duplicateKeys !== 0;
        joinDescriptions += `${joinTypes[joinIndex]} ${formatCount(rows)}${overflow ? ' overflow' : ''}; `;
        if (ctx.options.join === joinTypes[joinIndex]) {
          ctx.setReadout(
            'joinDiagnostics',
            `${formatCount(probes)} probes, max ${maximumProbe}; ${duplicateKeys} duplicate keys`
          );
        }
      }
      const lookupStart = offset;
      const lookupMatched = view.getUint32(lookupStart, true);
      const lookupMissing = view.getUint32(lookupStart + 4, true);
      const lookupProbes = view.getUint32(lookupStart + 8, true);
      const lookupMaximumProbe = view.getUint32(lookupStart + 12, true);
      offset += 16;
      if (ctx.options.join === 'lookup') {
        ctx.setReadout(
          'joinDiagnostics',
          `${formatCount(lookupProbes)} probes, max ${lookupMaximumProbe}; source-aligned output`
        );
      }
      const countyRows = view.getUint32(offset, true);
      offset += 4;
      const valid2016Rows = view.getUint32(offset, true);
      offset += 4;
      const valid2020Rows = view.getUint32(offset, true);
      offset += 4;
      const validRows = view.getUint32(offset, true);
      offset += 4;
      const nullRows = view.getUint32(offset, true);
      offset += 4;
      const changedRows = view.getUint32(offset, true);
      offset += 4;
      const meanChange = view.getFloat32(offset, true);
      offset += 4;
      const groupedCount = new Uint32Array(bytes.slice(offset, offset + 57 * 4));
      offset += 57 * 4;
      const groupedMeans = new Float32Array(bytes.slice(offset, offset + 57 * 4));
      offset += 57 * 4;
      const expressionValidRows = view.getUint32(offset, true);
      offset += 4;
      const meanDemocraticShare = view.getFloat32(offset, true);
      offset += 4;
      const meanRepublicanShare = view.getFloat32(offset, true);
      offset += 4;
      const meanNormalizedMargin = view.getFloat32(offset, true);
      offset += 4;
      const notAboveHalfRows = view.getUint32(offset, true);
      offset += 4;
      const notAboveHalfOrUnknownRows = view.getUint32(offset, true);
      offset += 4;
      const tieRows = view.getUint32(offset, true);
      offset += 4;
      const nonTieRows = view.getUint32(offset, true);
      offset += 4;
      const sortReadouts = sortVariants.map(variant => {
        const globalHead = new Uint32Array(bytes.slice(offset, offset + DISPLAY_ROWS * 4));
        offset += DISPLAY_ROWS * 4;
        const batchHead: number[] = [];
        for (const data of variant.batches.rowIndices.data.slice(0, 8)) {
          const length = Math.min(1, data.length);
          batchHead.push(...new Uint32Array(bytes.slice(offset, offset + length * 4)));
          offset += length * 4;
        }
        return {globalHead, batchHead};
      });
      const diagnosticSortReadouts = diagnosticSortVariants.map(variant => {
        const globalPreview = new Uint32Array(bytes.slice(offset, offset + DISPLAY_ROWS * 2 * 4));
        offset += DISPLAY_ROWS * 2 * 4;
        const batchHead: number[] = [];
        for (const data of variant.batches.rowIndices.data.slice(0, 8)) {
          const length = Math.min(1, data.length);
          batchHead.push(...new Uint32Array(bytes.slice(offset, offset + length * 4)));
          offset += length * 4;
        }
        return {globalPreview, batchHead};
      });
      const diagnosticValidRows = view.getUint32(offset, true);
      offset += 4;
      const diagnosticNullRows = view.getUint32(offset, true);
      offset += 4;
      const activeSortIndex = sortVariants.findIndex(
        variant =>
          variant.direction === ctx.options.sortDirection && variant.nulls === ctx.options.nulls
      );
      const selectedMode = ctx.options.nanMode;
      const selectedDirection = ctx.options.sortDirection;
      const selectedPlacement = ctx.options.nulls;
      const renderRow = (row: number): string => {
        const shareLabel = Number.isFinite(share2024[row])
          ? `${(share2024[row] * 100).toFixed(1)}%`
          : selectedMode === 'valid-nan'
            ? 'archived NaN payload'
            : 'null (archived NaN payload)';
        return `${countyLabel(names, fips, row)} (${shareLabel})`;
      };
      let globalRows: string[];
      let batchRows: string[];
      if (selectedMode === 'valid-nan') {
        const diagnosticIndex = diagnosticSortVariants.findIndex(
          variant => variant.direction === selectedDirection && variant.nans === selectedPlacement
        );
        const {globalPreview, batchHead} = diagnosticSortReadouts[diagnosticIndex];
        const firstRows = Array.from(globalPreview.slice(0, DISPLAY_ROWS), renderRow);
        const lastRows = Array.from(globalPreview.slice(DISPLAY_ROWS), renderRow);
        globalRows = [
          `First ${DISPLAY_ROWS} · NaNs ${selectedPlacement}:`,
          ...firstRows,
          `Last ${DISPLAY_ROWS}:`,
          ...lastRows
        ];
        batchRows = batchHead.map(
          row =>
            `${STATE_NAMES[stateFips[row]] ?? `State FIPS ${stateFips[row]}`} — ${renderRow(row)}`
        );
      } else {
        const {globalHead, batchHead} = sortReadouts[activeSortIndex];
        globalRows = Array.from(globalHead, renderRow);
        batchRows = batchHead.map(
          row =>
            `${STATE_NAMES[stateFips[row]] ?? `State FIPS ${stateFips[row]}`} — ${renderRow(row)}`
        );
      }
      ctx.setReadout(
        'sourceCoverage',
        `2016 ${formatCount(valid2016Rows)}; 2020 ${formatCount(valid2020Rows)}; 2024 ${formatCount(validRows)} / ${formatCount(countyRows)} finite`
      );
      ctx.setReadout(
        'missingCoverage',
        `${formatCount(nullRows)} explicit nulls; ${formatCount(lookupMissing)} lookup misses`
      );
      ctx.setReadout(
        'lookupStatus',
        `${formatCount(lookupMatched)} found; ${formatCount(lookupProbes)} probes, max ${lookupMaximumProbe}`
      );
      ctx.setReadout(
        'joinCounts',
        `${allJoinOverflow ? 'INVALID QUERY CONTRACT — ' : ''}${joinDescriptions.trim()}`
      );
      ctx.setReadout(
        'joinCapacity',
        allJoinOverflow ? 'capacity or index overflow' : 'all outputs fit; unique keys'
      );
      ctx.setReadout(
        'validNullFilters',
        `isValid: 2016 ${formatCount(valid2016Rows)}, 2020 ${formatCount(valid2020Rows)}, 2024 ${formatCount(validRows)}; 2024 isNull: ${formatCount(nullRows)}`
      );
      ctx.setReadout(
        'changeSummary',
        `${formatCount(changedRows)} counties; unweighted mean ${meanChange.toFixed(2)} percentage points`
      );
      const supportedGroups = groupedCount.reduce((count, rows) => count + (rows > 0 ? 1 : 0), 0);
      const namedGroups = [
        {fips: 2, label: 'AK'},
        {fips: 6, label: 'CA'},
        {fips: 12, label: 'FL'},
        {fips: 15, label: 'HI'},
        {fips: 36, label: 'NY'},
        {fips: 48, label: 'TX'}
      ].map(
        ({fips: stateCode, label}) =>
          `${label} n=${groupedCount[stateCode]} mean=${groupedCount[stateCode] > 0 && Number.isFinite(groupedMeans[stateCode]) ? `${(groupedMeans[stateCode] * 100).toFixed(1)}%` : 'unavailable'}`
      );
      ctx.setReadout(
        'stateGroups',
        `${formatCount(validRows)} observed counties; ${supportedGroups} occupied state groups (57 FIPS slots, including gaps); county-unweighted finite-share means. ${namedGroups.join(' · ')}`
      );
      ctx.setReadout(
        'shareExpressions',
        `${formatCount(expressionValidRows)} valid counties · mean Democratic share ${(meanDemocraticShare * 100).toFixed(2)}% · mean Republican complement ${(meanRepublicanShare * 100).toFixed(2)}% · mean normalized margin ${meanNormalizedMargin.toFixed(4)}`
      );
      ctx.setReadout(
        'shareNullPredicates',
        `NOT(share > 0.5): ${formatCount(notAboveHalfRows)}; isNull OR NOT: ${formatCount(notAboveHalfOrUnknownRows)} including unknowns`
      );
      ctx.setReadout(
        'shareBoundary',
        `Exactly 0.5: ${formatCount(tieRows)} ties; notEqual(0.5): ${formatCount(nonTieRows)} finite non-ties (nulls stay unknown)`
      );
      ctx.setReadout('globalSortRows', globalRows.join('\n'));
      ctx.setReadout('batchSortRows', batchRows.join('\n'));
      ctx.setReadout(
        'nanRepresentation',
        selectedMode === 'source-null'
          ? `Source semantics: ${formatCount(validRows)} valid / ${formatCount(nullRows)} null; archived NaN payloads are invalid.`
          : `Diagnostic sidecar: ${formatCount(diagnosticValidRows)} valid / ${formatCount(diagnosticNullRows)} null. The ${formatCount(nullRows)} archived NaN payload rows remain unavailable observations.`
      );
      ctx.setReadout(
        'sortContract',
        selectedMode === 'source-null'
          ? 'Global sort compares all counties. State batches compare counties only within each state; null placement is explicit.'
          : 'The same full GPU permutations now treat archived NaN payloads as valid IEEE NaNs; only their NaN placement changes. Global and state-local outputs preserve county identity.'
      );
      ctx.setReadout(
        'activeJoin',
        allJoinOverflow
          ? 'INVALID QUERY CONTRACT — counts withheld'
          : ctx.options.join === 'lookup'
            ? `${formatCount(lookupMatched)} found / ${formatCount(lookupMissing)} absent`
            : joinDescriptions.split(';')[joinTypes.indexOf(ctx.options.join)].trim()
      );
      ctx.requestLayers();
    }
  );
  let destroyed = false;
  let encoded = false;
  let dirty = true;
  let latestSignature = '';
  let readbackSignature = '';
  ctx.setReadout('sourceCoverage', 'Reading local source…');
  ctx.setReadout('countyCount', `${formatCount(rowCount)} polygons / five-digit county FIPS`);
  ctx.setReadout(
    'sourceFootnote',
    '2024 local extraction is state-reconciled; CT planning regions do not crosswalk to legacy counties.'
  );
  ctx.setReadout('selectedCounty', 'Click a county to inspect its FIPS and 2024 support.');
  ctx.setReadout('joinCounts', 'GPU joins compiling');
  ctx.setReadout('joinDiagnostics', 'bounded unique-right index');
  ctx.setReadout('nanRepresentation', 'GPU validity counts compiling');

  return {
    getCompiledGraphs: () => [],
    encode(commandEncoder) {
      const signature = JSON.stringify(ctx.options);
      latestSignature = signature;
      if (dirty || !encoded) {
        for (const query of allQueries) query.encode(commandEncoder, {});
        for (const variant of diagnosticSortVariants) {
          const globalIndices = variant.global.globalRowIndices.data[0];
          const sourceBuffer = getPhysicalBuffer(globalIndices);
          commandEncoder.copyBufferToBuffer({
            sourceBuffer,
            sourceOffset: globalIndices.byteOffset,
            destinationBuffer: variant.globalPreview,
            destinationOffset: 0,
            size: DISPLAY_ROWS * 4
          });
          commandEncoder.copyBufferToBuffer({
            sourceBuffer,
            sourceOffset: globalIndices.byteOffset + (rowCount - DISPLAY_ROWS) * 4,
            destinationBuffer: variant.globalPreview,
            destinationOffset: DISPLAY_ROWS * 4,
            size: DISPLAY_ROWS * 4
          });
        }
        encoded = true;
        dirty = false;
        if (!summaryReader.isPending) readbackSignature = signature;
        summaryReader.request(commandEncoder);
      } else {
        if (!summaryReader.isPending) readbackSignature = signature;
        summaryReader.flush(commandEncoder);
      }
    },
    getLayers() {
      const lookupValues = getPhysicalBuffer(lookupQuery.matchMask.data[0]);
      const selectedJoin = ctx.options.join;
      const layerValues = lookupValues;
      return [
        new SpatialAnalysisPolygonLayer({
          id: `county-coverage-${selectedJoin}`,
          coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
          coordinateOrigin: [origin[0], origin[1], 0],
          triangles: polygonBuffers.triangles,
          features: polygonBuffers.triangleFeatures,
          vertexCount: polygonBuffers.vertexCount,
          values: layerValues,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: [
            [72, 47, 78, 245],
            [25, 170, 155, 235]
          ],
          noDataColor: [0, 0, 0, 0],
          opacity: 0.95
        }),
        ...(ctx.options.showOutline
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'county-query-outline',
                coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
                coordinateOrigin: [origin[0], origin[1], 0],
                segments: polygonBuffers.outline,
                valueIndices: polygonBuffers.outlineFeatures,
                instanceCount: polygonBuffers.outlineCount,
                widthPixels: 0.35,
                color: [245, 245, 245, 95]
              })
            ]
          : []),
        ...(selectedCountySegments > 0
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'county-query-selected-outline',
                coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
                coordinateOrigin: [origin[0], origin[1], 0],
                segments: selectionBuffer,
                instanceCount: selectedCountySegments,
                widthPixels: 2.1,
                color: [255, 255, 255, 245]
              })
            ]
          : [])
      ];
    },
    onClick({coordinate}) {
      if (!coordinate) return false;
      const row = countyLocator.locate(coordinate[0], coordinate[1]);
      if (row < 0) return false;
      const rings: [number, number][][] = [];
      for (
        let ring = layout.featureRingOffsets[row];
        ring < layout.featureRingOffsets[row + 1];
        ring++
      ) {
        const points: [number, number][] = [];
        for (
          let vertex = layout.ringOffsets[ring];
          vertex < layout.ringOffsets[ring + 1];
          vertex++
        ) {
          points.push([layout.lngLat[vertex * 2], layout.lngLat[vertex * 2 + 1]]);
        }
        rings.push(points);
      }
      const segments = projectRingsToSegments(rings, projectCounty);
      selectionBuffer.write(segments);
      selectedCountySegments = segments.length / 4;
      ctx.setReadout(
        'selectedCounty',
        `${countyLabel(names, fips, row)} · FIPS ${fips[row]} · ${share2024Validity[row] ? '2024 observed' : '2024 unavailable'}`
      );
      ctx.requestLayers();
      return true;
    },
    setOption() {
      latestSignature = JSON.stringify(ctx.options);
      dirty = true;
      summaryReader.markStale();
      ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      summaryReader.stop();
      allQueries.forEach(query => query.destroy());
      stateFrame.destroy();
      diagnosticStateFrame.destroy();
      diagnosticFrame.destroy();
      frame.destroy();
      observedFrame.destroy();
      resources.destroy();
    }
  };
}
