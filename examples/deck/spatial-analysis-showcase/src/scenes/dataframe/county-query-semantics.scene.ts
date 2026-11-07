// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {GLOBAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec} from '../scene';
import type {CountyQuerySemanticsOptions} from './county-query-semantics.compute';

export default defineScene<CountyQuerySemanticsOptions>({
  id: 'county-query-semantics',
  title: 'Where did the counties go?',
  chapter: 'dataframe',
  order: 10,
  summary:
    'A county FIPS audit runs GPU joins, lookup, validity filters, grouped summaries and full sorts over 3,109 contiguous-US county polygons and the locally aligned 2000–2024 election table. The frozen 2024 extraction has 2,859 finite two-party share observations; a missing result is not zero votes.',
  contributors: [
    'GPUDataFrame',
    'GPUExpression',
    'GPUDataFrameJoinQuery',
    'GPUDataFrameLookupQuery',
    'GPUDataFrameAggregationQuery',
    'GPUDataFrameGroupedAggregationQuery',
    'GPUDataFrameGlobalSortQuery',
    'GPUDataFrameSortQuery'
  ],
  datasets: [
    {id: 'us-counties', role: '2022 Census county-equivalent boundaries and five-digit FIPS'},
    {
      id: 'us-elections',
      role: 'MIT Election Data + Science Lab county returns, locally aligned by FIPS'
    }
  ],
  initialView: {longitude: -98, latitude: 39, zoom: 3.15},
  basemap: ground('paperSheet'),
  furniture: {
    ...GLOBAL_FURNITURE,
    title: {
      title: 'Where did the counties go?',
      subtitle: 'FIPS joins, explicit nulls and full GPU orderings',
      sample: '3,109 contiguous-US county-equivalent polygons',
      chips: ['Local extraction audit', 'Not a zero-vote map']
    },
    credit: 'US Census Bureau; MIT Election Data + Science Lab; MEDSL 2024 precinct returns',
    caveat:
      'Local 2024 preprocessing excludes states that fail its reconciliation threshold. CT’s nine 2022 planning regions differ from legacy county returns; no crosswalk is inferred. Alaska and Hawaii are outside this bundled geography.'
  },
  options: [
    {
      kind: 'select',
      id: 'join',
      label: 'Join result shown',
      group: 'FIPS query',
      apply: 'param',
      default: 'lookup',
      display: 'segmented',
      options: [
        {value: 'lookup', label: 'Lookup'},
        {value: 'inner', label: 'Inner'},
        {value: 'left', label: 'Left'},
        {value: 'semi', label: 'Semi'},
        {value: 'anti', label: 'Anti'}
      ],
      help: 'Switches the live join result readout. The map remains the source-aligned lookup match mask so accepted and missing county support stays visible.'
    },
    {
      kind: 'select',
      id: 'sortDirection',
      label: 'Full sort direction',
      group: 'Ordering',
      apply: 'param',
      default: 'ascending',
      display: 'segmented',
      options: [
        {value: 'ascending', label: 'Low first'},
        {value: 'descending', label: 'High first'}
      ],
      help: 'Selects a precompiled full global ordering and a full independent sort within each application-created state batch. The compact readout shows the first county from several state batches.'
    },
    {
      kind: 'select',
      id: 'nulls',
      label: 'Null / NaN placement',
      group: 'Ordering',
      apply: 'param',
      default: 'last',
      display: 'segmented',
      options: [
        {value: 'first', label: 'First'},
        {value: 'last', label: 'Last'}
      ],
      help: 'In source-null mode, moves explicitly invalid rows before or after valid rows. In the diagnostic mode, reuses this control for valid IEEE NaN placement.'
    },
    {
      kind: 'select',
      id: 'nanMode',
      label: 'Archived NaN interpretation',
      group: 'Ordering',
      apply: 'param',
      default: 'source-null',
      display: 'segmented',
      options: [
        {value: 'source-null', label: 'Unavailable / null'},
        {value: 'valid-nan', label: 'Valid NaN diagnostic'}
      ],
      help: 'The default uses an analytical validity sidecar derived from source finiteness. The diagnostic reuses the exact same float32 payloads and rows but marks every 2024 share valid, isolating NaN placement; those NaNs are still unavailable observations, not measured shares.'
    },
    {
      kind: 'toggle',
      id: 'showOutline',
      label: 'County boundary hairlines',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Toggles fine county outlines to help locate small polygons against the muted atlas ground.'
    }
  ],
  legends: (_state): LegendSpec[] => [
    {
      id: 'coverage',
      title: '2024 local extraction support',
      kind: 'categories',
      entries: [
        {color: [25, 170, 155, 235], label: 'Finite two-party share observed'},
        {color: [72, 47, 78, 245], label: 'Unavailable in the local table'}
      ],
      note: 'Absence is not zero votes. Teal marks a finite source value; plum marks a missing observation.'
    }
  ],
  readouts: [
    {id: 'countyCount', label: 'Polygon/FIPS rows', unit: 'count'},
    {id: 'selectedCounty', label: 'Selected county', layout: 'block'},
    {id: 'sourceCoverage', label: 'Finite 2024 observations'},
    {id: 'missingCoverage', label: 'Nulls / lookup misses'},
    {id: 'lookupStatus', label: 'Source-aligned lookup'},
    {id: 'joinCounts', label: 'Join output counts', layout: 'block'},
    {id: 'activeJoin', label: 'Selected join result'},
    {id: 'joinDiagnostics', label: 'Join index diagnostics'},
    {id: 'joinCapacity', label: 'Capacity / uniqueness'},
    {id: 'validNullFilters', label: 'GPU validity predicates'},
    {id: 'changeSummary', label: '2020→2024 share change', unit: 'percentage points'},
    {id: 'shareExpressions', label: '2024 share expressions', layout: 'block'},
    {id: 'shareNullPredicates', label: 'Three-valued NOT predicates', layout: 'block'},
    {id: 'shareBoundary', label: 'Exact cutoff and null boundary', layout: 'block'},
    {id: 'stateGroups', label: 'State-FIPS group summary'},
    {id: 'globalSortRows', label: 'Global full-sort leading rows', layout: 'block'},
    {id: 'batchSortRows', label: 'First county from state-batch sorts', layout: 'block'},
    {id: 'nanRepresentation', label: 'Active validity representation'},
    {id: 'sortContract', label: 'Ordering scope'},
    {id: 'sourceFootnote', label: 'Geography / source note', layout: 'block'}
  ],
  story: [
    {
      id: 'atlas',
      title: 'Begin with the geography key',
      headline: 'The boundary table has 3,109 county-equivalent rows',
      body: 'Each polygon keeps the bundled five-digit FIPS from the 2022 Census geography. The atlas is contiguous US plus DC; the outline gives the fine-grained support table a geographic frame. Election fields are aligned to those same rows by FIPS.',
      textAlternative:
        'A muted national county atlas distinguishes finite 2024 local two-party share records in teal from unavailable rows in dark plum.',
      evidence:
        'The source contains **{{countyCount}}** polygon/FIPS rows; **{{sourceCoverage}}** have finite 2024 values. {{selectedCounty}}',
      caveat:
        'Alaska, Hawaii and territories are not in this geometry. Connecticut uses nine planning regions, which do not match the legacy county reporting units in older election returns.',
      controls: ['showOutline'],
      readouts: ['countyCount', 'sourceCoverage', 'selectedCounty'],
      camera: {longitude: -98, latitude: 39, zoom: 3.15, transitionMs: 1200}
    },
    {
      id: 'nulls-and-lookup',
      title: 'Separate missing from zero',
      headline: 'The extraction leaves 250 unavailable 2024 values',
      body: 'The uploaded float32 columns retain their source NaN payloads, while aligned uint32 validity sidecars mark those rows unavailable in the analytical view. `isValid` and `isNull` run as GPU filters. The observed-right table contains only finite FIPS rows, so lookup misses are visible as missing support rather than invented zeroes.',
      textAlternative:
        'A teal-and-plum county map shows which local 2024 share values are finite and which are missing. A white outline marks a county selected by clicking. GPU readouts report the valid and null filters plus lookup matches and misses.',
      evidence: '**{{validNullFilters}}**; lookup reports **{{lookupStatus}}**.',
      caveat:
        'These are properties of the frozen local files. 2024 was rebuilt from precinct returns and states that fail the local 1.5% reconciliation rule were excluded; this is not a live upstream completeness estimate.',
      controls: [],
      readouts: ['missingCoverage', 'lookupStatus', 'validNullFilters']
    },
    {
      id: 'boolean-null-semantics',
      title: 'Boolean NOT preserves unknown support',
      headline: 'A null remains unknown until the query names it',
      body: 'The source `demTwoParty2024` values are **fractions from 0 to 1**. The GPU derives the Republican complement as `1 − share` and a normalized two-party margin as `(share − 0.5) / 0.5`; the profile reports unweighted county means over the 2,859 valid rows. `not(share > 0.5)` retains null as unknown, so a filter selects the 2,457 known rows at or below one half. Adding `isNull()` explicitly with `OR` includes the 250 unknown counties and yields 2,707 rows. An exact `share == 0.5` query finds no ties in this extract; `notEqual(0.5)` counts the 2,859 finite non-ties while leaving null rows unknown.',
      textAlternative:
        'The map still distinguishes observed and unavailable source rows. GPU readouts show the mean Democratic share, its Republican complement, a normalized margin, the filtered count for three-valued NOT, the count after explicitly including nulls, exact ties and finite non-ties.',
      evidence: '**{{shareExpressions}}**; **{{shareNullPredicates}}**; **{{shareBoundary}}**.',
      caveat:
        'Shares describe the locally bundled two-party county returns, not turnout, total ballots or voter behaviour. The summaries weight each county equally. The observed absence of exact ties does not demonstrate tie handling for other inputs.',
      controls: [],
      readouts: ['shareExpressions', 'shareNullPredicates', 'shareBoundary']
    },
    {
      id: 'join-definitions',
      title: 'Ask what the join keeps',
      headline: 'One key set produces four different row sets',
      body: 'The unique-right observed table drives real `innerJoin`, `leftJoin`, `semiJoin` and `antiJoin` queries, each with a 3,109-pair publication capacity and bounded hash probes. Change **Join result shown** below to inspect its count and diagnostics; the map keeps the aligned lookup mask as the geographic reference.',
      textAlternative:
        'The readout compares inner, left, semi and anti join output sizes and reports the selected operation’s hash probes, duplicate-key contract and capacity status. The map marks observed versus unavailable counties.',
      evidence: '**{{joinCounts}}**; selected result: **{{activeJoin}}**; {{joinCapacity}}.',
      caveat:
        'The sparse right table is an adapter view containing finite 2024 observations. It is intentionally not the source’s original batch layout.',
      controls: ['join'],
      readouts: ['joinCounts', 'activeJoin', 'joinDiagnostics', 'joinCapacity']
    },
    {
      id: 'change-and-groups',
      title: 'Carry nulls through a derived measure',
      headline: 'The GPU computes a nullable share change',
      body: 'The dataframe derives `(share2024 − share2020) × 100`, in percentage points. Its validity depends on both years, then the GPU aggregation reports an unweighted mean across supported counties. A second GPU `groupBy` counts rows by state FIPS and shows county counts and finite-share means for named states; Alaska and Hawaii are explicitly unavailable in this bundled geography.',
      textAlternative:
        'Readouts state how many counties support the 2020 to 2024 change, its county-weighted mean where every county receives equal weight, and a state-FIPS grouped county summary.',
      evidence: '**{{changeSummary}}**; **{{stateGroups}}**.',
      caveat:
        'County means weight each county equally rather than weighting by votes or population. Share is a fraction of votes cast for the two major parties, not turnout.',
      controls: [],
      readouts: ['changeSummary', 'stateGroups']
    },
    {
      id: 'full-order',
      title: 'Compare global and batch ordering',
      headline: 'The same payload can sort as null or NaN',
      body: '`sortByGlobal` orders every county together; `sortBy` runs a full order separately inside application-created state batches cut from the original county batch. The default keeps 250 archived NaN payloads invalid, so `Null placement` moves them as unavailable rows. Switch **Archived NaN interpretation** to the diagnostic view: it keeps the exact same float32 bits and county rows but marks those entries valid, making the reused **Null / NaN placement** control act on IEEE NaNs instead. Validity counts and first/last excerpts come from GPU outputs; the archived NaNs remain unavailable source observations in both views.',
      textAlternative:
        'GPU sort lists show global leading rows and first rows from state-local sorts. A mode switch preserves the source nulls or temporarily marks the same archived NaN payload rows valid; the displayed GPU validity counts distinguish the representations, and placement changes the actual ordering.',
      evidence:
        '**{{nanRepresentation}}** The global list shows the first and last rows in diagnostic mode; state lists show the first row from each local sort. {{sortContract}}',
      caveat:
        'Rows in different state batches are never compared in the batch-local ordering. Validity is a separate sidecar from IEEE finiteness: marking a NaN payload valid does not make it a source observation or a usable share.',
      controls: ['sortDirection', 'nulls', 'nanMode'],
      readouts: ['nanRepresentation', 'globalSortRows', 'batchSortRows', 'sortContract']
    }
  ],
  snippet: state => `const countyRows = frame
  .withColumn('change2020to2024', column('share2024')
    .subtract(column('share2020')).multiply(literal(100)))
  .filter(column('change2020to2024').isValid());
const expressions = frame
  .withColumn('republicanShare2024', column('share2024').negate().add(literal(1)))
  .withColumn('normalizedMargin2024', column('share2024')
    .subtract(literal(0.5)).divide(literal(0.5)));
const knownNonMajority = expressions.filter(not(column('share2024').greaterThan(literal(0.5))));
const unknownInclusive = expressions.filter(column('share2024').isNull()
  .or(not(column('share2024').greaterThan(literal(0.5)))));
const observed = countyRows.leftJoin(finite2024, {
  leftOn: 'fips', rightOn: 'fips', capacity: 3109
});
const globalOrder = frame.sortByGlobal('share2024', {
  direction: '${state.sortDirection}', nulls: '${state.nulls}'
});`,
  about: {
    what: 'A GPU dataframe audit of county FIPS joins, local election-record availability, null validity, grouped counts and full row ordering.',
    why: 'Join type and missing-value semantics determine which geographic units remain in an analysis and which denominator a summary describes.',
    howToRead:
      'Teal means a finite local 2024 observation; plum means unavailable. The map uses a source-aligned GPU lookup mask. Join counts, null-filter counts, a nullable derived summary and bounded sort heads come from GPU outputs. The ordering diagnostic may reinterpret archived NaN payload bits as valid NaNs, but never as observed shares.'
  },
  create: async ctx =>
    (await import('./county-query-semantics.compute')).createCountyQuerySemantics(ctx)
});
