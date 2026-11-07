// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {CITY_FRAMES} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {FLOW_CREDITS} from '../flows/flows-style';
import {defineScene} from '../scene';
import type {BixiCellFlowsOptions} from './bixi-cell-flows.types';

export default defineScene<BixiCellFlowsOptions>({
  id: 'bixi-cell-flows',
  create: async ctx => {
    const compute = await import('./bixi-cell-flows.compute');
    return compute.createBixiCellFlows(ctx);
  },
  title: 'Do boundary flows account for every trip?',
  chapter: 'cells',
  order: 6,
  summary:
    'Aggregate BIXI’s retained monthly station-pair ledger to H3 cells, compare conserved outgoing and incoming rides with immediate-neighbor boundary flows, and inspect how the answer changes with resolution.',
  contributors: ['addCellFlowTopologyRecipe', 'GPUH3CellIdentities'],
  datasets: [{id: 'bixi-flows', role: 'August 2024 station pairs and station coordinates'}],
  initialView: CITY_FRAMES.montreal,
  basemap: ground('night'),
  furniture: {
    title: {title: 'The ride is not the boundary', subtitle: 'Montreal · BIXI August 2024'},
    scaleBar: {units: 'metric'},
    credit: joinCredits(FLOW_CREDITS.bixi, CREDITS.openStreetMap),
    caveat:
      'The input is the retained station-pair ledger. Boundary flows include only immediately adjacent H3 cells.'
  },
  options: [
    {
      kind: 'select',
      id: 'resolution',
      label: 'H3 resolution',
      group: 'Cell system',
      apply: 'compile',
      default: '7',
      options: [
        {value: '6', label: '6 · broad cells'},
        {value: '7', label: '7 · neighbourhood scale'},
        {value: '8', label: '8 · fine cells'}
      ],
      help: 'Reassigns station endpoints and preaggregates the actual retained OD pairs to this resolution before the bounded GPU topology scan.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Selected cell display',
      group: 'Inspection',
      apply: 'param',
      default: 'od-endpoints',
      options: [
        {value: 'od-endpoints', label: 'All outgoing endpoint links'},
        {value: 'neighbor-boundaries', label: 'Immediate neighbour boundaries'}
      ],
      help: 'Click a cell mark to select its outgoing flows. Endpoint links connect cell anchors and are not traveled routes; boundary lines are GPU-produced shared H3 edges.'
    },
    {
      kind: 'button',
      id: 'previousOrigin',
      label: 'Previous origin cell',
      group: 'Inspection',
      help: 'Select the preceding H3 cell in stable key order.'
    },
    {
      kind: 'button',
      id: 'nextOrigin',
      label: 'Next origin cell',
      group: 'Inspection',
      help: 'Select the next H3 cell in stable key order.'
    }
  ],
  story: [
    {
      id: 'retained-ledger',
      title: 'Start with the retained pairs',
      headline: 'The source ledger has a defined threshold.',
      textAlternative:
        'Montreal BIXI stations and H3 cell anchors are shown over a dark city basemap. Counts describe the retained August station-pair table.',
      body: 'The monthly source table retains station pairs with at least three rides in August 2024. BIXI preprocessing drops trips under 60 seconds and trips without station or borough. The separate hourly table uses a threshold of ten rides per pair and folds rarer rows into residual hourly counts; that threshold does not define this monthly pair table. The GPU topology analysis therefore describes the retained monthly table, not every ride in the month.',
      evidence:
        '{{stationCount}} stations contribute {{sourcePairs}} retained station pairs and {{sourcePairRides}} rides. The monthly source records {{sameStationRides}} same-station rides separately; {{rareExcludedRides}} additional cross-station rides are outside the retained pair table.',
      caveat:
        'Same-station rides are a separate source category. Different stations can map to one H3 cell and still form a same-cell aggregate pair.',
      options: {resolution: '7', display: 'od-endpoints'},
      controls: ['resolution', 'display'],
      readouts: ['sourcePairs', 'sourcePairRides', 'sameStationRides', 'rareExcludedRides'],
      stage: 'source'
    },
    {
      id: 'conserved-cell-totals',
      title: 'Compare outgoing and incoming totals',
      headline: 'Every valid pair contributes at both ends.',
      textAlternative:
        'Readouts compare GPU-summed outgoing and incoming rides with the retained input mass, immediate-neighbor flows, same-cell flows and non-neighbor flows.',
      body: '`addCellFlowTopologyRecipe` validates each cell-pair row and computes outgoing and incoming totals from the same valid rows. The equality is checked after GPU readback. The edge sum is a subset: only OD pairs between immediate H3 neighbors contribute to a directed boundary edge.',
      evidence:
        'The retained valid input mass is {{validInputMass}} rides. GPU outgoing totals sum to {{gpuOutgoingTotal}} (difference from input: {{outgoingMinusValidInput}}) and incoming totals to {{gpuIncomingTotal}} (difference from input: {{incomingMinusValidInput}}); outgoing minus incoming: {{conservationDifference}}. Valid rows partition into {{sameCellRides}} same-cell, {{neighborRides}} neighbour and {{nonNeighborRides}} non-neighbour rides.',
      caveat:
        'The category partition uses CPU-preaggregated cell-pair inputs, while validity, cell totals and directed edge flows come from the GPU recipe. Invalid rows are counted and excluded from all totals.',
      options: {resolution: '7', display: 'od-endpoints'},
      controls: ['resolution', 'display'],
      readouts: [
        'validInputMass',
        'outgoingMinusValidInput',
        'incomingMinusValidInput',
        'accountingChart'
      ],
      stage: 'conserve'
    },
    {
      id: 'one-origin',
      title: 'Inspect one origin cell',
      headline: 'Endpoint links and boundary flows answer different questions.',
      textAlternative:
        'Click a cell mark to select its origin. One view draws all retained outgoing destination-cell links; the other highlights only GPU boundary segments for its immediate H3 neighbours. Charts list endpoint destinations and neighbor-edge rides.',
      body: 'Select a cell by clicking its mark. Outgoing endpoint links connect the selected cell anchor to every destination cell represented in the retained table. Change the display to neighbour boundaries to see only immediate shared edges with their GPU-computed flow values. These straight links summarize endpoints, not actual trip paths.',
      evidence:
        '{{selectedCell}} sends {{selectedOutgoing}} rides across {{selectedDestinationCount}} destination cells; {{selectedNeighborCount}} neighbouring edges carry {{selectedNeighborRides}} rides. The destination and edge charts show the corresponding breakdowns.',
      caveat:
        'Pairs whose endpoints fall in the same H3 cell contribute to cell totals but have no visible center-to-center link or inter-cell boundary.',
      options: {resolution: '7', display: 'od-endpoints'},
      controls: ['resolution', 'display', 'previousOrigin', 'nextOrigin'],
      readouts: ['selectedCell', 'selectedOutgoing', 'destinationChart', 'neighborChart'],
      stage: 'select'
    },
    {
      id: 'resolution-sensitivity',
      title: 'Change the cell scale',
      headline: 'Neighbourhood depends on resolution.',
      textAlternative:
        'The same retained station-pair mass is reaggregated at H3 resolutions six, seven or eight. Cell and neighbor-flow counts update with each selected resolution.',
      body: 'Change H3 resolution to regroup the same station endpoints. More detailed cells create a different set of immediate neighbors, so the edge-flow share can change even while outgoing and incoming totals still conserve the valid pair mass.',
      evidence:
        'At resolution {{selectedResolution}}, {{cellCount}} cells represent the retained stations and {{neighborRides}} rides cross immediate cell boundaries; GPU outgoing and incoming sums remain {{gpuOutgoingTotal}} and {{gpuIncomingTotal}}.',
      caveat:
        'This is a sensitivity to H3 resolution and neighbor definition, not a change in trip behaviour. The source pair threshold and preprocessing remain fixed.',
      options: {resolution: '7', display: 'neighbor-boundaries'},
      controls: ['resolution', 'display'],
      readouts: ['selectedResolution', 'neighborRides', 'neighborShare', 'conservationDifference'],
      stage: 'resolution'
    }
  ],
  readouts: [
    {id: 'stationCount', label: 'Stations in the source', format: 'integer'},
    {id: 'cellCount', label: 'H3 cells with stations', format: 'integer'},
    {id: 'selectedResolution', label: 'Selected H3 resolution', format: 'integer'},
    {id: 'sourcePairs', label: 'Retained station pairs', format: 'integer'},
    {id: 'sourcePairRides', label: 'Rides in retained pairs', format: 'integer'},
    {
      id: 'sameStationRides',
      label: 'Same-station rides',
      format: 'integer',
      help: 'Source property, separate from the station-pair table.'
    },
    {
      id: 'rareExcludedRides',
      label: 'Cross-station rides outside retained pairs',
      format: 'integer',
      help: 'Difference between all cross-station departures and the retained monthly station-pair mass.'
    },
    {id: 'sourceMonthlyRides', label: 'Cross-station rides in source', format: 'integer'},
    {id: 'gpuOutgoingTotal', label: 'GPU outgoing sum', format: 'integer', unit: 'rides'},
    {id: 'gpuIncomingTotal', label: 'GPU incoming sum', format: 'integer', unit: 'rides'},
    {id: 'validInputMass', label: 'GPU-valid input mass', format: 'integer', unit: 'rides'},
    {
      id: 'outgoingMinusValidInput',
      label: 'Outgoing minus valid input',
      format: 'integer',
      unit: 'rides'
    },
    {
      id: 'incomingMinusValidInput',
      label: 'Incoming minus valid input',
      format: 'integer',
      unit: 'rides'
    },
    {
      id: 'conservationDifference',
      label: 'Outgoing minus incoming',
      format: 'integer',
      unit: 'rides'
    },
    {
      id: 'sameCellRides',
      label: 'Same-cell aggregate rides',
      format: 'integer',
      help: 'Different source stations can map to one H3 cell; this does not mean the source station IDs were the same.'
    },
    {
      id: 'neighborRides',
      label: 'Immediate-neighbour rides',
      format: 'integer',
      help: 'Sum of actual GPU edge flows over directed edges between immediately adjacent H3 cells.'
    },
    {id: 'nonNeighborRides', label: 'Non-neighbour cell rides', format: 'integer'},
    {id: 'neighborShare', label: 'Share between immediate neighbors', format: 'percent'},
    {id: 'invalidPairs', label: 'Invalid cell-pair rows', format: 'integer'},
    {
      id: 'selectedCell',
      label: 'Selected origin cell',
      help: 'The borough shown is the most frequent BIXI station borough assigned to this H3 cell; borough boundaries do not define cells.'
    },
    {id: 'selectedOutgoing', label: 'Selected-cell outgoing rides', format: 'integer'},
    {id: 'selectedIncoming', label: 'Selected-cell incoming rides', format: 'integer'},
    {id: 'selectedDestinationCount', label: 'Outgoing destination cells', format: 'integer'},
    {id: 'selectedNeighborCount', label: 'Valid neighbouring edges', format: 'integer'},
    {id: 'selectedNeighborRides', label: 'Selected-cell neighbour rides', format: 'integer'},
    {id: 'accountingChart', label: 'Retained pair mass by cell relation', kind: 'chart'},
    {
      id: 'selectedOtherDestinationRides',
      label: 'Rides in other destinations',
      format: 'integer',
      unit: 'rides',
      help: 'Selected outgoing rides not shown among the 18 highest-count destination-cell links.'
    },
    {id: 'destinationChart', label: 'Top 18 outgoing destination-cell links', kind: 'chart'},
    {id: 'neighborChart', label: 'Outgoing immediate-neighbor edges', kind: 'chart'}
  ],
  legends: state => [
    {
      kind: 'categories',
      title: 'Cell-flow marks',
      entries: [
        {color: [107, 162, 183, 175] as const, label: 'Station-cell anchor'},
        {color: [250, 187, 91, 245] as const, label: 'Selected origin cell'},
        ...(state.display === 'neighbor-boundaries'
          ? [
              {color: [113, 214, 217, 230] as const, label: 'GPU shared boundary with flow'},
              {
                color: [94, 122, 132, 155] as const,
                label: 'GPU shared boundary without retained flow'
              }
            ]
          : [{color: [250, 187, 91, 145] as const, label: 'Outgoing OD endpoint link'}])
      ]
    }
  ],
  about: {
    what: '`addCellFlowTopologyRecipe` validates cell-pair keys, sums outgoing and incoming flow, and computes directed neighbor-edge flows and their GPU-produced boundaries. Station coordinates are assigned to H3 cells and the retained station-pair table is preaggregated by cell pair on the CPU to keep the bounded direct scan small.',
    why: 'Cell totals include same-cell, adjacent-cell and non-neighbor pairs. Immediate edge totals include only adjacent cells, so they should not be mistaken for all trips crossing a map boundary.',
    howToRead:
      'The source is BIXI Montreal’s August 2024 trip history. Preprocessing omits rides under 60 seconds and rows without station or borough. The monthly station-pair ledger retains pairs with at least three rides; a separate hourly table uses a ten-ride threshold and folds rarer pairs into residual hourly counts. The dashboard reports the retained monthly ledger and states excluded cross-station mass separately. Different stations in one H3 cell are same-cell pairs, not same-station rides. Outgoing links join cell endpoint anchors; they do not show ridden paths. H3 edge paths use the recipe’s GPU boundary coordinates.'
  }
});
