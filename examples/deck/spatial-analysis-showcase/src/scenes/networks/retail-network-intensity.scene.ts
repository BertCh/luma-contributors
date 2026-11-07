// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RetailNetworkIntensityOptions} from './retail-network-intensity.compute';

const DENSITY_CLASSES = makeClassTable({
  breaks: [0.2, 0.5, 1, 2, 4, 8],
  colors: [
    [133, 195, 219, 230],
    [98, 179, 192, 235],
    [119, 191, 141, 240],
    [220, 195, 94, 245],
    [236, 133, 67, 250],
    [189, 61, 83, 255],
    [137, 30, 85, 255]
  ],
  labels: ['<0.2', '0.2–0.5', '0.5–1', '1–2', '2–4', '4–8', '>8'],
  unit: 'mapped places per km of street',
  noData: {color: [100, 108, 123, 255], label: 'density withheld'},
  method:
    'GPU network KDE; triangular shortest-path kernel with event-specific mass normalization over edge-length quadrature.'
});

const CATEGORY_OPTIONS = [
  {value: 'retail', label: 'Retail'},
  {value: 'restaurant_cafe', label: 'Restaurants and cafés'},
  {value: 'grocery', label: 'Groceries'},
  {value: 'combined', label: 'Combined shops and food'}
] as const;

/** First- and second-order point-process analysis on a complete local Chicago street graph. */
export default defineScene<RetailNetworkIntensityOptions>({
  id: 'retail-network-intensity',
  title: 'Does shop clustering remain after intensity?',
  chapter: 'networks',
  order: 7,
  summary:
    'Estimate place intensity along a real Chicago street network, then measure pair interaction after reweighting by that varying first-order intensity.',
  contributors: ['GPUNetworkCostMatrix', 'GPUNetworkKernelDensity', 'GPUNetworkInhomogeneousK'],
  datasets: [
    {
      id: 'chicago-roads',
      role: 'OpenStreetMap drive network used as a local undirected street graph'
    },
    {id: 'chicago-places', role: 'Overture mapped retail, restaurant/café and grocery events'}
  ],
  initialView: {longitude: -87.64, latitude: 41.88, zoom: 12.0},
  basemap: ground('night', {labels: 'above'}),
  furniture: {
    title: {
      subtitle: 'Downtown Chicago · mapped places on streets',
      chips: ['GPU network intensity · Ang-corrected K']
    },
    scaleBar: {units: 'metric', latitude: 41.88},
    credit:
      'OpenStreetMap contributors (ODbL) · Overture Maps Foundation (CDLA-Permissive-2.0; mixed source licences)',
    caveat:
      'Mapped place coverage is not a business census or footfall measure. The local street graph and vertex snapping define this descriptive analysis.'
  },
  options: [
    {
      kind: 'select',
      id: 'category',
      label: 'Place category',
      group: 'Observed sample',
      apply: 'param',
      default: 'retail',
      options: CATEGORY_OPTIONS,
      help: 'The sample contains 96 eligible mapped places from this category inside the downtown study graph.'
    },
    {
      kind: 'slider',
      id: 'maximumSnapDistance',
      label: 'Maximum vertex snap',
      group: 'Observed sample',
      apply: 'param',
      min: 50,
      max: 150,
      step: 10,
      default: 80,
      unit: 'm',
      help: 'Places farther from every retained intersection are excluded. Events are assigned to the nearest intersection, not to an interpolated edge position.'
    },
    {
      kind: 'slider',
      id: 'sampleSeed',
      label: 'Deterministic sample seed',
      group: 'Observed sample',
      apply: 'param',
      min: 1,
      max: 30,
      step: 1,
      default: 1,
      help: 'Selects a reproducible 96-place sample from eligible rows. Repeated snaps to a single intersection remain distinct events.'
    },
    {
      kind: 'select',
      id: 'bandwidth',
      label: 'Network KDE bandwidth',
      group: 'First-order intensity',
      apply: 'param',
      default: '2000',
      options: [
        {value: '300', label: '300 m'},
        {value: '600', label: '600 m'},
        {value: '1000', label: '1,000 m'},
        {value: '1500', label: '1,500 m'},
        {value: '2000', label: '2,000 m'},
        {value: '3000', label: '3,000 m'}
      ],
      help: 'Selects the triangular shortest-path kernel bandwidth. Each event kernel is normalized by its GPU-computed edge-length mass over the full retained graph. For this reproducible sample, the default 2,000 m bandwidth has positive leave-one-out intensity at every sampled event across all four categories.'
    },
    {
      kind: 'slider',
      id: 'maximumRadius',
      label: 'Largest K distance',
      group: 'Second-order interaction',
      apply: 'param',
      min: 300,
      max: 2500,
      step: 100,
      default: 1500,
      unit: 'm',
      help: 'The curve includes 24 increasing network-distance thresholds up to this maximum.'
    },
    {
      kind: 'select',
      id: 'correction',
      label: 'Network shell correction',
      group: 'Second-order interaction',
      apply: 'param',
      default: 'ang',
      options: [
        {value: 'ang', label: 'Ang geometric correction'},
        {value: 'none', label: 'No geometric correction'}
      ],
      help: 'Ang correction accounts for how much network lies at each shortest-path distance shell. It is the recommended default for interpreting the complete-random identity.'
    },
    {
      kind: 'select',
      id: 'normalization',
      label: 'K normalization',
      group: 'Second-order interaction',
      apply: 'param',
      default: 'network-length',
      options: [
        {value: 'network-length', label: 'Network length'},
        {value: 'inverse-intensity', label: 'Inverse-intensity sum'},
        {value: 'inverse-intensity-squared', label: 'Inverse-intensity squared'}
      ],
      help: 'Changes the denominator of the intensity-reweighted pair sum. The complete-random identity line is shown only with network-length normalization.'
    },
    {
      kind: 'toggle',
      id: 'showPlaces',
      label: 'Show original and snapped places',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Yellow dots show source locations; cyan rings show assigned street intersections.'
    }
  ],
  story: [
    {
      id: 'local-network',
      headline:
        'The retained street boundary defines both distances and the intensity denominator.',
      title: 'Define a connected street study graph',
      body: 'The map shows the largest connected component of the OSM drive network induced inside the downtown study box. Distances use every retained street edge without a radius cutoff; the graph reports its own length and the node quadrature weights sum to that length.',
      evidence: '**{{studyBox}}** · **{{networkSummary}}** · **{{quadratureSummary}}**',
      caveat:
        'The local study boundary removes streets outside this box. Results describe this retained network and the simplified source graph, not all Chicago roads.',
      textAlternative:
        'A bounded downtown Chicago street graph is displayed with a readout of its nodes, edges, length and quadrature total.',
      controls: [],
      readouts: ['studyBox', 'networkSummary', 'quadratureSummary']
    },
    {
      id: 'snap-real-places',
      headline: 'Intersection snapping changes the locations used by the statistical model.',
      title: 'Make the event-to-network adjustment visible',
      body: 'Ninety-six Overture places are sampled reproducibly and snapped to their nearest retained intersection. Yellow dots keep the original point locations visible and cyan rings mark the network event nodes used by both GPU estimators.',
      evidence: '**{{eventSample}}** · **{{snapSummary}}**',
      caveat:
        'Vertex snapping moves mapped places. The median and maximum displacement are reported, and a selected place category may reflect source coverage rather than all businesses.',
      textAlternative:
        'Yellow original place points sit near cyan rings at their assigned intersections.',
      controls: ['category', 'maximumSnapDistance', 'sampleSeed', 'showPlaces'],
      readouts: ['eventSample', 'snapSummary']
    },
    {
      id: 'first-order-intensity',
      headline: 'Normalized kernels conserve event mass, while bandwidth changes local intensity.',
      title: 'Estimate where mapped places concentrate',
      body: 'The GPU network KDE uses shortest-path distance and a triangular kernel. Each event kernel is normalized by its integrated mass over the graph; half of each unique street-edge length is assigned to both endpoints for trapezoidal quadrature. Compare bandwidths from 300 to 3,000 m. The quadrature-weighted intensity integral should recover the 96 sampled events.',
      evidence: '**{{intensitySummary}}** · integral **{{kernelIntegral}}** · **{{solverStatus}}**',
      caveat:
        'The edge-length quadrature approximates a continuous integral on this simplified undirected road graph. It is not a planar KDE and it estimates mapped-place intensity, not customers or demand. Zero leave-one-out intensity remains zero and withholds K rather than being replaced by a floor.',
      textAlternative:
        'Street segments are colored by GPU-estimated mapped places per kilometre at the active network bandwidth.',
      controls: ['bandwidth'],
      readouts: ['intensitySummary', 'kernelIntegral', 'solverStatus']
    },
    {
      id: 'second-order-after-baseline',
      headline:
        'Intensity reweighting separates the fitted baseline from remaining pair structure.',
      title: 'Ask whether pair interaction remains',
      body: 'The intensity estimated at each event uses a leave-one-out GPU KDE query at its snapped node. The GPU inhomogeneous network K statistic reweights ordered event pairs by both endpoint intensities and includes an optional Ang shell correction. With Ang correction and network-length normalization, compare the curve with the complete-random identity.',
      evidence: '**{{kEndpoint}}** · **{{normalizationSummary}}**',
      caveat:
        'This is a descriptive estimate without a simulation envelope or formal test. An estimate above the identity can indicate excess interaction after the fitted first-order intensity, but this mapped sample and local graph do not establish a causal mechanism.',
      textAlternative:
        'A line chart compares the inhomogeneous K curve with the geometrically corrected complete-random identity when its normalization supports that comparison.',
      controls: ['maximumRadius', 'correction', 'normalization'],
      readouts: ['inhomogeneousK', 'kEndpoint', 'normalizationSummary', 'solverStatus']
    }
  ],
  legends: () => [
    getClassTableLegend(DENSITY_CLASSES, {
      id: 'network-place-intensity',
      title: 'GPU network intensity',
      note: 'Triangular shortest-path kernels; per-event mass normalized over edge-length quadrature.'
    }),
    {
      kind: 'categories',
      title: 'Mapped events',
      entries: [
        {color: [255, 225, 125, 220], label: 'Original mapped place'},
        {color: [41, 219, 205, 240], label: 'Assigned network intersection'}
      ],
      note: 'Every visible snap displacement is part of the analysis assumption.'
    }
  ],
  readouts: [
    {id: 'studyBox', label: 'Local study extent'},
    {id: 'networkSummary', label: 'Connected study network'},
    {id: 'quadratureSummary', label: 'Edge-length quadrature'},
    {id: 'eventSample', label: 'Mapped event sample'},
    {id: 'snapSummary', label: 'Vertex snap displacement'},
    {id: 'intensitySummary', label: 'Estimated event intensity'},
    {id: 'kernelIntegral', label: 'Integrated network intensity'},
    {id: 'solverStatus', label: 'Complete GPU result status'},
    {id: 'normalizationSummary', label: 'K denominators'},
    {id: 'kEndpoint', label: 'Inhomogeneous K at radius'},
    {id: 'inhomogeneousK', label: 'Intensity-reweighted K curve', kind: 'chart'}
  ],
  snippet: state =>
    `new GPUNetworkKernelDensity({distances: completeNodeToEventDistances, eventCount: 96, quadratureLengths: halfEdgeLengthAtNodes, bandwidth: ${state.bandwidth}});\nnew GPUNetworkInhomogeneousK({correction: '${state.correction}', normalization: '${state.normalization}', maximumPairCount: 96 * 96});`,
  about: {
    what: 'First-order network kernel intensity and second-order inhomogeneous K for real Overture places on a local Chicago street graph.',
    why: 'A map of dense first-order intensity can look clustered even when event pairs have no residual interaction. Inhomogeneous K reweights pairs by an estimated spatially varying baseline.',
    howToRead:
      'Inspect the mapped intensity bandwidth, the original-to-node snap displacement, and the K curve. The Ang-corrected complete-random identity is meaningful here only with network-length normalization; this is descriptive, not a simulation test.'
  },
  create: async ctx =>
    (await import('./retail-network-intensity.compute')).createRetailNetworkIntensity(ctx)
});
