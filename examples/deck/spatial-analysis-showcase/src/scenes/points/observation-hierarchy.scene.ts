// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHICAGO, findPlace} from '../../cartography/gazetteer';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {
  CLUSTER_COLORS,
  NOISE_COLOR,
  UNRESOLVED_COLOR,
  type ObservationHierarchyOptions
} from './observation-hierarchy.shared';
import {OBSERVATION_HIERARCHY_BOUNDS} from './observation-hierarchy-data';

const DEFAULT_OPTIONS: ObservationHierarchyOptions = {
  algorithm: 'optics-dbscan',
  minimumSamples: '8',
  minimumClusterSize: '12',
  extractionDistance: '150',
  xi: '0.05',
  predecessorCorrection: true
};
const ISLAND = findPlace(CHICAGO, 'northerly-island')!;

function getClusterLegend(labels: number[], hasNoise: boolean, complete: boolean): LegendSpec {
  return {
    kind: 'categories',
    title: 'Observed GPU cluster labels',
    layout: 'list',
    entries: complete
      ? [
          ...labels.map(label => ({
            color: CLUSTER_COLORS[label % CLUSTER_COLORS.length],
            label: `Cluster ${label + 1}`,
            shape: 'dot' as const
          })),
          ...(hasNoise ? [{color: NOISE_COLOR, label: 'Noise', shape: 'dot' as const}] : [])
        ]
      : [{color: UNRESOLVED_COLOR, label: 'Labels pending or withheld', shape: 'dot'}],
    note: 'Only labels present in this GPU result are listed. Colours repeat after eight labels; rings mark coincident source positions.'
  };
}

function getSnippet(state: ObservationHierarchyOptions): string {
  if (state.algorithm === 'hdbscan') {
    return `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUHierarchicalDensityClustering} from '@luma-contributors/analysis/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'northerly-hdbscan'});
graph.add(new GPUHierarchicalDensityClustering({
  spatialContext: projection.spatialContext,
  positions,                         // 246 source records, projected to local metres
  minimumSamples: ${Number(state.minimumSamples)},
  minimumClusterSize: ${Number(state.minimumClusterSize)},
  output: {labels, coreDistances, status: {count: clusterCount}}
}));
const compiled = await graph.compileAsync();
compiled.encode(commandEncoder, {parameters: undefined});`;
  }
  const extraction = state.algorithm === 'optics-xi' ? 'xi' : 'dbscan';
  const extractionOptions =
    extraction === 'xi'
      ? `xi: ${Number(state.xi)},
  minimumClusterSize: ${Number(state.minimumClusterSize)},
  predecessorCorrection: ${state.predecessorCorrection},`
      : `extractionDistance: ${Number(state.extractionDistance)},`;
  return `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUOPTICS} from '@luma-contributors/analysis/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'northerly-optics'});
graph.add(new GPUOPTICS({
  positions,                         // 246 source records, projected to local metres
  minimumSamples: ${Number(state.minimumSamples)},
  maximumDistance: 1000,
  extractionMethod: '${extraction}',
  ${extractionOptions}
  output: {ordering, coreDistances, reachability, predecessors, labels, status}
}));
const compiled = await graph.compileAsync();
compiled.encode(commandEncoder, {parameters: undefined});`;
}

export default defineScene<ObservationHierarchyOptions>({
  id: 'observation-hierarchy',
  title: 'How do observation clusters take shape?',
  chapter: 'points',
  order: 6,
  summary:
    'Trace all 246 archived bird observations in a Northerly Island frame through OPTICS reachability, DBSCAN and Xi extraction, then compare HDBSCAN EOM clusters.',
  contributors: ['GPUOPTICS', 'GPUHierarchicalDensityClustering'],
  datasets: [{id: 'chicago-nature', role: 'iNaturalist observations (2023)'}],
  initialView: {
    longitude: ISLAND.lngLat[0],
    latitude: ISLAND.lngLat[1],
    zoom: 13.4,
    pitch: 0,
    bearing: 0
  },
  options: [
    {
      kind: 'select',
      id: 'algorithm',
      label: 'Clustering method',
      group: 'Model',
      apply: 'compile',
      default: DEFAULT_OPTIONS.algorithm,
      help: 'OPTICS creates a visit ordering and reachability curve; DBSCAN cuts that ordering at a distance, Xi extracts valleys, and HDBSCAN selects persistent clusters by EOM.',
      options: [
        {value: 'optics-dbscan', label: 'OPTICS + DBSCAN cut'},
        {value: 'optics-xi', label: 'OPTICS + Xi extraction'},
        {value: 'hdbscan', label: 'HDBSCAN EOM'}
      ]
    },
    {
      kind: 'select',
      id: 'minimumSamples',
      label: 'Minimum samples',
      group: 'Model',
      apply: 'compile',
      default: DEFAULT_OPTIONS.minimumSamples,
      help: 'The k-th neighbour distance defines each point’s core distance. Smaller values are more sensitive to local structure.',
      options: [
        {value: '4', label: '4'},
        {value: '8', label: '8'},
        {value: '16', label: '16'}
      ]
    },
    {
      kind: 'select',
      id: 'minimumClusterSize',
      label: 'Minimum cluster size',
      group: 'Extraction',
      apply: 'compile',
      default: DEFAULT_OPTIONS.minimumClusterSize,
      help: 'Xi and HDBSCAN use this minimum membership when retaining clusters.',
      disabledWhen: state => state.algorithm === 'optics-dbscan',
      options: [
        {value: '8', label: '8 records'},
        {value: '12', label: '12 records'},
        {value: '16', label: '16 records'},
        {value: '32', label: '32 records'}
      ]
    },
    {
      kind: 'select',
      id: 'extractionDistance',
      label: 'DBSCAN extraction distance',
      group: 'Extraction',
      apply: 'compile',
      default: DEFAULT_OPTIONS.extractionDistance,
      help: 'Only used by OPTICS + DBSCAN; distances are planar metres in the local frame.',
      disabledWhen: state => state.algorithm !== 'optics-dbscan',
      options: [
        {value: '75', label: '75 m'},
        {value: '150', label: '150 m'},
        {value: '300', label: '300 m'}
      ]
    },
    {
      kind: 'select',
      id: 'xi',
      label: 'Xi steepness',
      group: 'Extraction',
      apply: 'compile',
      default: DEFAULT_OPTIONS.xi,
      help: 'Only used by Xi extraction. A smaller value accepts shallower reachability valleys.',
      disabledWhen: state => state.algorithm !== 'optics-xi',
      options: [
        {value: '0.05', label: '0.05'},
        {value: '0.1', label: '0.10'},
        {value: '0.2', label: '0.20'}
      ]
    },
    {
      kind: 'toggle',
      id: 'predecessorCorrection',
      label: 'Xi predecessor correction',
      group: 'Extraction',
      apply: 'compile',
      default: DEFAULT_OPTIONS.predecessorCorrection,
      disabledWhen: state => state.algorithm !== 'optics-xi',
      help: 'The optional Xi correction uses predecessor reachability to revise cluster boundaries.'
    }
  ],
  readouts: [
    {id: 'archiveRows', label: 'Archived source rows'},
    {id: 'frameRows', label: 'Bird records in frame'},
    {id: 'coordinateReuse', label: 'Exact coordinate reuse'},
    {id: 'algorithmStatus', label: 'GPU algorithm status'},
    {id: 'clusterSummary', label: 'Cluster sizes', emphasis: 'tile'},
    {id: 'reachabilityChart', label: 'OPTICS reachability', kind: 'chart'},
    {id: 'coreDistanceChart', label: 'HDBSCAN core distances', kind: 'chart'},
    {id: 'clusterSizes', label: 'GPU label counts', kind: 'chart'}
  ],
  pipeline: [
    {
      id: 'project',
      label: 'Local frame',
      detail: 'Project the selected source coordinates to metres around Northerly Island.'
    },
    {
      id: 'order',
      label: 'OPTICS order',
      detail: 'Visit points by core and reachability distance; retain the actual GPU ordering.'
    },
    {
      id: 'extract',
      label: 'Extract clusters',
      detail: 'Read DBSCAN or Xi labels, or compare HDBSCAN EOM labels.'
    },
    {
      id: 'summarize',
      label: 'Read back',
      detail: 'Copy bounded label, distance, ordering and status arrays for the map and charts.'
    }
  ],
  legends: (_state, data) => {
    const labels = Array.isArray(data.clusterLabels)
      ? data.clusterLabels.filter((label): label is number => typeof label === 'number')
      : [];
    return [getClusterLegend(labels, data.hasNoise === true, data.labelsComplete === true)];
  },
  snippet: getSnippet,
  about: {
    what: '`GPUOPTICS` computes core distances and a reachability ordering for the selected records. DBSCAN extraction cuts that structure at a chosen distance; Xi extraction identifies steep valleys. `GPUHierarchicalDensityClustering` builds a hierarchy and selects clusters with the fixed EOM rule. The map colours come from GPU-returned labels, and the reachability chart reorders GPU distances by the returned OPTICS visit sequence.',
    why: 'A point map can look clumped without explaining which records connect, or how a cluster boundary was chosen. Reachability exposes density structure before a single cut is applied; comparing extraction rules makes that modelling choice visible.',
    howToRead:
      'OPTICS roots have infinite reachability and appear as gaps. A DBSCAN distance cut groups connected valleys below that threshold; Xi searches for steep valleys; HDBSCAN EOM selects persistent groups without an epsilon cut. Shared-coordinate outlines identify exact coincident source positions. These are clusters in observation records, not bird abundance or habitat.'
  },
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'How do observation clusters take shape?',
      subtitle: 'Reachability and extraction in a local metre frame'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.iNaturalist, CREDITS.cityOfChicago),
    caveat:
      'The mapped records follow observer effort and access; cluster labels describe this archive frame, not animal abundance or habitat.'
  },
  annotations: [
    {
      kind: 'frame',
      id: 'northerly-island-analysis-frame',
      bounds: [...OBSERVATION_HIERARCHY_BOUNDS],
      text: 'Northerly Island analysis frame'
    }
  ],
  create: async ctx =>
    (await import('./observation-hierarchy.compute')).createObservationHierarchy(ctx),
  story: [
    {
      id: 'bounded-frame',
      title: 'Start with the observation frame',
      headline: 'Every selected bird record remains in the analysis',
      textAlternative:
        'Northerly Island frame with all 246 archived bird records; rings identify exact coordinate pairs shared by multiple records.',
      body: 'The archive contains **{{archiveRows}}** rows; this bounded frame selects **{{frameRows}}** bird records in original source order. **{{coordinateReuse}}**. The GPU receives every selected row, including repeated positions, after a local metre projection.',
      caveat:
        'The source archive applied a stated positional-accuracy filter of at most 250 metres, but the filtered accuracy for each row is not stored. Neighbourhood distances are a sensitivity demonstration at scales below that source bound.',
      optionsMode: 'fresh',
      options: DEFAULT_OPTIONS,
      controls: ['algorithm'],
      readouts: ['archiveRows', 'frameRows', 'coordinateReuse', 'algorithmStatus'],
      camera: {longitude: -87.607, latitude: 41.8595, zoom: 13.4, pitch: 0, bearing: 0}
    },
    {
      id: 'distance-cut',
      title: 'Move the DBSCAN cut',
      headline: 'One reachability ordering supports different cuts',
      textAlternative:
        'The same bird records are coloured by GPU OPTICS labels, alongside reachability plotted in visit order and a GPU cluster-size chart.',
      body: 'OPTICS keeps an ordering and a reachability value for each visit. The DBSCAN extraction labels connected valleys under the selected distance in metres; change that distance to see which valleys join or become noise. The chart preserves the GPU visit order, and infinite root distances appear as gaps.',
      caveat:
        'A cut below the source accuracy bound shows parameter sensitivity, not positional certainty.',
      optionsMode: 'fresh',
      options: {...DEFAULT_OPTIONS, extractionDistance: '75'},
      controls: ['extractionDistance'],
      readouts: ['algorithmStatus', 'clusterSummary', 'reachabilityChart', 'clusterSizes'],
      camera: {longitude: -87.607, latitude: 41.8595, zoom: 13.6, pitch: 0, bearing: 0}
    },
    {
      id: 'xi-valleys',
      title: 'Read steep valleys instead',
      headline: 'Xi extracts nested density changes from the ordering',
      textAlternative:
        'GPU OPTICS reachability ordering remains visible while Xi extraction colours the mapped observations by its extracted clusters.',
      body: 'Xi searches for steep changes in the reachability curve and returns disjoint innermost clusters. The Xi control sets the steepness threshold; Minimum cluster size sets the smallest retained group. Toggle predecessor correction to see how the boundary rule changes extracted labels.',
      caveat:
        'An extracted valley is an algorithmic density pattern, not a biological population boundary.',
      optionsMode: 'fresh',
      options: {...DEFAULT_OPTIONS, algorithm: 'optics-xi', xi: '0.1'},
      controls: ['xi', 'minimumClusterSize', 'predecessorCorrection'],
      readouts: ['algorithmStatus', 'clusterSummary', 'reachabilityChart', 'clusterSizes'],
      camera: {longitude: -87.607, latitude: 41.8595, zoom: 13.6, pitch: 0, bearing: 0}
    },
    {
      id: 'persistent-groups',
      title: 'Compare a hierarchy',
      headline: 'HDBSCAN EOM selects groups without a distance cut',
      textAlternative:
        'HDBSCAN EOM GPU cluster labels colour the same bird records, with core-distance and cluster-size charts.',
      body: 'HDBSCAN builds a hierarchy and selects groups using excess of mass (EOM). The core-distance histogram shows the GPU distances that feed the hierarchy; the size chart counts its returned labels. Increase **Minimum cluster size** to require broader groups.',
      caveat:
        'This output exposes labels and core distances; it does not expose membership probabilities or a condensed-tree view.',
      optionsMode: 'fresh',
      options: {...DEFAULT_OPTIONS, algorithm: 'hdbscan', minimumClusterSize: '16'},
      controls: ['minimumSamples', 'minimumClusterSize'],
      readouts: ['algorithmStatus', 'clusterSummary', 'coreDistanceChart', 'clusterSizes'],
      camera: {longitude: -87.607, latitude: 41.8595, zoom: 13.6, pitch: 0, bearing: 0}
    },
    {
      id: 'explore',
      title: 'Explore the density choices',
      headline: 'Cluster membership follows the selected rule and scale',
      textAlternative:
        'The GPU-derived cluster map and algorithm-specific charts update as the clustering method and its scale controls change.',
      body: 'Choose OPTICS + DBSCAN, OPTICS + Xi or HDBSCAN EOM, then vary the relevant scale and size settings. The GPU outputs drive the points, label counts and distance charts. Compare like-for-like parameter choices and keep the observation process in view.',
      caveat:
        'These 246 records describe observed reporting in one archive frame. They cannot estimate bird abundance, habitat quality or population boundaries.',
      optionsMode: 'fresh',
      options: DEFAULT_OPTIONS,
      controls: [
        'algorithm',
        'minimumSamples',
        'extractionDistance',
        'xi',
        'predecessorCorrection'
      ],
      readouts: ['algorithmStatus', 'clusterSummary', 'reachabilityChart', 'clusterSizes'],
      camera: {longitude: -87.607, latitude: 41.8595, zoom: 13.6, pitch: 0, bearing: 0}
    }
  ]
});
