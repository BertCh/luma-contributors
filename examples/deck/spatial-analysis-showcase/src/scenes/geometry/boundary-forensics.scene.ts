// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';

type BoundaryForensicsOptions = {
  view: 'source' | 'encoding' | 'normalized' | 'precision';
  gridSize: string;
};

export default defineScene<BoundaryForensicsOptions>({
  id: 'boundary-forensics',
  title: 'Footprint boundary forensics',
  chapter: 'geometry',
  order: 10,
  summary:
    'Use a fixed set of real Chicago building footprints to distinguish ring encoding, canonical structural equality and topology-aware precision reduction in a local projected metre frame.',
  contributors: [
    'GPUNormalizePolygons',
    'GPUCanonicalPolygonEquality',
    'GPUPolygonPrecisionReduction'
  ],
  datasets: [
    {
      id: 'chicago-buildings',
      role: 'City of Chicago building footprint boundaries with source building IDs'
    }
  ],
  initialView: {longitude: -87.6365, latitude: 41.882, zoom: 16.2},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'When is a footprint the same geometry?',
      subtitle: 'Chicago Loop · selected source records'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cityOfChicago),
    caveat:
      'The grid step is an analysis operation, not a statement of source accuracy. These source footprints include legacy digitization dates; canonical equality tests structural coordinate serialization, not general topological equality.'
  },
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Geometry view',
      group: 'Evidence',
      apply: 'param',
      default: 'source',
      options: [
        {value: 'source', label: 'Source footprints'},
        {value: 'encoding', label: 'Alternate ring encoding'},
        {value: 'normalized', label: 'GPU normalized output'},
        {value: 'precision', label: 'GPU precision output'}
      ],
      help: 'Switches between the actual source outlines, the same rings with a different starting vertex and winding, GPU normalized output, and GPU precision-reduced output.'
    },
    {
      kind: 'select',
      id: 'gridSize',
      label: 'Precision grid',
      group: 'Precision reduction',
      apply: 'compile',
      default: '1',
      options: [0.1, 1, 5, 20, 50].map(value => ({value: String(value), label: `${value} m`})),
      help: 'Runs GPUPolygonPrecisionReduction on a true local projected metre grid. Coarse grids can move, reshape or collapse small footprints.'
    }
  ],
  story: [
    {
      id: 'source-records',
      title: 'Start with source boundaries',
      headline: 'Twelve footprints keep the ring edits inspectable',
      textAlternative:
        'A map shows a bounded set of actual Chicago building footprint outlines. The analysis uses their source building IDs and projects coordinates into a local metre frame.',
      body: 'The scene selects twelve source `bldg_id` records from the City of Chicago footprint extract. It projects their polygon rings into a local planar metre frame near the Loop, then keeps feature, polygon and ring offsets explicit for the GPU operators.',
      evidence:
        '**{{sourceBuildings}}** source records contribute **{{sourceVertices}}** stored vertices.',
      caveat:
        'A footprint is a plan-view boundary record; this sample does not establish when or how accurately each building was surveyed.',
      options: {view: 'source'},
      controls: [],
      readouts: ['sourceBuildings', 'sourceVertices'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 16.2, transitionMs: 1200}
    },
    {
      id: 'alternate-encoding',
      title: 'Same boundary, different ring encoding',
      headline: 'Starting vertex and winding can change',
      textAlternative:
        'Teal and amber outlines overlap to show the same footprint boundaries; differently placed start markers show a cyclic ring rotation and reversal in the coordinate encoding.',
      body: 'The amber version simulates a different coordinate serialization by rotating each ring and reversing its direction while preserving the same vertices and closure. The teal and amber outlines should occupy the same boundary; their start markers expose that the stored order changed.',
      evidence: 'The GPU receives **{{sourceVertices}}** source vertices in both encodings.',
      caveat:
        'Re-encoding a ring is a representation change. This step does not claim that different vertex sets or arbitrary topologically equal polygons are structurally identical.',
      options: {view: 'encoding'},
      controls: ['view'],
      readouts: ['sourceBuildings', 'sourceVertices'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 16.2, transitionMs: 900}
    },
    {
      id: 'normalize',
      title: 'Normalize, then compare structure',
      headline: 'Canonicalization makes this encoding pair comparable',
      textAlternative:
        'Violet GPU output outlines and start markers show normalized polygons; the equality readout reports whether the original and simulated ring encodings now have matching structures.',
      body: '`GPUNormalizePolygons` orients shells clockwise and holes counterclockwise, chooses canonical cyclic starts, then orders holes and polygon parts. `GPUCanonicalPolygonEquality` compares the two normalized outputs feature by feature. Normalization preserves feature order, duplicates, closure convention and vertex counts, so a different closure choice can remain distinct.',
      evidence: '**{{canonicalEquality}}** after GPU normalization.',
      caveat:
        'Canonical normalization establishes serialization equality for compatible encodings; it does not decide general topological equality for arbitrary polygon boundaries.',
      options: {view: 'normalized'},
      controls: ['view'],
      readouts: ['canonicalEquality'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 16.2, transitionMs: 900}
    },
    {
      id: 'precision-grid',
      title: 'Precision reduction changes the geometry',
      headline: 'A true lattice can move and collapse boundaries',
      textAlternative:
        'Faint source boundaries sit beneath orange GPU precision-reduced outlines; readouts show the actual output vertex count and projected area for the selected metre grid.',
      body: '`GPUPolygonPrecisionReduction` performs topology-aware hot-pixel noding and polygon reconstruction on the chosen local metre lattice. Compare the source outline with the GPU output as **Precision grid** changes from sub-metre to tens-of-metres spacing.',
      evidence:
        '**{{precisionVertices}}** remain; GPU output area and its change from the source are **{{precisionArea}}**. {{precisionStatus}}',
      caveat:
        'The displayed step is the operation’s grid size, not a claim about positional accuracy. A collapsed output is an effect of the selected grid and bounded topology pipeline.',
      options: {view: 'precision', gridSize: '1'},
      controls: ['gridSize', 'view'],
      readouts: ['precisionVertices', 'precisionArea', 'precisionStatus', 'precisionChart'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 16.2, transitionMs: 900}
    },
    {
      id: 'explore-grid',
      title: 'Read the lattice, not just the rounded coordinates',
      headline: 'Grid spacing controls the amount of change',
      textAlternative:
        'A map compares the source outlines with the selected GPU-reduced geometry; a bar chart compares input and output vertex counts for the active grid size.',
      body: 'Compare the fine **1 m** result with **50 m**. The GPU geometry, output area, vertex count, returned lattice size and status flags update from each compiled grid variant. Small outlines may disappear when the lattice is coarser than their dimensions.',
      evidence:
        '**{{precisionVertices}}** GPU vertices; projected area comparison: **{{precisionArea}}**. {{precisionStatus}}',
      caveat:
        'Do not infer ground truth from the most detailed coordinate set. The source portal provides legacy footprint records, and this operation illustrates precision handling rather than correcting known survey errors.',
      options: {view: 'precision', gridSize: '50'},
      controls: ['gridSize', 'view'],
      readouts: ['precisionVertices', 'precisionArea', 'precisionStatus', 'precisionChart'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 16.2, transitionMs: 800}
    }
  ],
  readouts: [
    {id: 'sourceBuildings', label: 'Selected source buildings', unit: 'records'},
    {id: 'sourceVertices', label: 'Stored source vertices', unit: 'vertices'},
    {id: 'canonicalEquality', label: 'Canonical structural equality', unit: 'feature pairs'},
    {id: 'precisionVertices', label: 'GPU output vertices', unit: 'vertices'},
    {
      id: 'precisionArea',
      label: 'GPU output area vs source',
      help: 'Source planar area is calculated from the projected input rings. Output area is a CPU shoelace summary of the coordinates read back from the GPU precision output; holes are subtracted.',
      unit: 'm²'
    },
    {id: 'precisionStatus', label: 'Precision pipeline status', unit: 'status'},
    {id: 'precisionChart', label: 'Source and output vertices', kind: 'chart'},
    {id: 'sourceIds', label: 'Source bldg_id records', layout: 'block'}
  ],
  legends: state => [
    {
      id: 'boundary-forensics',
      title: 'Geometry evidence',
      kind: 'categories',
      entries:
        state.view === 'encoding'
          ? [
              {color: [27, 123, 130, 210], label: 'source encoding / start marker'},
              {color: [227, 119, 67, 210], label: 'rotated and reversed encoding / start marker'}
            ]
          : state.view === 'normalized'
            ? [{color: [178, 142, 202, 230], label: 'GPU normalized output'}]
            : state.view === 'precision'
              ? [
                  {color: [88, 112, 123, 115], label: 'source boundary'},
                  {color: [224, 129, 76, 235], label: 'GPU precision output'}
                ]
              : [{color: [27, 123, 130, 210], label: 'City of Chicago source footprint'}],
      note: 'All displayed comparison outputs are derived from the selected source records.'
    }
  ],
  snippet: state =>
    `graph.add(new GPUNormalizePolygons({positions, ringOffsets, polygonOffsets, featureOffsets, output: normalized}));\n` +
    `graph.add(new GPUCanonicalPolygonEquality({left: {...normalized, featureOffsets}, right: {...normalizedReencoded, featureOffsets}, output: equality}));\n` +
    `graph.add(new GPUPolygonPrecisionReduction({polygons, gridSize: ${state.gridSize}, lattice, intersectionCapacity, output}));`,
  about: {
    what: '`GPUNormalizePolygons` canonicalizes polygon ring orientation and cyclic starts; `GPUCanonicalPolygonEquality` tests structural equality of normalized feature encodings; `GPUPolygonPrecisionReduction` reconstructs polygon topology on an explicit planar lattice.',
    why: 'Spatial workflows need to distinguish harmless serialization changes from structural differences and from actual coordinate changes introduced by a precision model.',
    howToRead:
      'Teal and amber ring-start markers show two encodings of the same source boundary. Violet outlines are GPU-normalized geometries; orange outlines are GPU precision-reduced outputs. Source planar area is calculated from projected input rings. Output planar area uses the shoelace formula on coordinates read back from the GPU output, subtracting hole rings.'
  },
  create: async ctx => (await import('./boundary-forensics.compute')).createBoundaryForensics(ctx)
});
