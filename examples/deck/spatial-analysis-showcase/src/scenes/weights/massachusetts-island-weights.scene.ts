// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MassachusettsIslandWeightsOptions} from './massachusetts-island-weights.compute';

export default defineScene<MassachusettsIslandWeightsOptions>({
  id: 'massachusetts-island-weights',
  title: 'Should Nantucket connect across the water?',
  chapter: 'weights',
  order: 95,
  summary:
    'Build strict queen adjacency on the 14 original Massachusetts county polygons, compare a planar polygon-distance tolerance, and add an explicitly authored nearest-centroid island bridge.',
  contributors: [
    'GPUContiguityWeights',
    'GPUFuzzyContiguity',
    'GPUIslandAttachment',
    'GPUSpatialWeightsValidation'
  ],
  datasets: [
    {
      id: 'us-counties',
      role: 'original Massachusetts county polygons, FIPS identities and source area-weighted centroid representatives'
    }
  ],
  initialView: {longitude: -70.95, latitude: 41.98, zoom: 6.6},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Massachusetts island weights',
      subtitle: 'Original county polygons · spherical CONUS Albers metre frame',
      sample: '14 original Massachusetts county features',
      chips: ['Topology scenario']
    },
    scaleBar: {units: 'metric'},
    credit: 'US Census county geometry · source FIPS retained',
    caveat:
      'A fuzzy edge is a distance rule, not a repair to county boundaries. A bridge is a separate graph intervention; neither changes source polygons or county identity.'
  },
  options: [
    {
      kind: 'select',
      id: 'scenario',
      label: 'Weights scenario',
      group: 'Construction',
      apply: 'compile',
      default: 'exact',
      display: 'segmented',
      help: 'Rebuilds the same 14 source features using strict queen, fuzzy polygon separation, or a nearest-centroid island attachment.',
      options: [
        {value: 'exact', label: 'Strict Queen'},
        {value: 'fuzzy', label: 'Fuzzy distance'},
        {value: 'bridge', label: 'Centroid bridge'}
      ]
    },
    {
      kind: 'select',
      id: 'toleranceKm',
      label: 'Polygon separation tolerance',
      group: 'Fuzzy contiguity',
      apply: 'compile',
      default: '12',
      display: 'segmented',
      help: 'A planar minimum-separation rule in the showcase spherical CONUS Albers metre frame. The implementation treats this as the diameter of equal round buffers (radius tolerance/2); zero is strict polygon contact, and no polygon is snapped or edited.',
      options: ['0', '8', '12', '16', '20'].map(value => ({value, label: `${value} km`}))
    },
    {
      kind: 'select',
      id: 'bridgeMode',
      label: 'Bridge direction',
      group: 'Centroid bridge',
      apply: 'compile',
      default: 'reciprocal',
      help: 'The nearest target is selected from the original strict-Queen graph. Reciprocal adds the reverse edge; outgoing-only leaves the attachment directed.',
      options: [
        {value: 'reciprocal', label: 'Reciprocal'},
        {value: 'outgoing', label: 'Island → neighbor'}
      ]
    }
  ],
  story: [
    {
      id: 'source-coast',
      title: 'Start with original county geography',
      headline: 'The map keeps 14 Massachusetts source identities intact',
      textAlternative:
        'Original Massachusetts county polygons are shown with Nantucket highlighted; no fuzzy or bridge intervention is overlaid.',
      body: 'The input is the bundled US Census county layer. This scene selects all 14 source rows whose state FIPS is 25 and retains their names and county FIPS. Polygon rings use the showcase spherical Albers CONUS frame on the mean Earth radius, in metres; distances are planar within that projected frame, not ellipsoidal EPSG:5070 distances. A preflight rejects non-finite or self-intersecting rings and never simplifies them. The GPU bridge uses the dataset’s source area-weighted centroid representative, reprojected to the same frame; a representative need not lie inside its county.',
      options: {scenario: 'exact'},
      controls: [],
      readouts: ['countyCount', 'vertexCount', 'featureStatus']
    },
    {
      id: 'strict-queen',
      title: 'Strict Queen asks whether boundaries touch',
      headline: 'A source island can remain a row with no neighbors',
      textAlternative:
        'The strict Queen baseline colors the Nantucket county polygon and reports its original adjacency row and GPU validation status.',
      body: '**`GPUContiguityWeights`** builds queen adjacency from shared source vertices with zero snap tolerance. This is an exact topological criterion on the projected float coordinates: proximity across water does not create a link. **`GPUSpatialWeightsValidation`** checks the produced CSR offsets, neighbor IDs, order, self-links and weights before it is used.',
      controls: ['scenario'],
      readouts: ['strictIslands', 'strictEdges', 'capacity', 'validation']
    },
    {
      id: 'fuzzy-separation',
      title: 'Fuzzy contiguity uses a distance rule',
      headline: 'A tolerance can connect nearby polygons without moving them',
      textAlternative:
        'Added fuzzy-contiguity links are drawn between unchanged county centroids and the selected minimum-separation tolerance is reported.',
      body: '**`GPUFuzzyContiguity`** compares filled polygon separation, including multipart outlines and holes, in the shared spherical Albers metre frame. Choose an authored tolerance from 0 to 20 km; an edge is marked as added only when positive separation tolerance was needed. This is a graph-construction scenario, not a claim that boundaries were inaccurate or repaired.',
      options: {scenario: 'fuzzy', toleranceKm: '12'},
      controls: ['scenario', 'toleranceKm'],
      readouts: ['scenarioEdges', 'finalEdges', 'nantucketLinks', 'capacity']
    },
    {
      id: 'nearest-bridge',
      title: 'A bridge is a separate modeling choice',
      headline: 'Attach an island row to its nearest source centroid',
      textAlternative:
        'The authored bridge link identifies Nantucket’s nearest county centroid, its projected distance, the chosen direction and graph validation result.',
      body: '**`GPUIslandAttachment`** first finds rows with no original strict-Queen neighbors, then chooses the nearest finite representative among all rows using the source area-weighted centroids projected into the same spherical Albers frame. Reciprocal mode adds both directions; outgoing-only adds island → target. The red line is a constructed link over water. It does not represent a ferry, road, shared boundary or observed interaction.',
      options: {scenario: 'bridge', bridgeMode: 'reciprocal'},
      controls: ['scenario', 'bridgeMode'],
      readouts: ['attachment', 'attachmentDistance', 'scenarioEdges', 'finalEdges']
    },
    {
      id: 'validate-choice',
      title: 'Check the altered relation before using it',
      headline: 'Validation tests CSR structure, not whether a bridge is substantively right',
      textAlternative:
        'The final view compares strict and scenario validation counts, output capacity, remaining islands and the exact constructed Nantucket edge.',
      body: 'The before/after validators report rows with CSR invariant violations. Capacity readouts include the exact required directed slot count and any overflow flags. Compare strict Queen, fuzzy separation and nearest-centroid attachment as explicit alternatives while the source polygons stay fixed. A valid matrix is structurally usable; its substantive suitability depends on the spatial process being modeled.',
      options: {scenario: 'bridge', bridgeMode: 'reciprocal'},
      controls: ['scenario', 'toleranceKm', 'bridgeMode'],
      readouts: ['validation', 'islandCount', 'strictEdges', 'finalEdges']
    }
  ],
  readouts: [
    {id: 'countyCount', label: 'Source county features', format: 'integer'},
    {id: 'vertexCount', label: 'Projected source vertices', format: 'integer'},
    {id: 'featureStatus', label: 'Polygon input status'},
    {id: 'strictIslands', label: 'Strict Queen islands', format: 'integer'},
    {id: 'islandCount', label: 'Scenario islands', format: 'integer'},
    {id: 'strictEdges', label: 'Strict Queen directed slots', format: 'integer'},
    {id: 'finalEdges', label: 'Scenario directed slots', format: 'integer'},
    {id: 'scenarioEdges', label: 'Added directed slots', format: 'integer'},
    {id: 'nantucketLinks', label: 'Nantucket added neighbors'},
    {id: 'attachment', label: 'Nearest centroid target'},
    {id: 'attachmentDistance', label: 'Centroid distance'},
    {id: 'validation', label: 'Strict / scenario invalid rows'},
    {id: 'capacity', label: 'CSR capacity and overflow'}
  ],
  legends: state => [
    {
      kind: 'categories',
      title: 'Map elements',
      entries: [
        {color: [221, 128, 60, 230], label: 'Nantucket · source county polygon'},
        {color: [42, 91, 123, 190], label: 'Other Massachusetts source county'},
        {
          color: state.scenario === 'bridge' ? [191, 84, 40, 230] : [0, 137, 123, 230],
          label:
            state.scenario === 'bridge'
              ? 'Authored nearest-centroid bridge'
              : 'Fuzzy-distance added edge'
        }
      ],
      note: 'Links change the weights graph; source county boundaries remain unchanged.'
    }
  ],
  about: {
    what: 'A GPU strict-Queen baseline is compared with filled-polygon distance contiguity and explicit nearest-centroid island attachment on original Massachusetts county polygons.',
    why: 'A geometric tolerance and a constructed bridge answer different modeling questions. Both can change neighbors without editing source geometry.',
    howToRead:
      'The map retains source county identities. Added edges come directly from GPU slot flags. Strict and scenario validation counts report structural CSR errors; centroid distance is computed in the same spherical Albers metre frame as the GPU inputs.'
  },
  create: async context =>
    (await import('./massachusetts-island-weights.compute')).createMassachusettsIslandWeights(
      context
    )
});
