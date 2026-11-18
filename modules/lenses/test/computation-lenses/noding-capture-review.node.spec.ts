// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {expandLensRecordEnvelope} from '../../src/computation-lenses/record-encoding';

const SCENARIOS = [
  'buffer-surface',
  'polygon-overlay',
  'spherical-polygon-overlay',
  'polygon-precision-reduction',
  'building-adjacency-recipe',
  'building-tessellation-recipe'
];

it.each(
  SCENARIOS
)('R6 refreshed gpu-spatial-analysis.%s records both noding candidate-loss inputs', scenario => {
  const envelope = expandLensRecordEnvelope(
    JSON.parse(
      readFileSync(
        new URL(
          `../../../../docs/lens-records/gpu-spatial-analysis.${scenario}.json`,
          import.meta.url
        ),
        'utf8'
      )
    )
  );
  const record = envelope['record'] as {
    graphs: {
      nodes: {
        id: string;
        source: string;
        bindings: {name: string; access: string; length: number}[];
      }[];
    }[];
    refs: {sources: {id: string; text: string}[]};
  };
  const publications = record.graphs
    .flatMap(graph => graph.nodes)
    .filter(node => node.id.includes('noding') && node.id.endsWith('-status-candidates'));
  expect(publications.length).toBeGreaterThan(0);
  for (const publication of publications) {
    expect(publication.bindings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({name: 'crossingCandidateOverflow', access: 'read', length: 1}),
        expect.objectContaining({name: 'incidenceCandidateOverflow', access: 'read', length: 1})
      ])
    );
    const source = record.refs.sources.find(source => source.id === publication.source);
    expect(source).toBeDefined();
    expect(source!.text).toContain(
      'crossingCandidateOverflow[crossingCandidateOverflowOffset] != 0u'
    );
    expect(source!.text).toContain(
      'incidenceCandidateOverflow[incidenceCandidateOverflowOffset] != 0u'
    );
  }
});
