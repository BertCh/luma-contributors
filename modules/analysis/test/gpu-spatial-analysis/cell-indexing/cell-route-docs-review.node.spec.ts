// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {
  scanLensDirectives,
  scanLensMarkers
} from '../../../../lenses/test/computation-docs/lens-directives';

const guidePath = 'docs/api/gpu-spatial-analysis/urban-morphology.md';
const lensPath = 'docs/api/gpu-spatial-analysis/urban-ecology-lenses.md';

it('B2 associates the reference result and graph with both public reference names', () => {
  const source = readFileSync(lensPath, 'utf8');
  const directives = scanLensDirectives(lensPath, source);
  const markers = scanLensMarkers(lensPath, source);
  for (const name of ['addBuildingTessellationRecipe', 'addBuildingTessellationReferenceRecipe']) {
    expect(markers.filter(marker => marker.name === name)).toHaveLength(1);
    for (const rung of ['result', 'graph']) {
      expect(
        directives.some(
          directive =>
            directive.sectionClasses.includes(name) &&
            directive.attributes['scenario'] ===
              'gpu-spatial-analysis.building-tessellation-recipe' &&
            directive.attributes['rung'] === rung
        ),
        `${name} ${rung}`
      ).toBe(true);
    }
  }
  const local = directives.filter(directive =>
    directive.sectionClasses.includes('addLocalBuildingTessellationRecipe')
  );
  expect(local).toHaveLength(2);
  expect(
    local.every(
      directive =>
        directive.attributes['scenario'] ===
        'gpu-spatial-analysis.local-building-tessellation-recipe'
    )
  ).toBe(true);
});

it('B3 distinguishes recipe-generated ring requirement status from caller-owned overlays', () => {
  const source = readFileSync(guidePath, 'utf8');
  const reference = source
    .split('### `addBuildingTessellationReferenceRecipe`')[1]!
    .split('### `addLocalBuildingTessellationRecipe`')[0]!;
  expect(reference).toContain('recipe-generated default');
  expect(reference).toContain('`requiredRingCount` is optional');
  expect(reference).toContain('`output.overlays[i].status.requiredRingCount`');
});
