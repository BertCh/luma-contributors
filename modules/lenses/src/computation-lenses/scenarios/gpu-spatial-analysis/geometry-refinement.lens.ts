// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

/** Repository-authored planar examples; these coordinates are illustrative, not observations. */
export const GEOMETRY_REFINEMENT_SCENARIOS = [
  {
    element: 'GPUNormalizePolygons',
    slug: 'normalize-polygons',
    title: 'Canonical clockwise rectangle rings'
  },
  {
    element: 'GPUCanonicalPolygonEquality',
    slug: 'canonical-polygon-equality',
    title: 'Equality of normalized rectangle encodings'
  },
  {
    element: 'GPUSubdividePolygons',
    slug: 'subdivide-polygons',
    title: 'Convex pieces and source provenance'
  },
  {
    element: 'GPUPolygonPrecisionReduction',
    slug: 'polygon-precision-reduction',
    title: 'Topology reconstruction on a unit lattice'
  },
  {
    element: 'GPULinePrecisionReduction',
    slug: 'line-precision-reduction',
    title: 'Preserving a collapsed line on a unit lattice'
  },
  {
    element: 'GPUSegmentSnap',
    slug: 'segment-snap',
    title: 'Ordered reference insertion along a line'
  },
  {
    element: 'GPULineLoopRemoval',
    slug: 'line-loop-removal',
    title: 'Chronological erasure of a crossing loop'
  },
  {
    element: 'GPURatioConcaveHull',
    slug: 'ratio-concave-hull',
    title: 'Border erosion of five planar points'
  },
  {
    element: 'GPUMaximumInscribedCircle',
    slug: 'maximum-inscribed-circle',
    title: 'Bounded search inside a rectangle'
  },
  {
    element: 'GPUConvexStraightSkeleton',
    slug: 'convex-straight-skeleton',
    title: 'Convex rectangle wavefront and inward offset'
  },
  {
    element: 'GPUNormalizeGeometry',
    slug: 'normalize-geometry',
    title: 'Normalize a mixed collection without losing its structure'
  },
  {
    element: 'GPUCanonicalGeometryEquality',
    slug: 'canonical-geometry-equality',
    title: 'Compare normalized mixed geometry roots'
  },
  {
    element: 'GPUVectorStraightSkeleton',
    slug: 'vector-straight-skeleton',
    title: 'Hole-aware event wavefront and inward offset'
  },
  {
    element: 'GPUOffsetCurveComponents',
    slug: 'offset-curve-components',
    title: 'Select connected offset components by side and distance'
  }
] as const;

for (const {element, slug, title} of GEOMETRY_REFINEMENT_SCENARIOS) {
  const id = `gpu-spatial-analysis.${slug}`;
  defineLensScenario({
    id,
    entry: 'gpu-spatial-analysis',
    element,
    title,
    audience: 'engineer',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    defaultRung: 'result',
    dataset: {
      id: 'geometry-refinement-illustrative/1',
      source:
        'Repository-authored rectangles, crossing lines and five-point sets in arbitrary planar units; at most eight vertices per input geometry.',
      licence: 'MIT (repository-authored illustrative inputs)'
    },
    async load() {
      const [{runS2Lens}, {buildGeometryRefinementScenario}] = await Promise.all([
        import('./s2-lens-run'),
        import('./geometry-refinement-examples')
      ]);
      return {
        run: device =>
          runS2Lens(device, {
            scenarioId: id,
            element,
            datasets: ['geometry-refinement-illustrative/1'],
            build: (graph, uploads) => buildGeometryRefinementScenario(element, graph, uploads)
          })
      };
    }
  });
}
