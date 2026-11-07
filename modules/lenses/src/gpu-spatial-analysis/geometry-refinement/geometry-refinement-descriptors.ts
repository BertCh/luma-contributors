// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../computation-pack/types';

/** Domain-owned discovery metadata; supported contracts deliberately remain explicit. @internal */
export const GEOMETRY_REFINEMENT_OPERATORS: SpatialFamilyOperators = {
  GPUNormalizeGeometry: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {positions: 'coordinate', children: 'category', status: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'node', label: 'Postorder geometry tree node', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Normalize leaf directions and ring winding, then sort nested children in GEOS type and coordinate order while preserving explicit empty nodes and stable source IDs.'
    },
    metadata: {
      backlog: 'GE01',
      supported:
        'Planar points, lines, polygons, multis and nested collections; maximum tree nesting 64.'
    }
  },
  GPUCanonicalGeometryEquality: {
    family: 'geometry',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {output: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'pair', label: 'Normalized tree root pair', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Compare normalized node types, cardinalities, coordinates and recursively ordered children with an explicit coordinate tolerance; source IDs do not participate in geometry equality.'
    },
    metadata: {
      backlog: 'GE01',
      supported:
        'Successful normalized trees; empty and nested collection structure is significant.'
    }
  },
  GPUVectorStraightSkeleton: {
    family: 'geometry',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {positions: 'coordinate', segments: 'coordinate', status: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'polygon', label: 'Wavefront polygon with holes', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Unit-speed edge planes advance linked wavefront vertices through edge collapse, reflex split, hole merge and simultaneous terminal ridge events; event budgets and numerical failure are explicit.'
    },
    metadata: {
      backlog: 'GE08',
      supported:
        'Small planar polygons with holes, concavity and collinear corners; float32 event ordering and caller capacities.'
    }
  },
  GPUOffsetCurveComponents: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {positions: 'coordinate', components: 'category', status: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'component', label: 'Selected offset line component', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Exact-predicate noding splits raw offsets, side and distance tests remove invalid portions, and ordered adjacency compacts retained multipart lines with source edge provenance.'
    },
    metadata: {
      backlog: 'GE05',
      supported:
        'Prepared offsets aligned with source lines; explicit distance and chord-error tolerance.'
    }
  },
  GPUNormalizePolygons: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'coordinate'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'Canonical polygon feature', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Exact ring orientation, cyclic rotation and lexicographic sorting produce canonical polygon coordinates and structure.'
    },
    metadata: {
      backlog: 'GE01',
      supported: 'finite polygon and multipolygon layouts, empty parts, original feature order'
    }
  },
  GPUCanonicalPolygonEquality: {
    family: 'geometry',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {output: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'feature', label: 'Normalized feature pair', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Compare polygon/ring structure and coordinates directly; no hash collision assumption.'
    },
    metadata: {
      backlog: 'GE01',
      supported: 'paired normalized polygon layouts, explicit coordinate tolerance'
    }
  },
  GPUSubdividePolygons: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'coordinate', pieces: 'category', status: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'piece', label: 'Convex subdivision piece', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Triangulate interiors and merge convex seeds; optionally clip seeds with an axis-aligned GPU work queue to enforce vertex limits while preserving source feature and polygon IDs.'
    },
    metadata: {
      backlog: 'GE02',
      supported: 'convex or rectilinear-queue partition; holes preserved by the piece union',
      excludes: 'PostGIS rectilinear piece ordering'
    }
  },
  GPULinePrecisionReduction: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'coordinate', status: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'line', label: 'Precision-reduced line', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Round to an explicit grid, remove repeated vertices and either drop or retain collapsed two-vertex lines.'
    },
    metadata: {backlog: 'GE03', supported: 'valid-output and keep-collapsed lines'}
  },
  GPUPolygonPrecisionReduction: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {'polygons.positions': 'coordinate', lattice: 'coordinate'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'Snap-rounded reconstructed polygon', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Materialize closures, hot-pixel node boundaries and reconstruct even-odd faces; publish actual lattice and topology capacity status.'
    },
    metadata: {
      backlog: 'GE03',
      supported: 'arbitrary positive float32 grid steps with observed extent-adjusted lattice'
    }
  },
  GPUSegmentSnap: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'coordinate', sourceEdges: 'category', counts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'line', label: 'Reference-snapped line', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Snap endpoint vertices, insert unique nearby reference vertices in projected segment order and report required capacity.'
    },
    metadata: {
      backlog: 'GE04',
      excludes: 'GEOS safety heuristics and a topology-validity guarantee'
    }
  },
  GPULineLoopRemoval: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'coordinate', sourceEdges: 'category', status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'line', label: 'Loop-erased curve', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Node intersections with exact predicates then chronologically erase revisited path suffixes while preserving edge provenance.'
    },
    metadata: {
      backlog: 'GE05',
      supported: 'loop-removal stage for prepared open curves',
      excludes: 'complete GEOS offset component selection'
    }
  },
  GPURatioConcaveHull: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {
      triangles: 'category',
      triangleValidity: 'category',
      boundaryEdges: 'category',
      edgeValidity: 'category'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'triangle', label: 'Retained hull triangle', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Erode exposed Delaunay edges using min + ratio * (max - min), preserving boundary connectivity; optional interior seeds form holes.'
    },
    metadata: {
      backlog: 'GE06',
      supported: 'ratio in [0,1], optional holes, bounded triangle removal',
      excludes: 'point and line geometry outputs for degenerate sets'
    }
  },
  GPUMaximumInscribedCircle: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {circles: 'measure', witnesses: 'coordinate', status: 'category'},
    stages: ['operator'],
    datumKinds: [
      {kind: 'feature', label: 'Inscribed circle and boundary witness', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Best-first quadtree branch-and-bound retains every unresolved signed-distance Lipschitz bound and distinguishes tolerance success from exhausted capacity or iterations.'
    },
    metadata: {
      backlog: 'GE07',
      error: 'geometric search bound; f32 arithmetic is not interval certified'
    }
  },
  GPUConvexStraightSkeleton: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      segments: 'coordinate',
      sourceEdges: 'category',
      offsetPositions: 'coordinate',
      status: 'category'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'polygon', label: 'Convex wavefront event sequence', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Unit-speed parallel edges generate bisector arcs; simultaneous collapses include terminal ridges and a requested inward offset.'
    },
    metadata: {
      backlog: 'GE08',
      supported: 'strictly convex simple polygons with at most 64 vertices',
      excludes: 'holes, nonconvex split events, collinear corners'
    }
  }
};
