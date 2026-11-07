// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {SpatialFamilyOperators} from '../computation-pack/types';

/** Domain-owned contracts for central computation-pack discovery. @internal */
export const GLOBAL_COORDINATE_OPERATORS: SpatialFamilyOperators = {
  GPUS2CellHierarchy: {
    family: 'cells',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {cells: 'category', status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'cell', label: 'S2 cell', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Integer Hilbert identities preserve parent/child/leaf ranges and cube-face edge neighbors at levels 0–30; invalid or unavailable levels have explicit status.'
    },
    metadata: {
      backlog: 'CR05',
      precision: 'exact integer keys',
      supported: 'parent, immediate children, leaf range, four edge neighbors'
    }
  },
  GPUS2CellMeasures: {
    family: 'cells',
    descriptor: {scope: 'local', reduction: 'sum'},
    variables: {measures: 'measure', status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'cell', label: 'S2 cell', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Analytic quadratic-UV increments avoid cancellation when measuring area and perimeter at levels 0–30.'
    },
    metadata: {backlog: 'CR05', units: 'steradians and radians', maximumLevel: 30}
  },
  GPUS2CapCovering: {
    family: 'cells',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {cells: 'category', counts: 'count', status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'cap', label: 'spherical cap', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Uniform S2 enumeration uses closest spherical boundary distance for exterior covers and antipodal distance for interior covers; counts expose incomplete capacity.'
    },
    metadata: {
      backlog: 'CR05',
      maximumLevel: 8,
      supported: 'uniform exterior/interior cap covers with angular safety margin'
    }
  },
  GPUSphericalConvexContains: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {contains: 'category', status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'point', label: 'unit direction', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Oriented great-circle halfspaces classify validated convex hemisphere polygons, with explicit open/closed tolerance-band boundaries and complementary regions.'
    },
    metadata: {
      backlog: 'CR04',
      maximumVertices: 32,
      supported: 'convex hemisphere polygons or their complements'
    }
  },
  GPUSphericalConvexIntersection: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'measure', counts: 'count', measures: 'measure', status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'pair', label: 'convex polygon pair', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Contained vertices and minor-arc intersections assemble convex spherical intersections; optional solid-angle measures supply geometric overlap fractions for prepared cross-family cell pairs.'
    },
    metadata: {
      backlog: 'CR04,CR05',
      maximumInputVertices: 32,
      maximumOutputVertices: 64,
      supported: 'closed convex intersection; source and target overlap fractions'
    }
  },
  GPUS2PolygonCovering: {
    family: 'cells',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'polygon', label: 'spherical polygon', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Adaptive mixed-level interior/exterior covering of prepared hemisphere shell/hole polygons and complements; capacity and visited-cell budgets report incomplete results.'
    },
    metadata: {backlog: 'CR05', maximumLevel: 15, maximumVisitedCells: 1048576}
  },
  GPUSphericalPolygonContains: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'polygon', label: 'spherical polygon', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Tangent winding classifies prepared nonconvex hemisphere rings and holes, with explicit complement and open/closed boundary semantics.'
    },
    metadata: {backlog: 'CR04', precision: 'f32 angular tolerance'}
  },
  GPUSphericalPolygonOverlay: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'polygon', label: 'spherical polygon', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Gnomonic projection, snap-rounded planar arrangement and sphere lifting support shell/hole Boolean operations with explicit symbolic complements.'
    },
    metadata: {
      backlog: 'CR04',
      precision: 'caller-supplied snap lattice',
      minimumChartCosine: 0.001
    }
  },
  GPUSphericalArcOverlay: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'polygon', label: 'spherical polygon', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Intrinsic minor-arc noding, reference-anchored parity and oriented ring assembly support globally distributed Boolean boundaries; uncertain separation or junctions fail explicitly.'
    },
    metadata: {backlog: 'CR04', maximumInputVertices: 128, maximumOutputVertices: 1024}
  }
};
