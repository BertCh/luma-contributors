// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';

export const CURATED_LANDSCAPE_METRICS_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-landscape-metrics',
  questions: ['shape-of-the-surface'],
  action: 'Measure landscape fragmentation',
  prominence: 2,
  inputs: [{name: 'landcover', kind: 'surface', fields: []}],
  parameters: [
    'connectivity',
    'edgeDepth',
    'searchRadius',
    'distanceModel',
    'excludeClass',
    'excludedClass'
  ].map(name => ({
    name,
    level: ['distanceModel', 'excludedClass'].includes(name) ? 'advanced' : 'basic',
    apply: 'rebuild',
    defaultPolicy: {from: 'constant'},
    ...(name === 'excludedClass' ? {activeWhen: {parameter: 'excludeClass', equals: [true]}} : {})
  })),
  requirements: [
    {
      id: 'planar-coordinates',
      severity: 'block',
      input: 'landcover',
      remedy:
        'Reproject the raster to a planar grid with consistent linear units before running landscape metrics'
    },
    {
      id: 'declared-coordinates',
      severity: 'block',
      input: 'landcover',
      remedy: 'Confirm the planar coordinate system and linear units of the raster grid'
    }
  ],
  outputs: ['cells', 'patches', 'classes', 'adjacency', 'distances', 'landscape'].map(tableId => ({
    tableId,
    kind: 'attributes',
    interpretation: {summary: 'row-count', cautions: []}
  })),
  notes: [
    'Select a decoded categorical raster band, not a numeric table converted to points. Valid integer class codes, including zero, retain their original values. Masked, nodata and excluded classes are barriers; all excluded cells rejects the run.',
    'Requires an entire planar, axis-aligned area grid with confirmed linear units. Areas and mesh use squared CRS units; perimeter, edge depth, search radius and nearest distances use CRS units. Positive cell sizes, cell area, domain extents, edge depth and search radius must fit normal Float32; zero depth and radius are supported. The default 1000-unit search radius is a stated constant, not a data-derived ecological scale.',
    'Four-connectivity is the default; eight-connectivity joins diagonal cells. Perimeters and adjacency always use four orthogonal sides. Outer-domain and background contacts count in perimeter. Contagion uses area-weighted conditional adjacency.',
    'Core area uses physical center-to-exterior depth strictly greater than edgeDepth, with the outer boundary closed. Boundary distances measure rectangle separation; cell-center mode is available for raster ENN conventions. The search radius restricts the proximity sum, not the nearest same-class distance.',
    'Null nearest distance means no other patch of the same class. Null proximity can mean zero-distance touching patches; null cohesion, aggregation or contagion can mean an undefined small-domain statistic. Diagnostics retain these cases. Map publication is withheld: the planar grid has no explicit transform to longitude/latitude. All outputs are tables; cell indices and patch_root_cell retain the source-grid identity.',
    'GPU-only bounded execution: at most 4096 cells, 32 classes, 256 patches and 1048576 patch-cell pairs. Class labels must converge before any metric is computed. Patch identifiers are scoped to the source id, selected band and grid; connectivity or grid changes can change them.'
  ]
};
