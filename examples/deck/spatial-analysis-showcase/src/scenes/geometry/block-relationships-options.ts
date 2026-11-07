// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type BlockRelationshipsOptions = {
  view: 'source' | 'bounds' | 'adjacency' | 'tessellation' | 'visibility';
  queryHalfExtent: number;
  boundsRelation: 'intersects' | 'within' | 'contains';
  inset: '0' | '0.4';
  sampleSpacing: string;
  sampleCapacity: string;
  visibilityDistance: string;
  visibilityEdgeCapacity: string;
  visibilityMetric:
    | 'degree'
    | 'reachability'
    | 'mean-hop-distance'
    | 'hop-closeness'
    | 'relative-asymmetry'
    | 'real-relative-asymmetry'
    | 'hillier-integration'
    | 'choice'
    | 'normalized-choice'
    | 'pyramid'
    | 'teklenburg'
    | 'penn'
    | 'maximum-depth';
};
