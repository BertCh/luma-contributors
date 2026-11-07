// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type RouteEditingOptions = {
  edit:
    | 'reverse'
    | 'cleanup'
    | 'precision'
    | 'vertex-snap'
    | 'segment-snap'
    | 'loop-revisit'
    | 'translation';
  tolerance: number;
  gridSize: number;
  focus: 'route' | 'downtown' | 'revisit';
};
