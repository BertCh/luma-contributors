// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type RouteWindowingOptions = {
  view: 'bounds' | 'subdivision' | 'inside' | 'outside';
  focus: 'route' | 'loop';
  method: 'convex' | 'rectilinear';
};
