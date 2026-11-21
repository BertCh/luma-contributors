// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Ordered float32 components of a GWR grid extent, after the row IDs. @internal */
export const GWR_GRID_EXTENT_COMPONENTS = [
  {name: 'minimumX', operation: 'min'},
  {name: 'minimumY', operation: 'min'},
  {name: 'maximumX', operation: 'max'},
  {name: 'maximumY', operation: 'max'}
] as const;
/** Number of components in the extent layout. @internal */
export const GWR_GRID_EXTENT_COMPONENT_COUNT = GWR_GRID_EXTENT_COMPONENTS.length;
