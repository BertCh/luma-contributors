// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// normalize(1,1+u*1e-6,1+v*1e-6), (u,v)=(-38,-3),(-38,33),(17,-6), then pack f32.
// A 70-digit Decimal projected-area oracle on these exact packed values gives
// -3.8270137717748720797311539363163239e-10: clockwise. See review-orientation.py in scratch.
export const OFF_AXIS_CLOCKWISE_TRIANGLE = [
  [0.5773581862449646, 0.5773361921310425, 0.5773563981056213],
  [0.5773512125015259, 0.5773292779922485, 0.577370285987854],
  [0.5773481726646423, 0.5773579478263855, 0.5773447155952454]
];
// normalize(1,1,1) projects to (0,0) in the triangle's local chart and lies strictly inside:
// the left edge is u=-38, the other edges cross u=0 at v=-5.07 and +6.05 (microunits).
export const OFF_AXIS_INTERIOR_QUERY = [Math.sqrt(1 / 3), Math.sqrt(1 / 3), Math.sqrt(1 / 3)];
// (u,v)=(-10,0) lies between the two sloping edges (v=-4.53 and +13.15 microunits).
const shiftedQuery = [1, 1 - 10e-6, 1];
export const OFF_AXIS_SHIFTED_INTERIOR_QUERY = shiftedQuery.map(
  value => value / Math.hypot(...shiftedQuery)
);
