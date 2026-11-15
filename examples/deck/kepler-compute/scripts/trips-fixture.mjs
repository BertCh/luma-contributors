// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const TRIP_FIXTURE_ORIGIN_MILLISECONDS = Date.UTC(2026, 0, 1);

export const makeTimedPath = latitude => ({type: 'LineString', coordinates: [
  [-74.03, latitude, 0, TRIP_FIXTURE_ORIGIN_MILLISECONDS],
  [-74, latitude, 0, TRIP_FIXTURE_ORIGIN_MILLISECONDS + 10000],
  [-73.97, latitude, 0, TRIP_FIXTURE_ORIGIN_MILLISECONDS + 20000]
]});

export const makeMultipartTimedPath = latitude => ({type: 'MultiLineString', coordinates: [
  makeTimedPath(latitude).coordinates,
  [[-74.03, latitude + 0.01, 100, TRIP_FIXTURE_ORIGIN_MILLISECONDS],
    [-73.97, latitude + 0.01, 200, TRIP_FIXTURE_ORIGIN_MILLISECONDS + 20000]]
]});

// One third along the second part, picked at default altitude zero in the top-down map.
export const MULTIPART_PICK_POSITION = [-74.01, 40.78];
export const MULTIPART_PICK_TIME_MILLISECONDS = TRIP_FIXTURE_ORIGIN_MILLISECONDS + 20000 / 3;
