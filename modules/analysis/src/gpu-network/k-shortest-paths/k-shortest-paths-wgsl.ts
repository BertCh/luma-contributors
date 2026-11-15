// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Packed Yen records live in storage, shared by the bounded merge and parallel spur searches. @internal */
export function getKShortestPathsWGSL(shape: {
  nodeCount: number;
  edgeCount: number;
  maximumPathLength: number;
  routeCount: number;
  candidateCapacity: number;
  spurCount: number;
  queryWords: number;
  recordWords: number;
}): string {
  return (
    IS_FINITE_VALUE_WGSL +
    `
const NONE: u32 = 0xffffffffu;
const NODE_COUNT: u32 = ${shape.nodeCount}u;
const EDGE_COUNT: u32 = ${shape.edgeCount}u;
const PATH_LENGTH: u32 = ${shape.maximumPathLength}u;
const ROUTE_COUNT: u32 = ${shape.routeCount}u;
const CANDIDATE_CAPACITY: u32 = ${shape.candidateCapacity}u;
const SPUR_COUNT: u32 = ${shape.spurCount}u;
const QUERY_WORDS: u32 = ${shape.queryWords}u;
const RECORD_WORDS: u32 = ${shape.recordWords}u;
fn accepted(base: u32, route: u32) -> u32 { return base + 5u + route * RECORD_WORDS; }
fn pending(base: u32, slot: u32) -> u32 { return accepted(base, ROUTE_COUNT + slot); }
fn proposed(base: u32, spur: u32) -> u32 { return accepted(base, ROUTE_COUNT + CANDIDATE_CAPACITY + spur); }
fn samePath(first: u32, second: u32) -> bool {
  let length = state[first + 1u];
  if (length != state[second + 1u]) { return false; }
  for (var slot = 0u; slot + 1u < length; slot++) {
    if (state[first + 2u + PATH_LENGTH + slot] != state[second + 2u + PATH_LENGTH + slot]) { return false; }
  }
  return true;
}
fn copyPath(source: u32, destination: u32) {
  let length = state[source + 1u]; state[destination] = state[source]; state[destination + 1u] = length;
  for (var slot = 0u; slot < length; slot++) {
    state[destination + 2u + slot] = state[source + 2u + slot];
    if (slot + 1u < length) { state[destination + 2u + PATH_LENGTH + slot] = state[source + 2u + PATH_LENGTH + slot]; }
  }
}
fn precedes(first: u32, second: u32) -> bool {
  let firstCost = bitcast<f32>(state[first]); let secondCost = bitcast<f32>(state[second]);
  if (firstCost != secondCost) { return firstCost < secondCost; }
  let firstLength = state[first + 1u]; let secondLength = state[second + 1u];
  for (var slot = 0u; slot + 1u < min(firstLength, secondLength); slot++) {
    let firstEdge = state[first + 2u + PATH_LENGTH + slot]; let secondEdge = state[second + 2u + PATH_LENGTH + slot];
    if (firstEdge != secondEdge) { return firstEdge < secondEdge; }
  }
  return firstLength < secondLength;
}`
  );
}
