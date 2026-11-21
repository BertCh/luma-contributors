// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
export {
  GPUAlternativeRoutes,
  getGPUAlternativeRoutesOutputLayout,
  decodeGPUAlternativeRoutes
} from './gpu-alternative-routes';
export type {GPUAlternativeRoutesProps} from './gpu-alternative-routes';
export type {
  RouteReadbackWords,
  RouteReadbackFloats
} from '../k-shortest-paths/k-shortest-paths-output';

/** Dimensions of the query-major alternative route table. */
export type GPUAlternativeRoutesOutputLayoutOptions = Parameters<
  typeof import('./gpu-alternative-routes').getGPUAlternativeRoutesOutputLayout
>[0];
/** Scalar formats and lengths, including the per-query completion column. */
export type GPUAlternativeRoutesOutputLayout = ReturnType<
  typeof import('./gpu-alternative-routes').getGPUAlternativeRoutesOutputLayout
>;
/** Typed columns or byte windows returned by buffer readback. */
export type GPUAlternativeRoutesReadback = Parameters<
  typeof import('./gpu-alternative-routes').decodeGPUAlternativeRoutes
>[0];
/** Retained routes with one completion certificate per query. */
export type GPUAlternativeRoutesDecodedResult = ReturnType<
  typeof import('./gpu-alternative-routes').decodeGPUAlternativeRoutes
>;

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeAlternativeRoutesOnCPU,
  CPUAlternativeRoutesOperation
} from './cpu-alternative-routes-operation';
export type {
  CPUAlternativeRoutesInput,
  CPUAlternativeRoutesOptions,
  CPUAlternativeRoutesResult
} from './cpu-alternative-routes-operation';
export {GPUAlternativeRoutesOperation} from './gpu-alternative-routes-operation';
export type {AlternativeRoutesOperationProps} from './gpu-alternative-routes-operation';
export {
  GPU_ALTERNATIVE_ROUTES_MAXIMUM_NODE_COUNT,
  GPU_ALTERNATIVE_ROUTES_MAXIMUM_ROUTE_COUNT,
  GPU_ALTERNATIVE_ROUTES_MAXIMUM_FRONTIER_SIZE,
  GPU_ALTERNATIVE_ROUTES_MAXIMUM_EXPANSIONS
} from './gpu-alternative-routes';
