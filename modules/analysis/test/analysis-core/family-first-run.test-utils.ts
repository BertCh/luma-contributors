// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Each module contains one literal lifecycle/example pair, including its invocation. Separate
// modules preserve the printed imports without colliding bindings or eager device creation.
export const loadTerrainExample = () => import('./family-first-run-terrain.test-utils');
export const loadDistanceExample = () => import('./family-first-run-distance.test-utils');
export const loadTimeExample = () => import('./family-first-run-time.test-utils');
export const loadPointCloudExample = () => import('./family-first-run-pointcloud.test-utils');
export const loadChangedTimeExample = () => import('./family-first-run-time-control.test-utils');
