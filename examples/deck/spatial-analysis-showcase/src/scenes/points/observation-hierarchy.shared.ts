// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type ObservationHierarchyOptions = {
  algorithm: 'optics-dbscan' | 'optics-xi' | 'hdbscan';
  minimumSamples: string;
  minimumClusterSize: string;
  extractionDistance: string;
  xi: string;
  predecessorCorrection: boolean;
};

export const CLUSTER_COLORS: readonly (readonly [number, number, number, number])[] = [
  [48, 115, 133, 255],
  [220, 124, 69, 255],
  [104, 139, 94, 255],
  [174, 119, 151, 255],
  [207, 164, 73, 255],
  [91, 128, 177, 255],
  [157, 112, 74, 255],
  [119, 137, 140, 255]
];

export const NOISE_COLOR: readonly [number, number, number, number] = [125, 132, 133, 230];
export const UNRESOLVED_COLOR: readonly [number, number, number, number] = [164, 156, 141, 220];
