// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type PointCloudChangeOptions = {
  view: 'epochs' | 'difference' | 'detection';
  alignment: 'surveyed' | 'local-icp';
  metric: 'point-to-point' | 'point-to-plane';
  radius: number;
  halfDepth: number;
  registrationError: number;
  uncertaintyCombination: 'linear' | 'quadrature';
  robust: 'none' | 'huber' | 'tukey';
  maximumDistance: number;
  pointSize: number;
};
