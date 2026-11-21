// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Live masks and permitted hash winners for the producer-to-smoother device regression. */
export function getFlowLossEvidenceFrame(loss: boolean) {
  return {
    mask: loss ? [1, 1, 0] : [1, 0, 0],
    retainedKeys: loss ? [1, 2] : [1]
  };
}
