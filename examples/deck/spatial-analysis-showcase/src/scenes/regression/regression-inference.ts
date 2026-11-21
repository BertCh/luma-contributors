// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Applied producer metadata, refreshed with every model result. */
export function getOLSInferenceLabel(model: {
  regularization?: {kind: string; lambda: number};
  inferenceMethod?: string;
}): string {
  if (model.inferenceMethod === 'ridge-approximate')
    return `ridge λ ${model.regularization?.lambda ?? 'unavailable'}; approximate inverse-penalized-normal-matrix inference`;
  return model.inferenceMethod ?? 'inference convention unavailable';
}
