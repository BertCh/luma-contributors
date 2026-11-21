// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const PANEL_ENTITY_COUNT = 4096;
export const PANEL_PERIOD_COUNT = 256;
export const PANEL_ROW_COUNT = PANEL_ENTITY_COUNT * PANEL_PERIOD_COUNT;
// Independent float64 pooled OLS on uploaded integer x,y gives beta=(1,2), residual=error.
// Let w=float64(float32(.2)). Per period u'u=4096 and u'Wu=2452.8*(w/.2).
// tr(W'W+WW)=4096*10*w*w, since all five directed edges have reciprocals.
// sigma2=1048576/(1048576-2). LM=(256*2452.8*(w/.2))^2 /
// (sigma2^2 * 256*4096*10*w*w) = 940032.0140521869.
export const PANEL_LM_ERROR = 940032.0140521869;

export function createLargePanelScene() {
  const predictors = new Float32Array(PANEL_ROW_COUNT);
  const response = new Float32Array(PANEL_ROW_COUNT);
  const errors = new Float32Array(PANEL_ROW_COUNT);
  const neighbors = new Uint32Array(PANEL_ENTITY_COUNT * 5);
  const offsets = Uint32Array.from({length: PANEL_ENTITY_COUNT + 1}, (_, entity) => entity * 5);
  const weights = new Float32Array(neighbors.length).fill(0.2);
  for (let entity = 0; entity < PANEL_ENTITY_COUNT; entity++) {
    neighbors.set(
      [
        (entity + PANEL_ENTITY_COUNT - 2) % PANEL_ENTITY_COUNT,
        (entity + PANEL_ENTITY_COUNT - 1) % PANEL_ENTITY_COUNT,
        (entity + 1) % PANEL_ENTITY_COUNT,
        (entity + 2) % PANEL_ENTITY_COUNT,
        (entity + PANEL_ENTITY_COUNT / 2) % PANEL_ENTITY_COUNT
      ].sort((first, second) => first - second),
      entity * 5
    );
    const predictor = (entity % 4 < 2 ? 1 : -1) + (entity % 8 < 4 ? 1 : -1);
    const error = entity < PANEL_ENTITY_COUNT / 2 ? 1 : -1;
    for (let period = 0; period < PANEL_PERIOD_COUNT; period++) {
      const row = period * PANEL_ENTITY_COUNT + entity;
      const sign = period % 2 === 0 ? 1 : -1;
      predictors[row] = sign * predictor;
      errors[row] = sign * error;
      response[row] = 1 + 2 * predictors[row] + errors[row];
    }
  }
  return {predictors, response, errors, offsets, neighbors, weights};
}
