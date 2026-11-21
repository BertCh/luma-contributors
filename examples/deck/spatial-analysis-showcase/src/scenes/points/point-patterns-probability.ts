// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Recover the attained Monte Carlo fraction from its rounded float32 readback. */
export function getGlobalEnvelopeProbability(value: number, simulationCount: number): number {
  const denominator = simulationCount + 1;
  return Math.round(value * denominator) / denominator;
}
