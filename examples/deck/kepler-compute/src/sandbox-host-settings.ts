// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Host preferences that travel inside the saved workspace bundle. */
export type SandboxHostSettings = {memoryBudgetBytes?: number};

export type SandboxHostSettingsProvider = {
  /** Only preferences the user changed; undefined or empty when everything is default. */
  get(): SandboxHostSettings | undefined;
  apply(settings: SandboxHostSettings): void;
};

let provider: SandboxHostSettingsProvider | undefined;

/** The host installs one provider at start-up; save and replay read it so every save path agrees. */
export function setSandboxHostSettingsProvider(
  next: SandboxHostSettingsProvider | undefined
): void {
  provider = next;
}

export function getSandboxHostSettings(): SandboxHostSettings | undefined {
  return provider?.get();
}

export function applySandboxHostSettings(settings: SandboxHostSettings): void {
  provider?.apply(settings);
}
