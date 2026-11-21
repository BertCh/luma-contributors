// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Requires at least one writable terminal while keeping the remaining outputs optional. @internal */
export type TerrainOutputSelection<Output, Keys extends keyof Output = keyof Output> = Output &
  {
    [Key in Keys]-?: Required<Pick<Output, Key>>;
  }[Keys];
