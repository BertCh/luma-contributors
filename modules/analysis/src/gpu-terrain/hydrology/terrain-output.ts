// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Select at least one accepted terminal while retaining optional sibling ports. @internal */
export type TerrainAtLeastOne<Ports, Terminal extends keyof Ports = keyof Ports> = Ports &
  {
    [Port in Terminal]-?: Required<Pick<Ports, Port>>;
  }[Terminal];
