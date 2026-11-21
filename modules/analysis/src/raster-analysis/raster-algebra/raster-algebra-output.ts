// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** At least one terminal output port, retaining completion for all optional siblings. @internal */
export type RasterAlgebraAtLeastOne<Ports> = {
  [Port in keyof Ports]-?: Required<Pick<Ports, Port>> & Partial<Omit<Ports, Port>>;
}[keyof Ports];
