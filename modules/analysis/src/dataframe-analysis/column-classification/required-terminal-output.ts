// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Select at least one defined terminal view while retaining optional sibling ports. @internal */
export type RequiredTerminalOutput<Ports> = {
  [Port in keyof Ports]-?: Required<Pick<Ports, Port>> & Partial<Omit<Ports, Port>>;
}[keyof Ports];
