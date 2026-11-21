// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** At least one of the actual terminal result choices, retaining optional sibling columns. @internal */
export type RequireRasterOutput<Columns, Keys extends keyof Columns = keyof Columns> = Columns &
  {
    [Key in Keys]-?: Required<Pick<Columns, Key>>;
  }[Keys];
