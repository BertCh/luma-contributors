// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Select at least one actual terminal column while retaining optional siblings. @internal */
export type RequiredGeometryOutput<
  Columns,
  Choices extends keyof Columns = keyof Columns
> = Columns & {[Column in Choices]-?: Required<Pick<Columns, Column>>}[Choices];
