// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
/** @internal Require an actual terminal result, excluding supplementary status/draw ports. */
export type RequireTrajectoryOutput<
  Output,
  Choice extends keyof Output = Exclude<keyof Output, 'status' | 'drawInstanceCount'>
> = Output &
  {
    [Port in Choice]-?: Required<Pick<Output, Port>>;
  }[Choice];
