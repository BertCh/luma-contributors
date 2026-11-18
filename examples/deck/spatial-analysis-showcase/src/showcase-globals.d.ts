// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {};

declare global {
  var spatialAnalysisShowcase:
    | {
        sceneId: string | null;
        ready: boolean;
        error: string | null;
        rebuildCount: number;
        selectScene: (id: string) => void;
      }
    | undefined;
}
