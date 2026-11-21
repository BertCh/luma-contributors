// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export function getCatalogIdentityKey(identity: {package: string; exportName: string}): string {
  return JSON.stringify([identity.package, identity.exportName]);
}
