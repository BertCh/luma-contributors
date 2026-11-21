// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export class DiscoveryBase {
  getCommandNodes() {
    return [];
  }
}
export class DiscoveryInherited extends DiscoveryBase {}
export class DiscoveryPass {
  getPasses() {
    return [];
  }
}
