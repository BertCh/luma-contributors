// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type CellNeighborhoodOptions = {
  destination: 'MDW' | 'JFK' | 'LAX' | 'SYD';
  resolution: string;
  pathCapacity: string;
  focus: 'journey' | 'boundary';
};
