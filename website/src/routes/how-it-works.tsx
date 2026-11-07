// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Outlet} from '@tanstack/react-router';

export const Route = createFileRoute('/how-it-works')({
  component: Outlet
});
