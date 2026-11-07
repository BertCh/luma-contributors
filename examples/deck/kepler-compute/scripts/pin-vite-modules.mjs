// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Pin loaded development modules while other contributors edit the shared checkout. */
export async function pinViteModules(page) {
  await page.routeWebSocket('**/*', socket => {
    const server = socket.connectToServer();
    server.onMessage(message => {
      if (typeof message === 'string') {
        try {
          const packet = JSON.parse(message);
          if (packet.type === 'update' || packet.type === 'full-reload') return;
        } catch {
          // Non-JSON application messages still pass through.
        }
      }
      socket.send(message);
    });
    socket.onMessage(message => server.send(message));
  });
}
