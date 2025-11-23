import { contextBridge } from 'electron';

// Expose minimal API for future filesystem/IPC needs; stub for now.
contextBridge.exposeInMainWorld('timbreNative', {
  ping: () => 'pong',
});
