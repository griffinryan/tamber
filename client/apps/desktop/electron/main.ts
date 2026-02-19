import { app, BrowserWindow } from 'electron';
import path from 'path';
import url from 'url';

const isDev = process.env.VITE_DEV_SERVER_URL !== undefined;

const createWindow = async () => {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: '#050814',
    title: 'Timbre Client',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
    },
  });

  if (isDev && process.env.VITE_DEV_SERVER_URL) {
    await win.loadURL(process.env.VITE_DEV_SERVER_URL);
    win.webContents.openDevTools({ mode: 'detach' });
  } else {
    const indexPath = url.format({
      pathname: path.join(__dirname, '../dist/index.html'),
      protocol: 'file:',
      slashes: true,
    });
    await win.loadURL(indexPath);
  }
};

app.whenReady().then(() => {
  void createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
