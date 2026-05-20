/** White = selection. Composite matte: white opaque / transparent elsewhere (for CSS + destination-in). */

function luma(r, g, b) {
  return (0.299 * r + 0.587 * g + 0.114 * b) | 0;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = () => reject(new Error('mask load failed'));
    i.src = url;
  });
}

/** In-place. Selection → #fff opaque; off → transparent (for destination-in / CSS luminance). */
export function imageDataToCompositeMatte(data) {
  for (let i = 0; i < data.length; i += 4) {
    const a = data[i + 3];
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const sel = a > 8 && luma(r, g, b) >= 128;
    if (sel) {
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = 255;
    } else {
      data[i] = 0;
      data[i + 1] = 0;
      data[i + 2] = 0;
      data[i + 3] = 0;
    }
  }
}

/** In-place. Selection → #fff; off → #000; all alpha 255 (export / viz). */
export function imageDataToOpaqueBW(data) {
  for (let i = 0; i < data.length; i += 4) {
    const a = data[i + 3];
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const sel = a > 8 && luma(r, g, b) >= 128;
    const v = sel ? 255 : 0;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = 255;
  }
}

export async function normalizeMaskBlobToObjectUrl(blob) {
  const u0 = URL.createObjectURL(blob);
  const img = await loadImage(u0);
  URL.revokeObjectURL(u0);
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height);
  imageDataToCompositeMatte(d.data);
  ctx.putImageData(d, 0, 0);
  return new Promise((resolve, reject) => {
    c.toBlob(
      (b) => {
        if (!b) {
          reject(new Error('mask encode failed'));
          return;
        }
        resolve(URL.createObjectURL(b));
      },
      'image/png',
      1
    );
  });
}

export async function applyMaskMatteDestinationIn(ctx, fgW, fgH, maskUrl) {
  const img = await loadImage(maskUrl);
  const t = document.createElement('canvas');
  t.width = fgW;
  t.height = fgH;
  const tc = t.getContext('2d');
  tc.drawImage(img, 0, 0, fgW, fgH);
  const d = tc.getImageData(0, 0, fgW, fgH);
  imageDataToCompositeMatte(d.data);
  tc.putImageData(d, 0, 0);
  const prev = ctx.globalCompositeOperation;
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(t, 0, 0);
  ctx.globalCompositeOperation = prev;
}

export async function buildOpaqueBWMaskFgCanvas(maskUrl, fgW, fgH) {
  const img = await loadImage(maskUrl);
  const c = document.createElement('canvas');
  c.width = fgW;
  c.height = fgH;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, fgW, fgH);
  const d = ctx.getImageData(0, 0, fgW, fgH);
  imageDataToOpaqueBW(d.data);
  ctx.putImageData(d, 0, 0);
  return c;
}
