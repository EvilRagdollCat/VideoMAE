import numpy as np, cv2
import numpy as np

def _motion_energy(frames, down=2):
    # frames: (T,H,W,3) uint8
    T,H,W,_ = frames.shape
    h,w = H//down, W//down
    gray = np.empty((T,h,w), np.float32)
    for t in range(T):
        g = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        gray[t] = cv2.resize(g, (w,h), interpolation=cv2.INTER_AREA)
    diff = np.abs(np.diff(gray, axis=0))              # (T-1,h,w)
    eng  = diff.sum(axis=0)                           # (h,w)
    eng  = cv2.GaussianBlur(eng, (0,0), 3)
    return eng

def roi_by_tiles(frames, G=4, topk=3, margin=0.1, min_wh=64):
    """
    Calculate motion energy statistics in a GxG grid, take the union of the top-k cells as ROI
    """
    T,H,W,_ = frames.shape
    eng = _motion_energy(frames, down=2)
    # > Resize (map) the energy map back to the original image resolution to make tile-wise summation convenient
    eng_full = cv2.resize(eng, (W,H), interpolation=cv2.INTER_CUBIC)
    gh, gw = H//G, W//G
    tiles = []
    for i in range(G):
        for j in range(G):
            y1,y2 = i*gh, (i+1)*gh if i<G-1 else H
            x1,x2 = j*gw, (j+1)*gw if j<G-1 else W
            val = eng_full[y1:y2, x1:x2].sum()
            #tiles.append((val, i, j, y1,y2,x1,x2))
            tiles.append((val, y1,y2,x1,x2))
    tiles.sort(reverse=True, key=lambda x: x[0])
    use = tiles[:max(1, topk)]
    #y1 = min(t[3] for t in use); y2 = max(t[4] for t in use)
    #x1 = min(t[5] for t in use); x2 = max(t[6] for t in use)
    y1 = min(t[1] for t in use); y2 = max(t[2] for t in use)
    x1 = min(t[3] for t in use); x2 = max(t[4] for t in use)
    # > margin
    dy = int((y2-y1)*margin); dx = int((x2-x1)*margin)
    y1 = max(0, y1-dy); y2 = min(H, y2+dy)
    x1 = max(0, x1-dx); x2 = min(W, x2+dx)
    # > minimum size
    if (y2-y1) < min_wh: 
        c = (y1+y2)//2; y1 = max(0, c-min_wh//2); y2 = min(H, y1+min_wh)
    if (x2-x1) < min_wh: 
        c = (x1+x2)//2; x1 = max(0, c-min_wh//2); x2 = min(W, x1+min_wh)
    return y1,y2,x1,x2

def snap16(y1, x1, y2, x2, H, W, min_wh=96, snap=16):
    side = max(min_wh, max(y2-y1, x2-x1))
    side = int(np.ceil(side / snap) * snap)
    cy, cx = (y1+y2)//2, (x1+x2)//2
    y1 = max(0, min(H - side, cy - side//2)); y2 = y1 + side
    x1 = max(0, min(W - side, cx - side//2)); x2 = x1 + side
    return y1, x1, y2, x2

def pick_active_tile(frames_rgb, grid=4):
    """
    frames_rgb: (T, H, W, 3) uint8
    return: (y1, x1, y2, x2) the edges of the subimage with the highest score
    """
    T, H, W, _ = frames_rgb.shape
    # > Motion energy: the difference in gray scale of consecutive frames
    E = np.zeros((H, W), dtype=np.float32)
    prev = None
    for t in range(T):
        g = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            d = cv2.GaussianBlur(d, (5, 5), 0)
            E += d.astype(np.float32)
        prev = g

    # > 4x4 grids summation, get the maximum
    hstep, wstep = H // grid, W // grid
    Et = E[:grid*hstep, :grid*wstep].reshape(grid, hstep, grid, wstep).sum(axis=(1,3))
    i, j = np.unravel_index(np.argmax(Et), Et.shape)
    y1, y2 = i * hstep, (i + 1) * hstep
    x1, x2 = j * wstep, (j + 1) * wstep
    return y1, x1, y2, x2

def expand_and_snap(y1, x1, y2, x2, H, W, min_wh=96, snap=16):
    """
    Expand the tile to square ROI, with side length >= min_wh and get aligned with the ViT patch size (dividable by 16)
    """
    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2
    side = max(min_wh, max(y2 - y1, x2 - x1))
    # dividable by 16
    side = int(np.ceil(side / snap) * snap)

    y1 = max(0, cy - side // 2)
    x1 = max(0, cx - side // 2)
    y2 = min(H, y1 + side)
    x2 = min(W, x1 + side)
    # 
    y1 = max(0, y2 - side)
    x1 = max(0, x2 - side)
    return y1, x1, y2, x2
