// Lightweight animated hex-grid + circuit background, echoing the reference
// "blue tech" mood board without depending on any external asset or library.
(function () {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, hexes = [], nodes = [];

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
    buildField();
  }

  function hexPoints(cx, cy, r) {
    const pts = [];
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i - Math.PI / 6;
      pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
    return pts;
  }

  function buildField() {
    hexes = [];
    nodes = [];
    const r = 46;
    const dx = r * 1.73;
    const dy = r * 1.5;
    const cols = Math.ceil(w / dx) + 2;
    const rows = Math.ceil(h / dy) + 2;
    for (let row = -1; row < rows; row++) {
      for (let col = -1; col < cols; col++) {
        const cx = col * dx + (row % 2 ? dx / 2 : 0);
        const cy = row * dy;
        if (Math.random() > 0.86) {
          hexes.push({
            cx, cy, r: r * 0.94,
            phase: Math.random() * Math.PI * 2,
            speed: 0.4 + Math.random() * 0.6,
            baseAlpha: 0.03 + Math.random() * 0.05,
          });
        }
      }
    }
    // sparse floating "circuit node" dots with connecting lines
    const nodeCount = Math.floor((w * h) / 60000);
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
        r: 1 + Math.random() * 1.5,
      });
    }
  }

  let t = 0;
  function draw() {
    t += 0.01;
    ctx.clearRect(0, 0, w, h);

    // pulsing hex outlines
    hexes.forEach(hx => {
      const alpha = hx.baseAlpha + 0.04 * Math.sin(t * hx.speed + hx.phase);
      ctx.beginPath();
      const pts = hexPoints(hx.cx, hx.cy, hx.r);
      pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      ctx.closePath();
      ctx.strokeStyle = `rgba(79,209,255,${Math.max(alpha, 0.015)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // drifting nodes + nearby connecting lines
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
    });
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 130) {
          ctx.strokeStyle = `rgba(79,209,255,${0.09 * (1 - d / 130)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(125,211,252,0.5)';
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  requestAnimationFrame(draw);
})();
