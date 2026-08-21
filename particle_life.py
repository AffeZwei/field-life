"""
Particle Life -- colored dots that attract/repel each other by color.

Everything is one file. Run it:   python3 particle_life.py
It writes particle_life.gif in the same folder.
"""

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- settings
SEED          = 7        # change this for a different world
N             = 800      # number of particles
COLORS        = 6        # number of particle types (colors)
R_MAX         = 0.10     # particles farther apart than this ignore each other
BETA          = 0.30     # inside BETA*R_MAX everyone repels everyone (personal space)
FORCE         = 10.0     # global strength of the interactions
DT            = 0.02     # time step
FRICTION_HALF = 0.040    # velocity halves every this many seconds
FRAMES        = 400      # frames in the gif
SIZE          = 500      # gif size in pixels
DOT           = 3        # dot radius in pixels

PALETTE = [(255, 80, 80), (255, 170, 60), (245, 235, 90),
           (90, 220, 120), (80, 180, 255), (200, 110, 255)]

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- the world
# Positions live in a [0,1) x [0,1) square that wraps around like Pac-Man.
pos   = rng.random((N, 2))
vel   = np.zeros((N, 2))
color = rng.integers(0, COLORS, N)

# THE INTERACTION MATRIX: matrix[a, b] is how color a feels about color b.
# Positive = attracted to it, negative = pushed away from it.
# Note it is NOT symmetric: red can chase green while green flees red.
matrix = rng.uniform(-1.0, 1.0, (COLORS, COLORS))


def force(r, attraction):
    """
    How hard a particle is pushed, given the distance r (as a fraction of R_MAX)
    and how much it likes the other color.

      r < BETA        -> always pushed away, hard (so dots never collapse)
      BETA < r < 1    -> pulled by `attraction`, strongest halfway, fading to 0 at r=1
    """
    close = r < BETA
    near  = ~close
    out = np.zeros_like(r)
    out[close] = r[close] / BETA - 1.0                                   # repulsion
    out[near]  = attraction[near] * (1.0 - np.abs(2.0 * r[near] - 1.0 - BETA) / (1.0 - BETA))
    return out


def step():
    """Advance the simulation by one tick."""
    global pos, vel

    # Vector from every particle to every other particle (N x N x 2).
    delta = pos[None, :, :] - pos[:, None, :]
    delta -= np.round(delta)                 # wrap: always use the shortest way around
    dist = np.hypot(delta[:, :, 0], delta[:, :, 1])

    # Who is close enough to matter, and not myself?
    mask = (dist > 0) & (dist < R_MAX)

    r = np.where(mask, dist / R_MAX, 1.0)    # normalized distance, 1.0 = "no effect"
    attraction = matrix[color[:, None], color[None, :]]
    strength = np.where(mask, force(r, attraction), 0.0)

    # Push along the unit vector towards the other particle.
    direction = delta / np.where(dist[:, :, None] == 0, 1.0, dist[:, :, None])
    accel = (strength[:, :, None] * direction).sum(axis=1) * R_MAX * FORCE

    vel *= 0.5 ** (DT / FRICTION_HALF)       # friction, so things settle down
    vel += accel * DT
    pos = (pos + vel * DT) % 1.0             # move, and wrap around the edges


# A gif can only hold a fixed list of colors, so we draw straight into a
# paletted image: index 0 is the background, index c+1 is particle color c.
GIF_PALETTE = [10, 10, 16] + [v for rgb in PALETTE for v in rgb]


def render():
    """Draw the current state as one image."""
    img = Image.new("P", (SIZE, SIZE), 0)
    img.putpalette(GIF_PALETTE)
    draw = ImageDraw.Draw(img)
    px = pos * SIZE
    for (x, y), c in zip(px, color):
        draw.ellipse([x - DOT, y - DOT, x + DOT, y + DOT], fill=int(c) + 1)
    return img


if __name__ == "__main__":
    print("interaction matrix (row = who feels, column = about whom):")
    print(np.round(matrix, 2))

    frames = []
    for i in range(FRAMES):
        step()
        frames.append(render())
        if i % 50 == 0:
            print(f"frame {i}/{FRAMES}")

    frames[0].save("particle_life.gif", save_all=True, append_images=frames[1:],
                   duration=40, loop=0, optimize=True)
    print("wrote particle_life.gif")
