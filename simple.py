import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import time


a1, m1, v1, x1 = np.array([0.5, 1, 0, 1])
a2, m2, v2, x2 = np.array([2, 100, -1, 3])
E = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
timestamp = 0
stats = [[timestamp, x1, v1, x2, v2]]

while True:
    x_delta, v_delta = x1 - 0, v1 - 0
    if x_delta == 0 and v_delta <= 0:
        raise ValueError(f'Error: x_1: {x_delta}, v_1: {v_delta}')
    elif v_delta >= 0:
        t1 = -np.inf
    else:
        t1 = x_delta / (-v_delta)

    x_delta, v_delta = x2 - (x1 + a1), v2 - v1
    if x_delta == 0 and v_delta <= 0:
        raise ValueError(f'Error: x_2: {x_delta}, v_2: {v_delta}')
    elif v_delta >= 0:
        t2 = -np.inf
    else:
        t2 = x_delta / (-v_delta)

    if t1 < 0 and t2 < 0:
        break
    elif 0 < t1 and t1 == t2:
        raise ValueError(f'Error: t1: {t1}, t2: {t2}')
    elif (t2 < 0 < t1) or (0 < t1 < t2):
        t_delta = t1
        x1 = 0
        x2 += v2 * t_delta
        v1 = -v1
        timestamp += t_delta
        stats.append([timestamp, x1, v1, x2, v2])
    elif (t1 < 0 < t2) or (0 < t2 < t1):
        t_delta = t2
        x1 += v1 * t_delta
        x2 += v2 * t_delta
        P = m1 * v1 + m2 * v2
        v1 = (P * m1 - np.sqrt(P**2 * m1**2 - m1 * (m1 + m2) * (P**2 - 2 * E * m2))) / (m1 * (m1 + m2))
        v2 = (P - m1 * v1) / m2
        timestamp += t_delta
        stats.append([timestamp, x1, v1, x2, v2])
    else:
        raise ValueError(f'Error: t1: {t1}, t2: {t2}')

stats.append([timestamp + 1, x1 + v1 * 1, v1, x2 + v2 * 1, v2])

df = pd.DataFrame(stats, columns=['timestamp', 'x1', 'v1', 'x2', 'v2'])


# video size
col = 3840
row = 2160
k1 = col / 6

# background image
background = np.zeros((row, col, 3), dtype=np.uint8)

# right simulation
pr = np.array([col * 0, row * 1])
kr = lambda x, y: (pr + np.array([k1, -k1]) * [x, y]).astype(np.int64)

# linear interpolation of position with constant fps
fps = 60
df_x_interp = pd.DataFrame(columns=['timestamp', 'x1', 'x2'])
df_x_interp['timestamp'] = np.arange(df['timestamp'][0], df['timestamp'].values[-1], 1 / fps)
df_x_interp['x1'] = np.interp(df_x_interp['timestamp'], df['timestamp'], df['x1'])
df_x_interp['x2'] = np.interp(df_x_interp['timestamp'], df['timestamp'], df['x2'])


# terminal rendering
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def render_frame(x1, x2, width=120, height=30):
    # Create empty frame
    frame = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw ground line
    ground_y = height - 2
    for x in range(width):
        frame[ground_y][x] = '━'

    # Calculate scale factor to map physical coordinates to terminal space
    max_x = max(max(df_x_interp['x1']), max(df_x_interp['x2'])) + a2
    scale = (width - 4) / max_x
    v_scale = scale * 0.5

    def draw_box(x_pos, y_pos, size_x, size_y):
        # Fill box
        for y in range(y_pos, y_pos + size_y):
            for x in range(x_pos, x_pos + size_x):
                if 0 <= x < width and 0 <= y < height:
                    frame[y][x] = '█'

    # Draw left box (m1)
    box1_x = int(x1 * scale) + 2
    box1_y = ground_y - int(a1 * v_scale)
    box1_size_x = int(a1 * scale)
    box1_size_y = int(a1 * v_scale)
    draw_box(box1_x, box1_y, box1_size_x, box1_size_y)

    # Draw right box (m2)
    box2_x = int(x2 * scale) + 2
    box2_y = ground_y - int(a2 * v_scale)
    box2_size_x = int(a2 * scale)
    box2_size_y = int(a2 * v_scale)
    draw_box(box2_x, box2_y, box2_size_x, box2_size_y)

    # Add mass labels
    m1_label = f"m1={m1}kg"
    m2_label = f"m2={m2}kg"
    for i, c in enumerate(m1_label):
        if box1_x + i < width:
            frame[box1_y - 1][box1_x + i] = c
    for i, c in enumerate(m2_label):
        if box2_x + i < width:
            frame[box2_y - 1][box2_x + i] = c

    # Convert frame to string
    return '\n'.join(''.join(row) for row in frame)


# terminal animation

for i in range(len(df_x_interp)):
    clear_screen()
    print(render_frame(df_x_interp['x1'][i], df_x_interp['x2'][i]))
    time.sleep(1 / fps)
