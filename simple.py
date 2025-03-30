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

# linear interpolation of position with constant fps
fps = 60
df_x_interp = pd.DataFrame(columns=['timestamp', 'x1', 'x2'])
df_x_interp['timestamp'] = np.arange(df['timestamp'][0], df['timestamp'].values[-1], 1 / fps)
df_x_interp['x1'] = np.interp(df_x_interp['timestamp'], df['timestamp'], df['x1'])
df_x_interp['x2'] = np.interp(df_x_interp['timestamp'], df['timestamp'], df['x2'])


def render_frame(x1, x2, width, height):
    # Create empty frame
    frame = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw ground line
    ground_y = height

    # Calculate scale factor to map physical coordinates to terminal space
    max_x = max(max(df_x_interp['x1']), max(df_x_interp['x2'])) + a2
    scale = width / max_x  # reduce margin from 4 to 2
    v_scale = scale * 0.5

    def draw_box(x_pos, y_pos, size_x, size_y):
        # Fill box with three types of blocks for smoother appearance
        start_x = x_pos * 2
        end_x = (x_pos + size_x) * 2

        for y in range(y_pos, y_pos + size_y):
            if 0 <= y < height:
                # Handle first character (left edge)
                first_x = start_x // 2
                if 0 <= first_x < width:
                    frame[y][first_x] = '▌' if start_x % 2 == 0 else '█'

                # Handle middle characters (full blocks)
                for x in range((start_x + 1) // 2, end_x // 2):
                    if 0 <= x < width:
                        frame[y][x] = '█'

                # Handle last character (right edge)
                last_x = (end_x - 1) // 2
                if last_x != first_x and 0 <= last_x < width:
                    frame[y][last_x] = '▐' if end_x % 2 == 1 else '█'

    # Draw left box (m1)
    box1_x = max(0, int(x1 * scale))  # remove +2 margin, use max to prevent negative position
    box1_y = ground_y - int(a1 * v_scale)
    box1_size_x = int(a1 * scale)
    box1_size_y = int(a1 * v_scale)
    draw_box(box1_x, box1_y, box1_size_x, box1_size_y)

    # Draw right box (m2)
    box2_x = max(0, int(x2 * scale))  # remove +2 margin, use max to prevent negative position
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
# if width is not enough, print error
width = 120
height = 30
if os.get_terminal_size().columns < width:
    print("Error: width is not enough")
    exit()

for i in range(len(df_x_interp)):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(render_frame(df_x_interp['x1'][i], df_x_interp['x2'][i], width, height))
    time.sleep(1 / fps)
