import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


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

fig = plt.figure(figsize=(16, 10))

ax = fig.add_subplot(2, 1, 1)
ax.plot(df['timestamp'], df['x1'], '-', label='x1', color='steelblue')
ax.plot(df['timestamp'][1:-1], df['x1'][1:-1], 'o', color=(100/255, 220/255, 100/255))
ax.plot(df['timestamp'], df['x2'], '-', label='x2', color='orange')
ax.plot(df['timestamp'][1:-1], df['x2'][1:-1], 'o', label='collision', color=(100/255, 200/255, 100/255))
ax.legend(loc='lower right')
ax.grid()
ax.set_title(f'm1={m1:.0f}kg, m2={m2:.0f}kg, collisions={len(df)-2}\n\n\nPosition')
# ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')

ax = fig.add_subplot(2, 1, 2)
ax.plot(df['timestamp'], df['v1'], '-', label='v1', color='steelblue')
ax.plot(df['timestamp'][1:-1], df['v1'][1:-1], 'o', color=(100/255, 220/255, 100/255))
ax.plot(df['timestamp'], df['v2'], '-', label='v2', color='orange')
ax.plot(df['timestamp'][1:-1], df['v2'][1:-1], 'o', label='collision', color=(100/255, 220/255, 100/255))
ax.legend(loc='lower right')
ax.grid()
ax.set_title(f'Velocity')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (m/s)')

plt.savefig('output.png', dpi=300)
# plt.show()

# k = 1
# m1 = 1
# m2 = m1 * (10 ** (2 * k))
# N = int(np.ceil((np.pi / np.atan(np.sqrt(m1 / m2))) - 1))
# pi_appr = (N + 1) / (10**k)
# print(f'm1: {m1}')
# print(f'm2: {m2}')
# print(f'N: {N}')
# print(f'π_appr: (N+1)/10^k: {pi_appr}')
# print(f'π_real: {np.pi}')
# print(f'δ: ±{pi_appr - np.pi}')

# video size
col = 3840
row = 2160
k1 = col // 10

# background image
background = np.zeros((row, col, 3), dtype=np.uint8) + 255

# right simulation
pr = np.array([col * 0.39, row * 0.8])
kr = lambda x, y: (pr + np.array([k1, -k1]) * [x, y]).astype(np.int64)
# y axis
background = cv2.line(background, kr(0, 0), kr(0, 3), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# y arrow
background = cv2.line(background, kr(0, 3), kr(-0.1, 2.9), (0, 0, 0), 3, lineType=cv2.LINE_AA)
background = cv2.line(background, kr(0, 3), kr(0.1, 2.9), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# y character
background = cv2.putText(background, 'y(m)', kr(-0.5, 2.9), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
# x axis
background = cv2.line(background, kr(0, 0), kr(6, 0), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# x arrow
background = cv2.line(background, kr(6, 0), kr(5.9, -0.1), (0, 0, 0), 3, lineType=cv2.LINE_AA)
background = cv2.line(background, kr(6, 0), kr(5.9, 0.1), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# x character
background = cv2.putText(background, 'x(m)', kr(5.7, -0.3), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
# origin
background = cv2.putText(background, 'O', kr(-0.15, -0.15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

# left plot
pl = kr(-2, 1.5)
kl = lambda x, y: (pl + np.array([k1, -k1]) * [x, y]).astype(np.int64)
r = 1.3
# circle, AntiAlias
background = cv2.circle(background, kl(0, 0), int(r * k1), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# y axis
background = cv2.line(background, kl(0, 0), kl(0, 1.3 * r), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# y arrow
background = cv2.line(background, kl(0, 1.3 * r), kl(-0.05 * r, 1.25 * r), (0, 0, 0), 3, lineType=cv2.LINE_AA)
background = cv2.line(background, kl(0, 1.3 * r), kl(0.05 * r, 1.25 * r), (0, 0, 0), 3, lineType=cv2.LINE_AA)

# y character
img_m1v1 = cv2.imread('m1v1.jpg')
# resize img_m1v1
tmp_h = 60
img_m1v1 = cv2.resize(img_m1v1, (int(img_m1v1.shape[1] * tmp_h / img_m1v1.shape[0]), tmp_h))
# paste img_m1v1 to background
background[
    pl[1] - int(1.3 * r * k1) : pl[1] - int(1.3 * r * k1) + img_m1v1.shape[0],
    pl[0] - 50 - img_m1v1.shape[1] : pl[0] - 50,
] = img_m1v1

# x axis
background = cv2.line(background, kl(0, 0), kl(1.3 * r, 0), (0, 0, 0), 3, lineType=cv2.LINE_AA)
# x arrow
background = cv2.line(background, kl(1.3 * r, 0), kl(1.25 * r, 0.05 * r), (0, 0, 0), 3, lineType=cv2.LINE_AA)
background = cv2.line(background, kl(1.3 * r, 0), kl(1.25 * r, -0.05 * r), (0, 0, 0), 3, lineType=cv2.LINE_AA)

# x character
img_m2v2 = cv2.imread('m2v2.jpg')
# resize img_m2v2
tmp_h = 60
img_m2v2 = cv2.resize(img_m2v2, (int(img_m2v2.shape[1] * tmp_h / img_m2v2.shape[0]), tmp_h))
# paste img_m2v2 to background
background[
    pl[1] + 50 : pl[1] + 50 + img_m2v2.shape[0],
    pl[0] + 50 + int(1.3 * r * k1) - img_m2v2.shape[1] : pl[0] + 50 + int(1.3 * r * k1),
] = img_m2v2


# plt.imshow(background)


# linear interpolation of position with constant fps
fps = 60
df_x_interp = pd.DataFrame(columns=['timestamp', 'x1', 'x2'])
df_x_interp['timestamp'] = np.arange(df['timestamp'][0], df['timestamp'].values[-1], 1 / fps)
df_x_interp['x1'] = np.interp(df_x_interp['timestamp'], df['timestamp'], df['x1'])
df_x_interp['x2'] = np.interp(df_x_interp['timestamp'], df['timestamp'], df['x2'])

# video writer
fourcc = cv2.VideoWriter_fourcc(*'avc1')
videoWriter = cv2.VideoWriter('output.mp4', fourcc, fps, (col, row))
j_old = 1
k2 = r / np.sqrt(2 * E)
for i in range(len(df_x_interp)):
    # draw collision lines in circle
    if df['timestamp'][j_old] <= df_x_interp['timestamp'][i]:
        j = j_old
        while df['timestamp'][j] <= df_x_interp['timestamp'][i]:
            background = cv2.line(
                background,
                kl(np.sqrt(m2) * df['v2'][j - 1] * k2, np.sqrt(m1) * df['v1'][j - 1] * k2),
                kl(np.sqrt(m2) * df['v2'][j] * k2, np.sqrt(m1) * df['v1'][j] * k2),
                (100, 220, 100),
                3,
                lineType=cv2.LINE_AA,
            )
            background = cv2.circle(
                background,
                kl(np.sqrt(m2) * df['v2'][j] * k2, np.sqrt(m1) * df['v1'][j] * k2),
                10,
                (100, 220, 100),
                -1,
                lineType=cv2.LINE_AA,
            )
            j += 1
        j_old = j
        # show approximation of π
        if j_old == df.shape[0] - 1:
            N = j_old - 1
            s = f'pi appr: {((N+1)*np.sqrt(m1/m2)):.6f}'
            text_size, _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            background = cv2.putText(
                background,
                s,
                kl(0, -1.4 * r) + [-text_size[0] + 200, text_size[1] // 2],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (100, 220, 100),
                3,
            )
            s = f'pi real: {np.pi:.6f}'
            text_size, _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            background = cv2.putText(
                background,
                s,
                kl(0, -1.55 * r) + [-text_size[0] + 200, text_size[1] // 2],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (100, 220, 100),
                3,
            )

    # show collisions
    s = f'collisions: {j_old-1}'
    text_size, _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    img = cv2.putText(
        background.copy(),
        s,
        kl(0, -1.2 * r) + [-text_size[0] // 2, text_size[1] // 2],
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (100, 220, 100),
        3,
    )

    # left box
    img = cv2.rectangle(img, kr(df_x_interp['x1'][i], 0), kr(df_x_interp['x1'][i] + a1, a1), (180, 130, 70), -1)
    # m = m1, loc = center
    s = f'm1={m1:.0f}kg'
    text_size, _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    img = cv2.putText(
        img,
        s,
        kr(df_x_interp['x1'][i] + a1 / 2, a1 / 2) + [-text_size[0] // 2, text_size[1] // 2],
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    # right box
    img = cv2.rectangle(img, kr(df_x_interp['x2'][i], 0), kr(df_x_interp['x2'][i] + a2, a2), (0, 165, 255), -1)
    # m = m2, loc = center
    s = f'm2={m2:.0f}kg'
    text_size, _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    img = cv2.putText(
        img,
        s,
        kr(df_x_interp['x2'][i] + a2 / 2, a2 / 2) + [-text_size[0] // 2, text_size[1] // 2],
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
    )
    videoWriter.write(img)
videoWriter.release()
