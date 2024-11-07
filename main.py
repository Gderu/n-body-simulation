from PyQt5 import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import time
import star_simulation_rust

N = 1000
WIDTH, HEIGHT = 800, 600
FPS = 30
dt = None

a = []
c = 0
before = time.time()

def update(positions, velocities, masses):
    accelerations = star_simulation_rust.calc_acceleration_brute_force(positions, masses)
    velocities += accelerations * dt
    positions += velocities * dt
    return accelerations


# def calc_acceleration(positions, masses, matrix, temp):
#     # Compute differences only for the unique pairs in the upper triangle
#     # a = time.time()
#     diff = positions[i, :] - positions[j, :]
#     dists_sqr = np.sum(diff ** 2, axis=1)
#     dists_sqr[dists_sqr < 1] = 1
#     # b = time.time()
#     # print(dists_sqr)
#     dists_cubed = dists_sqr[:, np.newaxis] ** -1.5
#     # c = time.time()
#
#     # Fill in the matrix for both upper and lower triangles
#     matrix[i, j, :] = G * diff * dists_cubed
#     # g = time.time()
#     test = G * diff * dists_cubed
#     # t = time.time()
#     matrix[j, i, :] = -temp[i * N + j, :]
#     # d = time.time()
#
#     # Adding weight for the masses
#     matrix *= masses[:, np.newaxis, np.newaxis]
#     res = np.sum(matrix, axis=0)
#     # e = time.time()
#     # print(f"dists: {b - a}")
#     # print(f"dists cubed: {c - b}")
#     # print(f"temp: {d - c}")
#     # print(f"matrix: {e - d}")
#     # print(f"not this: {g -c}")
#     # print(f"test: {t - g}")
#     # print(diff.shape)
#     # print(dists_cubed.shape)
#     return res


def update_draw_with_lines(positions, velocities, masses, bodies, saved_positions, lines, divide_by):
    for line, saved_pos, new_pos in zip(lines, saved_positions, positions):
        saved_pos.append(new_pos/divide_by)
        line.setData(pos=tuple(saved_pos))

    update(positions, velocities, masses)
    bodies.setData(pos=positions / divide_by)


def update_draw(positions, velocities, masses, bodies, divide_by):
    for i in range(10):
        acceleration = update(positions, velocities, masses)
    bodies.setData(pos=positions / divide_by)
    print((acceleration ** 2).sum(axis=1).max() ** 0.5)


def run_earth_moon_sun_system(w, t):
    global dt
    dt = 5e4
    w.addItem(gl.GLGridItem())

    # Sun, Earth, and the Moon
    positions = np.array([[0, 0, 0], [148.38e9, 0, 0], [148.38e9, 3.844e8, 0]], dtype=np.float64)
    velocities = np.array([[0, 0, 0], [0, 29722, 0], [0, 29722, 1.022e3]], dtype=np.float64)
    masses = np.array([1.989e30, 5.972e24, 7.348e22])
    colors = np.array([[249, 215, 28, 255], [79, 76, 176, 255], [70, 70, 70, 255]]) / 255
    line_colors = np.array([[0, 255, 0, 255], [255, 0, 0, 255], [255, 0, 255, 255]]) / 255
    sizes = np.array([0.1, 0.01, 0.005])

    velocities += star_simulation_rust.calc_acceleration_brute_force(positions, masses) * dt / 2
    bodies = gl.GLScatterPlotItem(pos=positions / 100e9, color=colors, size=sizes, pxMode=False)
    w.addItem(bodies)

    saved_positions = [[pos / 100e9] for pos in positions]
    lines = []
    for pos, line_color in zip(saved_positions, line_colors):
        line = gl.GLLinePlotItem(pos=tuple(pos), antialias=True, color=list(line_color), mode='line_strip')
        w.addItem(line)
        lines.append(line)

    t.timeout.connect(lambda: update_draw_with_lines(positions, velocities, masses, bodies, saved_positions, lines, 100e9))
    return


def n_stars(n: int, w: gl.GLViewWidget, t: QtCore.QTimer):
    global dt
    dt = 1e15
    # w.addItem(gl.GLGridItem())

    positions = (np.random.rand(n, 3) - 0.5) * n * 4.73e+16
    velocities = (np.random.rand(n, 3) - 0.5) * 0
    masses = np.random.rand(n) * 0.2 * 1.989e30
    sizes = masses / masses.max() / 4

    velocities += star_simulation_rust.calc_acceleration_brute_force(positions, masses) * dt / 2
    bodies = gl.GLScatterPlotItem(pos=positions / (1e16 * n), color=(1., 1., .7, 1.), size=sizes, pxMode=False)
    w.addItem(bodies)
    t.timeout.connect(lambda: update_draw(positions, velocities, masses, bodies, 1e16 * n))



def main():
    app = pg.mkQApp("N-body simulation")
    t = QtCore.QTimer()
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle("N-body simulation")

    # run_earth_moon_sun_system(w, t)
    n_stars(1000, w, t)

    t.start(1000 // FPS)
    pg.exec()


if __name__ == "__main__":
    # main()
    n = 1000000
    positions = np.random.rand(n, 3) - 0.5
    masses = np.random.rand(n)

    star_simulation_rust.test(positions, masses)
