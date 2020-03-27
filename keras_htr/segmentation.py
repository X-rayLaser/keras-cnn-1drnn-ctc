import numpy as np
import tensorflow as tf

from keras_htr.generators import binarize
from scipy import ndimage
import networkx as nx
import subprocess


class ConnectedComponent:
    def __init__(self, points):
        self.points = points

        self.y = [y for y, x in points]
        self.x = [x for y, x in points]
        self.top = min(self.y)
        self.bottom = max(self.y)

        self.left = min(self.x)
        self.right = max(self.x)

        self.height = self.bottom - self.top + 1
        self.width = self.right - self.left + 1

        self.center_x = np.array(self.x).mean()
        self.center_y = self.top + self.height // 2

    @property
    def bounding_box(self):
        return self.left, self.bottom, self.right, self.top

    def __contains__(self, point):
        y, x = point
        return x >= self.left and x <= self.right and y >= self.top and y <= self.bottom

    def visualize(self):
        a = np.zeros((self.bottom + 1, self.right + 1, 1))

        for y, x in self.points:
            a[y, x, 0] = 255

        tf.keras.preprocessing.image.array_to_img(a).show()


class Line:
    def __init__(self):
        self._components = []

    def add_component(self, component):
        self._components.append(component)

    def __iter__(self):
        for c in self._components:
            yield c

    @property
    def num_components(self):
        return len(self._components)

    @property
    def top(self):
        return min([component.top for component in self._components])

    @property
    def bottom(self):
        return max([component.bottom for component in self._components])

    @property
    def left(self):
        return min([component.left for component in self._components])

    @property
    def right(self):
        return max([component.right for component in self._components])

    @property
    def height(self):
        return self.bottom - self.height

    def __contains__(self, component):
        padding = 5
        return component.center_y >= self.top - padding and component.center_y < self.bottom + padding


def to_vertex(i, j, w):
    return i * w + j


def to_grid_cell(v, h, w):
    row = v // w
    col = v % w
    return row, col


def is_within_bounds(h, w, i, j):
    return i < h and i >= 0 and j < w and j >= 0


def make_edges(h, w, i, j):
    if j >= w:
        return []

    x = j
    y = i

    neighbors = []
    for l in [-1, 1]:
        for m in [-1, 1]:
            neighbors.append((y + l, x + m))

    vertices = [to_vertex(y, x, w) for y, x in neighbors if is_within_bounds(h, w, y, x)]

    u = to_vertex(i, j, w)
    edges = [(u, v) for v in vertices]
    return edges


def make_grid_graph(im):
    h, w = im.shape

    G = nx.Graph()

    for i in range(h):
        for j in range(w):
            for u, v in make_edges(h, w, i, j):
                row, col = to_grid_cell(v, h, w)
                if im[i, j] > 0 and im[row, col] > 0:
                    G.add_node(to_vertex(i, j, w))
                    G.add_node(u)
                    G.add_edge(u, v)

    return G


def get_connected_components(im):
    G = make_grid_graph(im)

    h, w = im.shape

    components = []
    for vertices in nx.connected_components(G):
        points = []
        for v in vertices:
            point = to_grid_cell(v, h, w)
            points.append(point)

        if len(points) > 0:
            components.append(ConnectedComponent(points))

    return components


def get_seam(signed_distance):
    s = ''
    h, w, _ = signed_distance.shape

    signed_distance = signed_distance.reshape(h, w)
    for row in signed_distance.tolist():
        s += ' '.join(map(str, row)) + '\n'

    with open('array.txt', 'w') as f:
        f.write('{} {}\n'.format(h, w))
        f.write(s)

    subprocess.call(['./carving'])

    with open('seam.txt') as f:
        s = f.read()

    row_indices = [int(v) for v in s.split(' ') if v != '']

    column_indices = list(range(w))

    return row_indices, column_indices


def visualize_map(m):
    h, w, _ = m.shape
    m = m.reshape(h, w)
    m = m + abs(m.min())

    c = m.max() / 255.0
    m = m / c

    m = 255 - m

    tf.keras.preprocessing.image.array_to_img(m.reshape(h, w, 1)).show()


def visualize_components(line):
    h = line.bottom + 1
    w = line.right + 1
    a = np.zeros((h, w, 1))
    for comp in line:
        for y, x in comp.points:
            a[y, x] = 255

    tf.keras.preprocessing.image.array_to_img(a.reshape(h, w, 1)).show()


def prepare_image():
    #img = tf.keras.preprocessing.image.load_img('iam_database/iam_database_formsA-D/a01-000u.png')
    img = tf.keras.preprocessing.image.load_img('screen.png')

    a = tf.keras.preprocessing.image.img_to_array(img)
    h, w, _ = a.shape

    a = binarize(a)
    x = a.reshape(h, w)

    return x // 255


def get_intersections(components, seam, lines):
    row_indices, column_indices = seam

    new_line = Line()

    for row, col in zip(row_indices, column_indices):
        point = (row, col)
        for component in components[:]:
            if point in component:
                add_to_new_line = True

                for line in lines:
                    if component in line:
                        line.add_component(component)
                        add_to_new_line = False
                        break
                if add_to_new_line:
                    new_line.add_component(component)

                components.remove(component)

    if new_line.num_components > 0:
        lines.append(new_line)


def seam_carving_segmentation():

    x = prepare_image()

    x_copy = x.copy()
    h, w = x.shape

    components = get_connected_components(x)

    lines = []
    xc = 1 - x
    signed_distance = ndimage.distance_transform_edt(xc) - ndimage.distance_transform_edt(x)
    signed_distance = signed_distance.reshape(h, w, 1)

    for i in range(h):
        if len(components) == 0:
            break

        seam = get_seam(signed_distance)
        row_indices, column_indices = seam
        signed_distance[row_indices, column_indices] = 255
        get_intersections(components, seam, lines)

        print('i', i, 'lines #:', len(lines), 'num components', len(components))

    for line in lines:
        visualize_components(line)
        input('press key\n')

    # todo: store components and line regions in R-trees
    # todo: compute all H seams in c++
    # todo: fast graph processing for large images
