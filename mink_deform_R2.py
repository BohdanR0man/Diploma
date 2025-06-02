import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.patches import Circle as MPLCircle
from scipy.spatial import ConvexHull
import time


def deformation_function(shape, direction_set):
    deformation_values = []
    for phi in direction_set:
        phi = np.array(phi)
        lambda_max = float('inf')

        for psi in direction_set:
            psi = np.array(psi)
            dot = np.dot(psi, phi)
            if dot == 0:
                continue

            support_val, _ = shape.support_function(psi)
            support_val -= np.dot(psi, shape.center)
            lambda_curr = support_val / dot

            if lambda_curr > 0 and lambda_curr < lambda_max:
                lambda_max = lambda_curr

        deformation_values.append(lambda_max)

    return deformation_values

def minkowski_functional(shape, direction_set):
    center = np.array(shape.get_center(), dtype=float)
    minkowski_values = []

    for phi in direction_set:
        phi = np.array(phi, dtype=float)
        norm_phi = np.linalg.norm(phi)
        if norm_phi < 1e-9:
            minkowski_values.append(0.0)
            continue

        unit_phi = phi / norm_phi
        max_value = 0.0

        for psi in direction_set:
            psi = np.array(psi, dtype=float)
            norm_psi = np.linalg.norm(psi)
            if norm_psi < 1e-9:
                continue

            unit_psi = psi / norm_psi
            support_val, _ = shape.support_function(unit_psi)
            support_val -= np.dot(unit_psi, center)

            if abs(support_val) < 1e-9:
                continue

            dot = np.dot(unit_psi, unit_phi)
            value = dot / support_val
            if value > max_value:
                max_value = value

        radius = 1.0 / max_value if max_value > 1e-9 else 0.0
        minkowski_values.append(radius)

    return minkowski_values

def plot_deformation_points(ax, direction_set, deformation_values, center, **plot_kwargs):
    points = np.array([center + deformation_values[i] * direction_set[i] for i in range(len(direction_set))])
    ax.scatter(points[:, 0], points[:, 1], **plot_kwargs)

def plot_minkowski_points(ax, direction_set, minkowski_values, center, **plot_kwargs):
    points = np.array([center + minkowski_values[i] * direction_set[i] for i in range(len(direction_set))])
    ax.scatter(points[:, 0], points[:, 1], **plot_kwargs)

class CircleShape():
    def __init__(self, center, radius):
        super().__init__()
        self.center = np.array(center)
        self.radius = radius

    def get_center(self):
        return self.center

    def support_function(self, direction):
        direction = np.array(direction)
        norm = direction / np.linalg.norm(direction)

        support_point = self.center + self.radius * norm
        support_value = np.dot(direction, support_point)
        return support_value, support_point

    def get_boundary_points(self, num_points=128):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return np.array([
            self.center + self.radius * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ])

    def plot(self, ax1, ax2, **kwargs):
        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault('fill', True)
        plot_kwargs.setdefault('alpha', 0.3)
        ax1.add_patch(MPLCircle(self.center, self.radius, **plot_kwargs))
        ax2.add_patch(MPLCircle(self.center, self.radius, **plot_kwargs))

        num_dirs = 120
        angles = np.linspace(0, 2 * np.pi, num_dirs, endpoint=False)
        direction_set = [np.array([np.cos(a), np.sin(a)]) for a in angles]

        deformation_vals = deformation_function(self, direction_set)
        plot_deformation_points(ax1, direction_set, deformation_vals, self.center, color="green", s=10)
        ax1.set_title('Deformation')
        ax1.set_aspect('equal')
        minkowski_vals = minkowski_functional(self, direction_set)
        plot_minkowski_points(ax2, direction_set, minkowski_vals, self.center, color="yellow", s=10)
        ax2.set_title('Minkowski Functional')
        ax2.set_aspect('equal')


class SquareShape():
    def __init__(self, center, side_length):
        super().__init__()
        self.center = np.array(center)
        self.side_length = side_length

    def get_center(self):
        return self.center

    def get_vertices(self):
        half_side = self.side_length / 2
        return np.array([
            [self.center[0] - half_side, self.center[1] - half_side],
            [self.center[0] + half_side, self.center[1] - half_side],
            [self.center[0] + half_side, self.center[1] + half_side],
            [self.center[0] - half_side, self.center[1] + half_side]
        ])

    def support_function(self, direction):
        direction = np.array(direction)
        half_side = self.side_length / 2
        max_dot = float('-inf')
        support_point = None

        for x in [-1, 1]:
            for y in [-1, 1]:
                vertex = self.center + np.array([x, y]) * half_side
                dot_product = np.dot(direction, vertex)
                if dot_product > max_dot:
                    max_dot = dot_product
                    support_point = vertex

        return max_dot, support_point

    def get_boundary_points(self, num_points=128):
        vertices = self.get_vertices()

        if num_points <= 4:
            return vertices

        points_per_edge = num_points // 4
        boundary_points = []

        for i in range(4):
            start = vertices[i]
            end = vertices[(i + 1) % 4]
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                point = (1 - t) * start + t * end
                boundary_points.append(point)

        return np.array(boundary_points)

    def plot(self, ax1, ax2, **kwargs):
        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault('fill', True)
        plot_kwargs.setdefault('alpha', 0.3)
        ax1.add_patch(MPLPolygon(self.get_vertices(), **plot_kwargs))
        ax2.add_patch(MPLPolygon(self.get_vertices(), **plot_kwargs))

        num_dirs = 120
        angles = np.linspace(0, 2 * np.pi, num_dirs, endpoint=False)
        direction_set = [np.array([np.cos(a), np.sin(a)]) for a in angles]

        deformation_vals = deformation_function(self, direction_set)
        plot_deformation_points(ax1, direction_set, deformation_vals, self.center, color="green", s=10)
        ax1.set_title('Deformation')
        ax1.set_aspect('equal')
        minkowski_vals = minkowski_functional(self, direction_set)
        plot_minkowski_points(ax2, direction_set, minkowski_vals, self.center, color="yellow", s=10)
        ax2.set_title('Minkowski Functional')
        ax2.set_aspect('equal')


class TransformedShape:
    def __init__(self, shape, A, b):
        self.shape = shape
        self.A = np.array(A)
        self.b = np.array(b)
        self.center = self.get_center()

    def get_center(self):
        return self.A @ self.shape.get_center() + self.b

    def support_function(self, direction):
        direction = np.array(direction)
        transformed_direction = self.A.T @ direction
        value, point = self.shape.support_function(transformed_direction)
        transformed_point = self.A @ point + self.b
        value += np.dot(self.b, direction)
        return value, transformed_point

    def get_boundary_points(self, num_points=128):
        boundary = self.shape.get_boundary_points(num_points)
        return (boundary @ self.A.T) + self.b

    def plot(self, ax1, ax2, **kwargs):
        fill_color = kwargs.pop('color', 'gray')
        fill_alpha = kwargs.pop('alpha', 0.3)

        points = self.get_boundary_points()
        try:
            hull = ConvexHull(points, qhull_options='QJ')
            hull_points = points[hull.vertices]
            ax1.add_patch(MPLPolygon(hull_points, closed=True, color=fill_color, alpha=fill_alpha))
            ax2.add_patch(MPLPolygon(hull_points, closed=True, color=fill_color, alpha=fill_alpha))
        except:
            ax1.scatter(points[:, 0], points[:, 1], color=fill_color, alpha=fill_alpha)
            ax2.scatter(points[:, 0], points[:, 1], color=fill_color, alpha=fill_alpha)

        num_dirs = 120
        angles = np.linspace(0, 2 * np.pi, num_dirs, endpoint=False)
        direction_set = [np.array([np.cos(a), np.sin(a)]) for a in angles]

        start = time.time()
        deformation_vals = deformation_function(self, direction_set)
        plot_deformation_points(ax1, direction_set, deformation_vals, self.center, color="purple", s=10)
        ax1.set_title('Deformation')
        ax1.set_aspect('equal')
        end = time.time()
        print(f"Час виконання: {end - start:.4f} секунд")

        start = time.time()
        minkowski_vals = minkowski_functional(self, direction_set)
        plot_minkowski_points(ax2, direction_set, minkowski_vals, self.center, color="orange", s=10)
        ax2.set_title('Minkowski Functional')
        ax2.set_aspect('equal')
        end = time.time()
        print(f"Час виконання matrix: {end - start:.4f} секунд")


class MinkowskiSum:
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2
        self.center = self.get_center()

    def get_center(self):
        print(self.shape1.get_center() + self.shape2.get_center())
        return self.shape1.get_center() + self.shape2.get_center()

    def support_function(self, direction):
        value1, point1 = self.shape1.support_function(direction)
        value2, point2 = self.shape2.support_function(direction)
        return value1 + value2, point1 + point2

    def compute_boundary_points(self, num_points=128):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = []

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            value, point = self.support_function(direction)
            points.append(point)

        return np.array(points)

    def plot(self, ax1, ax2, xlim=(-5, 5), ylim=(-5, 5), **kwargs):
        self.shape1.plot(ax1, ax2, color='blue', label='Shape 1')
        self.shape2.plot(ax1, ax2, color='red', label='Shape 2')

        boundary_points = self.compute_boundary_points()

        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault('fill', True)
        plot_kwargs.setdefault('alpha', 0.3)
        plot_kwargs.setdefault('color', 'green')
        plot_kwargs.setdefault('label', 'Minkowski Sum')

        try:
            hull = ConvexHull(boundary_points)
            hull_points = boundary_points[hull.vertices]
            ax1.add_patch(MPLPolygon(hull_points, **plot_kwargs))
            ax2.add_patch(MPLPolygon(hull_points, **plot_kwargs))
        except Exception as e:
            print(f"ConvexHull failed: {e}")
            ax1.scatter(boundary_points[:, 0], boundary_points[:, 1], color=plot_kwargs['color'], label=plot_kwargs['label'])
            ax2.scatter(boundary_points[:, 0], boundary_points[:, 1], color=plot_kwargs['color'], label=plot_kwargs['label'])

        num_dirs = 360
        angles = np.linspace(0, 2 * np.pi, num_dirs, endpoint=False)
        direction_set = [np.array([np.cos(a), np.sin(a)]) for a in angles]
        start = time.time()
        deformation_vals = deformation_function(self, direction_set)
        plot_deformation_points(ax1, direction_set, deformation_vals, self.center, color="blue", s=10)
        ax1.set_title('Deformation')
        ax1.set_aspect('equal')
        end = time.time()
        print(f"Час виконання сум: {end - start:.4f} секунд")

        start = time.time()
        minkowski_vals = minkowski_functional(self, direction_set)
        plot_minkowski_points(ax2, direction_set, minkowski_vals, self.center, color="red", s=10)
        ax2.set_title('Minkowski Functional')
        ax2.set_aspect('equal')
        end = time.time()
        print(f"Час виконання сум: {end - start:.4f} секунд")

        return boundary_points


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    circle = CircleShape((0, 0), 1.0)
    square = SquareShape((2, 2), 2.0)

    msum = MinkowskiSum(circle, square)
    msum.plot(ax1, ax2)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    base_shape = SquareShape((2, 2), 2.0)
    base_shape.plot(ax1, ax2, color='blue')

    A = np.array([[1, 2], [2, 1]])
    b = np.array([1, 0])
    transformed_shape = TransformedShape(base_shape, A, b)
    transformed_shape.plot(ax1, ax2)

    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 10)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    circle = CircleShape(center=(1, 1), radius=2)
    square = SquareShape(center=(2, 2), side_length=3)
    A = np.array([[1, 2], [2, 1]])
    A1 = np.array([[1, 2], [-1, -1]])
    b = np.array([0, 0])

    msum = MinkowskiSum(TransformedShape(circle, A, b), TransformedShape(square, A1, b))

    msum.plot(ax1, ax2, xlim=(0, 25), ylim=(-6, 12))
    plt.show()
