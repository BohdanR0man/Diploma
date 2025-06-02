import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, distance_matrix
import time
import tracemalloc


class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def support_function(self, direction):
        direction = direction / np.linalg.norm(direction)
        return np.dot(direction, self.center) + self.radius

    def reference_point(self, direction):
        direction = direction / np.linalg.norm(direction)
        return self.center + self.radius * direction

    # def reference_plane(self, direction):
    #     point = self.reference_point(direction)
    #     normal = direction
    #     d = -np.dot(normal, point)
    #     return normal, d

    def reference_plane(self, direction):
        direction = direction / np.linalg.norm(direction)
        normal = direction
        h_value = self.support_function(direction)
        d = -h_value
        return normal, d

    def generate_sphere_points(self, num_points=100):
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)

        sphere_points = []
        for theta in u:
            for phi in v:
                x = self.radius * np.sin(phi) * np.cos(theta) + self.center[0]
                y = self.radius * np.sin(phi) * np.sin(theta) + self.center[1]
                z = self.radius * np.cos(phi) + self.center[2]
                sphere_points.append([x, y, z])

        return np.array(sphere_points)


class Cube:
    def __init__(self, center, side_length):
        self.center = np.array(center)
        self.side_length = side_length
        self.half_side = side_length / 2

    def support_function(self, direction):
        direction = direction / np.linalg.norm(direction)

        max_dot = float('-inf')

        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertex = self.center + np.array([x, y, z]) * self.half_side
                    dot_product = np.dot(direction, vertex)
                    if dot_product > max_dot:
                        max_dot = dot_product

        return max_dot

    def reference_point(self, direction):
        direction = direction / np.linalg.norm(direction)

        max_dot = float('-inf')
        max_vertex = None

        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertex = self.center + np.array([x, y, z]) * self.half_side
                    dot_product = np.dot(direction, vertex)
                    if dot_product > max_dot:
                        max_dot = dot_product
                        max_vertex = vertex

        return max_vertex

    # def reference_plane(self, direction):
    #     point = self.reference_point(direction)
    #     normal = direction
    #     d = -np.dot(normal, point)
    #     return normal, d

    def reference_plane(self, direction):
        direction = direction / np.linalg.norm(direction)
        normal = direction
        h_value = self.support_function(direction)
        d = -h_value
        return normal, d

    def generate_cube_points(self, num_points=60):
        num_points_per_face = int(num_points / 6)
        points = []
        r = self.side_length / 2
        lin = np.linspace(-r, r, num_points_per_face)

        for u in lin:
            for v in lin:
                points.append([self.center[0] + r, self.center[1] + u, self.center[2] + v])
                points.append([self.center[0] - r, self.center[1] + u, self.center[2] + v])

                points.append([self.center[0] + u, self.center[1] + r, self.center[2] + v])
                points.append([self.center[0] + u, self.center[1] - r, self.center[2] + v])

                points.append([self.center[0] + u, self.center[1] + v, self.center[2] + r])
                points.append([self.center[0] + u, self.center[1] + v, self.center[2] - r])

        return np.array(points)


class SumFigure:
    def __init__(self, *figures):
        self.figures = figures

    def support_function(self, direction):
        direction = direction / np.linalg.norm(direction)
        return sum(fig.support_function(direction) for fig in self.figures)

    def reference_plane(self, direction):
        direction = direction / np.linalg.norm(direction)
        normal = direction
        h_value = self.support_function(direction)
        d = -h_value
        return normal, d

    def generate_shifted_points(self, directions, num_points=100):
        F_points = self.figures[0].generate_sphere_points(num_points)
        G_points = self.figures[1].generate_cube_points(num_points)

        sum_points = np.array([fp + gp for fp in F_points for gp in G_points])
        return sum_points



def generate_directions(n_points):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack((x, y, z), axis=1)

def find_plane_intersection(plane1, plane2, plane3):
    normals = np.array([plane1[0], plane2[0], plane3[0]])
    d_values = np.array([-plane1[1], -plane2[1], -plane3[1]])

    if np.linalg.det(normals) != 0:
        return np.linalg.solve(normals, d_values)
    return None

def hausdorff_distance(set1, set2):
    d_matrix1 = distance_matrix(set1, set2)
    d_matrix2 = distance_matrix(set2, set1)
    return max(np.max(np.min(d_matrix1, axis=1)), np.max(np.min(d_matrix2, axis=1)))


class ReferencePlanes:
    def __init__(self, figure, n_planes=10):
        self.figure = figure
        self.directions = generate_directions(n_planes)
        self.planes = self.reference_planes()
        # self.reference_points = self.get_reference_points()

    def reference_planes(self):
        return [self.figure.reference_plane(d) for d in self.directions]

    # def get_reference_points(self):
    #     return np.array([self.figure.reference_point(d) for d in self.directions])

class Approximate:
    def __init__(self, reference_planes):
        self.planes = reference_planes.planes
        self.directions = reference_planes.directions
        # self.reference_points = reference_planes.reference_points
        self.neighbors = self.find_neighbors()
        self.intersections = self.compute_intersections()
        self.hull = self.build_hull()

    def find_neighbors(self):
        hull = ConvexHull(self.directions)
        return [tuple(sorted(simplex)) for simplex in hull.simplices]

    def compute_intersections(self):
        intersections = []
        for i1, i2, i3 in self.neighbors:
            p1, p2, p3 = self.planes[i1], self.planes[i2], self.planes[i3]
            point = find_plane_intersection(p1, p2, p3)
            if point is not None:
                intersections.append(point)
        return np.array(intersections)

    def build_hull(self):
        if len(self.intersections) > 3:
            try:
                return ConvexHull(self.intersections)
            except Exception as e:
                print(f"Ошибка ConvexHull: {e}")
        return None


def plot_approximation(approx):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(
    #     approx.reference_points[:, 0],
    #     approx.reference_points[:, 1],
    #     approx.reference_points[:, 2],
    #     color='g', s=50, label='Reference Points'
    # )

    if len(approx.intersections) > 0:
        ax.scatter(
            approx.intersections[:, 0],
            approx.intersections[:, 1],
            approx.intersections[:, 2],
            color='r', s=20, label='Plane Intersections'
        )

    # cube = Cube(center=[3, 2, 1], side_length=10)
    # cube_pts = cube.generate_cube_points(100)
    # sphere = Sphere(center=[0, 0, 0], radius=5)
    # sphere_pts = sphere.generate_sphere_points(100)
    #
    # sum_figure = SumFigure(sphere, cube)
    # sum_pts = sum_figure.generate_shifted_points(generate_directions(300), num_points=20)
    # ax.scatter(sphere_pts[:, 0], sphere_pts[:, 1], sphere_pts[:, 2], color='green', s=5, alpha=0.5,
    #            label='Сумма (смещённые сферы)')

    # Plot the approximation faces
    if approx.hull:
        for simplex in approx.hull.simplices:
            poly3d = [[approx.intersections[i] for i in simplex]]
            ax.add_collection3d(
                Poly3DCollection(poly3d, facecolors='r', linewidths=1, edgecolors='k', alpha=0.3)
            )

    limit = np.max(np.abs(approx.intersections)) * 1.2
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()


def main_sphere():
    print("Построение сферы...")
    sphere = Sphere(center=[0, 0, 0], radius=5)
    reference_planes = ReferencePlanes(sphere, n_planes=300)
    approx = Approximate(reference_planes)
    plot_approximation(approx)
    print(f"Количество точек пересечения: {len(approx.intersections)}")

    origin_points = sphere.generate_sphere_points(num_points=100)
    haus_dist = hausdorff_distance(origin_points, approx.intersections)
    print(f"Hausdorff расстояние (сфера): {haus_dist:.4f}")

def main_cube():
    print("Построение куба...")
    cube = Cube(center=[2, 2, 2], side_length=10)
    reference_planes = ReferencePlanes(cube, n_planes=300)
    approx = Approximate(reference_planes)
    plot_approximation(approx)
    print(f"Количество точек пересечения: {len(approx.intersections)}")

    origin_points = cube.generate_cube_points(100)
    haus_dist = hausdorff_distance(origin_points, approx.intersections)
    print(f"Hausdorff расстояние (сфера): {haus_dist:.4f}")

def main_sum_sphere_cube():
    print("Построение суммы сферы и куба...")
    sphere = Sphere(center=[0, 0, 0], radius=5)
    cube = Cube(center=[3, 2, 1], side_length=10)
    sum_figure = SumFigure(sphere, cube)
    reference_planes = ReferencePlanes(sum_figure, n_planes=300)
    approx = Approximate(reference_planes)
    plot_approximation(approx)

    origin_points = sum_figure.generate_shifted_points(generate_directions(300), num_points=20)
    haus_dist = hausdorff_distance(origin_points, approx.intersections)
    print(f"Hausdorff расстояние (сумма): {haus_dist:.4f}")


if __name__ == "__main__":
    tracemalloc.start()
    start = time.time()
    main_sphere()
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Час виконання: {end - start:.4f} секунд")
    print(f"Використано пам’яті: {current / 1024:.2f} KB (пік: {peak / 1024:.2f} KB)")
    tracemalloc.start()
    start = time.time()
    main_cube()
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Час виконання: {end - start:.4f} секунд")
    print(f"Використано пам’яті: {current / 1024:.2f} KB (пік: {peak / 1024:.2f} KB)")
    tracemalloc.start()
    start = time.time()
    main_sum_sphere_cube()
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Час виконання: {end - start:.4f} секунд")
    print(f"Використано пам’яті: {current / 1024:.2f} KB (пік: {peak / 1024:.2f} KB)")


