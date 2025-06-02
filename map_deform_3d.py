import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm


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

            support_val, *_ = shape.support_function(psi)
            support_val -= np.dot(psi, shape.center)
            lambda_curr = support_val / dot

            if lambda_curr > 0 and lambda_curr < lambda_max:
                lambda_max = lambda_curr

        deformation_values.append(lambda_max)
    return deformation_values


def generate_directions_3d(num_directions):
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(num_directions)))
    theta = np.linspace(0, np.pi, int(np.sqrt(num_directions)))

    directions = []
    for p in phi:
        for t in theta:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            directions.append([x, y, z])

    return np.array(directions)


class Deform3D:
    def __init__(self, center, support_function):
        self.center = np.array(center)
        self.support_function = support_function


class Reachability3D:
    def __init__(self, A, x0_center, radius_func, F_radius, time_span):
        self.A = np.array(A)
        self.x0_center = np.array(x0_center)
        self.radius_func = radius_func
        self.F_radius = F_radius
        self.t0, self.t1 = time_span

    def support_function_X0(self, t, psi):
        psi = psi / np.linalg.norm(psi)

        max_dot = float('-inf')

        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertex = self.x0_center + np.array([x, y, z]) * (self.radius_func(t) / 2)
                    dot_product = np.dot(psi, vertex)
                    if dot_product > max_dot:
                        max_dot = dot_product

        return max_dot

    def support_function_F(self, s, psi):
        F_center = np.array([10, 10, 10])
        return np.dot(F_center, psi) + self.F_radius * np.linalg.norm(psi)

    def support_function_R(self, t, psi):
        exp1 = expm(self.A * (t - self.t0))
        psi1 = exp1.T @ psi
        part1 = self.support_function_X0(self.t0, psi1)

        def integrand(s):
            exp2 = expm(self.A * (t - s))
            psi2 = exp2.T @ psi
            return self.support_function_F(s, psi2)

        part2, _ = quad(integrand, self.t0, t)
        return part1 + part2

    def center_R(self, t):
        expA = expm(self.A * (t - self.t0))
        term1 = expA @ self.x0_center
        F_center = np.array([10, 10, 10])

        def integrand(s):
            return expm(self.A * (t - s)) @ F_center

        term2 = quad_vec(integrand, self.t0, t)[0]
        return term1 + term2

    def visualize(self):

        directions = generate_directions_3d(500)
        time_points = np.linspace(self.t0, self.t1, 7)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

        for i, t in enumerate(time_points):
            center = self.center_R(t)
            print(f"t={t:.2f}: center = {center}")
            support_vals = {}
            for psi in directions:
                support_vals[tuple(psi)] = self.support_function_R(t, psi)

            shape = Deform3D(
                center=self.center_R(t),
                support_function=lambda psi: (support_vals[tuple(psi)], None)
            )

            deformations = deformation_function(shape, directions)

            boundary_points = []
            for deformation, psi in zip(deformations, directions):
                if deformation != float('inf') and deformation > 0:
                    point = deformation * np.array(psi) + self.center_R(t)
                    boundary_points.append(point)

            if boundary_points:
                boundary = np.array(boundary_points)
                ax.scatter(boundary[:, 0], boundary[:, 1], boundary[:, 2],
                           c=[colors[i]], alpha=0.6, s=20, label=f't={t:.2f}')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim([-2, 22])
        ax.set_ylim([-2, 22])
        ax.set_zlim([-2, 22])

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 10:
            step = len(handles) // 5
            ax.legend(handles[::step], labels[::step])
        else:
            ax.legend()
        plt.tight_layout()
        plt.show()


def time_dependent_radius(t):
    return 1


def main():
    A = np.array([
        [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
        [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
        [0, 0, 1]
    ])

    x0_center = [0, 0, 0]
    F_radius = 1
    t_span = [0, 1]

    system = Reachability3D(A, x0_center, time_dependent_radius, F_radius, t_span)

    system.visualize()


if __name__ == "__main__":
    main()
