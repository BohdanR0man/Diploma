import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm

class SphereShape:
    def __init__(self, center, radius):
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

class CubeShape:
    def __init__(self, center, side_length):
        self.center = np.array(center)
        self.side_length = side_length

    def get_center(self):
        return self.center

    def support_function(self, direction):
        direction = np.array(direction)
        half_side = self.side_length / 2
        max_dot = float('-inf')
        support_point = None

        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertex = self.center + np.array([x, y, z]) * half_side
                    dot_product = np.dot(direction, vertex)
                    if dot_product > max_dot:
                        max_dot = dot_product
                        support_point = vertex

        return max_dot, support_point

class MinkowskiSum:
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2
        self.center = self.get_center()

    def get_center(self):
        return self.shape1.get_center() + self.shape2.get_center()

    def support_function(self, direction):
        value1, point1 = self.shape1.support_function(direction)
        value2, point2 = self.shape2.support_function(direction)
        return value1 + value2, point1 + point2


def generate_directions(num_directions):
    phi = np.linspace(0, np.pi, num_directions)
    theta = np.linspace(0, 2 * np.pi, num_directions)
    directions = []
    for p in phi:
        for t in theta:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            directions.append([x, y, z])
    return np.array(directions)

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
            support_val -= np.dot(psi, shape.get_center())
            lambda_curr = support_val / dot
            if 0 < lambda_curr < lambda_max:
                lambda_max = lambda_curr
        deformation_values.append(lambda_max)
    return deformation_values


class Deform:
    def __init__(self, center, support_function):
        self.center = np.array(center)
        self.support_function = support_function

    def get_center(self):
        return self.center


class Reachability3D:
    def __init__(self, A, x0_shape_A, F_radius_A, B, x0_shape_B, F_radius_B, time_span):
        self.A = np.array(A)
        self.X0_shape_A = x0_shape_A
        self.F_radius_A = F_radius_A
        self.B = np.array(B)
        self.X0_shape_B = x0_shape_B
        self.F_radius_B = F_radius_B
        self.t0, self.t1 = time_span

    def support_function_X0_A(self, psi):
        return self.X0_shape_A.support_function(psi)[0]

    def support_function_X0_B(self, psi):
        return self.X0_shape_B.support_function(psi)[0]

    def support_function_F_A(self, s, psi):
        F_center_A = np.array([10, 10, 10])
        return np.dot(F_center_A, psi) + self.F_radius_A * np.linalg.norm(psi)

    def support_function_F_B(self, s, psi):
        F_center_B = np.array([13, 13, 13])
        return np.dot(F_center_B, psi) + self.F_radius_B * np.linalg.norm(psi)

    def support_function_R_A(self, t, psi):
        exp1 = expm(self.A * (t - self.t0))
        psi1 = exp1.T @ psi
        part1 = self.support_function_X0_A(psi1)

        def integrand(s):
            exp2 = expm(self.A * (t - s))
            psi2 = exp2.T @ psi
            return self.support_function_F_A(s, psi2)

        part2, _ = quad(integrand, self.t0, t)
        return part1 + part2

    def support_function_R_B(self, t, psi):
        exp1 = expm(self.B * (t - self.t0))
        psi1 = exp1.T @ psi
        part1 = self.support_function_X0_B(psi1)

        def integrand(s):
            exp2 = expm(self.B * (t - s))
            psi2 = exp2.T @ psi
            return self.support_function_F_B(s, psi2)

        part2, _ = quad(integrand, self.t0, t)
        return part1 + part2

    def center_R_A(self, t):
        expA = expm(self.A * (t - self.t0))
        x0_center = self.X0_shape_A.get_center()
        term1 = expA @ x0_center
        F_center_A = np.array([10, 10, 10])

        def integrand(s):
            return expm(self.A * (t - s)) @ F_center_A

        term2 = quad_vec(integrand, self.t0, t)[0]
        return term1 + term2

    def center_R_B(self, t):
        expB = expm(self.B * (t - self.t0))
        x0_center = self.X0_shape_B.get_center()
        term1 = expB @ x0_center
        F_center_B = np.array([13, 13, 13])

        def integrand(s):
            return expm(self.B * (t - s)) @ F_center_B

        term2 = quad_vec(integrand, self.t0, t)[0]
        return term1 + term2

    def visualize(self):
        directions = generate_directions(20)
        time_points = np.linspace(self.t0, self.t1, 10)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.viridis(np.linspace(1, 0, len(time_points)))
        for i, t in enumerate(time_points):
            support_vals_A = {
                tuple(psi): self.support_function_R_A(t, psi) for psi in directions
            }

            support_vals_B = {
                tuple(psi): self.support_function_R_B(t, psi) for psi in directions
            }

            shape_A = Deform(
                center=self.center_R_A(t),
                support_function=lambda psi: (support_vals_A[tuple(psi)], None)
            )

            shape_B = Deform(
                center=self.center_R_B(t),
                support_function=lambda psi: (support_vals_B[tuple(psi)], None)
            )

            deformations_A = deformation_function(shape_A, directions)
            boundary_A = np.array([deformation_A * psi + self.center_R_A(t) for deformation_A, psi in zip(deformations_A, directions)])
            x_A, y_A, z_A = boundary_A[:, 0], boundary_A[:, 1], boundary_A[:, 2]
            t_A = np.full_like(x_A, t)
            ax.plot(x_A, y_A, z_A, color="lime", alpha = 0.4)

            deformations_B = deformation_function(shape_B, directions)
            boundary_B = np.array([deformation_B * psi + shape_B.get_center() for deformation_B, psi in zip(deformations_B, directions)])
            x_B, y_B, z_B = boundary_B[:, 0], boundary_B[:, 1], boundary_B[:, 2]
            t_B = np.full_like(x_B, t)
            ax.scatter(x_B, y_B, z_B, color=[colors[i]], alpha = 0.7, label=f't={t:.2f}')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim([-2, 35])
        ax.set_ylim([-2, 35])
        ax.set_zlim([-2, 35])

        ax.legend()

        plt.tight_layout()
        plt.show()

        # def visualize(self):
        #     from matplotlib.animation import FuncAnimation
        #
        #     directions = generate_directions(20)
        #     time_points = np.linspace(self.t0, self.t1, 10)
        #
        #     fig = plt.figure(figsize=(10, 8))
        #     ax = fig.add_subplot(111, projection='3d')
        #
        #     def animate(frame):
        #         ax.clear()
        #         t = time_points[frame]
        #
        #         support_vals_A = {
        #             tuple(psi): self.support_function_R_A(t, psi) for psi in directions
        #         }
        #
        #         support_vals_B = {
        #             tuple(psi): self.support_function_R_B(t, psi) for psi in directions
        #         }
        #
        #         shape_A = Deform(
        #             center=self.center_R_A(t),
        #             support_function=lambda psi: (support_vals_A[tuple(psi)], None)
        #         )
        #
        #         shape_B = Deform(
        #             center=self.center_R_B(t),
        #             support_function=lambda psi: (support_vals_B[tuple(psi)], None)
        #         )
        #
        #         deformations_A = deformation_function(shape_A, directions)
        #         boundary_A = np.array([deformation_A * psi + shape_A.get_center()
        #                                for deformation_A, psi in zip(deformations_A, directions)])
        #
        #         deformations_B = deformation_function(shape_B, directions)
        #         boundary_B = np.array([deformation_B * psi + shape_B.get_center()
        #                                for deformation_B, psi in zip(deformations_B, directions)])
        #
        #
        #         ax.scatter(boundary_A[:, 0], boundary_A[:, 1], boundary_A[:, 2], color="green", alpha=0.7, s=50, label='Shape A')
        #         ax.plot(boundary_B[:, 0], boundary_B[:, 1], boundary_B[:, 2], color="red", alpha=0.7, linewidth=3, label='Shape B')
        #
        #         ax.set_xlabel("x1")
        #         ax.set_ylabel("x2")
        #         ax.set_zlabel("x3")
        #         ax.set_title(f"Time evolution: t = {t:.3f}")
        #         ax.legend()
        #
        #         ax.set_xlim([-50, 50])
        #         ax.set_ylim([-50, 50])
        #         ax.set_zlim([-50, 50])
        #
        #     anim = FuncAnimation(fig, animate, frames=len(time_points),
        #                          interval=100, repeat=True)
        #     plt.show()
        #     return anim


    def check_info(self, directions, t):
        support_A_vals = []
        support_B_vals = []

        for psi in directions:
            support_A = self.support_function_R_A(t, psi)
            support_B = self.support_function_R_B(t, psi)
            support_A_vals.append(support_A)
            support_B_vals.append(support_B)

        intersect = True
        A_contains_B = True
        B_contains_A = True

        for i, psi in enumerate(directions):
            diff_AB = support_A_vals[i] + self.support_function_R_B(t, -psi)
            if diff_AB < 0:
                intersect = False

            if support_B_vals[i] > support_A_vals[i]:
                A_contains_B = False
            if support_A_vals[i] > support_B_vals[i]:
                B_contains_A = False

        return intersect, A_contains_B, B_contains_A

    def results_info(self, directions):
        time_points = np.linspace(self.t0, self.t1, 10)
        intersect_happened = False
        A_contains_B_happened = False
        B_contains_A_happened = False

        for t in time_points:
            intersect, A_contains_B, B_contains_A = self.check_info(directions, t)

            if not intersect_happened and intersect:
                print(f"Перший дотик на момент часу: t = {t:.4f}")
                intersect_happened = True

            if not A_contains_B_happened and A_contains_B:
                print(f"Множина A повністю поглинає B у момент: t = {t:.4f}")
                A_contains_B_happened = True

            if not B_contains_A_happened and B_contains_A:
                print(f"Множина B повністю поглинає A у момент: t = {t:.4f}")
                B_contains_A_happened = True


def main():
    A = np.array([
        [np.cos(np.pi / 6), np.sin(np.pi / 6), 0],
        [-np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
        [0, 0, 1]
    ])
    B = np.array([
        [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
        [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
        [0, 0, 1]
    ])
    sphere1 = SphereShape(center=[0, 9, 1.5], radius=1.0)
    sphere2 = SphereShape(center=[6, 0, 0], radius=1)
    cube = CubeShape(center=[0, 0, 0], side_length=1.0)
    mink_sum = MinkowskiSum(sphere2, cube)

    F_radius_A = 1.0
    F_radius_B = 1.0
    t_span = [0, 1]

    system = Reachability3D(A, sphere1, F_radius_A, B, mink_sum, F_radius_B, t_span)
    system.visualize()

    directions = generate_directions(20)
    system.results_info(directions)

if __name__ == "__main__":
    main()
