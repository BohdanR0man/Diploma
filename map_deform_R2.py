import numpy as np
import matplotlib.pyplot as plt
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

            support_val, _ = shape.support_function(psi)
            support_val -= np.dot(psi, shape.center)
            lambda_curr = support_val / dot

            if lambda_curr > 0 and lambda_curr < lambda_max:
                lambda_max = lambda_curr

        deformation_values.append(lambda_max)
    return deformation_values

def generate_directions(n_dim, num_directions):
    angles = np.linspace(0, 2 * np.pi, num_directions)
    return np.array([[np.cos(a), np.sin(a)] for a in angles])

class Deform:
    def __init__(self, center, support_function):
        self.center = np.array(center)
        self.support_function = support_function

class Reachability3D:
    def __init__(self, A, x0_center, x0_radius, F_radius, time_span):
        self.A = np.array(A)
        self.X0_center = np.array(x0_center)
        self.X0_radius = x0_radius
        self.F_radius = F_radius
        self.t0, self.t1 = time_span

    def support_function_X0(self, psi):
        return np.dot(self.X0_center, psi) + self.X0_radius * np.linalg.norm(psi)

    def support_function_F(self, s, psi):
        F_center = np.array([np.sin(s/2), np.cos(s/2)])
        return np.dot(F_center, psi) + self.F_radius * np.linalg.norm(psi)

    def support_function_R(self, t, psi):
        exp1 = expm(self.A * (t - self.t0))
        psi1 = exp1.T @ psi
        part1 = self.support_function_X0(psi1)

        def integrand(s):
            exp2 = expm(self.A * (t - s))
            psi2 = exp2.T @ psi
            return self.support_function_F(s, psi2)

        part2, _ = quad(integrand, self.t0, t)
        return part1 + part2

    def center_R(self, t):
        expA = expm(self.A * (t - self.t0))
        term1 = expA @ self.X0_center
        F_center = np.array([np.sin(t/2), np.cos(t/2)])

        def integrand(s):
            return expm(self.A * (t - s)) @ F_center

        term2 = quad_vec(integrand, self.t0, t)[0]
        return term1 + term2

    def visualize(self):
        directions = generate_directions(2, 100)
        time_points = np.linspace(self.t0, self.t1, 15)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for t in time_points:
            support_vals = {
                tuple(psi): self.support_function_R(t, psi)
                for psi in directions
            }

            shape = Deform(
                center=self.center_R(t),
                support_function=lambda psi: (support_vals[tuple(psi)], None)
            )

            deformations = deformation_function(shape, directions)
            boundary = np.array([deformation * psi for deformation, psi in zip(deformations, directions)])

            x, y = boundary[:, 0], boundary[:, 1]
            z = np.full_like(x, t)
            ax.plot(z, x, y, alpha=0.7)

        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
        plt.tight_layout()
        plt.show()

def main():
    A = [[0, 1], [-1, 0]]
    x0_center = [0, 0]
    x0_radius = 1.0
    F_radius = 1.0
    t_span = [0, 2 * np.pi]

    system = Reachability3D(A, x0_center, x0_radius, F_radius, t_span)
    system.visualize()

if __name__ == "__main__":
    main()
