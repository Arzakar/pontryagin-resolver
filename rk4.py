import numpy as np
import math as m



def rk4_step(equations: np.ndarray, current_state: np.ndarray, current_dif_var: float, step: float, constants: dict, nonstdvars: dict) -> np.ndarray:
    eq = equations
    y = current_state
    t = current_dif_var
    h = step
    k = np.zeros(shape=(eq.size, 4))

    # j - порядок по методу РК4
    for j in range(4):
        # i - порядковый номер уравнения в системе
        for i in range(eq.size):
            if j == 0:
                k[i, j] = h * eq[i](t, y, constants, nonstdvars)
                continue
            
            # Определяем коэффициент параметров
            # Для 2 и 3 порядка - половина шага
            # Для четвёрого порядка - полный шаг)
            coeff = 0.5 if j in (1, 2) else 1

            t_j = t + h * coeff
            y_j = y + k[:, j - 1] * coeff

            k[i, j] = h * eq[i](t_j, y_j, constants, nonstdvars)

    dy = (k[:, 0] + 2 * k[:, 1] + 2 * k[:, 2] + k[:, 3]) / 6

    return y + dy