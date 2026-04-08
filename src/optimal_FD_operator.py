import numpy as np
from scipy.optimize import minimize

def initial_taylor_coeffs():
    # 4th-order staggered central difference initial guess
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    c3 = 0.0
    return np.array([c1, c2, c3], dtype=np.float64)

def discrete_symbol(kappa, c):
    c1, c2, c3 = c
    return 2.0 * (
        c1 * np.sin(0.5 * kappa) +
        c2 * np.sin(1.5 * kappa) +
        c3 * np.sin(2.5 * kappa)
    )

def dispersion_error(c, kappa_max, n_kappa=2000, weight_power=2):
    kappas = np.linspace(1e-8, kappa_max, n_kappa)
    D = discrete_symbol(kappas, c)
    rel = D / kappas - 1.0
    w = (kappas / kappa_max) ** weight_power
    return np.trapezoid(w * rel**2, kappas)

def low_k_constraints(c):
    c1, c2, c3 = c
    return (c1 + 3.0 * c2 + 5.0 * c3) - 1.0

def optimize_c(lambda_min, dx, n_kappa=2000, weight_power=2):
    kappa_max = 2.0 * np.pi * dx / lambda_min
    c0 = initial_taylor_coeffs()

    cons = ({
        'type': 'eq',
        'fun': low_k_constraints
    },)

    res = minimize(
        dispersion_error,
        c0,
        args=(kappa_max, n_kappa, weight_power),
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-14}
    )
    return c0, res, kappa_max

def check_res(c, lambda_min, dx, show_plot=False):
    kappa_max = 2.0 * np.pi * dx / lambda_min
    kappas = np.linspace(1e-8, kappa_max, 1000)
    D = discrete_symbol(kappas, c)
    rel = D / kappas - 1.0

    # 1. D(kappa)/kappa が 1 に近いか（最大誤差・平均誤差）
    max_abs_rel = np.max(np.abs(rel))
    mean_abs_rel = np.mean(np.abs(rel))
    rms_rel = np.sqrt(np.mean(rel**2))

    print("check_res: kappa_max = %g" % kappa_max)
    print("check_res: max |D/kappa - 1|  = %g" % max_abs_rel)
    print("check_res: mean |D/kappa - 1| = %g" % mean_abs_rel)
    print("check_res: rms |D/kappa - 1|  = %g" % rms_rel)

    # 2. kappa → 0 で一次微分条件を満たすか
    # 低波数 Taylor 展開を見て、係数の和が 1 に近いかチェック
    sum_coef = c[0] + 3.0 * c[1] + 5.0 * c[2]
    print("check_res: low-k sum c1+3c2+5c3 = %g (should be 1.0)" % sum_coef)

    # 3. 1e-8 での D/kappa の値
    kappa_small = 1e-8
    D_small = discrete_symbol(kappa_small, c)
    rel_small = D_small / kappa_small - 1.0
    print("check_res: D/kappa - 1 at kappa=1e-8 = %g" % rel_small)

    # 4. 描画したい場合
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))

        plt.subplot(1, 2, 1)
        plt.plot(kappas, D / kappas, label=r'$D(\kappa)/\kappa$')
        plt.axhline(1.0, color='k', ls='--', lw=1, label='ideal')
        plt.xlabel(r'$\kappa = k\Delta x$')
        plt.ylabel(r'$D(\kappa)/\kappa$')
        plt.title('dispersion measure')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(kappas, np.abs(rel), label=r'$|D/kappa - 1|$')
        plt.xlabel(r'$\kappa = k\Delta x$')
        plt.ylabel('abs dispersion error')
        plt.title('dispersion error')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return dict(
        kappa_max=kappa_max,
        max_abs_rel=max_abs_rel,
        mean_abs_rel=mean_abs_rel,
        rms_rel=rms_rel,
        low_k_sum=sum_coef,
        rel_small=rel_small
    )

if __name__ == "__main__":
    lambda_min = 1000.0  # m
    dx = 200.0           # m, dy = dx (正方形格子)

    c0, res, kappa_max = optimize_c(lambda_min, dx)

    # print("Initial guess (4th-order Taylor):", c0)

    print("Optimization:")
    print("kappa_max =", kappa_max)
    print("success =", res.success)
    print("coeffs =", res.x)
    print("objective =", res.fun)

    print("\nChecking:")
    stats = check_res(res.x, lambda_min, dx, show_plot=False)

    # 可視化も見たい場合は show_plot=True
    # check_res(res.x, lambda_min, dx, show_plot=True)