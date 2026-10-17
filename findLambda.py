import numpy as np
import sympy as sp


# Premenné funkcie
x, y = sp.symbols("x y", real=True)

# Vyber funkciu
function = (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Bod na vyhodnotenie Hessovej matice
x_point = 0.0
y_point = 0.0

# Hessova matica
H = sp.hessian(function, (x, y))

print("Symbolická Hessova matica:")
sp.pprint(H)

# Dosadenie bodu – funguje aj pri konštantnom Hessiane
H_point = H.subs({
    x: x_point,
    y: y_point
})

# Prevod na číselnú maticu
H_numeric = np.array(H_point.evalf(), dtype=float)

# Vlastné čísla symetrickej matice, zoradené vzostupne
eigenvalues = np.linalg.eigvalsh(H_numeric)
lambda_max = float(eigenvalues[-1])

print(f"\nBod: ({x_point}, {y_point})")
print("Hessova matica v tomto bode:")
print(H_numeric)

print(f"\nVlastné čísla: {eigenvalues}")
print(f"Najväčšie vlastné číslo: {lambda_max}")

# Hranicu kroku interpretujeme ako záruku len pre
# konštantný, kladne definitný Hessian.
if not H.free_symbols and np.all(eigenvalues > 0):
    alpha_limit = 2.0 / lambda_max

    print("\nHessian je konštantný a kladne definitný.")
    print(f"Podmienka pre pevný krok: 0 < alpha < {alpha_limit:.10f}")
else:
    print(
        "\nPozor: výsledok nie je zárukou bezpečného pevného kroku "
        "pre celý beh gradient descent."
    )

    if H.free_symbols:
        print(
            "Hessian závisí od súradníc. Vypočítaná lambda platí "
            "iba pre zadaný bod."
        )

    if not np.all(eigenvalues > 0):
        print(
            "Hessian v tomto bode nie je kladne definitný."
        )