import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# Step 1: user input
x = sp.symbols('x')
func_str = input("Enter function in x: ")  # e.g., "x**2 - 2" or "exp(x) - 2"
f_expr = sp.sympify(func_str)
f_prime_expr = sp.diff(f_expr, x)

# Step 2: turn into callable functions
f = sp.lambdify(x, f_expr, "numpy")
f_prime = sp.lambdify(x, f_prime_expr, "numpy")

# Step 3: Newton-Raphson iteration
x0 = float(input("Enter initial guess: "))  # e.g., 1.0
epsilon = float(input("Enter epsilon value : "))#1e-6
max_iter = 20
guesses = [x0]
converged = False

print("\nIteration Table")
print("{:<5} {:<15} {:<15} {:<15}".format("n", "x_n", "f(x_n)", "error"))

for i in range(max_iter):
    x_n = guesses[-1]
    y_n = f(x_n)

    # Check for division by zero
    if abs(f_prime(x_n)) < epsilon:
        print("Derivative is too close to zero. Stopping iteration.")
        break

    x_next = x_n - y_n/f_prime(x_n)
    error = abs(x_next - x_n)

    print("{:<5} {:<15.8f} {:<15.8f} {:<15.8f}".format(i, x_n, y_n, error))

    guesses.append(x_next)
    if error < epsilon:
        converged = True
        print(f"Converged after {i+1} iterations")
        break

if not converged and i == max_iter-1:
    print("Maximum iterations reached without convergence")

## Animation codes are start from here
# Step 4: visualization with animation
x_min = min(guesses) - 2
x_max = max(guesses) + 2
x_vals = np.linspace(x_min, x_max, 400)
y_vals = f(x_vals)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_vals, 'b', label="f(x)")
ax.axhline(0, color='black', linewidth=0.8)

# Plot elements
line, = ax.plot([], [], 'r-', lw=2, label="Tangent")   # tangent line
point, = ax.plot([], [], 'go', ms=8, label="Current point")        # current point
next_point, = ax.plot([], [], 'ro', ms=8, label="Next estimate")   # next point
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha="left", va="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
ax.legend()
ax.grid(True)
ax.set_title("Newton-Raphson Method Visualization")

# Set initial axis limits
y_range = max(y_vals) - min(y_vals)
ax.set_xlim(x_min, x_max)
ax.set_ylim(min(y_vals) - 0.1*y_range, max(y_vals) + 0.1*y_range)

def update(i):
    if i >= len(guesses)-1:
        # Final state - show the root
        root = guesses[-1]
        text.set_text(f"Converged!\nRoot: {root:.8f}\nf({root:.8f}) = {f(root):.2e}")
        return line, point, next_point, text

    x_n = guesses[i]
    y_n = f(x_n)
    slope = f_prime(x_n)

    # Calculate next point
    x_next = guesses[i+1]

    # Create tangent line
    tangent_x = np.linspace(x_n-1.5, x_n+1.5, 50)
    tangent_y = y_n + slope*(tangent_x - x_n)

    # Update plot elements
    line.set_data(tangent_x, tangent_y)
    point.set_data([x_n], [y_n])
    next_point.set_data([x_next], [0])
    text.set_text(f"Iteration {i}\nx = {x_n:.6f}\nf(x) = {y_n:.6f}\nNext: {x_next:.6f}")

    return line, point, next_point, text

# Create animation
ani = FuncAnimation(fig, update, frames=len(guesses), blit=False, repeat=False, interval=1500)

# Display in Colab - this is the key for Colab
plt.close(fig)  # Prevents duplicate display
display(HTML(ani.to_jshtml()))

print("Animation is displaying above. You can use the controls to play, pause, or step through the iterations.")

