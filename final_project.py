import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, bisect
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

# Task 1: Graphical Method and Absolute Error
def task_1():
    def f(x_val):
        try:
            # Safely evaluate the user-defined function with x replaced by x_val
            return eval(user_function, {"x": x_val, "np": np})
        except Exception as e:
            result_label.config(text=f"Invalid function. Error: {e}")
            return None

    # Get user inputs
    try:
        start = float(entry_start.get())
        end = float(entry_end.get())
        user_function = entry_function.get().strip()
    except ValueError:
        result_label.config(text="Invalid input. Please enter numeric values for the range.")
        return

    if not user_function:
        result_label.config(text="Please enter a valid function.")
        return

    # Generate x values within the specified range
    x_vals = np.linspace(start, end, 500)

    try:
        # Evaluate the function for all x values
        y_vals = [f(x_val) for x_val in x_vals]
    except Exception as e:
        result_label.config(text=f"Error evaluating function: {e}")
        return

    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {user_function}")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Graphical Method")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    # Approximate root using graphical method
    try:
        approximate_root = fsolve(lambda x: f(x), (start + end) / 2)[0]  # Initial guess = midpoint
        plt.scatter(approximate_root, f(approximate_root), color='red',
                    label=f"Approx Root: {approximate_root:.4f}")
        plt.legend()
        plt.show()

        # Calculate absolute error
        exact_root = fsolve(lambda x: f(x), (start + end) / 2)[0]  # Using fsolve as the "exact" root
        abs_error = abs(exact_root - approximate_root)
        result_label.config(text=f"Approx Root: {approximate_root:.4f}, Abs Error: {abs_error:.4e}")
    except Exception as e:
        result_label.config(text=f"Error finding root: {e}")


# Task 2: Comparison of Root-Finding Methods
def task_2():
    def f(x):
        try:
            # Safely evaluate the user-defined function with x replaced by x_val
            return eval(user_function, {"x": x, "np": np})
        except Exception as e:
            result_label.config(text=f"Invalid function. Error: {e}")
            return None

    # Bisection Method
    def bisection(a, b, tol=1e-6, max_iter=100):
        iterations = 0
        while (b - a) / 2 > tol and iterations < max_iter:
            c = (a + b) / 2
            if f(c) == 0:
                return c, iterations
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
            iterations += 1
        return (a + b) / 2, iterations

    # Secant Method
    def secant(x0, x1, tol=1e-6, max_iter=100):
        iterations = 0
        while abs(x1 - x0) > tol and iterations < max_iter:
            x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            x0, x1 = x1, x2
            iterations += 1
        return x1, iterations

    # Get user inputs
    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        user_function = entry_function.get().strip()
    except ValueError:
        result_label.config(text="Invalid input. Please enter numeric values for the interval.")
        return

    if not user_function:
        result_label.config(text="Please enter a valid function.")
        return

    # Solve using both methods
    try:
        bisection_root, bisection_iters = bisection(a, b)
        secant_root, secant_iters = secant(a, b)

        # Calculate relative errors
        exact_root = fsolve(lambda x: f(x), (a + b) / 2)[0]
        bisection_error = abs(bisection_root - exact_root) / abs(exact_root)
        secant_error = abs(secant_root - exact_root) / abs(exact_root)

        result_label.config(
            text=f"Bisection Root: {bisection_root:.4f}, Iterations: {bisection_iters}, Rel Error: {bisection_error:.4e}\n"
                 f"Secant Root: {secant_root:.4f}, Iterations: {secant_iters}, Rel Error: {secant_error:.4e}"
        )
    except Exception as e:
        result_label.config(text=f"Error solving equation: {e}")

# Task 3: Jacobi Method
def task_3():
    def jacobi(A, b, tol=1e-6, max_iter=7):
        n = len(A)
        x = np.zeros(n)  # Initial guess of zeros
        D = np.diag(A)
        R = A - np.diagflat(D)
        iterations = 0

        if np.any(D == 0):
            result_label.config(text="Error: Diagonal elements must be non-zero.")
            return None, None

        while iterations < max_iter:
            x_new = (b - np.dot(R, x)) / D
            if np.linalg.norm(x_new - x) < tol:
                return x_new, iterations
            x = x_new
            iterations += 1
        return x, iterations

    try:
        # Get matrix A and vector b from user input
        matrix_a_str = matrix_a_entry.get("1.0", tk.END).strip()
        vector_b_str = vector_b_entry.get("1.0", tk.END).strip()

        # Safely evaluate matrix A and vector b
        A = np.array(eval(matrix_a_str))
        b = np.array(eval(vector_b_str))

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimensions of A and b do not match.")

        solution, iterations = jacobi(A, b)

        if solution is not None:
            result_label.config(text=f"Jacobi Solution: {solution}, Iterations: {iterations}")

    except (ValueError, SyntaxError, NameError) as e:
        result_label.config(text=f"Invalid matrix or vector input: {e}")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")
# Task 4: LU Factorization
def task_4():
    try:
        # Get matrix A from user input
        matrix_a_str = matrix_a_entry.get("1.0", tk.END).strip()

        # Safely evaluate matrix A
        A = np.array(eval(matrix_a_str), dtype=np.float64)

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")

        n = A.shape[0]
        L = np.eye(n)
        U = A.copy()

        for i in range(n - 1):
            if U[i, i] == 0:
                raise ValueError("LU decomposition failed. Pivot is zero.")
            for j in range(i + 1, n):
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, :] -= factor * U[i, :]

        result_label.config(text=f"L:\n{L}\nU:\n{U}")

    except (ValueError, SyntaxError, NameError) as e:
        result_label.config(text=f"Invalid matrix input: {e}")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")
# Task 5: Polynomial Curve Fitting
def task_5():
    try:
        # Get data from user input
        x_str = x_entry.get("1.0", tk.END).strip()
        y_str = y_entry.get("1.0", tk.END).strip()

        x = np.array(eval(x_str), dtype=np.float64)
        y = np.array(eval(y_str), dtype=np.float64)

        # Check if the sizes of arrays match
        if len(x) != len(y):
            raise ValueError("Sizes of x and y arrays must match.")

        # Create matrix A and vector b
        A = np.vstack([x**2, x, np.ones(len(x))]).T
        b = y

        # Solve the system of equations using least squares method
        coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

        # Create a function for the approximating curve
        def f(x_val):
            return coefficients[0]*x_val**2 + coefficients[1]*x_val + coefficients[2]

        # Plot the graph
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = [f(x_i) for x_i in x_vals]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label="Original Data")
        plt.plot(x_vals, y_vals, color='red', label="Approximating Curve")
        plt.title("Polynomial Curve Fitting")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Display the coefficients
        result_label.config(text=f"Coefficients: {coefficients}")

    except (ValueError, SyntaxError, NameError) as e:
        result_label.config(text=f"Data input error: {e}")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")

def task_6():
    try:
        # Get data from user input
        x_str = x_entry.get("1.0", tk.END).strip()
        y_str = y_entry.get("1.0", tk.END).strip()

        x = np.array(eval(x_str), dtype=np.float64)
        y = np.array(eval(y_str), dtype=np.float64)

        # Check if the sizes of arrays match
        if len(x) != len(y):
            raise ValueError("Sizes of x and y arrays must match.")

         # Sort x values (CubicSpline requires x to be sorted)
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        y = y[sort_indices]

        # Perform cubic spline interpolation
        cs = CubicSpline(x, y)

        # Generate points for plotting the spline
        x_vals = np.linspace(min(x), max(x), 500)  # Increased density for smoother curve
        y_vals = cs(x_vals)

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, label="Original Data Points")
        plt.plot(x_vals, y_vals, color='red', label="Cubic Spline Interpolation")
        plt.title("Cubic Spline Interpolation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Provide example of evaluating the spline at a specific point
        eval_point_str = eval_point_entry.get()
        if eval_point_str:  # Check if entry is not empty
            try:
                eval_point = float(eval_point_str)
                eval_result = cs(eval_point)
                result_label.config(text=f"Spline value at {eval_point}: {eval_result}")
            except ValueError:
                result_label.config(text="Invalid evaluation point. Please enter a number.")
        else:
            result_label.config(text="Cubic spline plotted successfully.")

    except (ValueError, SyntaxError, NameError) as e:
        result_label.config(text=f"Data input error: {e}")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")

def task_7():
    try:
        # Get integral limits and function from user input
        a_str = entry_a.get()
        b_str = entry_b.get()
        n_str = n_entry.get()  # Get the number of subintervals
        user_function = entry_function.get().strip()

        a = float(a_str)
        b = float(b_str)
        n = int(n_str)  # Convert to integer

        # Define the function to integrate (using eval as in previous tasks)
        def f(x):
            try:
                return eval(user_function, {"x": x, "np": np})
            except Exception as e:
                result_label.config(text=f"Invalid function. Error: {e}")
                return None  # Return None to indicate an error

        # Check if the function evaluation was successful
        if f(a) is None or f(b) is None:
            return  # Exit the task if there was an error in the function

        # Calculate the width of each subinterval
        h = (b - a) / n

        # Calculate the sum of the function values at the endpoints and intermediate points
        integral_sum = 0.5 * (f(a) + f(b))  # Add the function values at the endpoints
        for i in range(1, n):
            integral_sum += f(a + i * h)  # Add the function values at the intermediate points

        # Multiply the sum by the width of the subinterval to get the approximate integral value
        approximate_integral = integral_sum * h

        # Calculate the exact integral value (using quad from scipy.integrate)
        exact_integral, _ = quad(f, a, b)

        # Calculate the absolute error
        absolute_error = abs(exact_integral - approximate_integral)

        # Display the results
        result_label.config(
            text=f"Approximate Integral: {approximate_integral:.6f}\nExact Integral: {exact_integral:.6f}\nAbsolute Error: {absolute_error:.6e}"
        )

    except (ValueError, SyntaxError, NameError) as e:
        result_label.config(text=f"Invalid input: {e}")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")

def task_8():
    try:
        # Get initial conditions, interval, step size and function from user input
        x0_str = entry_x0.get()
        y0_str = entry_y0.get()
        h_str = entry_h.get()
        x_end_str = entry_x_end.get()
        user_function = entry_function.get().strip()

        x0 = float(x0_str)
        y0 = float(y0_str)
        h = float(h_str)
        x_end = float(x_end_str)

        # Define the function dy/dx = f(x, y)
        def f(x, y):
            try:
                return eval(user_function, {"x": x, "y": y, "np": np})
            except Exception as e:
                result_label.config(text=f"Invalid function. Error: {e}")
                return None  # Return None to indicate an error

        # Check if the function evaluation was successful
        if f(x0, y0) is None:
            return  # Exit the task if there was an error in the function

        # Runge-Kutta 4th order method
        x = x0
        y = y0
        results = [(x, y)]  # Store the results at each step

        while x <= x_end:
            k1 = f(x, y)
            k2 = f(x + h/2, y + h/2*k1)
            k3 = f(x + h/2, y + h/2*k2)
            k4 = f(x + h, y + h*k3)

            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            x += h
            results.append((x, y))

        # Display the results
        result_text = "Runge-Kutta 4th Order Results:\n"
        for x, y in results:
            result_text += f"x = {x:.4f}, y = {y:.4f}\n"

        result_label.config(text=result_text)

    except (ValueError, SyntaxError, NameError) as e:
        result_label.config(text=f"Invalid input: {e}")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")

# GUI Interface
def main():
    root = tk.Tk()
    root.title("Computational Mathematics Final Project")

    tk.Label(root, text="Select a Task:", font=("Arial", 12)).grid(row=0, column=0, pady=10, sticky=tk.W)
    task_var = tk.StringVar(value="1")

    #Creating RadioButtons
    tasks = [("Task 1: Graphical Method", "1"),
             ("Task 2: Root-Finding Methods", "2"),
             ("Task 3: Jacobi Method", "3"),
             ("Task 4: LU Factorization", "4"),
             ("Task 5: Polynomial Curve Fitting", "5"),
             ("Task 6: Cubic Spline Interpolation", "6"),
             ("Task 7: Trapezoidal Rule", "7"),
             ("Task 8: Runge-Kutta 4th Order", "8")]

    for i, (text, value) in enumerate(tasks):
        tk.Radiobutton(root, text=text, variable=task_var, value=value).grid(row=i + 1, column=0, sticky=tk.W)

    global entry_start, entry_end, entry_a, entry_b, entry_function, result_label, matrix_a_entry, vector_b_entry, x_entry, y_entry, eval_point_entry, n_entry  # Include n_entry
    matrix_a_entry = None
    vector_b_entry = None
    x_entry = None
    y_entry = None
    eval_point_entry = None
    n_entry = None

    input_frame = tk.Frame(root)
    input_frame.grid(row=len(tasks) + 2, column=0, pady=10)
    
    def show_inputs():
        global matrix_a_entry, x_entry, y_entry, eval_point_entry, entry_a, entry_b, n_entry, entry_function, vector_b_entry, entry_x0, entry_y0, entry_h, entry_x_end

        for widget in input_frame.winfo_children():
            widget.destroy()
        
        #Creating conditions for choosing the task between 1 to 8
        task = task_var.get()
        if task == "1":
            tk.Label(input_frame, text="Start of Range:").grid(row=0, column=0, sticky=tk.W)
            global entry_start
            entry_start = tk.Entry(input_frame)
            entry_start.grid(row=0, column=1)

            tk.Label(input_frame, text="End of Range:").grid(row=1, column=0, sticky=tk.W)
            global entry_end
            entry_end = tk.Entry(input_frame)
            entry_end.grid(row=1, column=1)

            tk.Label(input_frame, text="Function (e.g., np.exp(-x) - x**3 + x):").grid(row=2, column=0, sticky=tk.W)
            global entry_function
            entry_function = tk.Entry(input_frame)
            entry_function.grid(row=2, column=1)

        elif task == "2":
            tk.Label(input_frame, text="Interval Start (a):").grid(row=0, column=0, sticky=tk.W)
            global entry_a
            entry_a = tk.Entry(input_frame)
            entry_a.grid(row=0, column=1)

            tk.Label(input_frame, text="Interval End (b):").grid(row=1, column=0, sticky=tk.W)
            global entry_b
            entry_b = tk.Entry(input_frame)
            entry_b.grid(row=1, column=1)

            tk.Label(input_frame, text="Function (e.g., x**5 - x**4 + x**3 - x + 1):").grid(row=2, column=0, sticky=tk.W)
            entry_function = tk.Entry(input_frame)
            entry_function.grid(row=2, column=1)

        elif task == "3":
            tk.Label(input_frame, text="Matrix A (e.g., [[1,1,1],[0,2,1],[1,1,1]]):").grid(row=0, column=0, sticky=tk.W)
            global matrix_a_entry
            matrix_a_entry = tk.Text(input_frame, height=5, width=30)
            matrix_a_entry.grid(row=0, column=1)

            tk.Label(input_frame, text="Vector b (e.g., [6,-4,-3]):").grid(row=1, column=0, sticky=tk.W)
            global vector_b_entry
            vector_b_entry = tk.Text(input_frame, height=5, width=30)
            vector_b_entry.grid(row=1, column=1)

        elif task == "4":
            tk.Label(input_frame, text="Matrix A (e.g., [[4,-2,-1],[-2,4,-2],[-1,-2,4]]):").grid(row=0, column=0, sticky=tk.W)

            matrix_a_entry = tk.Text(input_frame, height=5, width=30)
            matrix_a_entry.grid(row=0, column=1)
         
        elif task == "5":
            tk.Label(input_frame, text="x (e.g., [1, 2, 3]):").grid(row=0, column=0, sticky=tk.W)
            x_entry = tk.Text(input_frame, height=5, width=30)
            x_entry.grid(row=0, column=1)

            tk.Label(input_frame, text="y to x (e.g., [1, 4, 9]):").grid(row=1, column=0, sticky=tk.W)
            y_entry = tk.Text(input_frame, height=5, width=30)
            y_entry.grid(row=1, column=1)

        elif task == "6":
            tk.Label(input_frame, text="x (e.g., [0, 1, 2, 3]):").grid(row=0, column=0, sticky=tk.W)
            x_entry = tk.Text(input_frame, height=5, width=30)
            x_entry.grid(row=0, column=1)

            tk.Label(input_frame, text="y (e.g., [1, 3, 2, 4]):").grid(row=1, column=0, sticky=tk.W)
            y_entry = tk.Text(input_frame, height=5, width=30) 
            y_entry.grid(row=1, column=1)

        elif task == "7":
            tk.Label(input_frame, text="Interval Start (a):").grid(row=0, column=0, sticky=tk.W)
            entry_a = tk.Entry(input_frame)
            entry_a.grid(row=0, column=1)

            tk.Label(input_frame, text="Interval End (b):").grid(row=1, column=0, sticky=tk.W)
            entry_b = tk.Entry(input_frame)
            entry_b.grid(row=1, column=1)

            tk.Label(input_frame, text="Number of Subintervals (n):").grid(row=2, column=0, sticky=tk.W)
            n_entry = tk.Entry(input_frame)
            n_entry.grid(row=2, column=1)

            tk.Label(input_frame, text="Function (e.g., (np.log(x)/x):").grid(row=3, column=0, sticky=tk.W)
            entry_function = tk.Entry(input_frame)
            entry_function.grid(row=3, column=1)

        elif task == "8":
            tk.Label(input_frame, text="Initial x (x0):").grid(row=0, column=0, sticky=tk.W)
            entry_x0 = tk.Entry(input_frame)
            entry_x0.grid(row=0, column=1)

            tk.Label(input_frame, text="Initial y (y0):").grid(row=1, column=0, sticky=tk.W)
            entry_y0 = tk.Entry(input_frame)
            entry_y0.grid(row=1, column=1)

            tk.Label(input_frame, text="Step size (h):").grid(row=2, column=0, sticky=tk.W)
            entry_h = tk.Entry(input_frame)
            entry_h.grid(row=2, column=1)

            tk.Label(input_frame, text="End x (x_end):").grid(row=3, column=0, sticky=tk.W)
            entry_x_end = tk.Entry(input_frame)
            entry_x_end.grid(row=3, column=1)

            tk.Label(input_frame, text="Function dy/dx = f(x, y) (e.g., x**2 - y):").grid(row=4, column=0, sticky=tk.W)
            entry_function = tk.Entry(input_frame)
            entry_function.grid(row=4, column=1)
    
    #Displaying button for user's input
    tk.Button(root, text="Show Inputs", command=show_inputs).grid(row=len(tasks) + 1, column=0, pady=10)

    input_frame = tk.Frame(root)
    input_frame.grid(row=len(tasks) + 2, column=0, pady=10)

    result_label = tk.Label(root, text="", font=("Arial", 12))
    result_label.grid(row=len(tasks) + 3, column=0, pady=10)

    #Showing the system what has to be done when user choose exact task
    def execute_task():
        task = task_var.get()
        if task == "1":
            task_1()
        elif task == "2":
            task_2()
        elif task == "3":
            task_3()
        elif task == "4":
            task_4()
        elif task == "5":
            task_5()
        elif task == "6":
            task_6()
        elif task == "7":
            task_7()
        elif task == "8":
            task_8()

    #Displaying button for execution of task
    tk.Button(root, text="Execute", command=execute_task, font=("Arial", 12)).grid(row=len(tasks) + 4, column=0, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()