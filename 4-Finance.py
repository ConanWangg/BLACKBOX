import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
from scipy.stats import norm, binom

def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def monte_carlo_option_price(S, K, T, r, sigma, num_simulations):
    dt = T / 252
    S_t = np.zeros(num_simulations)
    for i in range(num_simulations):
        Z = np.random.normal(0, 1)
        S_t[i] = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(S_t - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

def heston_option_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, num_simulations):
    dt = T / 252
    S_t = np.zeros(num_simulations)
    v_t = np.zeros(num_simulations)
    v_t[0] = v0
    for i in range(1, num_simulations):
        Z1 = np.random.normal(0, 1)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        v_t[i] = v_t[i - 1] + kappa * (theta - v_t[i - 1]) * dt + sigma_v * np.sqrt(v_t[i - 1] * dt) * Z2
        S_t[i] = S * np.exp((r - 0.5 * v_t[i]) * dt + np.sqrt(v_t[i] * dt) * Z1)
    payoff = np.maximum(S_t - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

def binomial_option_price(S, K, T, r, sigma, num_steps):
    dt = T / num_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    S_t = np.zeros((num_steps + 1, num_steps + 1))
    S_t[0, 0] = S
    for i in range(1, num_steps + 1):
        S_t[i, 0] = S_t[i - 1, 0] * u
        for j in range(1, i + 1):
            S_t[i, j] = S_t[i - 1, j - 1] * d
    option_payoff = np.maximum(S_t - K, 0)
    option_price = np.sum(binom.pmf(np.arange(num_steps + 1), num_steps, p) * option_payoff[num_steps])
    option_price *= np.exp(-r * T)
    return option_price

def calculate_option_price():
    try:
        S = float(entry_S.get().strip())
        K = float(entry_K.get().strip())
        T = float(entry_T.get().strip())
        r = float(entry_r.get().strip())

        model = option_type.get()

        if model == "Black-Scholes":
            sigma = float(entry_sigma.get().strip())
            option_price = black_scholes_call(S, K, T, r, sigma)
        elif model == "Monte Carlo":
            sigma = float(entry_sigma_mc.get().strip())
            num_simulations = int(entry_num_simulations_mc.get().strip())
            option_price = monte_carlo_option_price(S, K, T, r, sigma, num_simulations)
        elif model == "Heston":
            v0 = float(entry_v0.get().strip())
            kappa = float(entry_kappa.get().strip())
            theta = float(entry_theta.get().strip())
            sigma_v = float(entry_sigma_v.get().strip())
            rho = float(entry_rho.get().strip())
            num_simulations = int(entry_num_simulations_heston.get().strip())
            option_price = heston_option_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, num_simulations)
        elif model == "Binomial":
            sigma = float(entry_sigma_binomial.get().strip())
            num_steps = int(entry_num_steps.get().strip())
            option_price = binomial_option_price(S, K, T, r, sigma, num_steps)
        else:
            raise ValueError("Invalid option pricing model selected.")
        
        messagebox.showinfo("Option Price", f"Estimated Call Option Price: {option_price:.2f}")
    except ValueError as e:
        messagebox.showerror("Error", f"Please enter valid numeric values for all parameters.\nError details: {e}")

def validate_inputs():
    model = option_type.get()
    required_fields = [entry_S, entry_K, entry_T, entry_r]
    
    if model == "Black-Scholes":
        required_fields.append(entry_sigma)
    elif model == "Monte Carlo":
        required_fields.extend([entry_sigma_mc, entry_num_simulations_mc])
    elif model == "Heston":
        required_fields.extend([entry_v0, entry_kappa, entry_theta, entry_sigma_v, entry_rho, entry_num_simulations_heston])
    elif model == "Binomial":
        required_fields.extend([entry_sigma_binomial, entry_num_steps])
    
    for field in required_fields:
        if not field.get().strip():
            messagebox.showerror("Error", "Please fill in all required fields.")
            return
    
    calculate_option_price()

def update_fields():
    model = option_type.get()
    
    # Hide all widgets first
    entry_sigma.grid_forget()
    entry_sigma_mc.grid_forget()
    entry_num_simulations_mc.grid_forget()
    entry_v0.grid_forget()
    entry_kappa.grid_forget()
    entry_theta.grid_forget()
    entry_sigma_v.grid_forget()
    entry_rho.grid_forget()
    entry_num_simulations_heston.grid_forget()
    entry_sigma_binomial.grid_forget()
    entry_num_steps.grid_forget()
    
    if model == "Black-Scholes":
        label_sigma.grid(row=5, column=0, sticky="w")
        entry_sigma.grid(row=5, column=1)
    elif model == "Monte Carlo":
        label_sigma_mc.grid(row=5, column=0, sticky="w")
        entry_sigma_mc.grid(row=5, column=1)
        label_num_simulations_mc.grid(row=6, column=0, sticky="w")
        entry_num_simulations_mc.grid(row=6, column=1)
    elif model == "Heston":
        label_v0.grid(row=5, column=0, sticky="w")
        entry_v0.grid(row=5, column=1)
        label_kappa.grid(row=6, column=0, sticky="w")
        entry_kappa.grid(row=6, column=1)
        label_theta.grid(row=7, column=0, sticky="w")
        entry_theta.grid(row=7, column=1)
        label_sigma_v.grid(row=8, column=0, sticky="w")
        entry_sigma_v.grid(row=8, column=1)
        label_rho.grid(row=9, column=0, sticky="w")
        entry_rho.grid(row=9, column=1)
        label_num_simulations_heston.grid(row=10, column=0, sticky="w")
        entry_num_simulations_heston.grid(row=10, column=1)
    elif model == "Binomial":
        label_sigma_binomial.grid(row=5, column=0, sticky="w")
        entry_sigma_binomial.grid(row=5, column=1)
        label_num_steps.grid(row=6, column=0, sticky="w")
        entry_num_steps.grid(row=6, column=1)

# GUI
root = tk.Tk()
root.title("Option Pricing")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

label_option_type = tk.Label(frame, text="Option Pricing Model:")
label_option_type.grid(row=0, column=0, columnspan=5, sticky="w")

option_type = tk.StringVar()
option_type.set("Black-Scholes")
option_type.trace_add('write', lambda *args: update_fields())

radio_button_bs = tk.Radiobutton(frame, text="Black-Scholes", variable=option_type, value="Black-Scholes")
radio_button_bs.grid(row=1, column=0, sticky="w")

radio_button_mc = tk.Radiobutton(frame, text="Monte Carlo Simulation", variable=option_type, value="Monte Carlo")
radio_button_mc.grid(row=1, column=1, sticky="w")

radio_button_heston = tk.Radiobutton(frame, text="Heston Model", variable=option_type, value="Heston")
radio_button_heston.grid(row=1, column=2, sticky="w")

radio_button_binomial = tk.Radiobutton(frame, text="Binomial Model", variable=option_type, value="Binomial")
radio_button_binomial.grid(row=1, column=3, sticky="w")

# Common Fields
label_S = tk.Label(frame, text="Stock Price (S):")
label_S.grid(row=2, column=0, sticky="w")
entry_S = tk.Entry(frame)
entry_S.grid(row=2, column=1)

label_K = tk.Label(frame, text="Strike Price (K):")
label_K.grid(row=3, column=0, sticky="w")
entry_K = tk.Entry(frame)
entry_K.grid(row=3, column=1)

label_T = tk.Label(frame, text="Time to Maturity (T in years):")
label_T.grid(row=4, column=0, sticky="w")
entry_T = tk.Entry(frame)
entry_T.grid(row=4, column=1)

label_r = tk.Label(frame, text="Risk-Free Rate (r):")
label_r.grid(row=4, column=2, sticky="w")
entry_r = tk.Entry(frame)
entry_r.grid(row=4, column=3)

# Model-specific Fields
label_sigma = tk.Label(frame, text="Volatility (sigma):")
label_sigma_mc = tk.Label(frame, text="Volatility (sigma):")
label_num_simulations_mc = tk.Label(frame, text="Number of Simulations:")
label_v0 = tk.Label(frame, text="Initial Variance (v0):")
label_kappa = tk.Label(frame, text="Mean Reversion Rate (kappa):")
label_theta = tk.Label(frame, text="Long-Term Variance (theta):")
label_sigma_v = tk.Label(frame, text="Volatility of Variance (sigma_v):")
label_rho = tk.Label(frame, text="Correlation (rho):")
label_num_simulations_heston = tk.Label(frame, text="Number of Simulations:")
label_sigma_binomial = tk.Label(frame, text="Volatility (sigma):")
label_num_steps = tk.Label(frame, text="Number of Steps:")

entry_sigma = tk.Entry(frame)
entry_sigma_mc = tk.Entry(frame)
entry_num_simulations_mc = tk.Entry(frame)
entry_v0 = tk.Entry(frame)
entry_kappa = tk.Entry(frame)
entry_theta = tk.Entry(frame)
entry_sigma_v = tk.Entry(frame)
entry_rho = tk.Entry(frame)
entry_num_simulations_heston = tk.Entry(frame)
entry_sigma_binomial = tk.Entry(frame)
entry_num_steps = tk.Entry(frame)

# Buttons
button_validate = tk.Button(frame, text="Validate Inputs", command=validate_inputs)
button_validate.grid(row=11, column=0, columnspan=4, pady=10)

# Initialize fields
update_fields()

root.mainloop()


