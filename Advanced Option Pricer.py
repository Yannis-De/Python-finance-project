"""
Advanced Option Pricer Pro - Version 2.0
Pricing d'options avec données temps réel, volatilité implicite, stratégies et analyse avancée
"""

import customtkinter as ctk
from tkinter import ttk, messagebox
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import threading
import json

# Configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class BlackScholesModel:
    """Modèle Black-Scholes étendu"""

    @staticmethod
    def calculate_d1_d2(S, K, T, r, sigma):
        """Calcule d1 et d2"""
        if T <= 0 or sigma <= 0:
            return 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Prix d'un Call européen"""
        if T <= 0:
            return max(S - K, 0)
        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price

    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Prix d'un Put européen"""
        if T <= 0:
            return max(K - S, 0)
        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """Calcule tous les Greeks"""
        if T <= 0:
            return {'Delta': 0, 'Gamma': 0, 'Vega': 0, 'Theta': 0, 'Rho': 0}

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)

        # Delta
        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        # Theta
        theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type='call'):
        """Calcule la volatilité implicite"""
        if T <= 0 or price <= 0:
            return 0

        def objective(sigma):
            if option_type == 'call':
                return BlackScholesModel.call_price(S, K, T, r, sigma) - price
            else:
                return BlackScholesModel.put_price(S, K, T, r, sigma) - price

        try:
            iv = brentq(objective, 0.001, 5.0, maxiter=100)
            return iv
        except:
            return 0


class OptionStrategies:
    """Stratégies d'options"""

    @staticmethod
    def straddle(S, K, T, r, sigma):
        """Long Straddle: Achat Call + Put au même strike"""
        call = BlackScholesModel.call_price(S, K, T, r, sigma)
        put = BlackScholesModel.put_price(S, K, T, r, sigma)
        return {
            'cost': call + put,
            'call': call,
            'put': put,
            'name': 'Long Straddle'
        }

    @staticmethod
    def strangle(S, K_call, K_put, T, r, sigma):
        """Long Strangle: Call OTM + Put OTM"""
        call = BlackScholesModel.call_price(S, K_call, T, r, sigma)
        put = BlackScholesModel.put_price(S, K_put, T, r, sigma)
        return {
            'cost': call + put,
            'call': call,
            'put': put,
            'K_call': K_call,
            'K_put': K_put,
            'name': 'Long Strangle'
        }

    @staticmethod
    def butterfly(S, K_low, K_mid, K_high, T, r, sigma, option_type='call'):
        """Butterfly Spread"""
        if option_type == 'call':
            low = BlackScholesModel.call_price(S, K_low, T, r, sigma)
            mid = BlackScholesModel.call_price(S, K_mid, T, r, sigma)
            high = BlackScholesModel.call_price(S, K_high, T, r, sigma)
        else:
            low = BlackScholesModel.put_price(S, K_low, T, r, sigma)
            mid = BlackScholesModel.put_price(S, K_mid, T, r, sigma)
            high = BlackScholesModel.put_price(S, K_high, T, r, sigma)

        cost = low - 2 * mid + high
        return {
            'cost': cost,
            'low': low,
            'mid': mid,
            'high': high,
            'name': f'{option_type.capitalize()} Butterfly'
        }


class YahooFinanceConnector:
    """Connecteur Yahoo Finance pour données temps réel"""

    @staticmethod
    def get_stock_data(ticker):
        """Récupère les données d'un ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)

            return {
                'ticker': ticker,
                'price': current_price,
                'volatility': volatility,
                'name': info.get('longName', ticker),
                'currency': info.get('currency', 'USD'),
                'history': hist
            }
        except Exception as e:
            return None

    @staticmethod
    def get_risk_free_rate():
        """Récupère le taux sans risque (Treasury 10Y)"""
        try:
            tnx = yf.Ticker("^TNX")
            rate = tnx.history(period="1d")['Close'].iloc[-1] / 100
            return rate
        except:
            return 0.05


class AdvancedOptionPricerGUI(ctk.CTk):
    """Interface graphique avancée"""

    def __init__(self):
        super().__init__()

        self.title("Advanced Option Pricer Pro v2.0")
        self.geometry("1600x950")

        self.colors = {
            'bg_primary': '#1a1a1a',
            'bg_secondary': '#2d2d2d',
            'accent': '#3b8ed0',
            'success': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'text': '#ffffff'
        }

        self.scenarios_history = []
        self.auto_refresh = False

        self.setup_ui()

    def setup_ui(self):
        """Configuration UI"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.create_sidebar()
        self.create_main_frame()

    def create_sidebar(self):
        """Sidebar avec contrôles"""
        sidebar = ctk.CTkScrollableFrame(self, width=380, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")

        # Header
        title = ctk.CTkLabel(sidebar, text="⚡ Option Pricer Pro",
                            font=ctk.CTkFont(size=26, weight="bold"))
        title.pack(pady=(20, 5))

        subtitle = ctk.CTkLabel(sidebar, text="Advanced Analytics & Real-Time Data",
                               font=ctk.CTkFont(size=11), text_color="gray")
        subtitle.pack(pady=(0, 20))

        # Yahoo Finance Section
        yf_frame = ctk.CTkFrame(sidebar, fg_color=self.colors['bg_secondary'])
        yf_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(yf_frame, text="📊 Données Temps Réel (Yahoo Finance)",
                    font=ctk.CTkFont(size=13, weight="bold")).pack(pady=10)

        ticker_frame = ctk.CTkFrame(yf_frame, fg_color="transparent")
        ticker_frame.pack(fill="x", padx=10, pady=5)

        self.ticker_entry = ctk.CTkEntry(ticker_frame, placeholder_text="Ex: AAPL, TSLA, ^GSPC")
        self.ticker_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        fetch_btn = ctk.CTkButton(ticker_frame, text="Fetch", width=70,
                                 command=self.fetch_market_data)
        fetch_btn.pack(side="right")

        self.stock_info_label = ctk.CTkLabel(yf_frame, text="",
                                            font=ctk.CTkFont(size=10), text_color="gray")
        self.stock_info_label.pack(pady=5)

        # Auto-refresh
        self.auto_refresh_var = ctk.BooleanVar(value=False)
        refresh_check = ctk.CTkCheckBox(yf_frame, text="Auto-refresh (30s)",
                                       variable=self.auto_refresh_var,
                                       command=self.toggle_auto_refresh)
        refresh_check.pack(pady=5)

        # Mode Section
        mode_frame = ctk.CTkFrame(sidebar, fg_color=self.colors['bg_secondary'])
        mode_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(mode_frame, text="Mode d'analyse",
                    font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)

        self.analysis_mode = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Simple", "Stratégies", "Vol Implicite"],
            command=self.change_mode
        )
        self.analysis_mode.set("Simple")
        self.analysis_mode.pack(fill="x", padx=10, pady=5)

        # Option Type
        ctk.CTkLabel(sidebar, text="Type d'option:", anchor="w",
                    font=ctk.CTkFont(size=11, weight="bold")).pack(padx=20, pady=(15, 5), anchor="w")

        self.option_type = ctk.CTkSegmentedButton(sidebar, values=["Call", "Put"])
        self.option_type.set("Call")
        self.option_type.pack(fill="x", padx=20, pady=5)

        # Parameters
        self.entries = {}
        params = [
            ("Prix du sous-jacent (S):", "100", "spot_price"),
            ("Strike (K):", "100", "strike"),
            ("Temps à maturité (années):", "1", "maturity"),
            ("Taux sans risque (%):", "5", "risk_free_rate"),
            ("Volatilité (%):", "20", "volatility")
        ]

        for label_text, default_value, key in params:
            ctk.CTkLabel(sidebar, text=label_text, anchor="w",
                        font=ctk.CTkFont(size=10)).pack(padx=20, pady=(10, 2), anchor="w")
            entry = ctk.CTkEntry(sidebar, placeholder_text=default_value)
            entry.insert(0, default_value)
            entry.pack(fill="x", padx=20, pady=2)
            entry.bind('<KeyRelease>', lambda e: self.calculate_all())
            self.entries[key] = entry

        # Strategy Parameters (hidden by default)
        self.strategy_frame = ctk.CTkFrame(sidebar, fg_color=self.colors['bg_secondary'])

        ctk.CTkLabel(self.strategy_frame, text="Paramètres Stratégie",
                    font=ctk.CTkFont(size=11, weight="bold")).pack(pady=5)

        self.strategy_type = ctk.CTkOptionMenu(
            self.strategy_frame,
            values=["Straddle", "Strangle", "Butterfly"],
            command=lambda x: self.calculate_all()
        )
        self.strategy_type.pack(fill="x", padx=10, pady=5)

        # Buttons
        calc_button = ctk.CTkButton(sidebar, text="📊 Calculer", height=45,
                                   command=self.calculate_all,
                                   font=ctk.CTkFont(size=14, weight="bold"),
                                   fg_color=self.colors['accent'])
        calc_button.pack(fill="x", padx=20, pady=15)

        # Export buttons
        export_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        export_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(export_frame, text="💾 Sauver Scénario",
                     command=self.save_scenario).pack(fill="x", pady=2)
        ctk.CTkButton(export_frame, text="📄 Export Excel",
                     command=self.export_excel).pack(fill="x", pady=2)

    def create_main_frame(self):
        """Frame principal"""
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        self.create_results_frame(main_frame)
        self.create_charts_frame(main_frame)

    def create_results_frame(self, parent):
        """Frame résultats"""
        results_container = ctk.CTkFrame(parent)
        results_container.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        results_container.grid_columnconfigure((0, 1, 2), weight=1)

        # Prix
        price_frame = ctk.CTkFrame(results_container)
        price_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(price_frame, text="Prix de l'option",
                    font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(10, 2))

        self.price_label = ctk.CTkLabel(price_frame, text="0.00 $",
                                       font=ctk.CTkFont(size=28, weight="bold"),
                                       text_color=self.colors['success'])
        self.price_label.pack(pady=(2, 10))

        # Greeks
        greeks_frame = ctk.CTkFrame(results_container)
        greeks_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(greeks_frame, text="Greeks",
                    font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(10, 5))

        self.greeks_labels = {}
        for greek in ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']:
            frame = ctk.CTkFrame(greeks_frame, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=1)

            ctk.CTkLabel(frame, text=f"{greek}:", width=50, anchor="w",
                        font=ctk.CTkFont(size=10)).pack(side="left")

            label = ctk.CTkLabel(frame, text="0.0000", anchor="e",
                               font=ctk.CTkFont(size=10, weight="bold"))
            label.pack(side="right")
            self.greeks_labels[greek] = label

        # Info supplémentaire
        info_frame = ctk.CTkFrame(results_container)
        info_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(info_frame, text="Analyse",
                    font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(10, 5))

        self.analysis_labels = {}
        for key in ['Moneyness', 'Vol Impl', 'Break-even', 'Prob ITM']:
            frame = ctk.CTkFrame(info_frame, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=1)

            ctk.CTkLabel(frame, text=f"{key}:", width=70, anchor="w",
                        font=ctk.CTkFont(size=10)).pack(side="left")

            label = ctk.CTkLabel(frame, text="-", anchor="e",
                               font=ctk.CTkFont(size=10, weight="bold"))
            label.pack(side="right")
            self.analysis_labels[key] = label

    def create_charts_frame(self, parent):
        """Frame graphiques"""
        charts_frame = ctk.CTkFrame(parent)
        charts_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        charts_frame.grid_columnconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(0, weight=1)

        self.notebook = ctk.CTkTabview(charts_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.notebook.add("Prix vs Spot")
        self.notebook.add("Prix vs Vol")
        self.notebook.add("Greeks")
        self.notebook.add("Payoff")
        self.notebook.add("Surface 3D")
        self.notebook.add("Heatmap")

        self.setup_charts()

    def setup_charts(self):
        """Configure les graphiques"""
        plt.style.use('dark_background')

        # Chart 1: Prix vs Spot
        self.fig1 = Figure(figsize=(7, 4.5), facecolor='#2d2d2d')
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.notebook.tab("Prix vs Spot"))
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

        # Chart 2: Prix vs Vol
        self.fig2 = Figure(figsize=(7, 4.5), facecolor='#2d2d2d')
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.notebook.tab("Prix vs Vol"))
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

        # Chart 3: Greeks
        self.fig3 = Figure(figsize=(7, 4.5), facecolor='#2d2d2d')
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.notebook.tab("Greeks"))
        self.canvas3.get_tk_widget().pack(fill="both", expand=True)

        # Chart 4: Payoff
        self.fig4 = Figure(figsize=(7, 4.5), facecolor='#2d2d2d')
        self.ax4 = self.fig4.add_subplot(111)
        self.canvas4 = FigureCanvasTkAgg(self.fig4, self.notebook.tab("Payoff"))
        self.canvas4.get_tk_widget().pack(fill="both", expand=True)

        # Chart 5: Surface 3D
        self.fig5 = Figure(figsize=(7, 4.5), facecolor='#2d2d2d')
        self.ax5 = self.fig5.add_subplot(111, projection='3d')
        self.canvas5 = FigureCanvasTkAgg(self.fig5, self.notebook.tab("Surface 3D"))
        self.canvas5.get_tk_widget().pack(fill="both", expand=True)

        # Chart 6: Heatmap
        self.fig6 = Figure(figsize=(7, 4.5), facecolor='#2d2d2d')
        self.ax6 = self.fig6.add_subplot(111)
        self.canvas6 = FigureCanvasTkAgg(self.fig6, self.notebook.tab("Heatmap"))
        self.canvas6.get_tk_widget().pack(fill="both", expand=True)

    def get_parameters(self):
        """Récupère les paramètres"""
        try:
            S = float(self.entries['spot_price'].get())
            K = float(self.entries['strike'].get())
            T = float(self.entries['maturity'].get())
            r = float(self.entries['risk_free_rate'].get()) / 100
            sigma = float(self.entries['volatility'].get()) / 100
            option_type = self.option_type.get().lower()
            return S, K, T, r, sigma, option_type
        except:
            return None

    def fetch_market_data(self):
        """Récupère données Yahoo Finance"""
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showwarning("Attention", "Entrez un ticker")
            return

        self.stock_info_label.configure(text="⏳ Chargement...")

        def fetch():
            data = YahooFinanceConnector.get_stock_data(ticker)
            if data:
                self.entries['spot_price'].delete(0, 'end')
                self.entries['spot_price'].insert(0, f"{data['price']:.2f}")

                self.entries['volatility'].delete(0, 'end')
                self.entries['volatility'].insert(0, f"{data['volatility']*100:.2f}")

                rate = YahooFinanceConnector.get_risk_free_rate()
                self.entries['risk_free_rate'].delete(0, 'end')
                self.entries['risk_free_rate'].insert(0, f"{rate*100:.2f}")

                info_text = f"✅ {data['name']} | ${data['price']:.2f} | Vol: {data['volatility']*100:.1f}%"
                self.stock_info_label.configure(text=info_text, text_color=self.colors['success'])

                self.calculate_all()
            else:
                self.stock_info_label.configure(text="❌ Erreur de récupération",
                                               text_color=self.colors['danger'])

        threading.Thread(target=fetch, daemon=True).start()

    def toggle_auto_refresh(self):
        """Active/désactive auto-refresh"""
        if self.auto_refresh_var.get():
            self.auto_refresh = True
            self.auto_refresh_loop()
        else:
            self.auto_refresh = False

    def auto_refresh_loop(self):
        """Boucle auto-refresh"""
        if self.auto_refresh:
            self.fetch_market_data()
            self.after(30000, self.auto_refresh_loop)

    def change_mode(self, value):
        """Change le mode d'analyse"""
        if value == "Stratégies":
            self.strategy_frame.pack(fill="x", padx=15, pady=10)
        else:
            self.strategy_frame.pack_forget()
        self.calculate_all()

    def calculate_all(self, event=None):
        """Calcule tout"""
        params = self.get_parameters()
        if not params:
            return

        S, K, T, r, sigma, option_type = params

        mode = self.analysis_mode.get()

        if mode == "Simple":
            self.calculate_simple_option(S, K, T, r, sigma, option_type)
        elif mode == "Stratégies":
            self.calculate_strategy(S, K, T, r, sigma)
        elif mode == "Vol Implicite":
            self.calculate_implied_vol(S, K, T, r, sigma, option_type)

        self.update_all_charts(S, K, T, r, sigma, option_type)

    def calculate_simple_option(self, S, K, T, r, sigma, option_type):
        """Calcul option simple"""
        if option_type == 'call':
            price = BlackScholesModel.call_price(S, K, T, r, sigma)
        else:
            price = BlackScholesModel.put_price(S, K, T, r, sigma)

        greeks = BlackScholesModel.calculate_greeks(S, K, T, r, sigma, option_type)

        self.price_label.configure(text=f"{price:.4f} $")

        for greek_name, value in greeks.items():
            self.greeks_labels[greek_name].configure(text=f"{value:.4f}")

        # Analyse additionnelle
        moneyness = S / K
        if moneyness > 1.05:
            money_status = "ITM"
        elif moneyness < 0.95:
            money_status = "OTM"
        else:
            money_status = "ATM"

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)
        prob_itm = norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)

        self.analysis_labels['Moneyness'].configure(text=f"{moneyness:.3f} ({money_status})")
        self.analysis_labels['Prob ITM'].configure(text=f"{prob_itm*100:.1f}%")
        self.analysis_labels['Vol Impl'].configure(text="N/A")

        if option_type == 'call':
            breakeven = K + price
        else:
            breakeven = K - price
        self.analysis_labels['Break-even'].configure(text=f"{breakeven:.2f}")

    def calculate_strategy(self, S, K, T, r, sigma):
        """Calcul stratégie"""
        strategy = self.strategy_type.get()

        if strategy == "Straddle":
            result = OptionStrategies.straddle(S, K, T, r, sigma)
            self.price_label.configure(text=f"{result['cost']:.4f} $")
            self.analysis_labels['Moneyness'].configure(text="Straddle")
            self.analysis_labels['Vol Impl'].configure(text=f"C:{result['call']:.2f}")
            self.analysis_labels['Break-even'].configure(text=f"±{result['cost']:.2f}")
            self.analysis_labels['Prob ITM'].configure(text="N/A")

            # Reset Greeks for strategy
            for greek in self.greeks_labels:
                self.greeks_labels[greek].configure(text="N/A")

        elif strategy == "Strangle":
            K_put = K * 0.95  # Put OTM
            K_call = K * 1.05  # Call OTM
            result = OptionStrategies.strangle(S, K_call, K_put, T, r, sigma)
            self.price_label.configure(text=f"{result['cost']:.4f} $")
            self.analysis_labels['Moneyness'].configure(text="Strangle")
            self.analysis_labels['Vol Impl'].configure(text=f"C:{result['call']:.2f}")
            self.analysis_labels['Break-even'].configure(text=f"P:{result['put']:.2f}")
            self.analysis_labels['Prob ITM'].configure(text=f"K:{K_put:.1f}/{K_call:.1f}")

            for greek in self.greeks_labels:
                self.greeks_labels[greek].configure(text="N/A")

        elif strategy == "Butterfly":
            K_low = K * 0.95
            K_mid = K
            K_high = K * 1.05
            result = OptionStrategies.butterfly(S, K_low, K_mid, K_high, T, r, sigma, 'call')
            self.price_label.configure(text=f"{result['cost']:.4f} $")
            self.analysis_labels['Moneyness'].configure(text="Butterfly")
            self.analysis_labels['Vol Impl'].configure(text=f"L:{result['low']:.2f}")
            self.analysis_labels['Break-even'].configure(text=f"H:{result['high']:.2f}")
            self.analysis_labels['Prob ITM'].configure(text=f"Max:{(K_high-K_mid-result['cost']):.2f}")

            for greek in self.greeks_labels:
                self.greeks_labels[greek].configure(text="N/A")

    def calculate_implied_vol(self, S, K, T, r, sigma, option_type):
        """Calcul volatilité implicite"""
        if option_type == 'call':
            market_price = BlackScholesModel.call_price(S, K, T, r, sigma)
        else:
            market_price = BlackScholesModel.put_price(S, K, T, r, sigma)

        iv = BlackScholesModel.implied_volatility(market_price, S, K, T, r, option_type)

        self.price_label.configure(text=f"{market_price:.4f} $")
        self.analysis_labels['Vol Impl'].configure(text=f"{iv*100:.2f}%")

    def update_all_charts(self, S, K, T, r, sigma, option_type):
        """Met à jour tous les graphiques"""
        mode = self.analysis_mode.get()

        if mode == "Stratégies":
            strategy = self.strategy_type.get()
            self.update_strategy_charts(S, K, T, r, sigma, strategy)
        else:
            self.update_price_vs_spot(S, K, T, r, sigma, option_type)
            self.update_price_vs_vol(S, K, T, r, sigma, option_type)
            self.update_greeks_chart(S, K, T, r, sigma, option_type)
            self.update_payoff_chart(S, K, T, r, sigma, option_type)
            self.update_3d_surface(S, K, T, r, option_type)
            self.update_heatmap(S, K, T, r, sigma, option_type)

    def update_strategy_charts(self, S, K, T, r, sigma, strategy):
        """Met à jour les graphiques pour les stratégies"""

        # Chart 1: Payoff de la stratégie
        self.ax1.clear()
        spot_range = np.linspace(S * 0.5, S * 1.5, 100)

        if strategy == "Straddle":
            result = OptionStrategies.straddle(S, K, T, r, sigma)
            payoffs = []
            for s in spot_range:
                call_payoff = max(s - K, 0)
                put_payoff = max(K - s, 0)
                payoffs.append(call_payoff + put_payoff - result['cost'])

            self.ax1.plot(spot_range, payoffs, linewidth=3, color='#3b8ed0', label='Straddle P&L')
            self.ax1.axhline(0, color='white', linestyle='-', alpha=0.3)
            self.ax1.axvline(K, color='#e74c3c', linestyle='--', linewidth=2, label='Strike')
            self.ax1.fill_between(spot_range, payoffs, 0, where=(np.array(payoffs) > 0),
                                 alpha=0.3, color='#2ecc71', label='Profit')
            self.ax1.fill_between(spot_range, payoffs, 0, where=(np.array(payoffs) < 0),
                                 alpha=0.3, color='#e74c3c', label='Loss')
            self.ax1.set_title('Straddle Payoff Diagram', fontsize=13, fontweight='bold', pad=15)

        elif strategy == "Strangle":
            K_put = K * 0.95
            K_call = K * 1.05
            result = OptionStrategies.strangle(S, K_call, K_put, T, r, sigma)
            payoffs = []
            for s in spot_range:
                call_payoff = max(s - K_call, 0)
                put_payoff = max(K_put - s, 0)
                payoffs.append(call_payoff + put_payoff - result['cost'])

            self.ax1.plot(spot_range, payoffs, linewidth=3, color='#9b59b6', label='Strangle P&L')
            self.ax1.axhline(0, color='white', linestyle='-', alpha=0.3)
            self.ax1.axvline(K_put, color='#e74c3c', linestyle='--', linewidth=2, label=f'Put Strike {K_put:.1f}')
            self.ax1.axvline(K_call, color='#f39c12', linestyle='--', linewidth=2, label=f'Call Strike {K_call:.1f}')
            self.ax1.fill_between(spot_range, payoffs, 0, where=(np.array(payoffs) > 0),
                                 alpha=0.3, color='#2ecc71', label='Profit')
            self.ax1.fill_between(spot_range, payoffs, 0, where=(np.array(payoffs) < 0),
                                 alpha=0.3, color='#e74c3c', label='Loss')
            self.ax1.set_title('Strangle Payoff Diagram', fontsize=13, fontweight='bold', pad=15)

        elif strategy == "Butterfly":
            K_low = K * 0.95
            K_mid = K
            K_high = K * 1.05
            result = OptionStrategies.butterfly(S, K_low, K_mid, K_high, T, r, sigma, 'call')
            payoffs = []
            for s in spot_range:
                low_payoff = max(s - K_low, 0)
                mid_payoff = -2 * max(s - K_mid, 0)
                high_payoff = max(s - K_high, 0)
                payoffs.append(low_payoff + mid_payoff + high_payoff - result['cost'])

            self.ax1.plot(spot_range, payoffs, linewidth=3, color='#e74c3c', label='Butterfly P&L')
            self.ax1.axhline(0, color='white', linestyle='-', alpha=0.3)
            self.ax1.axvline(K_low, color='#95a5a6', linestyle='--', alpha=0.6, label=f'K Low {K_low:.1f}')
            self.ax1.axvline(K_mid, color='#e74c3c', linestyle='--', linewidth=2, label=f'K Mid {K_mid:.1f}')
            self.ax1.axvline(K_high, color='#95a5a6', linestyle='--', alpha=0.6, label=f'K High {K_high:.1f}')
            self.ax1.fill_between(spot_range, payoffs, 0, where=(np.array(payoffs) > 0),
                                 alpha=0.3, color='#2ecc71', label='Profit')
            self.ax1.fill_between(spot_range, payoffs, 0, where=(np.array(payoffs) < 0),
                                 alpha=0.3, color='#e74c3c', label='Loss')
            self.ax1.set_title('Butterfly Spread Payoff Diagram', fontsize=13, fontweight='bold', pad=15)

        self.ax1.set_xlabel('Prix du sous-jacent à maturité', fontsize=11, fontweight='bold')
        self.ax1.set_ylabel('Profit / Loss', fontsize=11, fontweight='bold')
        self.ax1.legend(loc='best', fontsize=9)
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.fig1.tight_layout()
        self.canvas1.draw()

        # Chart 2: Coût des composants
        self.ax2.clear()
        if strategy == "Straddle":
            result = OptionStrategies.straddle(S, K, T, r, sigma)
            components = ['Call', 'Put', 'Total Cost']
            values = [result['call'], result['put'], result['cost']]
            colors = ['#3b8ed0', '#9b59b6', '#e74c3c']

        elif strategy == "Strangle":
            K_put = K * 0.95
            K_call = K * 1.05
            result = OptionStrategies.strangle(S, K_call, K_put, T, r, sigma)
            components = ['Call OTM', 'Put OTM', 'Total Cost']
            values = [result['call'], result['put'], result['cost']]
            colors = ['#3b8ed0', '#9b59b6', '#e74c3c']

        elif strategy == "Butterfly":
            K_low = K * 0.95
            K_mid = K
            K_high = K * 1.05
            result = OptionStrategies.butterfly(S, K_low, K_mid, K_high, T, r, sigma, 'call')
            components = ['Long Low', '2x Short Mid', 'Long High', 'Net Cost']
            values = [result['low'], -2*result['mid'], result['high'], result['cost']]
            colors = ['#2ecc71', '#e74c3c', '#2ecc71', '#f39c12']

        bars = self.ax2.bar(components, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        self.ax2.set_ylabel('Prix ($)', fontsize=11, fontweight='bold')
        self.ax2.set_title(f'{strategy} - Décomposition des coûts', fontsize=13, fontweight='bold', pad=15)
        self.ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'${height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                         fontweight='bold', fontsize=10)

        self.fig2.tight_layout()
        self.canvas2.draw()

        # Autres graphiques restent vides ou affichent des infos génériques
        for ax, canvas, title in [
            (self.ax3, self.canvas3, 'Greeks N/A pour stratégies'),
            (self.ax4, self.canvas4, 'Voir onglet Prix vs Spot'),
            (self.ax5, self.canvas5, 'Surface 3D N/A'),
            (self.ax6, self.canvas6, 'Heatmap N/A')
        ]:
            ax.clear()
            ax.text(0.5, 0.5, title, ha='center', va='center',
                   fontsize=14, color='gray', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw()

    def update_price_vs_spot(self, S, K, T, r, sigma, option_type):
        """Graphique Prix vs Spot"""
        self.ax1.clear()
        spot_range = np.linspace(S * 0.5, S * 1.5, 100)

        if option_type == 'call':
            prices = [BlackScholesModel.call_price(s, K, T, r, sigma) for s in spot_range]
            current_price = BlackScholesModel.call_price(S, K, T, r, sigma)
        else:
            prices = [BlackScholesModel.put_price(s, K, T, r, sigma) for s in spot_range]
            current_price = BlackScholesModel.put_price(S, K, T, r, sigma)

        self.ax1.plot(spot_range, prices, linewidth=3, color='#3b8ed0', label=f'{option_type.capitalize()} Price')
        self.ax1.axvline(S, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='Current Spot')
        self.ax1.axvline(K, color='#f39c12', linestyle='--', alpha=0.7, linewidth=2, label='Strike')
        self.ax1.scatter([S], [current_price], color='#2ecc71', s=100, zorder=5, label='Current Value')

        self.ax1.set_xlabel('Prix du sous-jacent', fontsize=11, fontweight='bold')
        self.ax1.set_ylabel('Prix de l\'option', fontsize=11, fontweight='bold')
        self.ax1.set_title('Prix de l\'option vs Prix du sous-jacent', fontsize=13, fontweight='bold', pad=15)
        self.ax1.legend(loc='best')
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.fig1.tight_layout()
        self.canvas1.draw()

    def update_price_vs_vol(self, S, K, T, r, sigma, option_type):
        """Graphique Prix vs Volatilité"""
        self.ax2.clear()
        vol_range = np.linspace(0.05, 1.5, 100)

        if option_type == 'call':
            prices = [BlackScholesModel.call_price(S, K, T, r, v) for v in vol_range]
        else:
            prices = [BlackScholesModel.put_price(S, K, T, r, v) for v in vol_range]

        self.ax2.plot(vol_range * 100, prices, linewidth=3, color='#9b59b6')
        self.ax2.axvline(sigma * 100, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='Current Vol')

        self.ax2.set_xlabel('Volatilité (%)', fontsize=11, fontweight='bold')
        self.ax2.set_ylabel('Prix de l\'option', fontsize=11, fontweight='bold')
        self.ax2.set_title('Prix de l\'option vs Volatilité', fontsize=13, fontweight='bold', pad=15)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        self.fig2.tight_layout()
        self.canvas2.draw()

    def update_greeks_chart(self, S, K, T, r, sigma, option_type):
        """Graphique Greeks"""
        self.ax3.clear()
        spot_range = np.linspace(S * 0.7, S * 1.3, 100)

        deltas, gammas, vegas = [], [], []
        for s in spot_range:
            greeks = BlackScholesModel.calculate_greeks(s, K, T, r, sigma, option_type)
            deltas.append(greeks['Delta'])
            gammas.append(greeks['Gamma'] * 100)
            vegas.append(greeks['Vega'])

        self.ax3.plot(spot_range, deltas, linewidth=2, color='#3b8ed0', label='Delta', marker='o', markersize=3)
        self.ax3.plot(spot_range, gammas, linewidth=2, color='#e74c3c', label='Gamma (×100)', marker='s', markersize=3)
        self.ax3.axvline(S, color='white', linestyle='--', alpha=0.5)

        self.ax3.set_xlabel('Prix du sous-jacent', fontsize=11, fontweight='bold')
        self.ax3.set_ylabel('Valeur', fontsize=11, fontweight='bold')
        self.ax3.set_title('Greeks vs Prix du sous-jacent', fontsize=13, fontweight='bold', pad=15)
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3, linestyle='--')
        self.fig3.tight_layout()
        self.canvas3.draw()

    def update_payoff_chart(self, S, K, T, r, sigma, option_type):
        """Graphique Payoff"""
        self.ax4.clear()
        spot_range = np.linspace(K * 0.5, K * 1.5, 100)

        if option_type == 'call':
            current_price = BlackScholesModel.call_price(S, K, T, r, sigma)
            intrinsic = np.maximum(spot_range - K, 0)
            option_values = [BlackScholesModel.call_price(s, K, T, r, sigma) for s in spot_range]
        else:
            current_price = BlackScholesModel.put_price(S, K, T, r, sigma)
            intrinsic = np.maximum(K - spot_range, 0)
            option_values = [BlackScholesModel.put_price(s, K, T, r, sigma) for s in spot_range]

        profit = np.array(option_values) - current_price

        self.ax4.plot(spot_range, intrinsic, linewidth=2, color='#95a5a6', linestyle='--', label='Valeur intrinsèque')
        self.ax4.plot(spot_range, option_values, linewidth=3, color='#3b8ed0', label='Valeur option')
        self.ax4.axhline(0, color='white', linestyle='-', alpha=0.3)
        self.ax4.axvline(K, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='Strike')

        self.ax4.fill_between(spot_range, profit, 0, where=(profit > 0), alpha=0.3, color='#2ecc71', label='Profit')
        self.ax4.fill_between(spot_range, profit, 0, where=(profit < 0), alpha=0.3, color='#e74c3c', label='Perte')

        self.ax4.set_xlabel('Prix du sous-jacent à maturité', fontsize=11, fontweight='bold')
        self.ax4.set_ylabel('Payoff', fontsize=11, fontweight='bold')
        self.ax4.set_title('Diagramme de Payoff', fontsize=13, fontweight='bold', pad=15)
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3, linestyle='--')
        self.fig4.tight_layout()
        self.canvas4.draw()

    def update_3d_surface(self, S, K, T, r, option_type):
        """Surface 3D volatilité"""
        self.ax5.clear()

        spots = np.linspace(S * 0.7, S * 1.3, 30)
        vols = np.linspace(0.1, 0.8, 30)
        X, Y = np.meshgrid(spots, vols)
        Z = np.zeros_like(X)

        for i in range(len(spots)):
            for j in range(len(vols)):
                if option_type == 'call':
                    Z[j, i] = BlackScholesModel.call_price(X[j, i], K, T, r, Y[j, i])
                else:
                    Z[j, i] = BlackScholesModel.put_price(X[j, i], K, T, r, Y[j, i])

        surf = self.ax5.plot_surface(X, Y * 100, Z, cmap='viridis', alpha=0.9, edgecolor='none')
        self.ax5.set_xlabel('Spot Price', fontsize=9, fontweight='bold')
        self.ax5.set_ylabel('Volatilité (%)', fontsize=9, fontweight='bold')
        self.ax5.set_zlabel('Prix Option', fontsize=9, fontweight='bold')
        self.ax5.set_title('Surface de Volatilité 3D', fontsize=11, fontweight='bold', pad=10)
        self.fig5.colorbar(surf, ax=self.ax5, shrink=0.5)
        self.fig5.tight_layout()
        self.canvas5.draw()

    def update_heatmap(self, S, K, T, r, sigma, option_type):
        """Heatmap sensibilité"""
        self.ax6.clear()

        spots = np.linspace(S * 0.7, S * 1.3, 20)
        times = np.linspace(0.1, 2, 20)

        heatmap_data = np.zeros((len(times), len(spots)))

        for i, t in enumerate(times):
            for j, s in enumerate(spots):
                if option_type == 'call':
                    heatmap_data[i, j] = BlackScholesModel.call_price(s, K, t, r, sigma)
                else:
                    heatmap_data[i, j] = BlackScholesModel.put_price(s, K, t, r, sigma)

        im = self.ax6.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', origin='lower',
                            extent=[spots[0], spots[-1], times[0], times[-1]])

        self.ax6.set_xlabel('Prix du sous-jacent', fontsize=11, fontweight='bold')
        self.ax6.set_ylabel('Temps à maturité (années)', fontsize=11, fontweight='bold')
        self.ax6.set_title('Heatmap: Prix vs Spot vs Temps', fontsize=13, fontweight='bold', pad=15)
        self.fig6.colorbar(im, ax=self.ax6, label='Prix de l\'option')
        self.fig6.tight_layout()
        self.canvas6.draw()

    def save_scenario(self):
        """Sauvegarde le scénario actuel"""
        params = self.get_parameters()
        if params:
            S, K, T, r, sigma, option_type = params
            scenario = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                'type': option_type
            }
            self.scenarios_history.append(scenario)
            messagebox.showinfo("Succès", f"Scénario sauvegardé ! Total: {len(self.scenarios_history)}")

    def export_excel(self):
        """Export vers Excel"""
        try:
            params = self.get_parameters()
            if not params:
                return

            S, K, T, r, sigma, option_type = params

            data = {
                'Paramètre': ['Spot', 'Strike', 'Maturité', 'Taux', 'Volatilité'],
                'Valeur': [S, K, T, r*100, sigma*100]
            }

            df = pd.DataFrame(data)
            filename = f"option_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)

            messagebox.showinfo("Succès", f"Exporté vers {filename}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur d'export: {str(e)}")


def main():
    """Lance l'application"""
    app = AdvancedOptionPricerGUI()
    app.calculate_all()
    app.mainloop()


if __name__ == "__main__":
    main()
