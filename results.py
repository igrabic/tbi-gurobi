from ast import Raise
from dataclasses import dataclass
import os
import json
from datetime import date, datetime, timedelta
import time
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


class Structure:
    pass


class Results:
    """Class to represent a result."""

    def __init__(self):
        # Colors for plotting
        self.DARK_BLUE = "#00313C"
        self.NAVY = "#012E40"
        self.TEAL = "#025959"
        self.GREEN = "#02735E"
        self.MINT = "#038C65"
        self.GREEN = "#00C389"
        self.colors = [self.DARK_BLUE, self.NAVY, self.TEAL, self.GREEN, self.MINT]

    def from_gurobi(self, model, p):
        """
        Populate Results fields from a solved Gurobi model.
        """
        # Extract variables by name from the model
        self.PV_P = [model.getVarByName(f"P_PV[{i}]").X for i in range(p.el_N_s)]
        self.BESS_P_chg = [
            model.getVarByName(f"P_chg_BESS[{i}]").X for i in range(p.el_N_s)
        ]
        self.BESS_P_dchg = [
            model.getVarByName(f"P_dch_BESS[{i}]").X for i in range(p.el_N_s)
        ]
        self.EGC_C_en = [
            model.getVarByName(f"C_en_EGC[{i}]").X for i in range(p.el_N_s)
        ]
        self.EGC_P_peak = [model.getVarByName(f"P_peak_EGC[{i}]").X for i in range(12)]
        self.alpha_PV = [
            model.getVarByName(f"alpha_PV_kW[{i}]").X for i in range(p.PV_N_arr)
        ]
        self.EGC_P_cap_new = model.getVarByName("Pcap_new_EGC").X
        self.batt_E_new = model.getVarByName("E_new_batt").X
        self.PV_pc_new = model.getVarByName("P_pc_new_PV").X
        self.EGC_Pcap_new = model.getVarByName("Pcap_new_EGC").X
        self.PC_P_new = model.getVarByName("P_new_PC").X
        self.BESS_E_0 = model.getVarByName("E_BESS_0").X

        self.objective_value = model.ObjVal if hasattr(model, "ObjVal") else None
        # Convert numpy arrays to lists if needed
        if hasattr(self.BESS_P_chg, "tolist"):
            self.BESS_P_chg = self.BESS_P_chg.tolist()
        if hasattr(self.BESS_P_dchg, "tolist"):
            self.BESS_P_dchg = self.BESS_P_dchg.tolist()
        if hasattr(self.EGC_C_en, "tolist"):
            self.EGC_C_en = self.EGC_C_en.tolist()
        if hasattr(self.EGC_P_peak, "tolist"):
            self.EGC_P_peak = self.EGC_P_peak.tolist()
        if hasattr(self.alpha_PV, "tolist"):
            self.alpha_PV = self.alpha_PV.tolist()

        self.p = p

    def from_sol(self, sol, p):
        n = p.el_N_s
        self.p = p
        self.BESS_P_chg = sol[0:n].tolist()
        self.BESS_P_dchg = sol[n : 2 * n].tolist()
        self.EGC_C_en = sol[2 * n : 3 * n].tolist()
        self.EGC_P_peak = sol[3 * n : 3 * n + 12].tolist()
        self.EGC_P_CPP = sol[3 * n + 12 : 3 * n + 24].tolist()
        self.alpha_PV = sol[3 * n + 24 : 3 * n + 24 + self.p.PV_N_arr].tolist()
        self.batt_E_new = sol[3 * n + 24 + self.p.PV_N_arr].tolist()
        self.PC_P_new = sol[3 * n + 24 + self.p.PV_N_arr + 1].tolist()
        self.EGC_P_cap_new = sol[3 * n + 24 + self.p.PV_N_arr + 2].tolist()
        self.BESS_E_0 = sol[3 * n + 24 + self.p.PV_N_arr + 3].tolist()

    def create_mockup_data(self):
        p = generate_data()
        n = p.el_N_s
        x = np.random.rand(3 * n + 24 + p.PV_N_arr + 3 + 1)
        self.from_sol(x, p)
        self.calculate_variables()
        self._calculate_costs()
        self.calculate_financials()

    def from_json(self, filename):
        """Load the result from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

            p = Structure()

            for key, value in data["p"].items():
                if isinstance(value, list):
                    print(f"Loading key {key}")
                    if all(isinstance(i, list) for i in value):
                        setattr(
                            p,
                            key,
                            np.array([list(map(float, sublist)) for sublist in value]),
                        )
                    else:
                        setattr(p, key, np.array(list(map(float, value))))
                elif isinstance(value, str):
                    if value.startswith("["):
                        values = value[:1:-1].split(" ")
                        values = list(map(float, values))
                        setattr(p, key, values)
                    else:
                        setattr(p, key, value)

                elif isinstance(value, (int, float)):
                    setattr(p, key, value)
                else:
                    raise ValueError(
                        f"Unsupported data type for key {key}: {type(value)}"
                    )
            self.p = p

            self.BESS_P_chg = np.array(list(map(float, data["BESS_P_chg"])))
            self.BESS_P_dchg = np.array(list(map(float, data["BESS_P_dchg"])))
            self.EGC_C_en = np.array(list(map(float, data["EGC_C_en"])))
            self.EGC_P_peak = np.array(list(map(float, data["EGC_P_peak"])))
            self.alpha_PV = np.array(list(map(float, data["alpha_PV"])))
            self.batt_E_new = float(data["batt_E_new"])
            self.PC_P_new = float(data["PC_P_new"])
            self.EGC_P_cap_new = float(data["EGC_P_cap_new"])
            self.BESS_E_0 = float(data["BESS_E_0"])
            self.PV_pc_new = float(data["PV_pc_new"]) if "PV_pc_new" in data else None
            self.PV_P = np.array(list(map(float, data["PV_P"])))
            self.x = np.concatenate(
                (
                    self.BESS_P_chg,
                    self.BESS_P_dchg,
                    self.EGC_C_en,
                    self.EGC_P_peak,
                    self.alpha_PV,
                    [self.batt_E_new],
                    [self.PC_P_new],
                    [self.EGC_P_cap_new],
                    [self.BESS_E_0],
                )
            )
            self.calculate_variables()
            self._calculate_costs()
            self.calculate_financials()

    def to_json(self, filename):
        """Save the result and parameters to a JSON file."""
        data = {
            "PV_P": self.PV_P,
            "PV_pc_new": self.PV_pc_new,
            "BESS_P_chg": self.BESS_P_chg,
            "BESS_P_dchg": self.BESS_P_dchg,
            "EGC_C_en": self.EGC_C_en,
            "EGC_P_peak": self.EGC_P_peak,
            "alpha_PV": self.alpha_PV,
            "batt_E_new": self.batt_E_new,
            "PC_P_new": self.PC_P_new,
            "EGC_P_cap_new": self.EGC_P_cap_new,
            "BESS_E_0": self.BESS_E_0,
            "p": {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)  # Convert numpy arrays to list
                    else (
                        value
                        if isinstance(value, (int, float, str, list, dict))
                        else str(value)
                    )  # Convert other non-serializable types to string
                )
                for key, value in vars(self.p).items()
            },
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def calculate_financials(self):
        """
        Calculate annual savings based on energy consumption and cost per kWh.
        """
        total_cost = self.total_cost
        base_cost = self.p.C_noinv_remain

        self.annual_savings = base_cost - total_cost
        self.annual_savings_percentage = (base_cost - total_cost) / base_cost * 100
        self.payback_period = (
            (
                self.investment_cost_batt
                + self.investment_cost_pc
                + self.investment_cost_grid
                + self.investment_cost_pv
            )
            * self.p.N_DL
            / self.annual_savings
        )

        # Savings calculation
        self.total_energy_savings = self.baseline_energy_cost - self.total_energy_cost
        self.peak_savings = self.baseline_peak_cost - self.optimized_peak_cost
        self.maint_difference = self.baseline_maint_cost - self.optimized_maint_cost
        self.inv_difference = -self.total_investment_cost

        self.baseline_degr_cost = 0
        self.degr_difference = self.baseline_degr_cost - self.optimized_degr_cost

        self.total_savings = (
            self.total_energy_savings
            + self.peak_savings
            + self.maint_difference
            + self.degr_difference
            + self.inv_difference
        )
        self.roi = (self.annual_savings * self.p.N_DL) / self.total_investment_cost

        # self.npv = (self.annual_savings * self.p.N_DL) / (
        #     1 + self.p.disc_rate
        # ) ** self.p.N_DL

    def run_tests(self):
        assert not np.isnan(self.P_EGC).any(), "P_EGC contains NaN values"

        print("total_cost", self.total_cost)
        print(
            "all components sum to: ",
            self.total_investment_cost
            + self.optimized_maint_cost
            + self.total_energy_cost
            + self.optimized_degr_cost
            + self.optimized_peak_cost,
        )
        try:
            assert np.isclose(
                self.total_cost,
                self.total_investment_cost
                + self.optimized_maint_cost
                + self.total_energy_cost
                + self.optimized_degr_cost,
                +self.optimized_peak_cost,
            )
        except AssertionError:
            print("total_cost", self.total_cost)
            print(
                "all components sum to: ",
                self.total_investment_cost
                + self.optimized_maint_cost
                + self.total_energy_cost
                + self.optimized_degr_cost,
            )
            Raise(
                AssertionError("Total cost does not match the sum of all components.")
            )

        print("annual", self.annual_savings)
        print(
            "all components sum to: ",
            self.total_energy_savings
            + self.peak_savings
            + self.maint_difference
            + self.degr_difference
            + self.inv_difference,
        )
        try:
            assert np.isclose(
                self.annual_savings,
                self.total_energy_savings
                + self.peak_savings
                + self.maint_difference
                + self.degr_difference
                + self.inv_difference,
            )
        except AssertionError:
            print("annual", self.annual_savings)
            print(
                "all components sum to: ",
                self.total_energy_savings
                + self.peak_savings
                + self.maint_difference
                + self.degr_difference
                + self.inv_difference,
            )
            raise (
                AssertionError("Total savings do not match the sum of all components.")
            )

        try:
            assert np.allclose(
                self.P_EGC,
                np.array(self.BESS_P_chg)
                - np.array(self.BESS_P_dchg)
                + np.array(self.p.EGC_P_fix)
                - self.PV_P,
            )
        except AssertionError:
            plt.plot(self.P_EGC, label="P_EGC", color=self.colors[0], linestyle="--")
            plt.plot(
                np.array(self.BESS_P_chg)
                - np.array(self.BESS_P_dchg)
                + np.array(self.p.EGC_P_fix)
                - self.PV_P
            )
            plt.show()
            err = self.P_EGC - (
                np.array(self.BESS_P_chg)
                - np.array(self.BESS_P_dchg)
                + np.array(self.p.EGC_P_fix)
                - self.PV_P
            )
            plt.plot(
                err,
                label="Error (P_EGC - calculated)",
                color=self.colors[1],
                linestyle="--",
            )
            plt.show()
            # Scatter plot of PV production vs. error
            plt.scatter(
                self.PV_P,
                err,
                color=self.MINT,
                alpha=0.7,
            )
            plt.xlabel("PV Production (kW)")
            plt.ylabel("Error (P_EGC - PV Production)")
            plt.title("Scatter Plot of PV Production vs. Error")
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig("./images/scatter_pv_vs_error.png", dpi=300)
            plt.show()
            raise AssertionError("P_EGC does not match expected values.")

    def print_financials(self):
        """
        Print the financial results of the optimization.
        """
        print("Annual Savings: ", self.annual_savings)
        print("Annual Savings Percentage: ", self.annual_savings_percentage)
        print("Payback period: ", self.payback_period)
        print("Peak Savings: ", self.peak_savings)
        print("Baseline Maintenance Cost: ", self.baseline_maint_cost)
        print("Investment Cost BESS: ", self.investment_cost_batt)
        print("Investment Cost PC: ", self.investment_cost_pc)
        print("Investment Cost Grid: ", self.investment_cost_grid)
        print("Investment Cost PV: ", self.investment_cost_pv)

        print("Total investment cost: ", self.total_investment_cost)

    def _calculate_costs(self):
        """
        Calculate the total cost based on energy consumption and cost per kWh.
        """
        # Calculate the total cost based on the given parameters

        BESS_f_chg = (
            365
            / self.p.N_days
            * self.p.batt_c_deg
            * self.p.el_t_s
            * np.ones((self.p.el_N_s, 1))
        )
        BESS_f_dch = (
            365
            / self.p.N_days
            * self.p.batt_c_deg
            * self.p.el_t_s
            * np.ones((self.p.el_N_s, 1))
        )
        EGC_f_en = 365 / self.p.N_days * np.ones((self.p.el_N_s, 1))

        EGC_f_peak = (
            12 / self.p.N_months * self.p.EGC_c_peak * np.ones((self.p.N_months, 1))
        )

        PV_f_alpha = (
            self.p.PV_c_inv / self.p.N_DL
            + self.p.PV_c_mnt
            + self.p.PV_c_rpl / self.p.PV_N_LT
        ) * np.ones((self.p.PV_N_arr, 1))
        # REP_f_alpha   = self.p.REP_C_en_full

        batt_f_new = self.p.batt_c_inv / self.p.N_DL + self.p.batt_c_mnt
        PC_f_new = (
            self.p.PC_c_inv / self.p.N_DL
            + self.p.PC_c_mnt
            + self.p.PC_c_rpl / self.p.PC_N_LT
        )
        EGC_f_cap_new = self.p.EGC_c_inv / self.p.N_DL
        BESS_f_E_0 = np.zeros((1, 1))

        f = np.vstack(
            (
                BESS_f_chg,
                BESS_f_dch,
                EGC_f_en,
                EGC_f_peak,
                PV_f_alpha,
                batt_f_new,
                PC_f_new,
                EGC_f_cap_new,
                BESS_f_E_0,
            )
        )

        self.total_cost = f.T @ self.x

        # Investment costs
        self.investment_cost_batt = self.p.batt_c_inv / self.p.N_DL * self.batt_E_new
        self.investment_cost_pc = self.p.PC_c_inv / self.p.N_DL * self.PC_P_new
        self.investment_cost_grid = self.p.EGC_c_inv / self.p.N_DL * self.EGC_P_cap_new
        self.investment_cost_pv = self.p.PV_c_inv / self.p.N_DL * np.sum(self.alpha_PV)
        self.total_investment_cost = (
            self.investment_cost_batt
            + self.investment_cost_pc
            + self.investment_cost_grid
            + self.investment_cost_pv
        )

        # Operational costs
        self.total_energy_cost = np.sum(self.EGC_C_en) * self.p.el_t_s
        self.baseline_energy_cost = (
            np.array(self.p.EGC_c_en_buy) @ np.array(self.p.EGC_P_fix) * self.p.el_t_s
        )

        self.baseline_peak_cost = np.sum(self.p.EGC_c_peak * self.p.EGC_P_fix_peak)
        self.optimized_peak_cost = np.sum(self.p.EGC_c_peak * self.EGC_P_peak)

        #      p.C_noinv_remain = np.sum(p.EGC_P_fix * p.EGC_c_en_buy) + np.sum(
        #     p.EGC_c_peak * p.EGC_EGC_P_fix_peak
        # )  # [€]

        assert np.isclose(
            self.baseline_energy_cost,
            np.sum(self.p.EGC_c_en_buy * self.p.EGC_P_fix) * self.p.el_t_s,
        ), "Baseline energy cost does not match expected value."

        print(
            "Baseline energy cost: ",
            self.baseline_energy_cost,
            np.sum(self.p.EGC_c_en_buy * self.p.EGC_P_fix) * self.p.el_t_s,
        )

        assert np.isclose(
            self.baseline_peak_cost,
            np.sum(self.p.EGC_c_peak * self.p.EGC_P_fix_peak),
        ), "Baseline peak cost does not match expected value."

        assert np.isclose(
            self.p.C_noinv_remain, self.baseline_energy_cost + self.baseline_peak_cost
        ), "Baseline cost does not match the sum of energy and peak costs."

        # Maintenance and degradation costs
        self.baseline_maint_cost = (
            self.p.PV_c_mnt * np.sum(self.p.PV_P_exist_peak)
            + self.p.batt_c_mnt * self.p.batt_E_exist
            + self.p.PC_c_mnt * self.p.PC_P_exist
        )
        self.optimized_maint_cost = (
            self.p.PV_c_mnt * np.sum(self.alpha_PV)
            + self.p.batt_c_mnt * self.batt_E_new
            + self.p.PC_c_mnt * self.PC_P_new
        )

        self.optimized_degr_cost = (
            (
                self.p.batt_c_deg
                * self.p.el_t_s
                * (np.sum(self.BESS_P_chg) + np.sum(self.BESS_P_dchg))
            )
            + (self.p.PV_c_rpl / self.p.PV_N_LT) * np.sum(self.alpha_PV)
            + +(self.p.PC_c_rpl / self.p.PC_N_LT) * self.PC_P_new
        )

    def _get_one_day(self, data, day_number):
        # Calculate samples per day (4 samples per hour * 24 hours)
        samples_per_day = self.p.el_N_s // 365
        # Calculate start and end indices for the requested day
        start_idx = (day_number - 1) * samples_per_day
        end_idx = start_idx + samples_per_day
        # Check if indices are within bounds
        if end_idx > len(data):
            raise ValueError(f"Day {day_number} exceeds the data range")

        # Extract and return the data for the requested day
        return data[start_idx:end_idx]

    def _get_one_week_data(self, vector, week_number):
        # Calculate samples per week (4 samples per hour * 24 hours * 7 days)
        samples_per_week = self.p.el_N_s // 52

        # Calculate start and end indices for the requested week
        start_idx = (week_number - 1) * samples_per_week
        end_idx = start_idx + samples_per_week
        # Check if indices are within bounds
        if end_idx > len(vector):
            raise ValueError(f"Week {week_number} exceeds the data range")

        # Extract and return the data for the requested week
        return vector[start_idx:end_idx]

    def plot_one_day(
        self, var_names=["BESS_P_chg"], day=1, save_path="./images/results_plot.png"
    ):
        """
        Plot the results of the optimization.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        vars = {}
        for var_name in var_names:
            if var_name.startswith("p."):
                attr = var_name[2:]
                vars[var_name] = getattr(self.p, attr, None)
            else:
                vars[var_name] = getattr(self, var_name, None)
        i = 0
        for var_name, var in vars.items():
            if var is None:
                raise ValueError(f"Variable {var_name} not found in results")
            plt.plot(self._get_one_day(var, day), label=var_name, color=self.colors[i])
            i += 1
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def calculate_variables(self):
        E_bess = np.zeros(len(self.BESS_P_chg))
        print(len(self.BESS_P_chg))
        assert len(E_bess) == self.p.el_N_s, "E_bess length does not match el_N_s"

        E_bess[0] = self.BESS_E_0
        for i in range(1, len(self.BESS_P_chg)):
            E_bess[i] = (
                E_bess[i - 1]
                + self.BESS_P_chg[i] * self.p.PC_eta_chg
                - self.BESS_P_dchg[i] * (1 / self.p.PC_eta_dch)
            )
        self.BESS_E = E_bess
        self.BESS_annual_throughput = np.sum(self.BESS_P_chg) * self.p.el_t_s
        self.BESS_n_cycles = (
            self.BESS_annual_throughput / (self.batt_E_new * self.p.batt_s_DoD)
            if self.batt_E_new > 0
            else 0
        )

        EGC_aux_1_1 = np.nan_to_num(
            np.diag(1 / np.array(self.p.el_t_s * self.p.EGC_c_en_buy).flatten()), nan=0
        )
        EGC_aux_1_2 = np.nan_to_num(
            np.diag(1 / np.array(self.p.el_t_s * self.p.EGC_c_en_sell).flatten()), nan=0
        )

        P_plus = EGC_aux_1_1 @ np.array(self.EGC_C_en).T
        P_minus = EGC_aux_1_2 @ np.array(self.EGC_C_en).T
        P_plus = np.clip(P_plus, 0, None)
        P_minus = np.clip(P_minus, None, 0)

        self.P_EGC = P_plus + P_minus

        self.annual_kwh_purchased = np.sum(P_plus) * self.p.el_t_s
        self.annual_kwh_sold = np.sum(P_minus) * self.p.el_t_s
        self.net_kwh_exchange = self.annual_kwh_purchased + self.annual_kwh_sold

        self.annual_kwh_purchased_baseline = np.sum(self.p.EGC_P_fix) * self.p.el_t_s

        self.avg_peak_reduction = np.mean(
            (self.p.EGC_P_fix_peak - self.EGC_P_peak) / self.p.EGC_P_fix_peak
        )

        # self.total_pv_production = (
        #     np.array(self.alpha_PV).reshape(len(self.alpha_PV), 1) * self.p.PV_P_ref
        # ).sum(axis=0)

        self.total_pv_production = self.p.PV_P_ref.T @ self.alpha_PV
        self.pv_total_production_kWh = (self.total_pv_production * self.p.el_t_s).sum()

        self.net_local_load = (
            self.total_pv_production
            + np.array(self.BESS_P_dchg)
            - np.array(self.BESS_P_chg)
        )

        self.pv_serving_load = np.minimum(self.p.EGC_P_fix, self.net_local_load)
        self.pv_serving_load_kWh = self.pv_serving_load.sum() * self.p.el_t_s

        self.pv_consumption_ratio = (
            self.pv_serving_load_kWh / self.pv_total_production_kWh
        ) * 100

        self.pv_charging_battery = np.minimum(
            np.clip(self.total_pv_production - self.p.EGC_P_fix, a_min=0, a_max=None),
            self.BESS_P_chg,
        )
        self.pv_charging_battery_kWh = self.pv_charging_battery.sum() * self.p.el_t_s

        self.exported_to_grid = self.total_pv_production - self.pv_serving_load
        self.exported_to_grid_kWh = self.exported_to_grid.sum() * self.p.el_t_s

    def plot_grid_interaction(self, timeframe="month", num=1):
        """
        Plot the grid interaction results.
        """
        start_date = datetime.strptime(self.p.start_date, "%Y-%m-%d")

        if timeframe.lower() == "day":
            P_EGC = self._get_one_day(self.P_EGC, num)
            print("P_EGC", P_EGC)
            EGC_P_fix = self._get_one_day(self.p.EGC_P_fix, num)
            dates = pd.date_range(
                start=start_date + timedelta(days=num - 1), periods=len(P_EGC), freq="H"
            )
        elif timeframe.lower() == "week":
            P_EGC = self._get_one_week_data(self.P_EGC, num)
            EGC_P_fix = self._get_one_week_data(self.p.EGC_P_fix, num)
            dates = pd.date_range(
                start=start_date + timedelta(weeks=num - 1),
                periods=len(P_EGC),
                freq="H",
            )
        elif timeframe.lower() == "month":
            P_EGC = self._get_one_month_data(self.P_EGC, num)
            EGC_P_fix = self._get_one_month_data(self.p.EGC_P_fix, num)
            dates = pd.date_range(start=f"2024-{num}-1", periods=len(P_EGC), freq="H")
        else:
            P_EGC = self.P_EGC
            EGC_P_fix = self.p.EGC_P_fix
            dates = pd.date_range(
                start=self.p.start_date, periods=len(self.P_EGC), freq="H"
            )

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plotting the grid interaction
        plt.plot(dates, P_EGC, label="Grid Interaction optimal", color=self.DARK_BLUE)
        plt.plot(
            dates,
            EGC_P_fix,
            label="Grid Interaction base",
            color=self.GREEN,
            linestyle="--",
        )

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Power (kW)")
        plt.title("Grid Interaction with Battery System")
        plt.grid()

        plt.tight_layout()
        plt.savefig("./images/im_exchange_profile_comparison_{}.png".format(timeframe))
        plt.show()

    def _get_one_month_data(self, data, month_number):
        """
        Get the data for a specific month.

        """
        start_idx = int(
            sum(self.p.days_in_month[: month_number - 1]) * (self.p.el_N_s / 365)
        )
        end_idx = int(sum(self.p.days_in_month[:month_number]) * (self.p.el_N_s / 365))
        month_data = data[start_idx:end_idx]
        return month_data

    def _get_peaks_monthly(self, data):
        """
        Get the monthly peaks from the data.
        """
        monthly_peaks = []
        for month in range(12):
            month_data = self._get_one_month_data(data, month + 1)
            peak = np.max(month_data)
            monthly_peaks.append(peak)

        return np.array(monthly_peaks)

    def plot_day_with_negative_prices(self, daynum=10):
        """
        Plot the grid interaction and highlight the day(s) with negative energy prices.
        """
        # Find indices (time steps) where energy price is negative
        negative_price_indices = np.where(np.array(self.p.EGC_c_en_buy) < 0)[0]
        if len(negative_price_indices) == 0:
            print("No negative prices found in EGC_c_en_buy.")
            return

        # Find the first day with negative price
        samples_per_day = self.p.el_N_s // 365
        day_numbers = np.unique(negative_price_indices // samples_per_day + 1)
        day = int(day_numbers[daynum - 1])
        print(f"First day with negative price: Day {day}")

        # Plot grid interaction for that day
        self.plot_one_day(["EGC_C_en", "p.EGC_c_en_buy"], day=day)
        self.plot_one_day(["p.EGC_c_en_buy"], day=day)
        self.plot_pv_production(day=day)
        self.plot_battery_operation(timeframe="day", num=day)
        self.plot_1_law(timeframe="day", num=day)

    def plot_1_law(self, timeframe="day", num=1):
        """
        Plot the 1-law results.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if timeframe.lower() == "day":
            EGC_P_fix = self._get_one_day(self.p.EGC_P_fix, num)
            P_EGC = self._get_one_day(self.P_EGC, num)
            BESS_P_chg = self._get_one_day(self.BESS_P_chg, num)
            BESS_P_dchg = self._get_one_day(self.BESS_P_dchg, num)
            PV_P = self._get_one_day(self.PV_P, num)

            # dates = pd.date_range(
            #     start=self.p.start_date + timedelta(days=num - 1),
            #     periods=len(EGC_P_fix),
            #     freq="H",
            # )
        elif timeframe.lower() == "week":
            EGC_P_fix = self._get_one_week_data(self.p.EGC_P_fix, num)
            P_EGC = self._get_one_week_data(self.P_EGC, num)
            BESS_P_chg = self._get_one_week_data(self.BESS_P_chg, num)
            BESS_P_dchg = self._get_one_week_data(self.BESS_P_dchg, num)
            PV_P = self._get_one_week_data(self.PV_P, num)
            # dates = pd.date_range(
            #     start=self.p.start_date + timedelta(weeks=num - 1),
            #     periods=len(EGC_P_fix),
            #     freq="H",
            # )
        else:
            EGC_P_fix = self.p.EGC_P_fix
            P_EGC = self.P_EGC
            BESS_P_chg = self.BESS_P_chg
            BESS_P_dchg = self.BESS_P_dchg
            PV_P = self.PV_P
            # dates = pd.date_range(
            #     start=self.p.start_date, periods=len(self.p.EGC_P_fix), freq="H"
            # )

        # Plotting the 1-law results
        plt.plot(
            P_EGC - EGC_P_fix - BESS_P_dchg + PV_P + BESS_P_chg,
            label="EGC P fix",
            color=self.DARK_BLUE,
        )

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Power (kW)")
        plt.title("Difference")
        plt.grid()

        plt.tight_layout()
        plt.savefig("./images/im_1_law_results.png")
        plt.show()

    def plot_peak_comparison(self):
        """
        Plot the peak comparison results.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        base_peaks = self.p.EGC_P_fix_peak

        # Plotting the peak comparison
        plt.plot(
            self.EGC_P_peak,
            label="Peak power optimized",
            color=self.DARK_BLUE,
            linestyle="--",
        )
        plt.plot(
            base_peaks, label="Peak power base case", color=self.GREEN, linestyle="--"
        )

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Power (kW)")
        plt.title("Peak Power Comparison")
        plt.grid()

        plt.tight_layout()
        plt.savefig("./images/im_cpp_comparison.png")
        plt.show()

    def plot_SOE(self, timeframe="DAY", num=1):
        """
        Plot the results of the optimization.
        """
        E_bess = self.BESS_E

        if timeframe.lower() == "day":
            E_bess = self._get_one_day(E_bess, num)
        if timeframe.lower() == "week":
            E_bess = self._get_one_week_data(E_bess, num)
        if timeframe.lower() == "all":
            pass

        plt.plot(E_bess, label="BESS Energy", color=self.colors[0])
        plt.hlines(
            y=self.batt_E_new,
            xmin=0,
            xmax=len(E_bess) - 1,
            color=self.MINT,
            linestyle="--",
            label="BESS Max Energy",
        )
        plt.tight_layout()
        plt.savefig("./images/results_plot.png")
        plt.show()

    def print_results(self):
        """
        Print the results of the optimization.
        """
        print("EGC_P_peak:", self.EGC_P_peak)
        print("alpha_PV:", self.alpha_PV)
        print("batt_E_new:", self.batt_E_new)
        print("PC_P_new:", self.PC_P_new)
        print("EGC_P_cap_new:", self.EGC_P_cap_new)
        print("BESS_E_0:", self.BESS_E_0)

    def plot_battery_operation(self, timeframe="day", num=1):
        """
        Plot the results of the optimization.
        """
        if isinstance(self.p.start_date, str):
            start_date_ordinal = datetime.strptime(
                self.p.start_date, "%Y-%m-%d"
            ).toordinal()
        else:
            start_date_ordinal = self.p.start_date.toordinal()

        if isinstance(self.p.end_date, str):
            end_date_ordinal = datetime.strptime(
                self.p.end_date, "%Y-%m-%d"
            ).toordinal()
        else:
            end_date_ordinal = self.p.end_date.toordinal()

        time = np.linspace(
            start_date_ordinal,
            end_date_ordinal,
            num=self.p.el_N_s,
        )

        baseline_grid_power = np.array(self.p.EGC_P_fix)
        discharge_power = np.array(self.BESS_P_dchg)
        actual_grid_power = np.array(self.P_EGC)
        soc_percent = (np.array(self.BESS_E) / self.batt_E_new) * 100
        pv_production = self.PV_P

        print("Shape of pv_production:", pv_production.shape)

        if timeframe.lower() == "day":
            baseline_grid_power = self._get_one_day(baseline_grid_power, num)
            discharge_power = self._get_one_day(discharge_power, num)
            actual_grid_power = self._get_one_day(actual_grid_power, num)
            soc_percent = self._get_one_day(soc_percent, num)
            pv_production = self._get_one_day(pv_production, num)
            time = self._get_one_day(time, num)
        elif timeframe.lower() == "week":
            baseline_grid_power = self._get_one_week_data(baseline_grid_power, num)
            discharge_power = self._get_one_week_data(discharge_power, num)
            actual_grid_power = self._get_one_week_data(actual_grid_power, num)
            soc_percent = self._get_one_week_data(soc_percent, num)
            pv_production = self._get_one_week_data(pv_production, num)
            time = self._get_one_week_data(time, num)

        fig, ax1 = plt.subplots(figsize=(14, 7))

        if np.sum(pv_production) == 0:
            ax1.stackplot(
                time,
                actual_grid_power,
                discharge_power,
                labels=[
                    "Actual Grid Supply",
                    "Battery Discharge Supply",
                    "PV Production",
                ],
                colors=[self.DARK_BLUE, self.GREEN, self.MINT],
                alpha=0.8,
            )
        else:
            ax1.stackplot(
                time,
                actual_grid_power,
                discharge_power,
                pv_production,
                labels=[
                    "Actual Grid Supply",
                    "Battery Discharge Supply",
                    "PV Production",
                ],
                colors=[self.DARK_BLUE, self.GREEN, self.MINT],
                alpha=0.8,
            )

        ax1.plot(
            time,
            baseline_grid_power,
            label="Baseline Grid Power",
            color=self.MINT,
            linestyle="--",
            linewidth=1.5,
        )

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title(
            "Industrial Load Serving: Actual Grid vs. Battery Supply with Optimal Dispatch"
        )
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Create a secondary Y-axis for SOC
        ax2 = ax1.twinx()
        ax2.plot(
            time,
            soc_percent,
            label="Battery SOC (%)",
            color=self.GREEN,
            linestyle="-",
            linewidth=2.0,
        )
        ax2.set_ylabel("Battery SOC (%)", color=self.GREEN)
        ax2.tick_params(axis="y", labelcolor=self.GREEN)
        ax2.set_ylim(0, 100)  # SOC is always between 0 and 100 percent

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # Place legend

        plt.tight_layout()  # Adjust layout to prevent labels overlapping
        plt.savefig(
            "./images/im_battery_operation_{}.png".format(timeframe), dpi=300
        )  # Save the plot
        plt.show()

    def plot_cost_comparison(self):
        years = 25

        time = range(years)  # Create a time range for the x-axis

        cumulative_cost_with_investment = (
            np.cumsum(np.ones((1, years)) * self.total_cost)
            + self.total_investment_cost * self.p.N_DL
        )

        cumulative_cost_baseline = np.cumsum(np.tile(self.p.C_noinv_remain, years))

        fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axes

        # Plot the cumulative cost lines
        ax.plot(
            time,
            cumulative_cost_baseline,
            label="Cumulative Cost (Existing Scenario)",
            color=self.DARK_BLUE,
            linewidth=2,
        )
        ax.plot(
            time,
            cumulative_cost_with_investment,
            label="Cumulative Cost (With Investment)",
            color=self.GREEN,
            linewidth=2,
        )

        payback_year = int(self.payback_period[0])

        ax.plot(
            payback_year,
            cumulative_cost_with_investment[payback_year],
            "go",
            markersize=8,
            label=f"Estimated Payoff Point ({ round(self.payback_period[0],2) } years)",
        )

        final_difference = (
            cumulative_cost_baseline[-1] - cumulative_cost_with_investment[-1]
        )
        ax.text(
            time[-1],
            cumulative_cost_with_investment[-1] - 1e6,
            f"Savings: €{final_difference:,.2f} after {years} years",
            color=self.NAVY,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.legend()
        plt.savefig("./images/im_payoff_period.png", dpi=300)
        plt.show()

    def plot_pv_production(self, day=1):
        """
        Plot the PV production for a specific day.
        """
        pv_production_max = self.total_pv_production
        pv_production_real = self.PV_P

        pv_production_max = self._get_one_day(pv_production_max, day)
        pv_production_real = self._get_one_day(pv_production_real, day)

        freq = "H" if self.p.el_t_s == 1 else f"{self.p.el_t_s}H"
        time = pd.date_range(
            start=self.p.start_date, periods=len(pv_production_real), freq=freq
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            time,
            pv_production_real,
            label="PV Production",
            color=self.MINT,
            linewidth=2,
        )
        ax.plot(
            time,
            pv_production_max,
            label="PV Production",
            color=self.DARK_BLUE,
            linewidth=2,
        )

        for i, alpha in enumerate(self.alpha_PV):
            individual_pv_production = alpha * self.p.PV_P_ref[i]
            ax.plot(
                time,
                self._get_one_day(individual_pv_production, day),
                label=f"PV Array {i+1}",
                linestyle="--",
                linewidth=1,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"PV Production for Day {day}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./images/im_aggregated_production_day.png", dpi=300)
        plt.show()

    def plot_all(self):
        self.plot_grid_interaction(timeframe="day", num=15)
        self.plot_grid_interaction(timeframe="week", num=3)
        self.plot_grid_interaction(timeframe="month", num=1)
        self.plot_battery_operation("day", 15)
        self.plot_peak_comparison()
        self.plot_cost_comparison()
        self.plot_pv_production(day=10)
        # self.plot_SOE(timeframe="day", num=10)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and visualize results.")
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="Path to the JSON file containing the results.",
    )
    args = parser.parse_args()

    # Example usage
    result = Results()
    if not args.path:
        print("No path provided, creating mockup data.")
        result.create_mockup_data()  # Create mockup data if no path is provided
    else:
        print(f"Loading results from {args.path}")
        if not os.path.exists(args.path):
            raise FileNotFoundError(f"The file {args.path} does not exist.")
        result.from_json(args.path)  # Load the results from the specified JSON file

    result.print_results()
    # result.plot_grid_interaction(timeframe="day", num=5)
    # result.plot_grid_interaction(timeframe="week", num=5)
    # result.plot_grid_interaction(timeframe="month", num=5)
    # result.plot_battery_operation(day=10)
    # result.print_financials()
    # result.plot_peak_comparison()
    # result.plot_cost_comparison()
    # for day in range(10, 20):
    #     result.plot_day_with_negative_prices(daynum=day)
    # result.plot_pv_production(day=16)
    result.plot_battery_operation("day", 16)
    # result.plot_one_day(var_names=["BESS_P_dchg", "BESS_P_chg"], day=16)
    # result.plot_all()
    result.run_tests()

    # Export results.p.EGC_c_en_buy to CSV
    if hasattr(result.p, "EGC_c_en_buy"):
        pd.DataFrame(result.p.EGC_c_en_buy).to_csv(
            "EGC_c_en_buy.csv", index=False, header=["EGC_c_en_buy"]
        )
        print("EGC_c_en_buy has been exported to EGC_c_en_buy.csv")
    else:
        print("EGC_c_en_buy attribute not found in results.p")
