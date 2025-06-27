from xml.parsers.expat import model
from requests import get
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from utils import *
import pandas as pd
from results import Results
from data_generation import generate_data


def solve_energy_hub_optimization():
    """
    This function formulates and solves the energy hub optimization problem
    based on the document "problem formulation TBI v0.21.docx" and Scenario 1 parameters.
    """

    p = generate_data(tbi="TBI2", scenario="Scenario 6")
    print("Data generation complete.")

    # Create a new model
    m = gp.Model("EnergyHubOptimization_v0_21")

    # --- Parameters from Scenario 1 ---
    print("Setting up parameters...")
    # General
    N_days = p.N_days  # Number of days in the planning horizon
    N_months = p.N_months
    ts_BESS = p.BESS_t_s  # Sampling time in hours
    Nh_BESS = p.el_N_s
    NDL = p.NDL if hasattr(p, "NDL") else 40  # Design Life in years
    N_payoff_max = (
        p.N_payoff_max if hasattr(p, "N_payoff_max") else 20
    )  # Maximum payoff period in years
    C_inv_max = (
        p.C_inv_max if hasattr(p, "C_inv_max") else 1000000
    )  # Maximum investment budget in Euros

    # PV System
    Narr_PV = p.PV_N_arr
    cinv_PV = p.PV_c_inv
    cmnt_PV = p.PV_c_mnt
    crpl_PV = p.PV_c_rpl
    NLT_PV = p.PV_N_LT
    cpc_inv_PV = 300
    cpc_mnt_PV = 30
    cpc_rpl_PV = 300
    Npc_LT_PV = 10
    P_pc_exist_PV = 0
    PV_alpha_ub = np.minimum(p.PV_P_max, p.PV_alpha_max)

    # PV production profile
    PV_ref = p.PV_P_ref
    Pexist_PV = (
        p.PV_P_exist.sum(axis=1) if len(p.PV_P_exist.shape) == 2 else p.PV_P_exist
    )

    # BESS
    s_DoD_batt = p.batt_s_DoD
    cinv_batt = p.batt_c_inv
    crpl_batt = p.batt_c_rpl
    N_cyc_batt = p.batt_N_cyc
    cdeg_batt = p.batt_c_deg
    cmnt_batt = p.batt_c_mnt
    E_exist_batt = p.batt_E_exist

    # BESS Power Converter
    cinv_PC = p.PC_c_inv
    crpl_PC = p.PC_c_rpl
    NLT_PC = p.PC_N_LT
    cdeg_PC = crpl_PC / NLT_PC
    cmnt_PC = p.PC_c_mnt
    P_exist_PC = p.PC_P_exist
    eta_chg_BESS = p.BESS_eta_chg if hasattr(p, "BESS_eta_chg") else 0.93
    eta_dch_BESS = p.BESS_eta_dchg if hasattr(p, "BESS_eta_dchg") else 0.93
    beta_chg_batt = p.batt_beta_chg if hasattr(p, "batt_beta_chg") else 1.0
    beta_dch_batt = p.batt_beta_dchg if hasattr(p, "batt_beta_dchg") else 1.0

    # Electrical Grid Connection (EGC)
    Pcap_exist_EGC = p.EGC_P_exist if hasattr(p, "EGC_P_exist") else 1046
    cinv_EGC = p.EGC_c_inv
    Pfix_EGC = p.EGC_P_fix

    # Electricity prices based on day/night tariff
    day_start_hour = p.el_day_start
    day_end_hour = p.el_day_end
    p_el_d_consume = p.p_el_d_consume
    p_el_n_consume = p.p_el_n_consume
    p_el_d_supply = p.p_el_d_supply
    p_el_n_supply = p.p_el_n_supply
    c_power_EGC = p.EGC_c_peak

    cen_buy_EGC = p.EGC_c_en_buy
    cen_sell_EGC = p.EGC_c_en_sell
    days_in_month = p.days_in_month

    print("Shapes of arrays and variables:")
    print(f"Pref_PV_a_resampled: {PV_ref.shape}")
    print(f"Pexist_PV: {Pexist_PV.shape}")
    print(f"cen_buy_EGC: {cen_buy_EGC.shape}")
    print(f"cen_sell_EGC: {cen_sell_EGC.shape}")
    print(f"Pfix_EGC: {Pfix_EGC.shape}")

    # --- Optimization Variables ---
    # Note: alpha_PV is now the installed capacity in kW
    alpha_PV = m.addMVar(
        Narr_PV, vtype=GRB.CONTINUOUS, name="alpha_PV_kW", lb=0.0, ub=PV_alpha_ub
    )
    P_PV = m.addMVar(Nh_BESS, vtype=GRB.CONTINUOUS, name="P_PV", lb=0.0)
    P_pc_new_PV = m.addVar(vtype=GRB.CONTINUOUS, name="P_pc_new_PV", lb=0.0)

    E_new_batt = m.addVar(vtype=GRB.CONTINUOUS, name="E_new_batt", lb=0.0)
    P_new_PC = m.addVar(vtype=GRB.CONTINUOUS, name="P_new_PC", lb=0.0)
    P_chg_BESS = m.addMVar(Nh_BESS, vtype=GRB.CONTINUOUS, name="P_chg_BESS", lb=0.0)
    P_dch_BESS = m.addMVar(Nh_BESS, vtype=GRB.CONTINUOUS, name="P_dch_BESS", lb=0.0)
    E_BESS_0 = m.addVar(vtype=GRB.CONTINUOUS, name="E_BESS_0", lb=0.0)

    Pcap_new_EGC = m.addVar(vtype=GRB.CONTINUOUS, name="Pcap_new_EGC", lb=0.0)
    C_en_EGC = m.addMVar(
        Nh_BESS, vtype=GRB.CONTINUOUS, name="C_en_EGC", lb=-GRB.INFINITY
    )
    P_peak_EGC = m.addMVar(N_months, vtype=GRB.CONTINUOUS, name="P_peak_EGC", lb=0.0)

    # --- Objective Function ---
    # Cost coefficients for variables
    f_alpha_PV = (cinv_PV / NDL) + cmnt_PV + (crpl_PV / NLT_PV)
    f_pc_new_PV = (cpc_inv_PV / NDL) + cpc_mnt_PV + (cpc_rpl_PV / Npc_LT_PV)
    f_new_batt = (cinv_batt / NDL) + cmnt_batt
    f_new_PC = (cinv_PC / NDL) + cmnt_PC + (crpl_PC / NLT_PC)
    f_cap_new_EGC = cinv_EGC / NDL

    # Annualized Investment, Maintenance, and Degradation costs for new components
    J_inv_mnt_deg = (
        f_alpha_PV * alpha_PV.sum()
        + f_pc_new_PV * P_pc_new_PV
        + f_new_batt * E_new_batt
        + f_new_PC * P_new_PC
        + f_cap_new_EGC * Pcap_new_EGC
    )

    # BESS degradation from operation
    J_deg_BESS = (
        (365 / N_days) * cdeg_batt * ts_BESS * (P_chg_BESS.sum() + P_dch_BESS.sum())
    )

    # Operational Costs
    J_opr = (365 / N_days) * C_en_EGC.sum() + (
        12 / N_months
    ) * c_power_EGC * P_peak_EGC.sum()

    m.setObjective(J_inv_mnt_deg + J_deg_BESS + J_opr, GRB.MINIMIZE)

    # --- Constraints ---
    # Intermediate expressions
    E_total_batt = E_exist_batt + E_new_batt
    P_total_PC = P_exist_PC + P_new_PC
    P_total_pc_PV = P_pc_exist_PV + P_pc_new_PV

    # PV-1.1: PV Size constraint
    print("Setting up PV-1.1: PV Size constraint...")
    P_new_PV_total = PV_ref.T @ alpha_PV

    assert P_new_PV_total.shape == (
        Nh_BESS,
    ), "P_new_PV_total should have shape (Nh_BESS,)"
    assert P_PV.shape == (Nh_BESS,), "P_PV should have shape (Nh_BESS,)"
    assert Pexist_PV.shape == (Nh_BESS,), "Pexist_PV should have shape (Nh_BESS,)"

    m.addConstrs(
        (P_PV[k] <= Pexist_PV[k] + P_new_PV_total[k] for k in range(Nh_BESS)),
        name="PV_1_1_Size",
    )

    # PV-2.1: PV power converter constraint
    print("Setting up PV-2.1: PV Power Converter constraint...")
    m.addConstrs((P_PV[k] <= P_total_pc_PV for k in range(Nh_BESS)), name="PV_2_1_PC")

    # BESS-1.0: Sequence repeatability
    print("Setting up BESS-1.0: Sequence repeatability constraint...")
    m.addConstr(
        eta_chg_BESS * ts_BESS * P_chg_BESS.sum()
        - (1 / eta_dch_BESS) * ts_BESS * P_dch_BESS.sum()
        == 0,
        "BESS_1_0_repeatability",
    )

    # BESS-2: State variables limits (dynamics and bounds)
    print("Setting up BESS-2: State variables limits...")
    E_bess = (
        lambda k: gp.quicksum(
            (P_chg_BESS[i] * eta_chg_BESS - P_dch_BESS[i] / eta_dch_BESS) * ts_BESS
            for i in range(k)
        )
        + E_BESS_0
    )

    L = np.tril(np.ones((Nh_BESS, Nh_BESS)))
    psi_ch = eta_chg_BESS * ts_BESS * L
    psi_dch = (1 / eta_dch_BESS) * ts_BESS * L

    # m.addConstrs(
    #     ((1 - s_DoD_batt) * E_total_batt <= E_bess(k) for k in range(Nh_BESS)),
    #     name="BESS_2_1_SoC_min",
    # )
    # m.addConstrs(
    #     (E_bess(k) <= E_total_batt for k in range(Nh_BESS)), name="BESS_2_2_SoC_max"
    # )

    m.addConstr(
        (psi_ch @ P_chg_BESS - psi_dch @ P_dch_BESS >= E_total_batt * (1 - s_DoD_batt)),
        "BESS_SoC_min",
    )
    m.addConstr(
        psi_ch @ P_chg_BESS - psi_dch @ P_dch_BESS <= E_total_batt, "BESS_SoC_max"
    )

    m.addConstr((1 - s_DoD_batt) * E_total_batt <= E_BESS_0, "BESS_SoC_0_min")
    m.addConstr(E_BESS_0 <= E_total_batt, "BESS_SoC_0_max")

    # BESS-3.0: Size of power converter
    print("Setting up BESS-3.0: Size of power converter constraint...")
    m.addConstrs(
        (P_chg_BESS[k] + P_dch_BESS[k] <= P_total_PC for k in range(Nh_BESS)),
        name="BESS_3_0_PC_size",
    )

    # BESS-4: C-rate limit
    print("Setting up BESS-4: C-rate limit constraints...")

    print("Shapes of variables in BESS-4:")
    print(f"P_chg_BESS shape: {P_chg_BESS.shape}")
    print(f"P_dch_BESS shape: {P_dch_BESS.shape}")
    # print(f"E_total_batt shape: {E_total_batt}")
    print(f"beta_chg_batt: {beta_chg_batt}")
    print(f"beta_dch_batt: {beta_dch_batt}")
    print(f"P_total_PC shape: {P_total_PC}")

    m.addConstr(
        (P_total_PC <= beta_chg_batt * E_total_batt),
        name="BESS_4_1_C_rate_chg",
    )
    m.addConstr(
        (P_total_PC <= beta_dch_batt * E_total_batt),
        name="BESS_4_2_C_rate_dch",
    )

    # EGC-1: Grid energy billing
    print("Setting up EGC-1: Grid energy billing constraints...")

    # Note: P_grid_EGC is the net power exchanged with the grid
    P_grid_EGC = P_chg_BESS - P_dch_BESS - P_PV + Pfix_EGC - Pexist_PV
    m.addConstrs(
        (C_en_EGC[k] >= cen_buy_EGC[k] * P_grid_EGC[k] for k in range(Nh_BESS)),
        name="EGC_1_1_buy_cost",
    )
    m.addConstrs(
        (C_en_EGC[k] >= cen_sell_EGC[k] * P_grid_EGC[k] for k in range(Nh_BESS)),
        name="EGC_1_2_sell_cost",
    )

    # EGC-2: Peak power billing (Low Voltage case)
    print("Setting up EGC-2: Peak power billing constraints...")
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    steps_per_day = int(24 / ts_BESS)
    step_of_year = 0
    for j in range(N_months):
        steps_in_month = int(days_in_month[j] * steps_per_day)
        month_indices = range(step_of_year, step_of_year + steps_in_month)
        m.addConstrs(
            (P_peak_EGC[j] >= P_grid_EGC[k] for k in month_indices),
            name=f"EGC_2_0_peak_pos_{j}",
        )
        m.addConstrs(
            (P_peak_EGC[j] >= -P_grid_EGC[k] for k in month_indices),
            name=f"EGC_2_1_peak_neg_{j}",
        )
        step_of_year += steps_in_month

    # EGC-3.0: Electrical grid connection enlargement
    print("Setting up EGC-3.0: Electrical grid connection enlargement constraints...")
    P_total_cap_EGC = Pcap_exist_EGC + Pcap_new_EGC
    m.addConstrs(
        (P_peak_EGC[j] <= P_total_cap_EGC for j in range(N_months)),
        name="EGC_3_0_cap_limit",
    )

    # FIN-2.0: Maximum budget
    print("Setting up FIN-2.0: Maximum budget constraint...")
    C_inv_total = (
        cinv_PV * alpha_PV.sum()
        + cpc_inv_PV * P_pc_new_PV
        + cinv_batt * E_new_batt
        + cinv_PC * P_new_PC
        + cinv_EGC * Pcap_new_EGC
    )
    m.addConstr(C_inv_total <= C_inv_max, "FIN_2_0_Max_Budget")

    # --- Solve ---
    m.Params.Method = 1  # Use dual simplex
    m.optimize()

    # --- Print Results ---
    if m.status == GRB.OPTIMAL:
        print("\nOptimal solution found!")
        print(f"Objective Value (Total Annual Cost): {m.ObjVal:.2f} €/year")
        print("\n--- Investments ---")
        for a in range(Narr_PV):
            print(f"PV Array {a+1} New Capacity: {alpha_PV[a].X:.2f} kW")
        print(f"New PV Power Converter Capacity: {P_pc_new_PV.X:.2f} kW")
        print(f"New Battery Capacity: {E_new_batt.X:.2f} kWh")
        print(f"New BESS Power Converter Capacity: {P_new_PC.X:.2f} kW")
        print(f"New Grid Capacity Enlargement: {Pcap_new_EGC.X:.2f} kW")
        print(f"Total Grid Capacity: {Pcap_exist_EGC + Pcap_new_EGC.X:.2f} kW")
        solution_filename = "my_model_solution.sol"
        m.write(solution_filename)
    else:
        print("No optimal solution found.")
        if m.status == GRB.INFEASIBLE:
            print(
                "Model is infeasible. Computing IIS to find conflicting constraints..."
            )
            m.computeIIS()
            m.write("model.ilp")
            print(
                "IIS written to model.ilp. You can open this file to see the irreducible inconsistent subsystem."
            )

    results = Results()
    results.from_gurobi(m, p)
    results.to_json("results.json")
    # results.plot_all()


if __name__ == "__main__":
    solve_energy_hub_optimization()
