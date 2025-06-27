from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
from datetime import date
import os
import json
from results import Results


def load_results(path="../data/mockup_results.json"):
    results = Results()
    # Load the results from a JSON file
    results.from_json(path)
    return results


def create_optimization_report():
    """
    Populates the optimization report template with example data and saves the result as a new Word document.
    """
    r = load_results("results.json")

    r.calculate_variables()
    r.calculate_financials()
    print(r._calculate_costs())
    r.print_results()
    # r.plot_one_day(["EGC_C_en", "BESS_P_chg", "BESS_P_dchg"])
    # r.plot_SOE("day", 1)
    r.plot_battery_operation(timeframe="day", num=120)
    r.plot_battery_operation(timeframe="week", num=20)
    r.print_financials()
    r.plot_cost_comparison()
    r.plot_grid_interaction("month", 6)
    r.plot_grid_interaction("week", 20)
    r.plot_grid_interaction("day", 120)
    r.plot_pv_production(120)
    r.plot_peak_comparison()

    template_path = "TBI 2 - template.docx"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    doc = DocxTemplate(template_path)

    image_path = "./images/Capture.png"
    my_image = InlineImage(
        doc, image_path, width=Mm(180)
    )  # 60mm wide, adjust as needed
    im_cpp_comparison = InlineImage(
        doc, "./images/im_cpp_comparison.png", width=Mm(180)
    )
    im_exchange_profile_comparison_day = InlineImage(
        doc, "./images/im_exchange_profile_comparison_day.png", width=Mm(180)
    )
    im_exchange_profile_comparison_week = InlineImage(
        doc, "./images/im_exchange_profile_comparison_week.png", width=Mm(180)
    )
    im_bess_profile_day = InlineImage(
        doc, "./images/im_battery_operation_day.png", width=Mm(180)
    )
    im_bess_profile_week = InlineImage(
        doc, "./images/im_battery_operation_week.png", width=Mm(180)
    )
    im_payoff_period = InlineImage(doc, "./images/im_payoff_period.png", width=Mm(180))
    im_aggregated_production_day = InlineImage(
        doc, "./images/im_aggregated_production_day.png", width=Mm(180)
    )

    context = {
        "client_name": "Metal Product d.o.o.",
        "total_bess_capacity_kwh": round(r.batt_E_new),
        "total_bess_power_kw": round(r.PC_P_new),
        "total_new_pv_capacity": round(sum(r.alpha_PV), 2),
        "annual_savings_amount": round(r.annual_savings[0], 2),
        "annual_savings_percentage": round(r.annual_savings_percentage[0], 2),
        "peak_demand_reduction_percentage": round(r.avg_peak_reduction * 100, 2),
        "panel_price": r.p.PV_c_inv,
        "battery_price": r.p.batt_c_inv,
        "converter_price": r.p.PC_c_inv,
        "energy_price_ht": max(r.p.EGC_c_en_buy),
        "energy_price_lt": min(r.p.EGC_c_en_buy),
        "peak_power_price": r.p.EGC_c_peak,
        "location_image": "my_image",
        "pv_arrays": [
            {
                "id": i + 1,
                "size_kwp_max": r.p.PV_alpha_max[i],
                "orientation": r.p.PV_orientation[i],
                "tilt": r.p.PV_inclination[i],
                "description": "",
                "new_capacity_kwp": r.alpha_PV[i],
            }
            for i in range(r.p.PV_N_arr)
        ],
        "pv_arrays_data": [
            {
                "id": i + 1,
                "size_kwp_max": r.p.PV_alpha_max[i],
                "orientation": r.p.PV_orientation[i],
                "tilt": r.p.PV_inclination[i],
                "description": "",
                "new_capacity_kwp": r.alpha_PV[i],
            }
            for i in range(r.p.PV_N_arr)
        ],
        "existing_pv_capacity": sum(r.p.PV_P_exist_peak),
        "total_pv_capacity": sum(r.p.PV_P_exist_peak) + sum(r.alpha_PV),
        "existing_battery_capacity": round(r.p.batt_E_exist),
        "existing_power_converter_power": round(r.p.PC_P_exist),
        "new_battery_capacity": round(r.batt_E_new),
        "new_power_converter_power": round(r.PC_P_new),
        "existing_grid_capacity": r.p.EGC_P_cap_exist,
        "pv_total_produced_mwh": round(
            r.pv_total_production_kWh / 1000, 2
        ),  # Convert to MWh
        "pv_consumption_ratio": round(r.pv_consumption_ratio, 2),
        "pv_serving_load_mwh": round(r.pv_serving_load_kWh / 1000, 2),  # Convert to MWh
        "pv_charging_battery_mwh": round(
            r.pv_charging_battery_kWh / 1000, 2
        ),  # Convert to MWh
        "pv_exported_to_grid_mwh": round(
            r.exported_to_grid_kWh / 1000, 2
        ),  # Convert to MWh
        "im_aggregated_production_day": im_aggregated_production_day,
        "annual_mwh_purchased_before": round(r.annual_kwh_purchased_baseline / 1000, 2),
        "annual_mwh_purchased": round(r.annual_kwh_purchased / 1000, 2),
        "annual_mwh_sold": round(r.annual_kwh_sold / 1000, 2),
        "net_mwh_exchange": round(r.net_kwh_exchange / 1000, 2),
        "net_savings": "{{ net_savings }}",
        # Mapped from im_exchange_profile_comparison in the context
        "im_exchange_profile_comparison_day": im_exchange_profile_comparison_day,
        "im_exchange_profile_comparison_week": im_exchange_profile_comparison_week,
        "bpp_1": round(r.p.EGC_P_fix_peak[0], 2),
        "bpp_2": round(r.p.EGC_P_fix_peak[1], 2),
        "bpp_3": round(r.p.EGC_P_fix_peak[2], 2),
        "bpp_4": round(r.p.EGC_P_fix_peak[3], 2),
        "bpp_5": round(r.p.EGC_P_fix_peak[4], 2),
        "bpp_6": round(r.p.EGC_P_fix_peak[5], 2),
        "bpp_7": round(r.p.EGC_P_fix_peak[6], 2),
        "bpp_8": round(r.p.EGC_P_fix_peak[7], 2),
        "bpp_9": round(r.p.EGC_P_fix_peak[8], 2),
        "bpp_10": round(r.p.EGC_P_fix_peak[9], 2),
        "bpp_11": round(r.p.EGC_P_fix_peak[10], 2),
        "bpp_12": round(r.p.EGC_P_fix_peak[11], 2),
        "pp_1": round(r.EGC_P_peak[0], 2),
        "pp_2": round(r.EGC_P_peak[1], 2),
        "pp_3": round(r.EGC_P_peak[2], 2),
        "pp_4": round(r.EGC_P_peak[3], 2),
        "pp_5": round(r.EGC_P_peak[4], 2),
        "pp_6": round(r.EGC_P_peak[5], 2),
        "pp_7": round(r.EGC_P_peak[6], 2),
        "pp_8": round(r.EGC_P_peak[7], 2),
        "pp_9": round(r.EGC_P_peak[8], 2),
        "pp_10": round(r.EGC_P_peak[9], 2),
        "pp_11": round(r.EGC_P_peak[10], 2),
        "pp_12": round(r.EGC_P_peak[11], 2),
        "im_peak_power_comparison": im_cpp_comparison,
        "annual_bess_throughput": round(r.BESS_annual_throughput, 2),
        "num_of_cycles": round(r.BESS_n_cycles),
        "im_bess_profile_day": im_bess_profile_day,
        "im_bess_profile_week": im_bess_profile_week,
        "pv_investment_cost": round(r.investment_cost_pv * r.p.N_DL, 2),
        "battery_investment_cost": round(r.investment_cost_batt * r.p.N_DL, 2),
        "pc_investment_cost": round(r.investment_cost_pc * r.p.N_DL, 2),
        "total_investment_cost": round(r.total_investment_cost * r.p.N_DL, 2),
        "baseline_energy_cost": round(r.baseline_energy_cost, 2),
        "optimized_energy_cost": round(r.total_energy_cost, 2),
        "energy_savings": round(r.total_energy_savings, 2),
        "baseline_peak_cost": round(r.baseline_peak_cost, 2),
        "optimized_peak_cost": round(r.optimized_peak_cost, 2),
        "peak_savings": round(r.peak_savings, 2),
        "baseline_maint_cost": round(r.baseline_maint_cost, 2),
        "optimized_maint_cost": round(r.optimized_maint_cost, 2),
        "maint_difference": round(r.maint_difference, 2),
        "baseline_degr_cost": round(r.baseline_degr_cost, 2),
        "optimized_degr_cost": round(r.optimized_degr_cost, 2),
        "degr_difference": round(r.degr_difference, 2),
        "baseline_total_cost": round(r.p.C_noinv_remain, 2),
        "optimized_total_cost": round(r.total_cost[0], 2),
        "total_savings": round(r.annual_savings[0], 2),
        "payback_period": round(r.payback_period[0], 2),
        "return_on_investment": round(r.roi[0], 2),
        "net_present_value": "{{ net_present_value }}",
        "disc_rate": "{{ disc_rate }}",
        "internal_rate_of_return": "{{ internal_rate_of_return }}",
        "lcoe": "{{ lcoe }}",
        "im_payoff_period": im_payoff_period,
        "self_sufficiency_percentage": round(r.pv_consumption_ratio, 2),
        # Geografski i vremenski parametri
        "longitude": r.p.longitude,
        "latitude": r.p.latitude,
        "el_t_s": r.p.el_t_s,
        # Opći ekonomski parametri projekta
        "N_DL": r.p.N_DL,
        "C_inv_max": r.p.C_inv_max,
        "N_payoff_max": r.p.N_payoff_max,
        # Parametri fotonaponskog (PV) sustava
        "PV_N_arr": r.p.PV_N_arr,
        "PV_orientation": r.p.PV_orientation,
        "PV_inclination": r.p.PV_inclination,
        "PV_c_inv": r.p.PV_c_inv,
        "PV_c_sub": r.p.PV_c_sub,
        "PV_c_mnt": r.p.PV_c_mnt,
        "PV_c_rpl": r.p.PV_c_rpl,
        "PV_N_LT": r.p.PV_N_LT,
        "PV_P_max": r.p.PV_P_max,
        "PV_alpha_min": r.p.PV_alpha_min,
        "PV_alpha_max": r.p.PV_alpha_max,
        # Parametri baterijskog sustava (BESS)
        "batt_N_cyc": r.p.batt_N_cyc,
        "batt_S_DoD": r.p.batt_s_DoD,
        "batt_beta_chg": r.p.batt_beta_chg,
        "batt_beta_dch": r.p.batt_beta_dch,
        "batt_c_inv": r.p.batt_c_inv,
        "batt_c_sub": r.p.batt_c_sub,
        "batt_c_mnt": r.p.batt_c_mnt,
        "batt_c_rpl": r.p.batt_c_rpl,
        "batt_E_new_min": r.p.batt_E_new_min,
        "batt_E_new_max": r.p.batt_E_new_max,
        # Parametri pretvarača (Power Converter - PC)
        "PC_eta_chg": r.p.PC_eta_chg,
        "PC_eta_dch": r.p.PC_eta_dch,
        "PC_c_inv": r.p.PC_c_inv,
        "PC_c_sub": r.p.PC_c_sub,
        "PC_c_mnt": r.p.PC_c_mnt,
        "PC_c_rpl": r.p.PC_c_rpl,
        "PC_N_LT": r.p.PC_N_LT,
        "PC_P_new_min": r.p.PC_P_new_min,
        "PC_P_new_max": r.p.PC_P_new_max,
        # Cijene električne energije i tarife
        "el_day_start": r.p.el_day_start,
        "el_day_end": r.p.el_day_end,
        "p_el_d_consume": r.p.p_el_d_consume,
        "p_el_n_consume": r.p.p_el_n_consume,
        "p_el_d_supply": r.p.p_el_d_supply,
        "p_el_n_supply": r.p.p_el_n_supply,
        # Parametri priključka na mrežu (EGC)
        "EGC_c_peak": r.p.EGC_c_peak,
        "EGC_c_cap_exist": r.p.EGC_P_cap_exist,
        "EGC_c_inv": r.p.EGC_c_inv,
        "EGC_P_cap_min": r.p.EGC_P_cap_min,
        "EGC_P_cap_max": r.p.EGC_P_cap_max,
    }

    doc.render(context)
    doc.save("optimization_report_2.docx")
    print("Document 'optimization_report_2.docx' created successfully!")


if __name__ == "__main__":
    # main()
    create_optimization_report()
