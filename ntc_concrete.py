import math

def calc_As(m_ed_Nmm, n_ed_N, b_mm, bw_mm, d_mm, h_mm, hf_mm, fcd, fyd, fck):
    y_s = d_mm - h_mm / 2.0
    m_eds = m_ed_Nmm - n_ed_N * y_s
    if m_eds <= 0: return 0.0 
    
    x_max = 0.45 * d_mm
    M_c_flange = 0.8 * b_mm * hf_mm * fcd * (d_mm - 0.4 * hf_mm)
    
    if m_eds <= M_c_flange or hf_mm >= 0.8 * x_max:
        mu = m_eds / (b_mm * d_mm**2 * fcd)
        if mu > 0.5: mu = 0.5 
        omega = 1 - math.sqrt(max(0, 1 - 2*mu))
        x = omega * d_mm / 0.8
        As = (0.8 * b_mm * x * fcd - n_ed_N) / fyd
    else:
        F_cf = (b_mm - bw_mm) * hf_mm * fcd
        M_f = F_cf * (d_mm - hf_mm / 2.0)
        
        m_eds_web = m_eds - M_f
        M_c_max_web = 0.8 * bw_mm * x_max * fcd * (d_mm - 0.4 * x_max)
        
        if m_eds_web <= M_c_max_web:
            mu_w = m_eds_web / (bw_mm * d_mm**2 * fcd)
            if mu_w > 0.5: mu_w = 0.5 
            omega_w = 1 - math.sqrt(max(0, 1 - 2*mu_w))
            x = omega_w * d_mm / 0.8
            As = (0.8 * bw_mm * x * fcd + F_cf - n_ed_N) / fyd
        else:
            d_prime = 40.0
            delta_M = m_eds_web - M_c_max_web
            As_comp = delta_M / (fyd * (d_mm - d_prime))
            As = (0.8 * bw_mm * x_max * fcd + F_cf - n_ed_N) / fyd + As_comp
            
    return max(0.0, As)

def calc_Mrd(As, n_ed_N, b_mm, bw_mm, d_mm, h_mm, hf_mm, fcd, fyd):
    x = (As * fyd + n_ed_N) / (0.8 * b_mm * fcd)
    if 0.8 * x <= hf_mm:
        if x < 0: x = 0
        if x > d_mm: x = d_mm 
        C_c = 0.8 * b_mm * x * fcd
        M_rd = C_c * (h_mm/2.0 - 0.4*x) + As * fyd * (d_mm - h_mm/2.0)
    else:
        F_cf = (b_mm - bw_mm) * hf_mm * fcd
        x = (As * fyd + n_ed_N - F_cf) / (0.8 * bw_mm * fcd)
        if x < 0: x = 0
        if x > d_mm: x = d_mm 
        C_cw = 0.8 * bw_mm * x * fcd
        M_rd = F_cf * (h_mm/2.0 - hf_mm/2.0) + C_cw * (h_mm/2.0 - 0.4*x) + As * fyd * (d_mm - h_mm/2.0)
    return max(0.0, M_rd / 1e6)

def choose_bars(As_req, phi_bar, as_min=0.0):
    As_target = max(As_req, as_min)
    area_phi = math.pi * (phi_bar**2)/4
    
    # Se non c'è sforzo, mettiamo almeno 1 ferro di base
    if As_target <= 0: 
        return 1, area_phi * 1
        
    n = math.ceil(As_target / area_phi)
    
    # Rimuoviamo il blocco dei 2 ferri: consentiamo n = 1
    if n < 1: n = 1 
    
    return n, n * area_phi

def build_envelope(arrays_of_scenarios):
    if not arrays_of_scenarios: return [], []
    num_points = len(arrays_of_scenarios[0])
    max_env = [-float('inf')] * num_points
    min_env = [float('inf')] * num_points
    for scenario in arrays_of_scenarios:
        for i in range(num_points):
            val = scenario[i]
            if val > max_env[i]: max_env[i] = val
            if val < min_env[i]: min_env[i] = val
    return max_env, min_env

def design_member_premium(m_ed_scenarios, n_ed_scenarios, v_ed_scenarios, lengths, has_shear_reinf, b_mm, h_mm, c_mm, fck, fyk, phi_long, phi_shear, bw_mm, hf_mm, is_solaio=False):
    m_max_env, m_min_env = build_envelope(m_ed_scenarios)
    n_max_env, n_min_env = build_envelope(n_ed_scenarios)
    v_max_env, v_min_env = build_envelope(v_ed_scenarios)
    
    num_webs = b_mm / 500.0 if is_solaio else 1.0
    bw_eff = bw_mm * num_webs if is_solaio else bw_mm
    
    m_req_inf = [max(0, m) for m in m_max_env]
    m_req_sup = [max(0, -m) for m in m_min_env]
    n_comp_array = [-min(0, n) * 1000 for n in n_min_env]
    v_ed_abs_local = [max(abs(v_max_env[i]), abs(v_min_env[i])) for i in range(len(v_max_env))]
    
    L_m = sum(lengths)
    gamma_c, gamma_s, alpha_cc = 1.5, 1.15, 0.85
    fcd = alpha_cc * fck / gamma_c
    fyd = fyk / gamma_s
    d = h_mm - c_mm
    fctm = 0.3 * (fck**(2/3))
    
    As_min = max(0.26 * (fctm / fyk) * bw_eff * d, 0.0013 * bw_eff * d)
    As_max = 0.04 * b_mm * h_mm
    num_points = len(m_req_sup)
    
    a_l_m = (0.9 * d) / 1000 * 0.5  
    L_bd_m = 40 * phi_long / 1000 
    extension_ratio = (a_l_m + L_bd_m) / L_m

    As_req_sup_arr = [calc_As(m_req_sup[i]*1e6, n_comp_array[i], b_mm, bw_eff, d, h_mm, hf_mm, fcd, fyd, fck) for i in range(num_points)]
    As_req_inf_arr = [calc_As(m_req_inf[i]*1e6, n_comp_array[i], b_mm, bw_eff, d, h_mm, hf_mm, fcd, fyd, fck) for i in range(num_points)]

    n_base_sup, As_base_sup = choose_bars(max(As_req_sup_arr + [0]) * 0.25, phi_long, As_min)
    n_base_inf, As_base_inf = choose_bars(max(As_req_inf_arr + [0]) * 0.30, phi_long, As_min) 
    
    def get_extended_regions(As_req_arr, As_base):
        regions = []
        in_region = False
        start_idx = 0
        
        for i in range(num_points):
            if As_req_arr[i] > As_base + 1e-5:
                if not in_region:
                    start_idx = i
                    in_region = True
            else:
                if in_region:
                    regions.append([start_idx, i - 1])
                    in_region = False
        if in_region:
            regions.append([start_idx, num_points - 1])
            
        final_regions = []
        idx_extension = math.ceil(extension_ratio * (num_points - 1))
        
        for r in regions:
            local_max_req = max(As_req_arr[r[0]:r[1]+1])
            n_ex, As_ex = choose_bars(local_max_req - As_base, phi_long)
            
            ext_start = max(0, r[0] - idx_extension)
            ext_end = min(num_points - 1, r[1] + idx_extension)
            
            final_regions.append({
                "start": ext_start,
                "end": ext_end,
                "n_bars": n_ex,
                "As_extra": As_ex,
                "teo_start_x": r[0] / (num_points - 1),
                "teo_end_x": r[1] / (num_points - 1)
            })
            
        return final_regions

    sup_regions = get_extended_regions(As_req_sup_arr, As_base_sup)
    inf_regions = get_extended_regions(As_req_inf_arr, As_base_inf)

    As_act_sup_teo = [As_base_sup] * num_points
    As_act_inf_teo = [As_base_inf] * num_points

    for r in sup_regions:
        for i in range(r["start"], r["end"] + 1):
            As_act_sup_teo[i] = max(As_act_sup_teo[i], As_base_sup + r["As_extra"])
            
    for r in inf_regions:
        for i in range(r["start"], r["end"] + 1):
            As_act_inf_teo[i] = max(As_act_inf_teo[i], As_base_inf + r["As_extra"])

    m_rd_sup_env = [calc_Mrd(As_act_sup_teo[i], n_comp_array[i], b_mm, bw_eff, d, h_mm, hf_mm, fcd, fyd) for i in range(num_points)]
    m_rd_inf_env = [calc_Mrd(As_act_inf_teo[i], n_comp_array[i], b_mm, bw_eff, d, h_mm, hf_mm, fcd, fyd) for i in range(num_points)]

    barre_disegno = [
        {"pos": "sup", "x_start": 0, "x_end": 1.0, "label": f"{n_base_sup}Ø{phi_long}"},
        {"pos": "inf", "x_start": 0, "x_end": 1.0, "label": f"{n_base_inf}Ø{phi_long}"}
    ]
    
    prefisso = "Nervatura Solaio" if is_solaio else "Trave"
    distinta_txt = [
        f"<b>FLESSIONE ({prefisso} - NTC 2018) - Inviluppo Scenari:</b>",
        f"• Sup. Passante: {n_base_sup}Ø{phi_long}", 
        f"• Inf. Passante: {n_base_inf}Ø{phi_long}"
    ]
    
    for r in sup_regions:
        x_s = max(0, r["teo_start_x"] - extension_ratio)
        x_e = min(1.0, r["teo_end_x"] + extension_ratio)
        barre_disegno.append({"pos": "sup_m", "x_start": x_s, "x_end": x_e, "label": f"+{r['n_bars']}Ø{phi_long}"})
    for r in inf_regions:
        x_s = max(0, r["teo_start_x"] - extension_ratio)
        x_e = min(1.0, r["teo_end_x"] + extension_ratio)
        barre_disegno.append({"pos": "inf_m", "x_start": x_s, "x_end": x_e, "label": f"+{r['n_bars']}Ø{phi_long}"})
        
    if sup_regions: distinta_txt.append(f"• Monconi Sup: {sum(r['n_bars'] for r in sup_regions)}Ø{phi_long}")
    if inf_regions: distinta_txt.append(f"• Monconi Inf: {sum(r['n_bars'] for r in inf_regions)}Ø{phi_long}")

    if max(As_req_sup_arr + As_req_inf_arr) > As_max:
        distinta_txt.append(f"<span style='color:#e74c3c; font-weight:bold;'>• ATTENZIONE: Armatura supera As,max (4%). Ingrandire sezione!</span>")


    # =========================================================================
    # REVISIONE PROGETTO TAGLIO E ZONE SOLAIO
    # =========================================================================
    v_ed_max_kN = max(v_ed_abs_local)
    idx_v = v_ed_abs_local.index(v_ed_max_kN) if v_ed_max_kN > 0 else 0
    sigma_cp = min(max(0, n_comp_array[idx_v]) / (bw_eff * h_mm), 0.2 * fcd)
    
    v_rd_pos, v_rd_neg = [], []
    w_staffe = 0
    distinta_txt.append(f"<br><b>VERIFICA A TAGLIO E ZONE CRITICHE:</b>")

    n_spans = len(lengths)
    n_pts_per_span = (num_points - 1) // n_spans + 1
    x_axis = []
    curr_x = 0.0
    support_xs = [0.0]
    
    for j, L_span in enumerate(lengths):
        for i in range(1 if j > 0 else 0, n_pts_per_span):
            x_axis.append(curr_x + (i / (n_pts_per_span - 1)) * L_span)
        curr_x += L_span
        support_xs.append(curr_x)
        
    if len(x_axis) != num_points: 
        x_axis = [(i / (num_points - 1)) * L_m for i in range(num_points)]
        support_xs = [0.0, L_m]

    support_zones = {sx: {'piena': 0.0, 'semi': 0.0} for sx in support_xs}

    if is_solaio:
        rho_1 = min(As_base_inf / (bw_eff * d), 0.02)
        k_val = min(1 + math.sqrt(200 / d), 2.0)
        
        def get_vrdc(width_mm):
            return (max(0.12 * k_val * (100 * rho_1 * fck)**(1/3) * width_mm * d, 
                        0.035 * k_val**1.5 * fck**0.5 * width_mm * d) + 0.15 * sigma_cp * width_mm * d) / 1000.0

        vrdc_corr = get_vrdc(bw_eff)
        vrdc_semi = get_vrdc((bw_eff + b_mm) / 2.0)
        vrdc_piena = get_vrdc(b_mm)

        for i, x in enumerate(x_axis):
            v = v_ed_abs_local[i]
            nearest_sx = min(support_xs, key=lambda sx: abs(x - sx))
            dist = abs(x - nearest_sx)
            
            if v > vrdc_semi: support_zones[nearest_sx]['piena'] = max(support_zones[nearest_sx]['piena'], dist)
            if v > vrdc_corr: support_zones[nearest_sx]['semi'] = max(support_zones[nearest_sx]['semi'], dist)

        # Arrotondamento ai 5 cm per adattamento ai blocchi (pignatte) e limite minimo pratico
        for sx in support_zones:
            piena = math.ceil(support_zones[sx]['piena'] / 0.05) * 0.05
            semi = math.ceil(support_zones[sx]['semi'] / 0.05) * 0.05
            
            # MINIMO COSTRUTTIVO OBBIGATORIO PER FASCIA PIENA AGLI APPOGGI: 15 cm
            piena = max(piena, 0.15)
            
            if semi < piena: semi = piena
            support_zones[sx]['piena'] = piena
            support_zones[sx]['semi'] = semi

        for i, x in enumerate(x_axis):
            nearest_sx = min(support_xs, key=lambda sx: abs(x - sx))
            dist = abs(x - nearest_sx)
            
            if dist <= support_zones[nearest_sx]['piena'] + 1e-5: v_rd_val = vrdc_piena
            elif dist <= support_zones[nearest_sx]['semi'] + 1e-5: v_rd_val = vrdc_semi
            else: v_rd_val = vrdc_corr
                
            v_rd_pos.append(v_rd_val)
            v_rd_neg.append(-v_rd_val)

        distinta_txt.append(f"• VRd,c (Fascia Corrente): {vrdc_corr:.1f} kN")
        if v_ed_max_kN > vrdc_piena:
            distinta_txt.append(f"<span style='color:#e74c3c; font-weight:bold;'>• TAGLIO NON VERIFICATO! (VEd = {v_ed_max_kN:.1f} > VRd,piena = {vrdc_piena:.1f} kN).</span>")
        else:
            distinta_txt.append("• Fasce di taglio (min. 15 cm per appoggio) calcolate dinamicamente.")
                 
        curr_x = 0.0
        for j, L_span in enumerate(lengths):
            sx_left = support_xs[j]
            sx_right = support_xs[j+1]
            
            piena_l = min(support_zones[sx_left]['piena'], L_span/2)
            semi_l = min(support_zones[sx_left]['semi'], L_span/2)
            piena_r = min(support_zones[sx_right]['piena'], L_span/2)
            semi_r = min(support_zones[sx_right]['semi'], L_span/2)
            
            base_x = curr_x / L_m
            L_norm = L_span / L_m
            
            if piena_l > 0: barre_disegno.append({"pos": "zona_staffe", "x_start": base_x, "x_end": base_x + piena_l/L_m, "label": "F. Piena"})
            if semi_l > piena_l: barre_disegno.append({"pos": "zona_staffe", "x_start": base_x + piena_l/L_m, "x_end": base_x + semi_l/L_m, "label": "F. Semip."})
            
            corr_start = semi_l
            corr_end = L_span - semi_r
            if corr_end > corr_start:
                barre_disegno.append({"pos": "zona_staffe", "x_start": base_x + corr_start/L_m, "x_end": base_x + corr_end/L_m, "label": "Corrente"})
                
            if semi_r > piena_r: barre_disegno.append({"pos": "zona_staffe", "x_start": base_x + corr_end/L_m, "x_end": base_x + (L_span-piena_r)/L_m, "label": "F. Semip."})
            if piena_r > 0: barre_disegno.append({"pos": "zona_staffe", "x_start": base_x + (L_span-piena_r)/L_m, "x_end": base_x + L_norm, "label": "F. Piena"})
            
            curr_x += L_span

    elif not has_shear_reinf:
        rho_1 = min(As_base_inf / (bw_eff * d), 0.02)
        k_val = min(1 + math.sqrt(200 / d), 2.0)
        v_rdc_N = max(0.12 * k_val * (100 * rho_1 * fck)**(1/3) * bw_eff * d, 0.035 * k_val**1.5 * fck**0.5 * bw_eff * d) + 0.15 * sigma_cp * bw_eff * d
        v_rdc_kN = v_rdc_N / 1000
        v_rd_pos, v_rd_neg = [v_rdc_kN] * num_points, [-v_rdc_kN] * num_points
        if v_ed_max_kN <= v_rdc_kN: 
            distinta_txt.append(f"• Verifica soddisfatta (VRd,c = {v_rdc_kN:.1f} kN)")
        else: 
            distinta_txt.append(f"<span style='color:#e74c3c; font-weight:bold;'>• NON VERIFICATA! (VRd,c = {v_rdc_kN:.1f} < {v_ed_max_kN:.1f} kN). Inserire staffe.</span>")
    
    else:
        l_c = min(1.5 * h_mm / 1000, max(lengths)/2 if lengths else L_m/2) 
        
        max_v_crit = max_v_corr = 0
        for i, x in enumerate(x_axis):
            v = v_ed_abs_local[i]
            dist = min([abs(x - sx) for sx in support_xs])
            if dist <= l_c: max_v_crit = max(max_v_crit, v)
            else: max_v_corr = max(max_v_corr, v)

        z_taglio = 0.9 * d
        v1 = 0.6 * (1 - fck/250)
        v_rcd_kN = (1.0 * bw_eff * z_taglio * v1 * fcd * 0.5) / 1000 
        
        Asw = 2 * math.pi * (phi_shear**2) / 4 
        s_req_crit = (Asw * z_taglio * fyd / (max_v_crit * 1000)) if max_v_crit > 0 else 300
        s_req_corr = (Asw * z_taglio * fyd / (max_v_corr * 1000)) if max_v_corr > 0 else 300
        
        s_max_crit = min(h_mm / 4, 175, 24 * phi_shear, 8 * phi_long) 
        s_crit_mm = math.floor(min(s_req_crit, s_max_crit) / 10) * 10
        s_corr_mm = math.floor(min(s_req_corr, 0.8 * d, 330) / 10) * 10
        
        if s_crit_mm < 50: s_crit_mm = 50
        if s_corr_mm < 50: s_corr_mm = 50
        s_crit_cm, s_corr_cm = int(s_crit_mm / 10), int(s_corr_mm / 10)

        for i, x in enumerate(x_axis):
            dist = min([abs(x - sx) for sx in support_xs])
            s_local = s_crit_mm if dist <= l_c + 1e-5 else s_corr_mm
            v_rsd_kN = (Asw * z_taglio * fyd / s_local) / 1000
            v_rd = min(v_rcd_kN, v_rsd_kN)
            v_rd_pos.append(v_rd)
            v_rd_neg.append(-v_rd)

        if v_ed_max_kN > v_rcd_kN:
            distinta_txt.append(f"<span style='color:#e74c3c; font-weight:bold;'>• ROTTURA BIELLE COMPRESSE! (VRcd = {v_rcd_kN:.1f} kN < VEd = {v_ed_max_kN:.1f} kN). Ingrandire sezione.</span>")
        else:
            distinta_txt.append(f"• Zone Critiche (L_c = {l_c*100:.0f} cm): Staffe Ø{phi_shear}/{s_crit_cm} cm")
            distinta_txt.append(f"• Zona Corrente: Staffe Ø{phi_shear}/{s_corr_cm} cm")

        curr_x = 0.0
        n_staffe = 0
        for j, L_span in enumerate(lengths):
            lc_eff = min(l_c, L_span/2)
            base_x = curr_x / L_m
            
            x_st = curr_x
            while x_st < curr_x + lc_eff - 1e-5:
                barre_disegno.append({"pos": "staffa_linea", "x_start": x_st/L_m, "x_end": x_st/L_m, "label": ""})
                x_st += s_crit_mm / 1000.0
            
            x_st = curr_x + lc_eff
            while x_st < curr_x + L_span - lc_eff - 1e-5:
                barre_disegno.append({"pos": "staffa_linea", "x_start": x_st/L_m, "x_end": x_st/L_m, "label": ""})
                x_st += s_corr_mm / 1000.0
                
            x_st = curr_x + L_span
            while x_st > curr_x + L_span - lc_eff + 1e-5:
                barre_disegno.append({"pos": "staffa_linea", "x_start": x_st/L_m, "x_end": x_st/L_m, "label": ""})
                x_st -= s_crit_mm / 1000.0
                
            lc_norm = lc_eff / L_m
            L_norm = L_span / L_m
            barre_disegno.append({"pos": "zona_staffe", "x_start": base_x, "x_end": base_x + lc_norm, "label": f"Ø{phi_shear}/{s_crit_cm}"})
            barre_disegno.append({"pos": "zona_staffe", "x_start": base_x + lc_norm, "x_end": base_x + L_norm - lc_norm, "label": f"Ø{phi_shear}/{s_corr_cm}"})
            barre_disegno.append({"pos": "zona_staffe", "x_start": base_x + L_norm - lc_norm, "x_end": base_x + L_norm, "label": f"Ø{phi_shear}/{s_crit_cm}"})
            
            curr_x += L_span
            n_staffe += 2 * (lc_eff / (s_crit_mm/1000)) + max(0, (L_span - 2*lc_eff) / (s_corr_mm/1000))

        l_staffa = 2 * (b_mm - 2*c_mm + h_mm - 2*c_mm) / 1000 + 0.2 
        w_staffe = n_staffe * l_staffa * (phi_shear**2 * 0.00617)

    kg_long = (phi_long**2) * 0.00617
    w_long = (n_base_sup + n_base_inf) * L_m * kg_long
    for r in sup_regions + inf_regions:
        x_s = max(0, r["teo_start_x"] - extension_ratio)
        x_e = min(1.0, r["teo_end_x"] + extension_ratio)
        w_long += r["n_bars"] * (x_e - x_s) * L_m * kg_long
        
    distinta_txt.append(f"<br><b style='color:#2c3e50; font-size:15px;'>PESO TOTALE STIMATO: ~{(w_long + w_staffe):.1f} kg</b>")

    fasce_solaio = None
    if is_solaio:
        sz_list = []
        for sx in sorted(support_zones.keys()):
            sz_list.append({"x": sx, "piena": support_zones[sx]['piena'], "semi": support_zones[sx]['semi']})
            
        fasce_solaio = {
            "L_tot": L_m,
            "lengths": lengths,
            "support_zones": sz_list,
            "b": b_mm / 1000.0, 
            "bw": bw_mm / 1000.0
        }

    return {
        "m_rd_sup": m_rd_sup_env,
        "m_rd_inf": m_rd_inf_env,
        "v_rd_pos": v_rd_pos,
        "v_rd_neg": v_rd_neg,
        "m_max": m_max_env,
        "m_min": m_min_env,
        "v_max": v_max_env,
        "v_min": v_min_env,
        "distinta": distinta_txt,
        "barre_disegno": barre_disegno,
        "fasce_solaio": fasce_solaio
    }
