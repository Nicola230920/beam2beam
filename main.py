from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any , Optional
from engine import Frame, Node, Member, Material, Section, generate_dxf_from_diagrams
import io
import ezdxf
import math

# Importiamo la logica Premium per la sagomatura
from ntc_concrete import design_member_premium, build_envelope

app = FastAPI(title="Structural Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NodeModel(BaseModel):
    id: int
    x: float
    y: float
    angle: float
    ext_type: str
    int_release: str
    supports: List[bool]
    nodal_loads: List[float]
    settlements: List[float]
    spring_k: List[float]

class MemberModel(BaseModel):
    id: int
    name: str
    node_i_id: int
    node_j_id: int
    E: float
    A: float
    I: float
    qx_i: float
    qy_i: float
    qx_j: float
    qy_j: float

class FrameRequest(BaseModel):
    nodes: List[NodeModel]
    members: List[MemberModel]

@app.post("/analyze")
def analyze_frame(payload: FrameRequest):
    frame = Frame()
    for n_data in payload.nodes:
        n = Node(n_data.id, n_data.x, n_data.y)
        n.angle, n.ext_type, n.int_release, n.supports, n.nodal_loads, n.settlements, n.spring_k = n_data.angle, n_data.ext_type, n_data.int_release, n_data.supports, n_data.nodal_loads, n_data.settlements, n_data.spring_k
        frame.nodes[n.id] = n

    for m_data in payload.members:
        mat = Material(f"Mat_{m_data.id}", m_data.E)
        sec = Section(f"Sec_{m_data.id}", m_data.A, m_data.I)
        if m_data.node_i_id not in frame.nodes or m_data.node_j_id not in frame.nodes:
            raise HTTPException(status_code=400, detail="Membro referenzia un nodo inesistente.")
        m = Member(m_data.id, frame.nodes[m_data.node_i_id], frame.nodes[m_data.node_j_id], mat, sec)
        m.name, m.qx_i, m.qy_i, m.qx_j, m.qy_j = m_data.name, m_data.qx_i, m_data.qy_i, m_data.qx_j, m_data.qy_j
        frame.members[m.id] = m

    try:
        U, tot_gdl = frame.solve()
        diagrams = frame.get_diagram_data(U)
        nodal_results = {node.id: {"u_x": float(U[node.gdl[0]]) if node.gdl[0] != -1 else float(node.settlements[0]), "u_y": float(U[node.gdl[1]]) if node.gdl[1] != -1 else float(node.settlements[1]), "phi": float(U[node.gdl[2]]) if node.gdl[2] != -1 else float(node.settlements[2])} for node in frame.nodes.values()}
        return {"status": "success", "tot_gdl": int(tot_gdl), "nodal_results": nodal_results, "diagrams": diagrams}
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

@app.post("/export-dxf")
async def export_dxf(request: Request):
    data = await request.json()
    doc = generate_dxf_from_diagrams(data.get("diagrams", {}))
    stream = io.StringIO()
    doc.write(stream)
    return Response(content=stream.getvalue(), media_type="application/dxf", headers={"Content-Disposition": "attachment; filename=Beam2Beam_Analisi.dxf"})

# =========================================================================
# ROTTA PREMIUM COMPLETAMENTE SEPARATA: L'AUTO-DESIGN SULL'INTERA ASTA
# =========================================================================
# =========================================================================
# ROTTA PREMIUM COMPLETAMENTE SEPARATA: L'AUTO-DESIGN SU TRAVE E INVILUPPO
# =========================================================================
# =========================================================================
# ROTTA PREMIUM COMPLETAMENTE SEPARATA: L'AUTO-DESIGN SULL'INTERA ASTA
# =========================================================================
class MemberDesignRequest(BaseModel):
    material: str = "cls"
    m_ed_scenarios: List[List[List[float]]]
    n_ed_scenarios: List[List[List[float]]]
    v_ed_scenarios: List[List[List[float]]]
    lengths: List[float]
    linf_custom: Optional[float] = 0.0
    is_fully_restrained: Optional[bool] = False

    # --- CAMPI CLS (Resi Opzionali) ---
    has_shear_reinf: Optional[bool] = None
    b: Optional[float] = None
    h: Optional[float] = None
    c: Optional[float] = None
    fck: Optional[float] = None
    fyk: Optional[float] = None
    phi_long: Optional[int] = None
    phi_shear: Optional[int] = None
    section_type: Optional[str] = None
    bw: Optional[float] = None
    hf: Optional[float] = None

    # --- CAMPI ACCIAIO NTC ---
    steel_grade: Optional[int] = None
    profile_name: Optional[str] = None
    profile_data: Optional[Dict[str, float]] = None          

@app.post("/premium/design-member")
def premium_design_member(payload: MemberDesignRequest):
    try:
        stitched_m_scenarios = []
        stitched_n_scenarios = []
        stitched_v_scenarios = []
        
        for sc_idx in range(len(payload.m_ed_scenarios)):
            current_m = []
            current_n = []
            current_v = []
            for j, m_array in enumerate(payload.m_ed_scenarios[sc_idx]):
                if j > 0: current_m.extend(m_array[1:])
                else: current_m.extend(m_array)
            for j, n_array in enumerate(payload.n_ed_scenarios[sc_idx]):
                if j > 0: current_n.extend(n_array[1:])
                else: current_n.extend(n_array)
            for j, v_array in enumerate(payload.v_ed_scenarios[sc_idx]):
                if j > 0: current_v.extend(v_array[1:])
                else: current_v.extend(v_array)
            
            stitched_m_scenarios.append(current_m)
            stitched_n_scenarios.append(current_n)
            stitched_v_scenarios.append(current_v)

        # --- BIFORCAZIONE ACCIAIO VS CLS ---
        # --- BIFORCAZIONE ACCIAIO VS CLS ---
        if payload.material == "acciaio":
            # 0. Otteniamo gli inviluppi massimi e minimi
            m_max, m_min = build_envelope(stitched_m_scenarios)
            v_max, v_min = build_envelope(stitched_v_scenarios)
            n_max, n_min = build_envelope(stitched_n_scenarios)
            
            # ---------------------------------------------------------
            # MOTORE ACCIAIO COMPLETO - NTC 2018 (Capitolo 4.2)
            # ---------------------------------------------------------
            gamma_M0 = 1.05
            gamma_M1 = 1.05
            fyk = float(payload.steel_grade)
            E = 210000.0 # Modulo Elastico (MPa)
            
            # Estraggo geometria dal database JS
            A_mm2 = payload.profile_data.get('A', 0) * 100.0  # da cm2 a mm2
            W_ply_cm3 = payload.profile_data.get('Wy_pl', 0)
            I_z_cm4 = payload.profile_data.get('Iz', 0)       # Inerzia asse debole
            
            prof_h = payload.profile_data.get('h', 0)
            prof_tf = payload.profile_data.get('tf', 0)
            prof_tw = payload.profile_data.get('tw', 0)
            
            # --- 1. Sforzo Normale e Instabilità a Compressione ---
            N_pl_Rd = (A_mm2 * fyk) / (1000.0 * gamma_M0) # Trazione (kN)
            
            # DETERMINAZIONE LUNGHEZZA LIBERA D'INFLESSIONE
            L_geom = max(payload.lengths) * 1000.0  # mm (Luce totale)
            L_inf = (payload.linf_custom * 1000.0) if (payload.linf_custom and payload.linf_custom > 0) else L_geom
            
            i_z = math.sqrt((I_z_cm4 * 10000.0) / A_mm2) if A_mm2 > 0 else 1.0
            lambda_z = L_inf / i_z
            lambda_rel_z = (lambda_z / math.pi) * math.sqrt(fyk / E)
            
            alpha_c = 0.49 # Fattore di imperfezione
            phi_z = 0.5 * (1 + alpha_c * (lambda_rel_z - 0.2) + lambda_rel_z**2)
            chi_z = min(1.0, 1.0 / (phi_z + math.sqrt(max(0, phi_z**2 - lambda_rel_z**2))))
            
            N_b_Rd = (chi_z * A_mm2 * fyk) / (1000.0 * gamma_M1) # Compressione (kN)
            
            # --- 2. Momento Flettente e Svergolamento ---
            M_c_Rd = (W_ply_cm3 * fyk) / (1000.0 * gamma_M0) # kNm
            
            if payload.is_fully_restrained:
                chi_LT = 1.0  # Ala vincolata dal solaio: nessuna instabilità
                M_b_Rd = M_c_Rd
            else:
                lambda_rel_LT = lambda_rel_z 
                alpha_LT = 0.34 
                phi_LT = 0.5 * (1 + alpha_LT * (lambda_rel_LT - 0.2) + lambda_rel_LT**2)
                chi_LT = min(1.0, 1.0 / (phi_LT + math.sqrt(max(0, phi_LT**2 - lambda_rel_LT**2))))
                M_b_Rd = M_c_Rd * chi_LT
            
            # --- 3. Taglio ---
            h_web = prof_h - 2 * prof_tf
            A_v = max(h_web * prof_tw, A_mm2 * 0.3) 
            V_c_Rd = (A_v * (fyk / math.sqrt(3))) / (1000.0 * gamma_M0) # kN
            
            # --- Calcolo Sollecitazioni Massime di Inviluppo ---
            m_ed_max = max(max(m_max), abs(min(m_min)))
            v_ed_max = max(max(v_max), abs(min(v_min)))
            
            n_ed_traz = max(max(n_max), 0.0)
            n_ed_comp = abs(min(min(n_min), 0.0))
            n_ed_max_abs = max(n_ed_traz, n_ed_comp)
            
            # --- Verifica Combinata (N + M) ---
            # N_Ed / N_Rd + M_Ed / M_Rd <= 1
            ratio_n = (n_ed_comp / N_b_Rd) if n_ed_comp > 0 else (n_ed_traz / N_pl_Rd)
            ratio_m = m_ed_max / M_b_Rd
            ratio_tot = ratio_n + ratio_m
            
            # --- Compilazione Array Grafici Plotly ---
            n_pts = len(m_max)
            m_rd_sup = [M_b_Rd] * n_pts
            m_rd_inf = [M_b_Rd] * n_pts
            v_rd_pos = [V_c_Rd] * n_pts
            v_rd_neg = [-V_c_Rd] * n_pts
            
            peso_kg_m = (A_mm2 / 100.0) * 0.785
            lunghezza_totale = sum(payload.lengths)
            peso_totale = peso_kg_m * lunghezza_totale
            
            distinta = [
                f"<b>PROFILO SELEZIONATO:</b> {payload.profile_name}",
                f"<b>GRADO ACCIAIO:</b> S{payload.steel_grade} (fyk = {fyk} MPa)",
                f"<b>PESO STRUTTURA:</b> {peso_totale:.1f} kg ({peso_kg_m:.1f} kg/m)",
                f"<br><b>CAPACITÀ PORTANTI SEZIONE (NTC 2018):</b>",
                f"- Trazione Pura (N_pl,Rd) = {N_pl_Rd:.1f} kN",
                f"- Compressione / Instab. (N_b,Rd) = {N_b_Rd:.1f} kN (χ_z = {chi_z:.2f})",
                f"- Momento Plastico (M_c,Rd) = {M_c_Rd:.1f} kNm",
                f"- Flesso-Torsionale (M_b,Rd) = {M_b_Rd:.1f} kNm (χ_LT = {chi_LT:.2f})",
                f"- Taglio Plastico (V_c,Rd) = {V_c_Rd:.1f} kN",
                f"<br><b>RISULTATI VERIFICHE (Inviluppo):</b>"
            ]
            
            # Se lo sforzo normale è quasi zero (sotto 0.1 kN), cambiamo l'etichetta testuale
            tipo_sollecitazione = "Flessione" if n_ed_max_abs < 0.1 else "Interazione N-M"
            
            if ratio_tot > 1.0:
                distinta.append(f"<span style='color:#e74c3c;'><b>• NON VERIFICATA!</b> {tipo_sollecitazione} oltre limite (Indice = {ratio_tot:.2f} > 1.0)</span>")
            else:
                distinta.append(f"<span style='color:#27ae60;'><b>• OK STRUTTURALE.</b> {tipo_sollecitazione} verificata (Indice = {ratio_tot:.2f} ≤ 1.0)</span>")
                
            if v_ed_max > V_c_Rd:
                distinta.append(f"<span style='color:#e74c3c;'><b>• TAGLIO NON VERIFICATO!</b> (V_Ed = {v_ed_max:.1f} > {V_c_Rd:.1f} kN)</span>")
            else:
                distinta.append(f"<span style='color:#27ae60;'><b>• Taglio Verificato</b> (V_Ed = {v_ed_max:.1f} ≤ {V_c_Rd:.1f} kN)</span>")
            
            result = {
                "m_max": m_max, "m_min": m_min,
                "m_rd_sup": m_rd_sup, "m_rd_inf": m_rd_inf,
                "v_max": v_max, "v_min": v_min,
                "v_rd_pos": v_rd_pos, "v_rd_neg": v_rd_neg,
                "distinta": distinta,
                "barre_disegno": [],
                "fasce_solaio": None
            }

        else:
            # ---------------------------------------------------------
            # MOTORE CEMENTO ARMATO (Usa il modulo originale intatto)
            # ---------------------------------------------------------
            result = design_member_premium(
                m_ed_scenarios=stitched_m_scenarios,
                n_ed_scenarios=stitched_n_scenarios,
                v_ed_scenarios=stitched_v_scenarios,
                lengths=payload.lengths,
                has_shear_reinf=payload.has_shear_reinf,
                b_mm=payload.b,
                h_mm=payload.h,
                c_mm=payload.c,
                fck=payload.fck,
                fyk=payload.fyk,
                phi_long=payload.phi_long,     
                phi_shear=payload.phi_shear,
                bw_mm=payload.bw if payload.section_type in ['T', 'Solaio'] else payload.b,
                hf_mm=payload.hf if payload.section_type in ['T', 'Solaio'] else payload.h,
                is_solaio=(payload.section_type == 'Solaio')
            )
            
        return {"status": "success", "data": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Errore: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Errore: {str(e)}")

# =========================================================================
# ROTTA PREMIUM DXF
# =========================================================================
# =========================================================================
# ROTTA PREMIUM DXF: ESPORTAZIONE CAD DEI DIAGRAMMI, SEZIONI E CARPENTERIA
# =========================================================================
class PremiumDXFRequest(BaseModel):
    name: str
    L: float
    h: float
    c: float
    m_max: List[float]
    m_min: List[float]
    m_rd_sup: List[float]
    m_rd_inf: List[float]
    v_max: List[float]
    v_min: List[float]
    v_rd_pos: List[float]
    v_rd_neg: List[float]
    distinta: List[str]
    barre_disegno: List[Dict[str, Any]]
    # --- NUOVI CAMPI PER SEZIONI E CARPENTERIA ---
    b: float
    bw: float
    hf: float
    sec_type: str
    phi_long: float
    has_shear: str
    fasce_solaio: Optional[Dict[str, Any]] = None
    bars_sx: Dict[str, int]
    bars_mid: Dict[str, int]
    bars_dx: Dict[str, int]

    # =========================================================================
# ROTTA PREMIUM NODI: METODO DELLE COMPONENTI (EC3)
# =========================================================================
# =========================================================================
# ROTTA PREMIUM NODI: METODO DELLE COMPONENTI (EUROCODICE 3 - EN 1993-1-8)
# =========================================================================
class JointDesignRequest(BaseModel):
    joint_type: str
    tp: float
    bp: float
    hp: float
    bolt_d: float
    bolt_class: str
    bolt_rows: int
    pitch: float
    # Nuovi parametri avanzati
    stiffener_t: Optional[float] = 0.0
    fck_concrete: Optional[float] = 25.0

@app.post("/premium/joint-stiffness")
def calculate_joint_stiffness(payload: JointDesignRequest):
    try:
        # Costanti del materiale
        E = 210000.0  # Modulo elastico acciaio in MPa
        Sj_ini = 0.0
        
        # --- 1. COMPONENTE k10: BULLONI A TRAZIONE ---
        # Calcolo dell'area resistente (filettata) del bullone
        A_res = math.pi * (payload.bolt_d / 2)**2 * 0.78 
        # Lunghezza di serraggio stimata (spessore piastre + rondelle)
        Lb = payload.tp + 15.0 
        # Rigidezza di una singola fila di bulloni
        k10_singolo = 1.6 * A_res / Lb 
        
        # Stimiamo le file tese (assumiamo la fila più bassa compressa e le altre tese)
        bulloni_tesi = 2 * max(1, payload.bolt_rows - 1)
        k10_totale = k10_singolo * bulloni_tesi

        # --- BIFORCAZIONE PER TIPOLOGIA DI NODO ---
        
        if payload.joint_type in ["end_plate", "stiffened_end_plate"]:
            # --- 2. COMPONENTE k5: PIASTRA IN FLESSIONE (Modello T-Stub) ---
            m = 40.0 # Distanza tra asse bullone e cordone di saldatura (mm)
            l_eff = payload.bp # Lunghezza efficace (larghezza piastra)
            
            # Rigidezza a flessione della piastra
            k5 = (0.9 * l_eff * payload.tp**3) / (m**3)
            
            # Se la piastra è irrigidita (Stiffened), la sua deformabilità crolla e la rigidezza esplode
            if payload.joint_type == "stiffened_end_plate":
                k5 *= 5.0 
            
            # Assemblaggio molle in Serie (1/Keq = 1/k5 + 1/k10)
            if k10_totale > 0 and k5 > 0:
                k_eq = 1.0 / ((1.0 / k10_totale) + (1.0 / k5))
            else:
                k_eq = 0.0
                
            # Braccio di leva 'z' (distanza centro compressione - baricentro trazione)
            z = payload.hp * 0.8  # mm
            
            # Sj_ini = E * z^2 * k_eq (Convertito da N*mm a kNm)
            Sj_ini = (E * (z**2) * k_eq) / 1e6

        elif payload.joint_type == "base_plate":
            # --- COMPONENTE k13: CALCESTRUZZO IN COMPRESSIONE ---
            # Modulo elastico del calcestruzzo di fondazione (EC2)
            Ec = 22000.0 * (payload.fck_concrete / 10.0)**0.3
            l_eff = payload.bp
            b_eff = payload.tp * 2 # Fascia di impronta della piastra
            
            k13 = (Ec * math.sqrt(l_eff * b_eff)) / E 
            
            if k10_totale > 0 and k13 > 0:
                k_eq = 1.0 / ((1.0 / k10_totale) + (1.0 / k13))
            else:
                k_eq = 0.0
                
            z = payload.hp * 0.8
            Sj_ini = (E * (z**2) * k_eq) / 1e6

        elif payload.joint_type == "splice_joint":
            # --- GIUNTO DI CONTINUITÀ (Trave-Trave) ---
            # La rigidezza rotazionale è dominata dal taglio e trazione sui bulloni dei coprigiunti
            z = payload.hp * 0.45
            Sj_ini = ((E * (z**2) * k10_totale) / 1e6) * 1.5

        elif payload.joint_type == "fin_plate":
            # --- PIASTRA D'ANIMA (Cerniera) ---
            # Ha una rigidezza leggermente superiore alla doppia squadretta, ma sempre trascurabile
            Sj_ini = 350.0

        elif payload.joint_type == "double_web_angle":
            # --- DOPPIA SQUADRETTA (Cerniera Pura) ---
            Sj_ini = 150.0 
            
        return {"status": "success", "Sj_ini": Sj_ini}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/premium/export-dxf")
async def export_premium_dxf(payload: PremiumDXFRequest):
    try:
        doc = ezdxf.new() 
        msp = doc.modelspace()
        
        L = float(payload.L)
        n = len(payload.m_max)
        if n == 0: 
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"detail": "Nessun dato disponibile."})
            
        X = [float(i * L / max(1, n - 1)) for i in range(n)]
        h_m = float(payload.h) / 1000.0
        c_m = float(payload.c) / 1000.0
        
        doc.layers.add("PREM_M_ED", color=1)    
        doc.layers.add("PREM_M_RD", color=3)    
        doc.layers.add("PREM_V_ED", color=6)    
        doc.layers.add("PREM_V_RD", color=2)    
        doc.layers.add("PREM_ARMATURE", color=5)
        doc.layers.add("PREM_STAFFE", color=8)  
        doc.layers.add("PREM_TESTI", color=7)
        # Nuovi layer
        doc.layers.add("PREM_SEZIONI", color=4)
        doc.layers.add("PREM_CARPENTERIA", color=9)
        
        max_M = max([abs(float(v)) for v in payload.m_max + payload.m_min + payload.m_rd_sup + payload.m_rd_inf] + [1e-5])
        scale_M = 2.0 / max_M
        max_V = max([abs(float(v)) for v in payload.v_max + payload.v_min + payload.v_rd_pos + payload.v_rd_neg] + [1e-5])
        scale_V = 2.0 / max_V
        
        y_momento, y_taglio, y_armature = 0.0, -6.0, -12.0
        
        def step_poly(x_arr, y_arr, y_off, scale):
            pts = []
            for i in range(len(x_arr)):
                if i > 0: pts.append((float(x_arr[i]), float(y_off + float(y_arr[i-1]) * scale)))
                pts.append((float(x_arr[i]), float(y_off + float(y_arr[i]) * scale)))
            return pts

        # --- 1. DIAGRAMMA MOMENTO ---
        pts_med = [(float(X[i]), float(y_momento - float(payload.m_max[i])*scale_M)) for i in range(n)] + \
                  [(float(X[i]), float(y_momento - float(payload.m_min[i])*scale_M)) for i in reversed(range(n))] + \
                  [(float(X[0]), float(y_momento - float(payload.m_max[0])*scale_M))]
        msp.add_lwpolyline(pts_med, dxfattribs={'layer': 'PREM_M_ED'})
        msp.add_lwpolyline(step_poly(X, payload.m_rd_sup, y_momento, scale_M), dxfattribs={'layer': 'PREM_M_RD'})
        msp.add_lwpolyline(step_poly(X, payload.m_rd_inf, y_momento, -scale_M), dxfattribs={'layer': 'PREM_M_RD'})
        msp.add_lwpolyline([(0.0, y_momento), (L, y_momento)], dxfattribs={'layer': 'PREM_TESTI'})
        
        idx_max_M = payload.m_max.index(max(payload.m_max))
        idx_min_M = payload.m_min.index(min(payload.m_min))
        msp.add_text(f"{payload.m_max[idx_max_M]:.1f}", dxfattribs={'layer':'PREM_TESTI', 'height': 0.12}).set_placement((X[idx_max_M], y_momento - payload.m_max[idx_max_M]*scale_M - 0.1))
        msp.add_text(f"{payload.m_min[idx_min_M]:.1f}", dxfattribs={'layer':'PREM_TESTI', 'height': 0.12}).set_placement((X[idx_min_M], y_momento - payload.m_min[idx_min_M]*scale_M + 0.1))

        for i in range(n):
            if i == 0 or payload.m_rd_sup[i] != payload.m_rd_sup[i-1]:
                msp.add_text(f"{payload.m_rd_sup[i]:.1f}", dxfattribs={'layer':'PREM_M_RD', 'height': 0.1}).set_placement((X[i], y_momento + payload.m_rd_sup[i]*scale_M + 0.05))
            if i == 0 or payload.m_rd_inf[i] != payload.m_rd_inf[i-1]:
                msp.add_text(f"{payload.m_rd_inf[i]:.1f}", dxfattribs={'layer':'PREM_M_RD', 'height': 0.1}).set_placement((X[i], y_momento - payload.m_rd_inf[i]*scale_M - 0.15))

        # --- 2. DIAGRAMMA TAGLIO ---
        pts_ved = [(float(X[i]), float(y_taglio + float(payload.v_max[i])*scale_V)) for i in range(n)] + \
                  [(float(X[i]), float(y_taglio + float(payload.v_min[i])*scale_V)) for i in reversed(range(n))] + \
                  [(float(X[0]), float(y_taglio + float(payload.v_max[0])*scale_V))]
        msp.add_lwpolyline(pts_ved, dxfattribs={'layer': 'PREM_V_ED'})
        msp.add_lwpolyline(step_poly(X, payload.v_rd_pos, y_taglio, scale_V), dxfattribs={'layer': 'PREM_V_RD'})
        msp.add_lwpolyline(step_poly(X, payload.v_rd_neg, y_taglio, scale_V), dxfattribs={'layer': 'PREM_V_RD'})
        msp.add_lwpolyline([(0.0, y_taglio), (L, y_taglio)], dxfattribs={'layer': 'PREM_TESTI'})
        
        idx_max_V = payload.v_max.index(max(payload.v_max))
        idx_min_V = payload.v_min.index(min(payload.v_min))
        msp.add_text(f"{payload.v_max[idx_max_V]:.1f}", dxfattribs={'layer':'PREM_TESTI', 'height': 0.12}).set_placement((X[idx_max_V], y_taglio + payload.v_max[idx_max_V]*scale_V + 0.1))
        msp.add_text(f"{payload.v_min[idx_min_V]:.1f}", dxfattribs={'layer':'PREM_TESTI', 'height': 0.12}).set_placement((X[idx_min_V], y_taglio + payload.v_min[idx_min_V]*scale_V - 0.2))

        for i in range(n):
            if i == 0 or payload.v_rd_pos[i] != payload.v_rd_pos[i-1]:
                msp.add_text(f"{payload.v_rd_pos[i]:.1f}", dxfattribs={'layer':'PREM_V_RD', 'height': 0.1}).set_placement((X[i], y_taglio + payload.v_rd_pos[i]*scale_V + 0.05))
            if i == 0 or payload.v_rd_neg[i] != payload.v_rd_neg[i-1]:
                msp.add_text(f"{abs(payload.v_rd_neg[i]):.1f}", dxfattribs={'layer':'PREM_V_RD', 'height': 0.1}).set_placement((X[i], y_taglio + payload.v_rd_neg[i]*scale_V - 0.15))

        # --- 3. SCHEMA ARMATURE ---
        msp.add_lwpolyline([(0.0, float(y_armature + h_m/2)), (L, float(y_armature + h_m/2))], dxfattribs={'layer': 'PREM_TESTI'})
        msp.add_lwpolyline([(0.0, float(y_armature - h_m/2)), (L, float(y_armature - h_m/2))], dxfattribs={'layer': 'PREM_TESTI'})
        
        for b in payload.barre_disegno:
            pos = str(b.get('pos', ''))
            x_s, x_e = float(b.get('x_start', 0.0)) * L, float(b.get('x_end', 0.0)) * L
            lbl = str(b.get('label', ''))
            
            if pos == 'staffa_linea':
                msp.add_line((x_s, float(y_armature - h_m/2 + c_m)), (x_s, float(y_armature + h_m/2 - c_m)), dxfattribs={'layer': 'PREM_STAFFE'})
            elif pos == 'zona_staffe':
                y_line = y_armature - h_m/2 - 0.4
                msp.add_line((x_s, y_line), (x_e, y_line), dxfattribs={'layer': 'PREM_TESTI'})
                msp.add_line((x_s, y_line-0.1), (x_s, y_line+0.1), dxfattribs={'layer': 'PREM_TESTI'})
                msp.add_line((x_e, y_line-0.1), (x_e, y_line+0.1), dxfattribs={'layer': 'PREM_TESTI'})
                msp.add_text(lbl, dxfattribs={'layer':'PREM_TESTI', 'height': 0.12}).set_placement((float((x_s+x_e)/2 - len(lbl)*0.04), float(y_line - 0.2)))
            else:
                y_val = y_armature
                if pos == 'sup': y_val += h_m/2 - c_m
                elif pos == 'inf': y_val -= h_m/2 - c_m
                elif pos == 'sup_m': y_val += h_m/2 - c_m - 0.05
                elif pos == 'inf_m': y_val -= h_m/2 - c_m - 0.05
                
                msp.add_line((x_s, float(y_val)), (x_e, float(y_val)), dxfattribs={'layer': 'PREM_ARMATURE'})
                txt_y = y_val + 0.05 if 'sup' in pos and pos != 'sup_m' else y_val - 0.15
                if pos == 'inf_m': txt_y = y_val + 0.05
                msp.add_text(lbl, dxfattribs={'layer':'PREM_TESTI', 'height': 0.12}).set_placement((float((x_s+x_e)/2 - len(lbl)*0.04), float(txt_y)))

        # --- 4. SEZIONI TRASVERSALI ---
        y_sezioni = y_armature - 4.0

        def draw_dxf_cross_section(x_offset, y_offset, title, n_sup, n_inf):
            b_val = payload.b / 1000.0
            h_val = payload.h / 1000.0
            bw_val = payload.bw / 1000.0
            hf_val = payload.hf / 1000.0
            c_val = payload.c / 1000.0
            phi_val = payload.phi_long / 1000.0
            
            # Titolo sezione
            msp.add_text(title, dxfattribs={'layer':'PREM_TESTI', 'height': 0.15}).set_placement((x_offset - b_val/2, y_offset + h_val + 0.2))
            
            # Profilo CLS
            if payload.sec_type == 'rect':
                msp.add_lwpolyline([(x_offset-b_val/2, y_offset), (x_offset+b_val/2, y_offset), (x_offset+b_val/2, y_offset+h_val), (x_offset-b_val/2, y_offset+h_val)], close=True, dxfattribs={'layer': 'PREM_SEZIONI'})
            elif payload.sec_type in ['T', 'Solaio']:
                msp.add_lwpolyline([(x_offset-b_val/2, y_offset+h_val-hf_val), (x_offset+b_val/2, y_offset+h_val-hf_val), (x_offset+b_val/2, y_offset+h_val), (x_offset-b_val/2, y_offset+h_val)], close=True, dxfattribs={'layer': 'PREM_SEZIONI'})
                msp.add_lwpolyline([(x_offset-bw_val/2, y_offset), (x_offset+bw_val/2, y_offset), (x_offset+bw_val/2, y_offset+h_val-hf_val), (x_offset-bw_val/2, y_offset+h_val-hf_val)], close=True, dxfattribs={'layer': 'PREM_SEZIONI'})
            
            w_eff = b_val if payload.sec_type == 'rect' else bw_val
            
            # Staffe
            if payload.has_shear == 'yes' and payload.sec_type != 'Solaio':
                st_x0 = x_offset - w_eff/2 + c_val - 0.005
                st_x1 = x_offset + w_eff/2 - c_val + 0.005
                st_y0 = y_offset + c_val - 0.005
                st_y1 = y_offset + h_val - c_val + 0.005
                msp.add_lwpolyline([(st_x0, st_y0), (st_x1, st_y0), (st_x1, st_y1), (st_x0, st_y1)], close=True, dxfattribs={'layer': 'PREM_STAFFE'})
            
            # Ferri
            def place_bars(n, y_val, lbl_offset, prefix):
                if n <= 0: return
                spacing = (w_eff - 2*c_val) / (n - 1) if n > 1 else 0
                start_x = x_offset - w_eff/2 + c_val if n > 1 else x_offset
                for i in range(n):
                    bx = start_x + i * spacing
                    msp.add_circle((bx, y_val), phi_val/2, dxfattribs={'layer': 'PREM_ARMATURE'})
                msp.add_text(f"{n}%%c{int(payload.phi_long)}", dxfattribs={'layer':'PREM_TESTI', 'height': 0.1}).set_placement((x_offset - 0.15, y_val + lbl_offset))

            place_bars(n_inf, y_offset + c_val, 0.05, 'Inf')
            place_bars(n_sup, y_offset + h_val - c_val, -0.15, 'Sup')
    
        draw_dxf_cross_section(0.0, y_sezioni, "Sezione Appoggio Sx", payload.bars_sx['sup'], payload.bars_sx['inf'])
        draw_dxf_cross_section(L/2, y_sezioni, "Sezione Mezzeria", payload.bars_mid['sup'], payload.bars_mid['inf'])
        draw_dxf_cross_section(L, y_sezioni, "Sezione Appoggio Dx", payload.bars_dx['sup'], payload.bars_dx['inf'])

        # --- 5. PIANTA CARPENTERIA (Se applicabile) ---
        if payload.fasce_solaio and payload.sec_type == 'Solaio':
            y_carp = y_sezioni - 6.0
            fs = payload.fasce_solaio
            b_m = fs['b']
            bw_m = fs['bw']
            lengths = fs.get('lengths', [L])
            sz = fs.get('support_zones', [])
            
            msp.add_text("PIANTA CARPENTERIA SOLAIO", dxfattribs={'layer':'PREM_TESTI', 'height': 0.25}).set_placement((0, y_carp + b_m + 0.5))
            msp.add_lwpolyline([(0, y_carp), (L, y_carp), (L, y_carp+b_m), (0, y_carp+b_m)], close=True, dxfattribs={'layer': 'PREM_CARPENTERIA'})
            
            num_rows = max(1, round(b_m / 0.50))
            interasse = b_m / num_rows
            b_pignatta = interasse - bw_m
            l_pignatta = 0.25
            
            curr_x = 0.0
            support_xs = [0.0]
            for l_span in lengths:
                curr_x += l_span
                support_xs.append(curr_x)
            
            # Travi intermedie
            for i in range(1, len(support_xs)-1):
                sx = support_xs[i]
                msp.add_lwpolyline([(sx-0.15, y_carp), (sx+0.15, y_carp), (sx+0.15, y_carp+b_m), (sx-0.15, y_carp+b_m)], close=True, dxfattribs={'layer': 'PREM_CARPENTERIA'})
                
            # Pignatte e fasce
            curr_x = 0.0
            for j, l_span in enumerate(lengths):
                x_start = curr_x
                x_end = curr_x + l_span
                has_rompi = (l_span > 4.5)
                x_rompi_s = x_start + l_span/2 - bw_m/2
                x_rompi_e = x_start + l_span/2 + bw_m/2
                
                sz_left = sz[j] if j < len(sz) else {'piena':0, 'semi':0}
                sz_right = sz[j+1] if j+1 < len(sz) else {'piena':0, 'semi':0}
                
                for row in range(num_rows):
                    y0 = y_carp + row * interasse + bw_m/2
                    y1 = y0 + b_pignatta
                    is_even = (row % 2 == 0)
                    
                    p_sx = sz_left.get('piena', 0)
                    s_sx = sz_left.get('semi', 0)
                    p_dx = sz_right.get('piena', 0)
                    s_dx = sz_right.get('semi', 0)
                    
                    off_s = p_sx if is_even else s_sx
                    off_e = p_dx if is_even else s_dx
                    
                    pign_s = x_start + off_s
                    pign_e = x_end - off_e
                    
                    segments = []
                    if pign_s < pign_e:
                        if has_rompi and pign_s < x_rompi_s and pign_e > x_rompi_e:
                            segments.append((pign_s, x_rompi_s))
                            segments.append((x_rompi_e, pign_e))
                        else:
                            segments.append((pign_s, pign_e))
                            
                    for seg in segments:
                        sx_rect = seg[0]
                        ex_rect = seg[1]
                        msp.add_lwpolyline([(sx_rect, y0), (ex_rect, y0), (ex_rect, y1), (sx_rect, y1)], close=True, dxfattribs={'layer': 'PREM_CARPENTERIA'})
                        cx = sx_rect
                        while cx < ex_rect - 0.05:
                            cx += l_pignatta
                            if cx < ex_rect:
                                msp.add_line((cx, y0), (cx, y1), dxfattribs={'layer': 'PREM_CARPENTERIA'})
                curr_x += l_span

        # --- 6. DISTINTA TESTUALE ---
        txt_x, txt_y = float(L + 1.5), float(y_momento + 1.5)
        msp.add_text(f"DISTINTA - {payload.name}", dxfattribs={'layer':'PREM_TESTI', 'height': 0.25}).set_placement((txt_x, txt_y))
        for line in payload.distinta:
            txt_y -= 0.35
            msp.add_text(str(line), dxfattribs={'layer':'PREM_TESTI', 'height': 0.15}).set_placement((txt_x, txt_y))

        stream = io.StringIO()
        doc.write(stream)
        return Response(content=stream.getvalue(), media_type="application/dxf", headers={"Content-Disposition": 'attachment; filename="Esecutivo_NTC.dxf"'})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": f"Errore DXF: {str(e)}"})
