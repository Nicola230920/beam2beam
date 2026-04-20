import math
import numpy as np
import ezdxf

class Material:
    def __init__(self, name, E):
        self.name = name
        self.E = E

class Section:
    def __init__(self, name, A, I):
        self.name = name
        self.A = A
        self.I = I

class Node:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x 
        self.y = y
        self.angle = 0.0
        self.ext_type = "Libero"
        self.int_release = "Nessuno (Incastro)"
        self.supports = [False, False, False]
        self.nodal_loads = [0.0, 0.0, 0.0] 
        self.gdl = [] 
        self.settlements = [0.0, 0.0, 0.0]
        self.spring_k = [0.0, 0.0, 0.0]

class Member:
    def __init__(self, member_id, node_i, node_j, material, section):
        self.id = member_id
        self.name = f"E{member_id}"
        self.node_i = node_i 
        self.node_j = node_j
        self.material = material 
        self.section = section
        self.qx_i = 0.0 
        self.qy_i = 0.0 
        self.qx_j = 0.0 
        self.qy_j = 0.0 

    @property
    def length(self):
        return math.hypot(self.node_j.x - self.node_i.x, self.node_j.y - self.node_i.y)

def get_local_matrices(E, A, I, L, qxi, qyi, qxj, qyj):
    EA_L = E * A / L
    EI_L3 = E * I / (L**3)
    EI_L2 = E * I / (L**2)
    EI_L = E * I / L
    
    K = np.array([
        [EA_L, 0, 0, -EA_L, 0, 0],
        [0, 12*EI_L3, 6*EI_L2, 0, -12*EI_L3, 6*EI_L2],
        [0, 6*EI_L2, 4*EI_L, 0, -6*EI_L2, 2*EI_L],
        [-EA_L, 0, 0, EA_L, 0, 0],
        [0, -12*EI_L3, -6*EI_L2, 0, 12*EI_L3, -6*EI_L2],
        [0, 6*EI_L2, 2*EI_L, 0, -6*EI_L2, 4*EI_L]
    ])
    
    F = np.zeros(6)
    F[0] = (2*qxi + qxj)*L/6.0
    F[3] = (qxi + 2*qxj)*L/6.0
    F[1] = (7*qyi + 3*qyj)*L/20.0
    F[4] = (3*qyi + 7*qyj)*L/20.0
    F[2] = (3*qyi + 2*qyj)*(L**2)/60.0
    F[5] = -(2*qyi + 3*qyj)*(L**2)/60.0
    
    return K, F

def condense_matrix(K, F, rel_indices):
    if not rel_indices:
        return K, F
    keep_indices = [i for i in range(6) if i not in rel_indices]
    
    K_KK = K[np.ix_(keep_indices, keep_indices)]
    K_KR = K[np.ix_(keep_indices, rel_indices)]
    K_RK = K[np.ix_(rel_indices, keep_indices)]
    K_RR = K[np.ix_(rel_indices, rel_indices)]
    
    F_K = F[keep_indices]
    F_R = F[rel_indices]
    
    try:
        K_RR_inv = np.linalg.inv(K_RR)
    except np.linalg.LinAlgError:
        return K, F 
        
    K_cond = K_KK - K_KR @ K_RR_inv @ K_RK
    F_cond = F_K - K_KR @ K_RR_inv @ F_R
    
    K_final = np.zeros((6, 6))
    F_final = np.zeros(6)
    
    for idx, k_idx in enumerate(keep_indices):
        F_final[k_idx] = F_cond[idx]
        for jdx, k_jdx in enumerate(keep_indices):
            K_final[k_idx, k_jdx] = K_cond[idx, jdx]
            
    return K_final, F_final

class Frame:
    def __init__(self):
        self.nodes = {} 
        self.members = {} 

    def solve(self):
        tot_gdl = 0
        for node in self.nodes.values():
            node.gdl = []
            for is_fixed in node.supports: 
                if is_fixed: node.gdl.append(-1) 
                else: node.gdl.append(tot_gdl); tot_gdl += 1

        if tot_gdl == 0: 
            if len(self.nodes) == 0: raise ValueError("Non ci sono nodi nel modello!")
            return np.array([]), 0

        K = np.zeros((tot_gdl, tot_gdl))
        F = np.zeros(tot_gdl)

        for node in self.nodes.values():
            th = math.radians(node.angle)
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            f_nodal = R @ np.array(node.nodal_loads)
            for i in range(3):
                idx = node.gdl[i]
                if idx != -1: F[idx] += f_nodal[i]

        master_member = {}

        for member in self.members.values():
            E, A, I, L = member.material.E, member.section.A, member.section.I, member.length
            if L == 0: continue

            dx = member.node_j.x - member.node_i.x
            dy = member.node_j.y - member.node_i.y
            c, s = dx / L, dy / L

            qxi = member.qx_i * c + member.qy_i * s
            qyi = -member.qx_i * s + member.qy_i * c
            qxj = member.qx_j * c + member.qy_j * s
            qyj = -member.qx_j * s + member.qy_j * c

            k_loc, f_eq_loc = get_local_matrices(E, A, I, L, qxi, qyi, qxj, qyj)

            T = np.array([
                [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]
            ])

            th_i, th_j = math.radians(member.node_i.angle), math.radians(member.node_j.angle)
            ci, si = math.cos(th_i), math.sin(th_i)
            cj, sj = math.cos(th_j), math.sin(th_j)
            
            R_sys_T = np.array([
                [ci, -si, 0, 0, 0, 0], [si, ci, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cj, -sj, 0], [0, 0, 0, sj, cj, 0], [0, 0, 0, 0, 0, 1]
            ])
            
            T_new = T @ R_sys_T 
            k_glob = T_new.T @ k_loc @ T_new
            f_eq_glob = T_new.T @ f_eq_loc

            rel_indices = []
            for idx, node in [(0, member.node_i), (3, member.node_j)]:
                if node.int_release != "Nessuno (Incastro)":
                    if node.id not in master_member:
                        master_member[node.id] = member.id 
                    elif master_member[node.id] != member.id: 
                        if node.int_release == "Cerniera Interna": rel_indices.append(idx + 2) 
                        elif node.int_release == "Doppio Pendolo Interno": rel_indices.append(idx + 1) 
                        elif node.int_release == "Pendolo Interno": rel_indices.extend([idx + 1, idx + 2]) 
                            
            if rel_indices:
                k_glob, f_eq_glob = condense_matrix(k_glob, f_eq_glob, rel_indices)

            u_ced = np.zeros(6)
            for i in range(3):
                if member.node_i.supports[i]: u_ced[i] = member.node_i.settlements[i]
                if member.node_j.supports[i]: u_ced[i+3] = member.node_j.settlements[i]
            
            f_ced = k_glob @ u_ced
            gdl_asta = member.node_i.gdl + member.node_j.gdl
            
            for r in range(6):
                gr = gdl_asta[r]
                if gr != -1:
                    F[gr] += f_eq_glob[r] - f_ced[r]
                    for col in range(6):
                        gc = gdl_asta[col]
                        if gc != -1:
                            K[gr, gc] += k_glob[r, col]

        for node in self.nodes.values():
            for i in range(3):
                idx = node.gdl[i]
                if idx != -1 and node.spring_k[i] > 0:
                    K[idx, idx] += node.spring_k[i]

        try:
            U = np.linalg.solve(K, F)
            return U, tot_gdl
        except np.linalg.LinAlgError:
            raise ValueError("Matrice singolare! Il telaio è labile o mal vincolato.")

    def get_diagram_data(self, U):
        master_member = {}
        for m in self.members.values():
            for idx, node in [(0, m.node_i), (3, m.node_j)]:
                if node.int_release != "Nessuno (Incastro)":
                    if node.id not in master_member: 
                        master_member[node.id] = m.id

        results = {}
        for m in self.members.values():
            E, A, I, L = m.material.E, m.section.A, m.section.I, m.length
            if L <= 0: continue

            u_g = np.zeros(6)
            for i in range(3):
                u_g[i]   = U[m.node_i.gdl[i]] if m.node_i.gdl[i] != -1 else m.node_i.settlements[i]
                u_g[i+3] = U[m.node_j.gdl[i]] if m.node_j.gdl[i] != -1 else m.node_j.settlements[i]

            dx, dy = m.node_j.x - m.node_i.x, m.node_j.y - m.node_i.y
            c, s = dx / L, dy / L
            
            q_ax_i = m.qx_i * c + m.qy_i * s
            q_tr_i = -m.qx_i * s + m.qy_i * c
            q_ax_j = m.qx_j * c + m.qy_j * s
            q_tr_j = -m.qx_j * s + m.qy_j * c
            
            # 1. Calcolo matrici NON condensate
            k_loc_uncond, f_eq_loc_uncond = get_local_matrices(E, A, I, L, q_ax_i, q_tr_i, q_ax_j, q_tr_j)

            rel_indices = []
            for idx, node in [(0, m.node_i), (3, m.node_j)]:
                if node.int_release != "Nessuno (Incastro)":
                    if master_member.get(node.id) != m.id: 
                        if node.int_release == "Cerniera Interna": rel_indices.append(idx + 2)
                        elif node.int_release == "Doppio Pendolo Interno": rel_indices.append(idx + 1)
                        elif node.int_release == "Pendolo Interno": rel_indices.extend([idx + 1, idx + 2])
            
            # 2. Condensazione per ricavare gli sforzi esatti nodali
            if rel_indices:
                k_loc, f_eq_loc = condense_matrix(k_loc_uncond, f_eq_loc_uncond, rel_indices)
            else:
                k_loc, f_eq_loc = k_loc_uncond, f_eq_loc_uncond

            T_mat = np.array([
                [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]
            ])
            th_i, th_j = math.radians(m.node_i.angle), math.radians(m.node_j.angle)
            R_sys_T = np.array([
                [math.cos(th_i), -math.sin(th_i), 0, 0, 0, 0], [math.sin(th_i), math.cos(th_i), 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, math.cos(th_j), -math.sin(th_j), 0], [0, 0, 0, math.sin(th_j), math.cos(th_j), 0], [0, 0, 0, 0, 0, 1]
            ])
            
            T_full = T_mat @ R_sys_T
            
            # Spostamenti nodali nel sistema locale
            u_loc = T_full @ u_g
            
            # Calcolo sforzi esatti (usando le matrici condensate per i diagrammi N, T, M)
            f_loc = k_loc @ u_loc - f_eq_loc
            
            # =========================================================
            # 3. RECUPERO DEI GRADI DI LIBERTÀ CONDENSATI (u_true)
            # =========================================================
            u_true = u_loc.copy()
            if rel_indices:
                keep_indices = [i for i in range(6) if i not in rel_indices]
                K_RK = k_loc_uncond[np.ix_(rel_indices, keep_indices)]
                K_RR = k_loc_uncond[np.ix_(rel_indices, rel_indices)]
                F_R = f_eq_loc_uncond[rel_indices]
                U_K = u_loc[keep_indices]
                
                try:
                    # Risolvo: U_Released = K_RR^-1 * (F_eq_R - K_RK * U_Kept)
                    U_R = np.linalg.inv(K_RR) @ (F_R - K_RK @ U_K)
                    for idx_enum, r_idx in enumerate(rel_indices):
                        u_true[r_idx] = U_R[idx_enum]
                except np.linalg.LinAlgError:
                    pass
            # =========================================================

            num_punti = 21
            x_loc = np.linspace(0, L, num_punti)

            xi = x_loc / L if L > 0 else np.zeros_like(x_loc)
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = x_loc * (1 - 2*xi + xi**2)
            N3 = 3*xi**2 - 2*xi**3
            N4 = x_loc * (xi**2 - xi)
            
            # 4. Funzioni di forma Hermite calcolate sui VERI spostamenti dell'asta
            v_val_nodi = u_true[1]*N1 + u_true[2]*N2 + u_true[4]*N3 + u_true[5]*N4 
            u_ax_val_nodi = u_true[0]*(1-xi) + u_true[3]*xi                      
            
            # 5. Sovrapposizione degli effetti del carico distribuito (Soluzione Particolare)
            v_q_uni = (q_tr_i * (x_loc**2) * ((L - x_loc)**2)) / (24 * E * I)
            v_q_tri = ((q_tr_j - q_tr_i) * (x_loc**2) * ((L - x_loc)**2) * (x_loc + 2*L)) / (120 * E * I * L) if L > 0 else 0
            v_val_carico = v_q_uni + v_q_tri
            
            q_ax_avg = (q_ax_i + q_ax_j) / 2
            u_ax_val_carico = (q_ax_avg * x_loc * (L - x_loc)) / (2 * E * A)
            
            # 6. Deformata Totale Assoluta
            v_val = v_val_nodi + v_val_carico
            u_ax_val = u_ax_val_nodi + u_ax_val_carico
            
            dx_glob = u_ax_val * c - v_val * s
            dy_glob = u_ax_val * s + v_val * c
            
            N_val = -f_loc[0] - (q_ax_i * x_loc + (q_ax_j - q_ax_i) * (x_loc**2) / (2*L))
            T_val = f_loc[1] + (q_tr_i * x_loc + (q_tr_j - q_tr_i) * (x_loc**2) / (2*L))
            M_val = -f_loc[2] + f_loc[1] * x_loc + q_tr_i * (x_loc**2) / 2 + (q_tr_j - q_tr_i) * (x_loc**3) / (6*L)

            X_base = m.node_i.x + x_loc * c
            Y_base = m.node_i.y + x_loc * s

            results[m.id] = {
                'name': m.name,
                'X_base': X_base.tolist(), 'Y_base': Y_base.tolist(),
                'N': N_val.tolist(), 'T': T_val.tolist(), 'M': M_val.tolist(),
                'dx_g': dx_glob.tolist(), 'dy_g': dy_glob.tolist(),
                'c': c, 's': s, 'L': L
            }
            
        return results

# =========================================================================
# NUOVA FUNZIONE PER ESPORTARE IN DXF
# =========================================================================
def generate_dxf_from_diagrams(diagrams_data):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layer Setup
    doc.layers.add("STRUTTURA", color=7)       # Bianco/Nero
    doc.layers.add("SFORZO_NORMALE", color=5)  # Blu
    doc.layers.add("TAGLIO", color=3)          # Verde
    doc.layers.add("MOMENTO", color=1)         # Rosso
    doc.layers.add("DEFORMATA", color=6)       # Magenta

    # Troviamo i massimi assoluti per scalare i diagrammi graficamente nel CAD
    global_max = 1e-5
    max_L = 1e-5
    max_disp = 1e-9
    
    for d in diagrams_data.values():
        n_max = max(abs(np.array(d['N'])) / 1000)
        t_max = max(abs(np.array(d['T'])) / 1000)
        m_max = max(abs(np.array(d['M'])) / 1000)
        disp = max(np.hypot(d['dx_g'], d['dy_g']))
        global_max = max(global_max, n_max, t_max, m_max)
        max_L = max(max_L, d['L'])
        max_disp = max(max_disp, disp)

    # Fattori di scala grafici proporzionali al disegno
    sc = (max_L * 0.25) / global_max
    sc_disp = (max_L * 0.10) / max_disp

    def draw_diagram_polygon(X, Y, vals, scale_factor, c, s, layer, invert_m=False):
        dir = -1 if invert_m else 1
        pts = []
        for i in range(len(X)):
            nx = X[i] - vals[i] * scale_factor * s * dir
            ny = Y[i] + vals[i] * scale_factor * c * dir
            pts.append((nx, ny))
            
        # Disegna il poligono chiuso del diagramma
        poly_pts = [(X[0], Y[0])] + pts + [(X[-1], Y[-1])]
        msp.add_lwpolyline(poly_pts, dxfattribs={'layer': layer, 'closed': True})
        
        # Disegna le linee di tratteggio verticali
        for i in range(len(X)):
            msp.add_line((X[i], Y[i]), pts[i], dxfattribs={'layer': layer})

    for m_id, d in diagrams_data.items():
        X, Y = d['X_base'], d['Y_base']
        c, s = d['c'], d['s']
        
        # Disegna l'asta base
        msp.add_line((X[0], Y[0]), (X[-1], Y[-1]), dxfattribs={'layer': 'STRUTTURA'})
        
        vals_N = np.array(d['N']) / 1000
        vals_T = np.array(d['T']) / 1000
        vals_M = np.array(d['M']) / 1000
        dx_g = np.array(d['dx_g'])
        dy_g = np.array(d['dy_g'])

        # Disegna diagrammi
        draw_diagram_polygon(X, Y, vals_N, sc, c, s, "SFORZO_NORMALE")
        draw_diagram_polygon(X, Y, vals_T, sc, c, s, "TAGLIO")
        draw_diagram_polygon(X, Y, vals_M, sc, c, s, "MOMENTO", invert_m=True)

        # Disegna deformata
        def_pts = [(X[i] + dx_g[i]*sc_disp, Y[i] + dy_g[i]*sc_disp) for i in range(len(X))]
        msp.add_lwpolyline(def_pts, dxfattribs={'layer': 'DEFORMATA'})

    return doc
