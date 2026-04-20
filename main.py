from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from engine import Frame, Node, Member, Material, Section, generate_dxf_from_diagrams
import io

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
    
    # 1. Ricostruzione Nodi
    for n_data in payload.nodes:
        n = Node(n_data.id, n_data.x, n_data.y)
        n.angle = n_data.angle
        n.ext_type = n_data.ext_type
        n.int_release = n_data.int_release
        n.supports = n_data.supports
        n.nodal_loads = n_data.nodal_loads
        n.settlements = n_data.settlements
        n.spring_k = n_data.spring_k
        frame.nodes[n.id] = n

    # 2. Ricostruzione Aste
    for m_data in payload.members:
        mat = Material(f"Mat_{m_data.id}", m_data.E)
        sec = Section(f"Sec_{m_data.id}", m_data.A, m_data.I)
        if m_data.node_i_id not in frame.nodes or m_data.node_j_id not in frame.nodes:
            raise HTTPException(status_code=400, detail="Membro referenzia un nodo inesistente.")
        
        node_i = frame.nodes[m_data.node_i_id]
        node_j = frame.nodes[m_data.node_j_id]
        
        m = Member(m_data.id, node_i, node_j, mat, sec)
        m.name = m_data.name
        m.qx_i = m_data.qx_i
        m.qy_i = m_data.qy_i
        m.qx_j = m_data.qx_j
        m.qy_j = m_data.qy_j
        frame.members[m.id] = m

    # 3. Soluzione e Post-Processing
    try:
        U, tot_gdl = frame.solve()
        diagrams = frame.get_diagram_data(U)
        
        nodal_results = {}
        for node in frame.nodes.values():
            # FIX CRITICO: Casting da numpy.float64 a float standard Python per permettere la serializzazione JSON
            u_x = float(U[node.gdl[0]]) if node.gdl[0] != -1 else float(node.settlements[0])
            u_y = float(U[node.gdl[1]]) if node.gdl[1] != -1 else float(node.settlements[1])
            phi = float(U[node.gdl[2]]) if node.gdl[2] != -1 else float(node.settlements[2])
            nodal_results[node.id] = {"u_x": u_x, "u_y": u_y, "phi": phi}

        return {
            "status": "success",
            "tot_gdl": int(tot_gdl),
            "nodal_results": nodal_results,
            "diagrams": diagrams
        }
    except ValueError as e:
        # Errore matrice singolare (struttura labile)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Qualsiasi altro errore interno
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

@app.post("/export-dxf")
async def export_dxf(request: Request):
    data = await request.json()
    diagrams = data.get("diagrams", {})
    
    # Chiama la funzione nel backend per generare il file
    doc = generate_dxf_from_diagrams(diagrams)
    
    # Salva il file in memoria anziché su disco fisico
    stream = io.StringIO()
    doc.write(stream)
    
    # Ritorna il file al browser pronto per il download
    return Response(
        content=stream.getvalue(), 
        media_type="application/dxf",
        headers={"Content-Disposition": "attachment; filename=Beam2Beam_Analisi.dxf"}
    )
