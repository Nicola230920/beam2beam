<!DOCTYPE html>
<html lang="it">
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-ZV9DCTDZHW"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-ZV9DCTDZHW');
</script>
    <meta charset="UTF-8">
    <title>Beam2Beam | Calcolatore Strutturale Gratuito Online</title>
<meta name="description" content="Risolutore di telai 2D online gratuito. Calcola reazioni vincolari, sforzo normale, taglio, momento flettente e deformata. Disegna il tuo schema strutturale interattivamente.">
<meta name="keywords" content="FEA, elementi finiti, calcolo strutturale, telaio 2D, ingegneria civile, diagrammi taglio momento">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; height: 100vh; overflow: hidden; background-color: #f0f2f5; }
        .toolbar { padding: 12px; background: #2c3e50; border-bottom: 1px solid #1a252f; display: flex; gap: 10px; align-items: center; color: white; }
        .toolbar button { padding: 8px 15px; cursor: pointer; border-radius: 4px; border: none; background: #34495e; color: white; transition: background 0.2s; }
        .toolbar button:hover { background: #465c71; }
        .btn-solve { background-color: #27ae60 !important; font-weight: bold; margin-left: auto; }
        .btn-solve:hover { background-color: #2ecc71 !important; }
        .btn-save { background-color: #2980b9 !important; font-weight: bold; margin-left: 0; }
        .btn-save:hover { background-color: #3498db !important; }
        .btn-dxf { background-color: #8e44ad !important; font-weight: bold; }
        
        .btn-premium { background-color: #f1c40f !important; color: #2c3e50 !important; font-weight: bold; border: 2px solid #f39c12 !important; margin-left: 10px; }
        .btn-premium:hover { background-color: #f39c12 !important; color: white !important; }

        .main-container { display: flex; flex: 1; overflow: hidden; }
        .sidebar { width: 320px; background: #fff; border-right: 1px solid #ddd; padding: 15px; overflow-y: auto; box-shadow: 2px 0 5px rgba(0,0,0,0.05); }
        .sidebar h4 { margin: 20px 0 10px; font-size: 13px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #eee; padding-bottom: 5px;}
        .form-row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
        .form-row label { width: 85px; font-size: 12px; font-weight: 600; color: #34495e; }
        .form-row input, .form-row select { flex: 1; padding: 5px; border: 1px solid #dcdde1; border-radius: 4px; font-size: 13px; }
        .sidebar button { width: 100%; padding: 8px; margin-top: 10px; cursor: pointer; border-radius: 4px; border: 1px solid #dcdde1; background: #f8f9fa; font-weight: 600; }
        .sidebar button:hover { background: #e9ecef; }
        
        .canvas-container { flex: 1; position: relative; background: #fff; overflow: hidden; }
        canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
        #statusBar { padding: 8px 15px; background: #fff; font-size: 12px; border-top: 1px solid #ddd; color: #7f8c8d; }

        #resultsModal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 1000; justify-content: center; align-items: center; }
        .modal-content { background: #fff; width: 98%; height: 96%; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .modal-header { padding: 10px 20px; background: #2c3e50; color: white; display: flex; justify-content: space-between; align-items: center; height: 40px; }
        .modal-body { flex: 1; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 10px; padding: 10px; background: #e2e6ea; height: calc(100% - 60px); }
        .plot-container { background: white; border-radius: 6px; box-shadow: inset 0 0 5px rgba(0,0,0,0.05); width: 100%; height: 100%; position: relative; }
        .close-btn { color: white; font-size: 28px; cursor: pointer; background: none; border: none; font-weight: bold;}
        .close-btn:hover { color: #e74c3c; }

        #premiumModal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(44, 62, 80, 0.95); z-index: 2000; justify-content: center; align-items: center; }
        .premium-content { background: #fff; width: 90%; max-width: 1200px; height: 85%; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 0 40px rgba(241, 196, 15, 0.4); }
        .premium-header { padding: 15px 20px; background: #f39c12; color: #2c3e50; font-weight: bold; font-size: 18px; display: flex; justify-content: space-between; align-items: center; }
        .premium-body { display: flex; flex: 1; overflow: hidden; }
        .premium-sidebar { width: 340px; padding: 20px; background: #fdfbf7; border-right: 1px solid #f1c40f; overflow-y: auto; }
        .premium-results { flex: 1; padding: 20px; display: flex; flex-direction: column; gap: 15px; overflow-y: auto; background: #ecf0f1; scroll-behavior: smooth; }
        
        .section-preview { background: #fff; border: 1px solid #f1c40f; border-radius: 4px; margin-bottom: 15px; padding: 10px; display: flex; justify-content: center; }
        .distinta-box { background: #fff; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 14px; border-left: 5px solid #f39c12; border: 1px solid #ddd; line-height: 1.5; }

        .interactive-section { display: flex; gap: 15px; background: #fff; padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin-top: 10px;}
        .interactive-controls { flex: 1; }
        .interactive-canvas { flex: 1.5; display: flex; justify-content: center; align-items: center; background: #fafafa; border-radius: 4px; border: 1px inset #eee; }
        .range-slider { width: 100%; margin: 10px 0 20px 0; }
        .domain-box { padding: 10px; margin-top: 15px; background: #e8f8f5; border-left: 4px solid #1abc9c; font-weight: bold; font-size: 13px; }

        .cross-sections-container { display: flex; gap: 10px; margin-top: 15px; overflow-x: auto; padding-bottom: 10px; }
        .cs-plot { flex: 1; min-width: 250px; background: #fff; border: 1px solid #ddd; border-radius: 5px; height: 280px; }

        #cookie-banner { position: fixed; bottom: 0; left: 0; width: 100%; background: #2c3e50; color: white; padding: 15px; text-align: center; z-index: 10000; display: none; box-sizing: border-box; border-top: 2px solid #34495e; font-size: 14px; }
        #cookie-banner button { background: #27ae60; color: white; border: none; padding: 8px 20px; margin-left: 15px; cursor: pointer; border-radius: 4px; font-weight: bold; transition: background 0.2s; }
        #cookie-banner button:hover { background: #2ecc71; }
        
        .goog-te-banner-frame { display: none !important; }
        body { top: 0px !important; }
    </style>
</head>
<body>

    <div class="toolbar">
        <button onclick="setTool('Seleziona')">Seleziona</button>
        <button onclick="setTool('Disegna Asta')">Disegna Asta</button>
        <button onclick="setTool('Cancella Elemento')" style="color: #e74c3c;">Cancella Elemento</button>
        <button onclick="setTool('Cancella Carico')" style="color: #e67e22;">Cancella Carico</button>
        <button onclick="openSectionCalcModal()" style="background-color: #3498db; font-weight: bold; margin-left: 10px;">📐 GEOMETRIA SEZIONE</button>
        
        <div id="google_translate_element"></div>
        <button class="btn-solve" onclick="runSolve()">RISOLVI E GRAFICA</button>
        <button class="btn-save" onclick="saveScenario()">💾 SALVA COMBINAZIONE</button>
        <button class="btn-dxf" onclick="exportDXF()">ESPORTA IN CAD</button><button onclick="openNodeModal()" style="background-color: #e67e22; color: white; font-weight: bold; margin-left: 10px; border: 2px solid #d35400; border-radius: 4px; cursor: pointer; padding: 8px 15px;">🔗 PROGETTO NODI</button>
        <button class="btn-premium" onclick="openPremiumModal()"> VERIFICHE</button>
    </div>

    <div class="main-container">
        <div class="sidebar">
            <h4>Coordinate Nodo</h4>
            <div class="form-row"><label>X (m):</label><input type="number" id="entry_x" value="5.0"></div>
            <div class="form-row"><label>Y (m):</label><input type="number" id="entry_y" value="4.0"></div>
            <button onclick="addNodeFromCoords()">Aggiungi Nodo</button>

            <h4>Vincoli Esterni</h4>
            <div class="form-row">
                <select id="combo_vincolo_ext">
                    <option>Incastro</option><option>Cerniera</option><option>Carrello</option>
                    <option>Pendolo</option><option>Doppio Pendolo</option><option selected>Libero</option>
                </select>
            </div>
            <div class="form-row"><label>Angolo (°):</label><input type="number" id="entry_node_angle" value="0.0"></div>
            
            <h4>Svincoli Interni</h4>
            <div class="form-row">
                <select id="combo_vincolo_int">
                    <option selected>Nessuno (Incastro)</option><option>Cerniera Interna</option>
                    <option>Pendolo Interno</option><option>Doppio Pendolo Interno</option>
                </select>
            </div>
            <button onclick="setTool('Applica Vincolo')">Applica Vincoli</button>

            <h4>Proprietà Asta (Analisi Elastica)</h4>
            <div class="form-row"><label>E (Pa):</label><input type="number" id="entry_e" value="210e9"></div>
            <div class="form-row"><label>A (m²):</label><input type="number" id="entry_a" value="0.01"></div>
            <div class="form-row"><label>I (m⁴):</label><input type="number" id="entry_i" value="0.0001"></div>

            <h4>Carichi Nodali (kN, kNm)</h4>
            <div class="form-row"><label>Fx:</label><input type="number" id="entry_fx" value="10.0"></div>
            <div class="form-row"><label>Fy:</label><input type="number" id="entry_fy" value="-2.0"></div>
            <div class="form-row"><label>Mz:</label><input type="number" id="entry_mz" value="0.0"></div>
            <button onclick="setTool('Applica Forza Concentrata')">Applica Forza/Momento</button>
            

            <h4>Vincoli Elastici (kN/m)</h4>
            <div class="form-row"><label>Kx:</label><input type="number" id="entry_kx" value="0.0"></div>
            <div class="form-row"><label>Ky:</label><input type="number" id="entry_ky" value="75000.0"></div>
            <div class="form-row"><label>Kφ:</label><input type="number" id="entry_kphi" value="0.0"></div>
            <button onclick="setTool('Applica Molla')">Applica Molla</button>

            <h4>Carichi Distribuiti (kN/m)</h4>
            <div class="form-row"><label>Qx INI:</label><input type="number" id="entry_qx_i" value="0.0"></div>
            <div class="form-row"><label>Qx FIN:</label><input type="number" id="entry_qx_j" value="0.0"></div>
            <div class="form-row"><label>Qy INI:</label><input type="number" id="entry_qy_i" value="-20.0"></div>
            <div class="form-row"><label>Qy FIN:</label><input type="number" id="entry_qy_j" value="-20.0"></div>
            <button onclick="openLoadAnalysisModal()" style="background-color: #f39c12; color: white; margin-top: 5px; border-color: #d35400;">📊 Analisi dei Carichi </button>
            <button onclick="setTool('Applica Carico Distribuito')">Applica Carico Asta</button>
            
            <h4>Cedimenti Vincolari (m, rad)</h4>
            <div class="form-row"><label>δx:</label><input type="number" id="entry_ced_x" value="0.0"></div>
            <div class="form-row"><label>δy:</label><input type="number" id="entry_ced_y" value="0.0"></div>
            <div class="form-row"><label>φz:</label><input type="number" id="entry_ced_phi" value="0.0"></div>
            <button onclick="setTool('Applica Cedimento')">Applica Cedimento</button>
        </div>

        <div class="canvas-container" id="canvasContainer">
            <canvas id="mainCanvas"></canvas>
        </div>
    </div>
    <div id="statusBar">Pronto. Strumento attivo: Seleziona</div>

    <div id="resultsModal">
        <div class="modal-content">
            <div class="modal-header">
                <span>Diagrammi Sollecitazioni: Valori e Massimi/Minimi</span>
                <button class="close-btn" onclick="document.getElementById('resultsModal').style.display='none'">&times;</button>
            </div>
            <div class="modal-body">
                <div class="plot-container" id="plot_N"></div>
                <div class="plot-container" id="plot_T"></div>
                <div class="plot-container" id="plot_M"></div>
                <div class="plot-container" id="plot_Def"></div>
            </div>
        </div>
    </div>

<!-- MODALE ANALISI CARICHI SLU -->
   <div id="loadAnalysisModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 3000; justify-content: center; align-items: center;">
    <div style="background: #fff; width: 600px; border-radius: 8px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); display: flex; flex-direction: column;">
        <div style="padding: 15px 20px; background: #f39c12; color: #2c3e50; display: flex; justify-content: space-between; align-items: center; font-weight: bold; border-bottom: 2px solid #e67e22;">
            <span>📊 Analisi Carichi: Favorevole vs Sfavorevole</span>
            <button onclick="document.getElementById('loadAnalysisModal').style.display='none'" style="background:none; border:none; color:#2c3e50; font-size:24px; cursor:pointer;">&times;</button>
        </div>
        
        <div style="padding: 20px; background: #ecf0f1; display: flex; flex-direction: column; gap: 12px; max-height: 85vh; overflow-y: auto;">
            
            <div style="background: #fff; padding: 12px; border-radius: 6px; border-left: 5px solid #2980b9;">
                <h4 style="margin: 0 0 10px 0; color: #2980b9; font-size:13px;">📐 Struttura G1 (Peso Proprio)</h4>
                
                <div class="form-row">
                    <label>Tipo Solaio:</label>
                    <select id="g1_calc_type" onchange="toggleG1Inputs()" style="flex:1;">
                        <option value="laterocemento">Laterocemento (Travetti)</option>
                        <option value="piena">Soletta Piena C.A.</option>
                        <option value="legno">Solaio in Legno</option>
                        <option value="acciaio">Solaio in Acciaio</option>
                    </select>
                </div>

                <div id="ui_g1_laterocemento" style="display:flex; flex-direction:column; gap:5px; margin-top:5px;">
                    <div class="form-row"><label>S. Cappa (cm):</label><input type="number" id="g1_s" value="4" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>H Pignatta (cm):</label><input type="number" id="g1_h" value="20" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>Interasse (cm):</label><input type="number" id="g1_i" value="50" oninput="computeG1Analytic()"></div>
                </div>

                <div id="ui_g1_piena" style="display:none; margin-top:5px;">
                    <div class="form-row"><label>Spessore (cm):</label><input type="number" id="g1_h_piena" value="20" oninput="computeG1Analytic()"></div>
                </div>

                <div id="ui_g1_legno" style="display:none; flex-direction:column; gap:5px; margin-top:5px;">
                    <div class="form-row"><label>S. Cappa (cm):</label><input type="number" id="g1_s" value="4" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>B Travetto (cm):</label><input type="number" id="g1_b_legno" value="10" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>H Travetto (cm):</label><input type="number" id="g1_h_legno" value="20" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>Interasse (cm):</label><input type="number" id="g1_i_legno" value="60" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>S. Tavolato (cm):</label><input type="number" id="g1_s_legno" value="3" oninput="computeG1Analytic()"></div>
                </div>

                <div id="ui_g1_acciaio" style="display:none; flex-direction:column; gap:5px; margin-top:5px;">
                    <div class="form-row">
                        <label>Profilo:</label>
                        <select id="g1_steel_prof" onchange="computeG1Analytic()" style="flex:1;"></select>
                    </div>
                    <div class="form-row"><label>Interasse (cm):</label><input type="number" id="g1_i_steel" value="100" oninput="computeG1Analytic()"></div>
                    <div class="form-row"><label>S. Getto/Lam. (cm):</label><input type="number" id="g1_s_steel" value="10" oninput="computeG1Analytic()"></div>
                </div>
            </div>

            <div style="background: #fff; padding: 12px; border-radius: 6px; border-left: 5px solid #16a085;">
                <h4 style="margin: 0 0 10px 0; color: #16a085; font-size:13px;">🏠 Permanenti G2</h4>
                <div style="display:grid; grid-template-columns: 1fr 1fr; font-size: 11px;">
                    <label><input type="checkbox" id="chk_pav" onchange="computeG1Analytic()" checked> Pavimento (0.4)</label>
                    <label><input type="checkbox" id="chk_mas" onchange="computeG1Analytic()" checked> Massetto (0.8)</label>
                    <label><input type="checkbox" id="chk_int" onchange="computeG1Analytic()" checked> Intonaco (0.3)</label>
                </div>
            </div>

            <div style="background: #fff; padding: 12px; border-radius: 6px; border-left: 5px solid #3498db;">
                <h4 style="margin: 0 0 10px 0; color: #3498db; font-size:13px;">❄️ Neve e Uso</h4>
                <div class="form-row"><label>Uso (Qk):</label><select id="qk_type" onchange="computeG1Analytic()"><option value="2.0">A-Resid. (2.0)</option><option value="3.0">B-Uffici (3.0)</option></select></div>
                <div class="form-row"><label>Zona Neve:</label>
                    <select id="zona_neve" onchange="computeG1Analytic()">
                        <option value="1_alp">Zona I-Alpina</option><option value="2">Zona II</option><option value="3">Zona III</option>
                    </select>
                </div>
                <div class="form-row"><label>Altitudine (m):</label><input type="number" id="altitudine" value="200" oninput="computeG1Analytic()"></div>
            </div>

            <div style="background: #2c3e50; color: white; padding: 15px; border-radius: 6px;">
                <div class="form-row"><label>Influenza (m):</label><input type="number" id="inf_width" value="4.0" oninput="computeG1Analytic()" style="color:black;"></div>
                <div style="margin-top:10px; border-top:1px solid #444; padding-top:10px;">
                    <div style="display:flex; justify-content: space-between; font-weight:bold; color:#e74c3c;">
                        <span>qy SFAVOREVOLE:</span><span id="res_slu_sfav">0.00 kN/m</span>
                    </div>
                    <div style="display:flex; justify-content: space-between; font-weight:bold; color:#2ecc71; margin-top:5px;">
                        <span>qy FAVOREVOLE:</span><span id="res_slu_fav">0.00 kN/m</span>
                    </div>
                </div>
            </div>

            <div style="display: flex; gap: 10px;">
                <button onclick="applyAnalyzedLoad('sfav')" style="flex:1; padding:12px; background:#e74c3c; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">⬇️ APPLICA SFAVOREVOLE</button>
                <button onclick="applyAnalyzedLoad('fav')" style="flex:1; padding:12px; background:#2ecc71; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">⬇️ APPLICA FAVOREVOLE</button>
            </div>
        </div>
    </div>
</div>
                   
<!-- MODALE CALCOLATORE GEOMETRIA SEZIONE -->
    <div id="sectionCalcModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 3000; justify-content: center; align-items: center;">
        <div style="background: #fff; width: 600px; border-radius: 8px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); display: flex; flex-direction: column;">
            <div style="padding: 15px 20px; background: #3498db; color: white; display: flex; justify-content: space-between; align-items: center; font-weight: bold;">
                <span>📐 Calcolatore Proprietà Sezione</span>
                <button onclick="document.getElementById('sectionCalcModal').style.display='none'" style="background:none; border:none; color:white; font-size:24px; cursor:pointer;">&times;</button>
            </div>
            <div style="display: flex; padding: 20px; gap: 20px; background: #ecf0f1;">
                <div style="flex: 1; display: flex; flex-direction: column; gap: 10px;">
                    <div class="form-row">
                        <label style="width:100px;">Forma:</label>
                        <select id="calc_shape" onchange="updateCalcUI()">
                            <option value="rect">Rettangolare</option>
                            <option value="t">Sezione a T</option>
                            <option value="circle">Circolare</option>
                            <option value="solaio">Solaio</option>
                            <option value="acciaio">Acciaio (Sagomario)</option>
                        </select>
                    </div>
                    
                    <div class="form-row" id="row_calc_steel" style="display:none;">
                        <label style="width:100px; color:#d35400; font-weight:bold;">Profilo:</label>
                        <select id="calc_steel_profile" onchange="computeSectionProperties()" style="border: 2px solid #d35400; font-weight:bold;"></select>
                    </div>

                    <div class="form-row" id="row_calc_b"><label style="width:100px;">Base (cm):</label><input type="number" id="calc_b" value="30" oninput="computeSectionProperties()"></div>
                    <div class="form-row" id="row_calc_h"><label style="width:100px;">Altezza (cm):</label><input type="number" id="calc_h" value="50" oninput="computeSectionProperties()"></div>
                    <div class="form-row" id="row_calc_bw" style="display:none;"><label style="width:100px;">Sp. Anima (cm):</label><input type="number" id="calc_bw" value="20" oninput="computeSectionProperties()"></div>
                    <div class="form-row" id="row_calc_hf" style="display:none;"><label style="width:100px;">Sp. Ala (cm):</label><input type="number" id="calc_hf" value="15" oninput="computeSectionProperties()"></div>
                    <div class="form-row" id="row_calc_d" style="display:none;"><label style="width:100px;">Diametro (cm):</label><input type="number" id="calc_d" value="40" oninput="computeSectionProperties()"></div>
                    
                    <div style="background: #e8f8f5; border-left: 4px solid #1abc9c; padding: 10px; margin-top: 10px;">
                        <div style="font-size:12px; color:#7f8c8d;">Area (A):</div>
                        <div style="font-weight:bold; font-size:16px; color:#2c3e50;" id="out_calc_a">0.0000 m²</div>
                        <div style="font-size:12px; color:#7f8c8d; margin-top:5px;">Inerzia Flettente (I):</div>
                        <div style="font-weight:bold; font-size:16px; color:#2c3e50;" id="out_calc_i">0.0000 m⁴</div>
                    </div>
                    <button onclick="applySectionProperties()" style="padding:10px; background:#27ae60; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer; margin-top:10px;">✔️ APPLICA AD ASTA</button>
                </div>
                <div style="flex: 1; background: #fff; border: 1px solid #bdc3c7; border-radius: 4px; display: flex; justify-content: center; align-items: center;" id="calcSvgContainer">
                </div>
            </div>
        </div>
    </div>

    <!-- MODALE PREMIUM -->
    <div id="premiumModal">
        <div class="premium-content">
            <div class="premium-header">
                <span id="premiumTitle">💎 PROGETTO ESECUTIVO NTC</span>
                <button class="close-btn" style="color:#2c3e50;" onclick="document.getElementById('premiumModal').style.display='none'">&times;</button>
            </div>
            <div class="premium-body">
                <div class="premium-sidebar">
                    
                    <div class="form-row" style="align-items:flex-start;">
                        <label style="margin-top:5px;">Scenari:</label>
                        <div id="scenariosList" style="flex:1; max-height:80px; overflow-y:auto; background:#fff; border:1px solid #dcdde1; border-radius:4px; padding:5px; font-size:12px;">
                            <i>Nessuno. Verrà usato l'ultimo calcolo.</i>
                        </div>
                    </div>
                    
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 15px 0;">
                    
                    <div class="section-preview" id="svgContainer"></div>

                    <div class="form-row"><label style="color:#d35400; font-weight:bold;">Materiale:</label>
                        <select id="prem_material" onchange="updateSectionUI()" style="border: 2px solid #d35400; font-weight:bold;">
                            <option value="cls">Cemento Armato</option>
                            <option value="acciaio">Acciaio Strutturale</option>
                        </select>
                    </div>
    

                    <!-- CAMPI CEMENTO ARMATO -->
                    <div id="fields_cls">
                        <div class="form-row"><label>fck (MPa):</label><input type="number" id="prem_fck" value="25"></div>
                        <div class="form-row"><label>fyk (MPa):</label><input type="number" id="prem_fyk" value="450"></div>
                        
                        <div class="form-row"><label>Sezione:</label>
                            <select id="prem_sec_type" onchange="updateSectionUI()">
                                <option value="rect">Rettangolare</option>
                                <option value="T">Sezione a "T"</option>
                                <option value="Solaio">Solaio (NTC)</option>
                            </select>
                        </div>

                        <div class="form-row"><label id="lbl_b">b (Base Sup):</label><input type="number" id="prem_b" value="1000" oninput="drawSection(); updateInteractiveSection();"></div>
                        <div class="form-row"><label>h (Altezza):</label><input type="number" id="prem_h" value="240" oninput="drawSection(); updateInteractiveSection();"></div>
                        
                        <div id="extra_t_params" style="display:none;">
                            <div class="form-row"><label>bw (Base Inf):</label><input type="number" id="prem_bw" value="120" oninput="drawSection(); updateInteractiveSection();"></div>
                            <div class="form-row"><label>hf (h Ala):</label><input type="number" id="prem_hf" value="50" oninput="drawSection(); updateInteractiveSection();"></div>
                        </div>

                        <div class="form-row"><label>c (Coprif.):</label><input type="number" id="prem_c" value="35" oninput="updateInteractiveSection();"></div>
                        <div class="form-row"><label>Ø Longit.:</label><input type="number" id="prem_phi_long" value="16" oninput="updateInteractiveSection();"></div>
                        <div class="form-row"><label>Ø Staffe:</label><input type="number" id="prem_phi_shear" value="8"></div>
                        <div class="form-row"><label>Taglio:</label>
                            <select id="prem_shear">
                                <option value="yes">Sì (Travi/Pilastri)</option>
                                <option value="no">No (Solette)</option>
                            </select>
                        </div>
                    </div>

                    <!-- CAMPI ACCIAIO -->
                    <div id="fields_steel" style="display:none; background-color:#f4f6f7; padding:10px; border-radius:5px; border: 1px dashed #bdc3c7; margin-bottom:10px;">
    <div class="form-row"><label>Grado:</label>
        <select id="prem_steel_grade">
            <option value="235">S235</option>
            <option value="275">S275</option>
            <option value="355" selected>S355</option>
            <option value="420">S420</option>
            <option value="460">S460</option>
        </select>
    </div>
    <div class="form-row"><label>Profilo:</label>
        <select id="prem_steel_profile" onchange="drawSection()"></select>
    </div>
    
    <div style="margin-top:10px; border-top:1px solid #ddd; padding-top:10px;">
        <div class="form-row">
            <label title="Lunghezza tra i ritegni laterali">L_inf (m):</label>
            <input type="number" id="prem_steel_linf" value="0" step="0.1">
        </div>
        <div style="font-size:10px; color:#7f8c8d; margin-bottom:5px;">(Usa 0 per l'intera luce tra i nodi)</div>
        <div class="form-row">
            <label style="font-size:11px;">Ala vincolata?</label>
            <input type="checkbox" id="prem_steel_restrained" style="width:auto;">
        </div>
    </div>
</div>

                <div style="margin-top: 20px; display: flex; flex-direction: column; gap: 10px; padding: 10px; border-top: 1px solid #eee;">
    <button onclick="runPremiumDesign()" style="padding:12px; background:#27ae60; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer; font-size:14px;">
        🚀 ESEGUI VERIFICA NTC
    </button>
    
    <div id="premiumStatus" style="font-size: 13px; font-weight: bold; text-align: center; min-height: 20px; color: #7f8c8d;">
    </div>
    
    <button id="btnPremiumDXF" onclick="exportPremiumDXF()" style="display:none; padding:10px; background:#8e44ad; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">
        📥 ESPORTA ESECUTIVO DXF
    </button>
</div>
                </div><div class="premium-results" id="resultsScrollArea">
                    <div id="premiumPlot" style="flex:none; background:#fff; border:1px solid #ddd; border-radius:5px; height:300px;"></div>
                    <div id="premiumShearPlot" style="flex:none; background:#fff; border:1px solid #ddd; border-radius:5px; height:300px; margin-top:10px;"></div>
                    <div id="carpentryPlot" style="background:#fff; border:1px solid #ddd; border-radius:5px; min-height:250px; margin-top:10px; display:none;"></div>
                    <div id="rebarPlot" style="margin-top:10px;"></div>
                    <div id="stirrupPlot" style="margin-top:10px;"></div>
                    
                    <div class="interactive-section" id="interactiveSectionModule" style="display:none;">
                        <div class="interactive-controls">
                            <h4 style="margin-top:0; color:#2c3e50;">🔬 Analisi Dominio Rottura</h4>
                            <p style="font-size:11px; color:#7f8c8d;">Varia i ferri per osservare la rotazione dell'asse neutro e il passaggio tra rottura duttile e fragile (NTC 2018).</p>
                            
                            <label style="font-size:13px; font-weight:bold;">Armatura Superiore (<span id="lbl_as_sup">2</span> Ø<span class="lbl_phi">16</span>)</label>
                            <input type="range" id="slider_as_sup" class="range-slider" min="0" max="20" value="2" oninput="updateInteractiveSection()">
                            
                            <label style="font-size:13px; font-weight:bold;">Armatura Inferiore (<span id="lbl_as_inf">4</span> Ø<span class="lbl_phi">16</span>)</label>
                            <input type="range" id="slider_as_inf" class="range-slider" min="1" max="20" value="4" oninput="updateInteractiveSection()">
                            
                            <div class="domain-box" id="domain_text">
                                Calcolo in corso...
                            </div>
                            <div style="font-size:14px; font-weight:bold; color:#e74c3c; margin-top:10px;" id="mrd_text">
                                M_Rd = 0.0 kNm
                            </div>
                        </div>
                        <div class="interactive-canvas">
                            <canvas id="strainCanvas" width="450" height="250"></canvas>
                        </div>
                    </div>

                    <!-- NUOVO MODULO: SEZIONI TRASVERSALI -->
                    <div id="crossSectionsModule" style="display:none; margin-top:15px;">
                        <h4 style="color:#2c3e50; border-bottom: 2px solid #f1c40f; padding-bottom:5px; margin-bottom:10px;">Particolari Costruttivi - Sezioni Trasversali</h4>
                        <div class="cross-sections-container">
                            <div id="csPlotLeft" class="cs-plot"></div>
                            <div id="csPlotMid" class="cs-plot"></div>
                            <div id="csPlotRight" class="cs-plot"></div>
                        </div>
                    </div>

                    <div class="distinta-box" id="distintaBox" style="margin-top:10px;">
                        <i>La distinta armature ottimizzata (con lunghezze di ancoraggio) apparirà qui dopo il calcolo...</i>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="cookie-banner">
        Questo sito utilizza i cookie per migliorare la tua esperienza e per il funzionamento di servizi di terze parti.
        <button onclick="acceptCookies()">Accetto</button>
    </div>

    <script>
        const steelProfiles = {
            "IPE 120": {h:120, b:64, tw:4.4, tf:6.3, A:13.21, Iy:317.8, Wy_pl:60.73, Iz:27.67, Wz_pl:13.58},
            "IPE 140": {h:140, b:73, tw:4.7, tf:6.9, A:16.43, Iy:541.2, Wy_pl:88.34, Iz:44.92, Wz_pl:19.25},
            "IPE 160": {h:160, b:82, tw:5.0, tf:7.4, A:20.09, Iy:869.3, Wy_pl:123.9, Iz:68.31, Wz_pl:26.10},
            "IPE 180": {h:180, b:91, tw:5.3, tf:8.0, A:23.95, Iy:1317, Wy_pl:166.4, Iz:100.9, Wz_pl:34.60},
            "IPE 200": {h:200, b:100, tw:5.6, tf:8.5, A:28.48, Iy:1943, Wy_pl:220.6, Iz:142.4, Wz_pl:44.61},
            "IPE 220": {h:220, b:110, tw:5.9, tf:9.2, A:33.37, Iy:2772, Wy_pl:285.4, Iz:204.9, Wz_pl:58.11},
            "IPE 240": {h:240, b:120, tw:6.2, tf:9.8, A:39.12, Iy:3892, Wy_pl:366.6, Iz:283.6, Wz_pl:73.92},
            "IPE 270": {h:270, b:135, tw:6.6, tf:10.2, A:45.95, Iy:5790, Wy_pl:484.0, Iz:419.9, Wz_pl:96.95},
            "IPE 300": {h:300, b:150, tw:7.1, tf:10.7, A:53.81, Iy:8356, Wy_pl:628.4, Iz:603.8, Wz_pl:125.2},
            "IPE 330": {h:330, b:160, tw:7.5, tf:11.5, A:62.61, Iy:11770, Wy_pl:804.3, Iz:788.1, Wz_pl:153.7},
            "IPE 360": {h:360, b:170, tw:8.0, tf:12.7, A:72.73, Iy:16270, Wy_pl:1019, Iz:1043, Wz_pl:191.1},
            "IPE 400": {h:400, b:180, tw:8.6, tf:13.5, A:84.46, Iy:23130, Wy_pl:1307, Iz:1318, Wz_pl:229.0},
            "HEA 120": {h:114, b:120, tw:5.0, tf:8.0, A:25.34, Iy:606.2, Wy_pl:119.5, Iz:230.9, Wz_pl:58.85},
            "HEA 140": {h:133, b:140, tw:5.5, tf:8.5, A:31.42, Iy:1033, Wy_pl:173.5, Iz:389.3, Wz_pl:84.85},
            "HEA 160": {h:152, b:160, tw:6.0, tf:9.0, A:38.77, Iy:1673, Wy_pl:245.1, Iz:615.6, Wz_pl:117.6},
            "HEA 180": {h:171, b:180, tw:6.0, tf:9.5, A:45.25, Iy:2510, Wy_pl:324.9, Iz:924.6, Wz_pl:156.5},
            "HEA 200": {h:190, b:200, tw:6.5, tf:10.0, A:53.83, Iy:3692, Wy_pl:429.5, Iz:1336, Wz_pl:203.8},
            "HEA 220": {h:210, b:220, tw:7.0, tf:11.0, A:64.34, Iy:5410, Wy_pl:568.5, Iz:1955, Wz_pl:270.6},
            "HEA 240": {h:230, b:240, tw:7.5, tf:12.0, A:76.84, Iy:7763, Wy_pl:744.6, Iz:2769, Wz_pl:351.7},
            "HEA 260": {h:250, b:260, tw:7.5, tf:12.5, A:86.82, Iy:10450, Wy_pl:919.8, Iz:3668, Wz_pl:430.2},
            "HEA 300": {h:290, b:300, tw:8.5, tf:14.0, A:112.5, Iy:18260, Wy_pl:1383, Iz:6310, Wz_pl:641.2},
            "HEB 120": {h:120, b:120, tw:6.5, tf:11.0, A:34.01, Iy:864.4, Wy_pl:165.2, Iz:317.5, Wz_pl:80.97},
            "HEB 140": {h:140, b:140, tw:7.0, tf:12.0, A:42.96, Iy:1509, Wy_pl:245.4, Iz:549.7, Wz_pl:119.8},
            "HEB 160": {h:160, b:160, tw:8.0, tf:13.0, A:54.25, Iy:2492, Wy_pl:354.0, Iz:889.2, Wz_pl:170.0},
            "HEB 180": {h:180, b:180, tw:8.5, tf:14.0, A:65.25, Iy:3831, Wy_pl:481.4, Iz:1363, Wz_pl:231.0},
            "HEB 200": {h:200, b:200, tw:9.0, tf:15.0, A:78.08, Iy:5696, Wy_pl:642.5, Iz:2003, Wz_pl:305.8},
            "HEB 220": {h:220, b:220, tw:9.5, tf:16.0, A:91.04, Iy:8091, Wy_pl:827.0, Iz:2843, Wz_pl:393.9},
            "HEB 240": {h:240, b:240, tw:10.0, tf:17.0, A:106.0, Iy:11260, Wy_pl:1053, Iz:3923, Wz_pl:498.4},
            "HEB 260": {h:260, b:260, tw:10.0, tf:17.5, A:118.4, Iy:14920, Wy_pl:1283, Iz:5135, Wz_pl:602.2},
            "HEB 300": {h:300, b:300, tw:11.0, tf:19.0, A:149.1, Iy:25170, Wy_pl:1869, Iz:8563, Wz_pl:870.1}
        };

        function updateSectionUI() {
            let matSel = document.getElementById('prem_material');
            let material = matSel ? matSel.value : 'cls';
            
            if(material === 'acciaio') {
                document.getElementById('fields_cls').style.display = 'none';
                document.getElementById('fields_steel').style.display = 'block';
                if(document.getElementById('interactiveSectionModule')) document.getElementById('interactiveSectionModule').style.display = 'none';
                
                let sel = document.getElementById('prem_steel_profile');
                if(sel.options.length === 0) {
                    for(let p in steelProfiles) sel.add(new Option(p, p));
                }
            } else {
                document.getElementById('fields_cls').style.display = 'block';
                document.getElementById('fields_steel').style.display = 'none';
                if(document.getElementById('interactiveSectionModule')) document.getElementById('interactiveSectionModule').style.display = 'flex';
                
                let type = document.getElementById('prem_sec_type').value;
                let extra = document.getElementById('extra_t_params');
                let lblB = document.getElementById('lbl_b');
                
                if (type === 'rect') {
                    extra.style.display = 'none';
                    lblB.innerText = "b (Base):";
                } else {
                    extra.style.display = 'block';
                    lblB.innerText = (type === 'Solaio') ? "b (Interasse):" : "b (Base Sup):";
                    if(type === 'Solaio' && document.getElementById('prem_b').value == 300) {
                        document.getElementById('prem_b').value = 1000;
                        document.getElementById('prem_h').value = 240;
                        document.getElementById('prem_shear').value = 'no';
                    }
                }
            }
            drawSection();
            if(material === 'cls' && document.getElementById('interactiveSectionModule') && document.getElementById('interactiveSectionModule').style.display !== 'none') {
                updateInteractiveSection();
            }
        }

        function drawSection() {
            let container = document.getElementById('svgContainer');
            if(!container) return;
            let svgW = 200, svgH = 150, pad = 20;
            let material = document.getElementById('prem_material') ? document.getElementById('prem_material').value : 'cls';
            let html = `<svg width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">`;

            if(material === 'acciaio') {
                let profName = document.getElementById('prem_steel_profile').value;
                if(!profName || !steelProfiles[profName]) return;
                let p = steelProfiles[profName];
                let maxDim = Math.max(p.b, p.h);
                let sc = (svgH - 2*pad) / (maxDim || 1);
                
                // Manteniamo gli spessori visibili anche per profili piccoli per non far sparire l'ala nel disegno
                let sw = p.b * sc, sh = p.h * sc, stw = Math.max(2, p.tw * sc), stf = Math.max(2, p.tf * sc);
                let offX = (svgW - sw)/2, offY = (svgH - sh)/2;
                
                html += `<path d="M ${offX},${offY} h ${sw} v ${stf} h ${-(sw-stw)/2} v ${sh-2*stf} h ${(sw-stw)/2} v ${stf} h ${-sw} v ${-stf} h ${(sw-stw)/2} v ${-(sh-2*stf)} h ${-(sw-stw)/2} z" fill="#bdc3c7" stroke="#2c3e50" stroke-width="1.5"/>`;
                html += `<text x="${offX + sw/2}" y="${offY - 5}" font-size="11" text-anchor="middle" fill="#e74c3c" font-weight="bold">b=${p.b}</text>`;
                html += `<text x="${offX - 5}" y="${offY + sh/2}" font-size="11" text-anchor="middle" transform="rotate(-90 ${offX-5},${offY+sh/2})" fill="#e74c3c" font-weight="bold">h=${p.h}</text>`;
            } else {
                let type = document.getElementById('prem_sec_type').value;
                let b = parseInt(document.getElementById('prem_b').value) || 300;
                let h = parseInt(document.getElementById('prem_h').value) || 500;
                let bw = parseInt(document.getElementById('prem_bw').value) || 120;
                let hf = parseInt(document.getElementById('prem_hf').value) || 50;
                
                let maxDim = Math.max(b, h);
                let sc = (svgH - 2*pad) / (maxDim || 1);
                let sw = b * sc, sh = h * sc, shf = hf * sc, sbw = bw * sc;
                let offX = (svgW - sw)/2, offY = pad;

                if (type === 'rect') {
                    html += `<rect x="${offX}" y="${offY}" width="${sw}" height="${sh}" fill="#bdc3c7" stroke="#2c3e50" stroke-width="2"/>`;
                    html += `<text x="${offX + sw/2}" y="${offY - 5}" font-size="12" text-anchor="middle" fill="#e74c3c" font-weight="bold">b</text>`;
                    html += `<text x="${offX - 5}" y="${offY + sh/2}" font-size="12" text-anchor="middle" transform="rotate(-90 ${offX-5},${offY+sh/2})" fill="#e74c3c" font-weight="bold">h</text>`;
                } else if (type === 'T') {
                    html += `<path d="M ${offX},${offY} h ${sw} v ${shf} h ${-(sw-sbw)/2} v ${sh-shf} h ${-sbw} v ${-(sh-shf)} h ${-(sw-sbw)/2} z" fill="#bdc3c7" stroke="#2c3e50" stroke-width="2"/>`;
                    html += `<text x="${offX + sw/2}" y="${offY - 5}" font-size="12" text-anchor="middle" fill="#e74c3c" font-weight="bold">b</text>`;
                    html += `<text x="${offX + sw + 10}" y="${offY + shf/2}" font-size="10" fill="#2980b9" font-weight="bold">hf</text>`;
                    html += `<text x="${offX + sw/2}" y="${offY + sh + 15}" font-size="11" text-anchor="middle" fill="#27ae60" font-weight="bold">bw</text>`;
                    html += `<text x="${offX - 5}" y="${offY + sh/2}" font-size="12" text-anchor="middle" transform="rotate(-90 ${offX-5},${offY+sh/2})" fill="#e74c3c" font-weight="bold">h</text>`;
                } else if (type === 'Solaio') {
                    let sw2 = sw/2, sbw2 = sbw, gap = sw2 - sbw2;
                    html += `<path d="M ${offX},${offY} h ${sw} v ${shf} h ${-gap/2} v ${sh-shf} h ${-sbw2} v ${-(sh-shf)} h ${-gap} v ${sh-shf} h ${-sbw2} v ${-(sh-shf)} h ${-gap/2} z" fill="#bdc3c7" stroke="#2c3e50" stroke-width="1.5"/>`;
                    html += `<text x="${svgW/2}" y="${offY - 5}" font-size="10" text-anchor="middle" fill="#e74c3c" font-weight="bold">b (1 metro = 2 nervature)</text>`;
                    html += `<text x="${svgW/2}" y="${offY + sh + 15}" font-size="10" text-anchor="middle" fill="#27ae60" font-weight="bold">bw (singolo travetto)</text>`;
                }
            }
            html += `</svg>`;
            container.innerHTML = html;
        }
        function updateInteractiveSection() {
            let mod = document.getElementById('interactiveSectionModule');
            if (mod.style.display === 'none') return; 

            let phi = parseFloat(document.getElementById('prem_phi_long').value) || 16;
            let n_sup = parseInt(document.getElementById('slider_as_sup').value);
            let n_inf = parseInt(document.getElementById('slider_as_inf').value);
            
            document.getElementById('lbl_as_sup').innerText = n_sup;
            document.getElementById('lbl_as_inf').innerText = n_inf;
            document.querySelectorAll('.lbl_phi').forEach(el => el.innerText = phi);

            let is_solaio = document.getElementById('prem_sec_type').value === 'Solaio';
            let b = parseFloat(document.getElementById('prem_b').value) || 300;
            let h = parseFloat(document.getElementById('prem_h').value) || 500;
            let bw = parseFloat(document.getElementById('prem_bw').value) || 120;
            
            if (is_solaio) bw = bw * (b/500.0); 
            if (document.getElementById('prem_sec_type').value === 'rect') bw = b;

            let c = parseFloat(document.getElementById('prem_c').value) || 35;
            let fck = parseFloat(document.getElementById('prem_fck').value) || 25;
            let fyk = parseFloat(document.getElementById('prem_fyk').value) || 450;

            let d = h - c;
            let d_prime = c;
            
            let fcd = 0.85 * fck / 1.5;
            let fyd = fyk / 1.15;
            let Es = 200000;

            let As_sup = n_sup * Math.PI * Math.pow(phi, 2) / 4;
            let As_inf = n_inf * Math.PI * Math.pow(phi, 2) / 4;

            let x = 1.0;
            let eps_c = 0.0, eps_s = 0.0, eps_s_prime = 0.0;
            let sigma_s_inf = 0.0, sigma_s_sup = 0.0;
            let found = false;

            for(let i = 1; i < h; i += 0.5) {
                x = i;
                eps_c = 0.0035;
                eps_s = 0.0035 * (d - x) / x;
                
                if (eps_s > 0.010) {
                    eps_s = 0.010;
                    eps_c = 0.010 * x / (d - x);
                }
                
                eps_s_prime = eps_c * (x - d_prime) / x; 
                
                sigma_s_inf = Math.min(eps_s * Es, fyd);
                sigma_s_sup = Math.max(Math.min(eps_s_prime * Es, fyd), -fyd);
                
                let Cc = 0.8 * bw * x * fcd; 
                let Cs = As_sup * sigma_s_sup;
                let Ts = As_inf * sigma_s_inf;
                
                if (Cc + Cs >= Ts) {
                    found = true;
                    break;
                }
            }

            let domainBox = document.getElementById('domain_text');
            let color = "#1abc9c";
            let campo = "";
            
            if (x >= h) {
                campo = "Campo 5 (Compressione Uniforme)"; color = "#34495e";
            } else if (eps_s >= 0.010 && eps_c < 0.0035) {
                campo = "Campo 1/2 (Rottura Duttile Acciaio)"; color = "#2ecc71";
            } else if (eps_s >= fyd/Es && eps_c >= 0.0034) {
                campo = "Campo 3 (Flessione Duttile Bilanciata)"; color = "#f1c40f";
            } else if (eps_s < fyd/Es && eps_c >= 0.0034) {
                campo = "Campo 4 (Rottura Fragile Calcestruzzo)"; color = "#e74c3c";
            } else {
                campo = "Stato misto / Oltre limiti"; color = "#95a5a6";
            }

            domainBox.innerHTML = `${campo}<br><span style="font-weight:normal; font-size:11px;">Asse x = ${x.toFixed(1)} mm | ε_c = ${(eps_c*1000).toFixed(2)}‰ | ε_s = ${(eps_s*1000).toFixed(2)}‰</span>`;
            domainBox.style.borderLeftColor = color;

            let Mrd = (0.8 * bw * x * fcd * (d - 0.4*x) + As_sup * Math.max(0, sigma_s_sup) * (d - d_prime)) / 1e6;
            document.getElementById('mrd_text').innerText = `Momento Resistente (M_Rd) = ${Mrd.toFixed(1)} kNm`;

            let canvas = document.getElementById('strainCanvas');
            let ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            let drawW = 80, drawH = 200;
            let offX = 50, offY = 25;
            
            ctx.fillStyle = "#ecf0f1";
            ctx.strokeStyle = "#2c3e50";
            ctx.lineWidth = 2;
            ctx.fillRect(offX, offY, drawW, drawH);
            ctx.strokeRect(offX, offY, drawW, drawH);

            ctx.fillStyle = "#e67e22";
            let c_draw = (c / h) * drawH;
            if(n_sup > 0) {
                ctx.beginPath(); ctx.arc(offX + drawW/2 - 20, offY + c_draw, 4, 0, 2*Math.PI); ctx.fill();
                ctx.beginPath(); ctx.arc(offX + drawW/2 + 20, offY + c_draw, 4, 0, 2*Math.PI); ctx.fill();
            }
            if(n_inf > 0) {
                ctx.fillStyle = "#2980b9";
                ctx.beginPath(); ctx.arc(offX + drawW/2 - 20, offY + drawH - c_draw, 5, 0, 2*Math.PI); ctx.fill();
                ctx.beginPath(); ctx.arc(offX + drawW/2 + 20, offY + drawH - c_draw, 5, 0, 2*Math.PI); ctx.fill();
            }

            let naY = offY + (x / h) * drawH;
            ctx.strokeStyle = "#7f8c8d";
            ctx.setLineDash([5, 5]);
            ctx.beginPath(); ctx.moveTo(offX - 20, naY); ctx.lineTo(canvas.width - 20, naY); ctx.stroke();
            ctx.setLineDash([]);

            let strainX = 250;
            let maxEps = Math.max(eps_c, eps_s);
            let scaleEps = 100 / (maxEps || 0.010);

            let x_c = strainX + eps_c * scaleEps;
            let x_s = strainX - eps_s * scaleEps;

            ctx.fillStyle = "rgba(231, 76, 60, 0.4)";
            ctx.beginPath();
            ctx.moveTo(strainX, naY);
            ctx.lineTo(x_c, offY);
            ctx.lineTo(strainX, offY);
            ctx.closePath();
            ctx.fill();

            let dY = offY + (d / h) * drawH;
            ctx.fillStyle = "rgba(52, 152, 219, 0.4)";
            ctx.beginPath();
            ctx.moveTo(strainX, naY);
            ctx.lineTo(x_s, dY);
            ctx.lineTo(strainX, dY);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = "#34495e";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(strainX, offY); ctx.lineTo(strainX, offY + drawH); ctx.stroke();

            ctx.strokeStyle = "#2c3e50";
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(x_c, offY); ctx.lineTo(x_s, dY); ctx.stroke();

            ctx.fillStyle = "#e74c3c";
            ctx.font = "12px Arial";
            ctx.fillText(`εc = -${(eps_c*1000).toFixed(1)}‰`, x_c + 5, offY + 10);
            
            ctx.fillStyle = "#2980b9";
            ctx.fillText(`εs = ${(eps_s*1000).toFixed(1)}‰`, x_s - 65, dY + 5);
        }

        // --- NUOVO MODULO: DISEGNO SEZIONI TRASVERSALI ---
        function drawCrossSection(divId, title, b_mm, h_mm, bw_mm, hf_mm, sec_type, c_mm, phi, n_sup, n_inf, has_shear) {
            let shapes = [];
            let annotations = [];

            // Adatta il ViewBox in base alle proporzioni per mantenere il disegno bello grande
            let maxDim = Math.max(b_mm, h_mm);
            let pad = maxDim * 0.15;
            let margin = {l:20, r:20, t:40, b:20};

            // Disegno Calcestruzzo
            let clsColor = '#ecf0f1';
            let clsLine = {color: '#2c3e50', width: 2};

            if (sec_type === 'rect') {
                shapes.push({ type: 'rect', x0: -b_mm/2, y0: 0, x1: b_mm/2, y1: h_mm, fillcolor: clsColor, line: clsLine });
            } else if (sec_type === 'T' || sec_type === 'Solaio') {
                // Ala
                shapes.push({ type: 'rect', x0: -b_mm/2, y0: h_mm - hf_mm, x1: b_mm/2, y1: h_mm, fillcolor: clsColor, line: clsLine });
                // Anima
                shapes.push({ type: 'rect', x0: -bw_mm/2, y0: 0, x1: bw_mm/2, y1: h_mm - hf_mm, fillcolor: clsColor, line: clsLine });
                // Linea tratteggiata per connessione ala-anima
                shapes.push({ type: 'line', x0: -bw_mm/2, y0: h_mm - hf_mm, x1: bw_mm/2, y1: h_mm - hf_mm, line: {color: '#bdc3c7', dash: 'dash', width: 1} });
            }

            // Disegno Staffe
            let w_eff = (sec_type === 'rect') ? b_mm : bw_mm;
            if (has_shear === 'yes' && sec_type !== 'Solaio') {
                let st_x0 = -w_eff/2 + c_mm - 5;
                let st_x1 = w_eff/2 - c_mm + 5;
                let st_y0 = c_mm - 5;
                let st_y1 = h_mm - c_mm + 5;
                shapes.push({ type: 'rect', x0: st_x0, y0: st_y0, x1: st_x1, y1: st_y1, line: {color: '#95a5a6', width: 2}, fillcolor: 'transparent' });
            }

            // Funzione per piazzare ferri
            const placeBars = (n, y_val, color, text_y_offset, label_txt) => {
                if (n <= 0) return;
                let spacing = (w_eff - 2 * c_mm) / (n > 1 ? n - 1 : 1);
                let start_x = -w_eff/2 + c_mm;
                if (n === 1) start_x = 0;

                for (let i = 0; i < n; i++) {
                    let bx = start_x + i * spacing;
                    shapes.push({ type: 'circle', x0: bx - phi/2, y0: y_val - phi/2, x1: bx + phi/2, y1: y_val + phi/2, fillcolor: color, line: {color: 'black'} });
                }
                
                annotations.push({ x: 0, y: y_val + text_y_offset, text: `<b>${n}Ø${phi}</b>`, showarrow: false, font: {color: color, size: 11} });
            };

            // Disegna ferri inferiori e superiori
            placeBars(n_inf, c_mm, '#2980b9', 30, 'Inf');
            placeBars(n_sup, h_mm - c_mm, '#e67e22', -30, 'Sup');

            // Quotature
            annotations.push({ x: 0, y: h_mm + pad*0.4, text: `b = ${b_mm}`, showarrow: false });
            annotations.push({ x: -b_mm/2 - pad*0.4, y: h_mm/2, text: `h = ${h_mm}`, showarrow: false, textangle: -90 });
            if (sec_type !== 'rect') annotations.push({ x: 0, y: -pad*0.4, text: `bw = ${bw_mm}`, showarrow: false });

            let layout = {
                title: { text: title, font: {size: 13, color: '#34495e'} },
                shapes: shapes,
                annotations: annotations,
                xaxis: { range: [-b_mm/2 - pad, b_mm/2 + pad], showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { range: [-pad, h_mm + pad], scaleanchor: "x", scaleratio: 1, showgrid: false, zeroline: false, showticklabels: false },
                margin: margin,
                plot_bgcolor: '#fdfbf7',
                paper_bgcolor: '#fff',
                dragmode: false
            };

            Plotly.newPlot(divId, [], layout, {displayModeBar: false});
        }


        const canvas = document.getElementById('mainCanvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        let nodes = {}, members = {};
        let nodeCounter = 1, memberCounter = 1;
        let scale = 50.0, offsetX = 100, offsetY = 500, isPanning = false, lastPanX = 0, lastPanY = 0;
        let currentTool = "Seleziona", selectedNodeI = null, selectedObject = null;
        
        let lastDiagrams = null; 
        let lastPremiumData = null; 
        let savedScenarios = [];
        let selectedMembers = []; 

        function resizeCanvas() {
            width = canvas.parentElement.clientWidth; height = canvas.parentElement.clientHeight;
            canvas.width = width; canvas.height = height; redraw();
        }
        window.addEventListener('resize', resizeCanvas);
        
        function setTool(tool) {
            currentTool = tool; selectedNodeI = null;
            document.getElementById('statusBar').innerText = `Strumento attivo: ${tool}`; redraw();
        }

        function checkGeometryReset() {
            if (savedScenarios.length > 0) {
                savedScenarios = [];
                document.getElementById('statusBar').innerText = "Geometria modificata: scenari precedenti eliminati per coerenza.";
            }
        }

        function realToPixel(x, y) { return { px: offsetX + x * scale, py: offsetY - y * scale }; }
        function pixelToReal(px, py) { return { x: (px - offsetX) / scale, y: -(py - offsetY) / scale }; }

        function addNodeFromCoords() {
            let x = parseFloat(document.getElementById('entry_x').value), y = parseFloat(document.getElementById('entry_y').value);
            if(isNaN(x) || isNaN(y)) return alert("Coordinate non valide");
            nodes[nodeCounter] = { id: nodeCounter, x, y, angle: 0.0, ext_type: "Libero", int_release: "Nessuno (Incastro)", supports: [false, false, false], nodal_loads: [0,0,0], settlements: [0,0,0], spring_k: [0,0,0] };
            checkGeometryReset();
            document.getElementById('statusBar').innerText = `Creato Nodo N${nodeCounter} a (${x}, ${y})m`; nodeCounter++; redraw();
        }

        function drawGrid() {
            ctx.strokeStyle = "#f1f2f6"; ctx.lineWidth = 1; ctx.beginPath();
            let boundsMin = pixelToReal(0, height), boundsMax = pixelToReal(width, 0);
            let step = (width / scale > 50) ? 5 : 1;
            for(let i = Math.floor(boundsMin.x); i <= Math.ceil(boundsMax.x); i+=step) {
                let p1 = realToPixel(i, boundsMin.y), p2 = realToPixel(i, boundsMax.y);
                ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py);
            }
            for(let i = Math.floor(boundsMin.y); i <= Math.ceil(boundsMax.y); i+=step) {
                let p1 = realToPixel(boundsMin.x, i), p2 = realToPixel(boundsMax.x, i);
                ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py);
            }
            ctx.stroke();
        }

        function drawArrow(ctx, fromx, fromy, tox, toy) {
            let headlen = 8, angle = Math.atan2(toy - fromy, tox - fromx);
            ctx.beginPath(); ctx.moveTo(fromx, fromy); ctx.lineTo(tox, toy);
            ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
            ctx.moveTo(tox, toy); ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
            ctx.stroke();
        }

        function redraw() {
            ctx.clearRect(0, 0, width, height); drawGrid();
            for (let id in members) {
                let m = members[id], ni = nodes[m.node_i_id], nj = nodes[m.node_j_id], p1 = realToPixel(ni.x, ni.y), p2 = realToPixel(nj.x, nj.y);
                let isSelected = selectedMembers.includes(id.toString()) || (selectedObject && selectedObject.type === 'member' && selectedObject.id == id);
                ctx.strokeStyle = isSelected ? "#f39c12" : "#2c3e50";
                ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py); ctx.stroke();
                let midX = (p1.px + p2.px)/2, midY = (p1.py + p2.py)/2;
                ctx.fillStyle = ctx.strokeStyle; ctx.font = "bold 12px Arial"; ctx.fillText(m.name, midX, midY - 15);
                let qx_i = m.qx_i / 1000, qx_j = m.qx_j / 1000, qy_i = m.qy_i / 1000, qy_j = m.qy_j / 1000;
                if (qy_i !== 0 || qy_j !== 0 || qx_i !== 0 || qx_j !== 0) {
                    let dx = p2.px - p1.px, dy = p2.py - p1.py, len = Math.hypot(dx, dy);
                    if (len > 0) {
                        let nx = -dy/len, ny = dx/len, tx = dx/len, ty = dy/len, scaleLoad = 0.5; 
                        if (qy_i !== 0 || qy_j !== 0) {
                            ctx.fillStyle = "rgba(255, 0, 0, 0.2)"; ctx.strokeStyle = "red"; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(p1.px + nx * qy_i * scaleLoad, p1.py + ny * qy_i * scaleLoad);
                            ctx.lineTo(p2.px + nx * qy_j * scaleLoad, p2.py + ny * qy_j * scaleLoad);
                            ctx.lineTo(p2.px, p2.py); ctx.lineTo(p1.px, p1.py); ctx.closePath(); ctx.fill(); ctx.stroke();
                        }
                        if (qx_i !== 0 || qx_j !== 0) {
                            let off = 10; ctx.fillStyle = "rgba(230, 126, 34, 0.2)"; ctx.strokeStyle = "#d35400"; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(p1.px - nx * off - tx * qx_i * scaleLoad, p1.py - ny * off - ty * qx_i * scaleLoad);
                            ctx.lineTo(p2.px - nx * off - tx * qx_j * scaleLoad, p2.py - ny * off - ty * qx_j * scaleLoad);
                            ctx.lineTo(p2.px - nx * off, p2.py - ny * off); ctx.lineTo(p1.px - nx * off, p1.py - ny * off);
                            ctx.closePath(); ctx.fill(); ctx.stroke();
                        }
                    }
                }
            }
            for (let id in nodes) {
                let n = nodes[id], p = realToPixel(n.x, n.y), color = (selectedNodeI == id || (selectedObject && selectedObject.type === 'node' && selectedObject.id == id)) ? "#e67e22" : "#3498db";
                ctx.save(); ctx.translate(p.px, p.py); ctx.rotate(-n.angle * Math.PI / 180);
                if (n.int_release === "Cerniera Interna") { ctx.fillStyle = "white"; ctx.strokeStyle = "black"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(0, 0, 8, 0, 2*Math.PI); ctx.fill(); ctx.stroke(); }
                else if (n.int_release === "Doppio Pendolo Interno") { ctx.fillStyle = "white"; ctx.strokeStyle = "white"; ctx.beginPath(); ctx.rect(-4, -8, 8, 16); ctx.fill(); ctx.stroke(); ctx.strokeStyle = "black"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(-4, -8); ctx.lineTo(-4, 8); ctx.stroke(); ctx.beginPath(); ctx.moveTo(4, -8); ctx.lineTo(4, 8); ctx.stroke(); }
                else if (n.int_release === "Pendolo Interno") { ctx.fillStyle = "white"; ctx.strokeStyle = "white"; ctx.beginPath(); ctx.rect(-8, -4, 16, 8); ctx.fill(); ctx.stroke(); ctx.strokeStyle = "black"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(-6, 0); ctx.lineTo(6, 0); ctx.stroke(); ctx.beginPath(); ctx.arc(-6, 0, 3, 0, 2*Math.PI); ctx.fill(); ctx.stroke(); ctx.beginPath(); ctx.arc(6, 0, 3, 0, 2*Math.PI); ctx.fill(); ctx.stroke(); }
                if (n.ext_type === "Incastro") { ctx.fillStyle = "gray"; ctx.strokeStyle = "black"; ctx.lineWidth = 1; ctx.beginPath(); ctx.rect(-12, 6, 24, 10); ctx.fill(); ctx.stroke(); ctx.beginPath(); ctx.moveTo(-8, 16); ctx.lineTo(-12, 20); ctx.moveTo(0, 16); ctx.lineTo(-4, 20); ctx.moveTo(8, 16); ctx.lineTo(4, 20); ctx.stroke(); }
                else if (n.ext_type === "Cerniera") { ctx.fillStyle = "lightgray"; ctx.strokeStyle = "black"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, 6); ctx.lineTo(-10, 18); ctx.lineTo(10, 18); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(-15, 18); ctx.lineTo(15, 18); ctx.stroke(); }
                else if (n.ext_type === "Carrello") { ctx.fillStyle = "lightgray"; ctx.strokeStyle = "black"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, 6); ctx.lineTo(-8, 14); ctx.lineTo(8, 14); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.fillStyle = "gray"; ctx.beginPath(); ctx.arc(-4, 16, 2, 0, 2*Math.PI); ctx.fill(); ctx.beginPath(); ctx.arc(0, 16, 2, 0, 2*Math.PI); ctx.fill(); ctx.beginPath(); ctx.arc(4, 16, 2, 0, 2*Math.PI); ctx.fill(); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(-12, 18); ctx.lineTo(12, 18); ctx.stroke(); }
                else if (n.ext_type === "Pendolo") { ctx.strokeStyle = "black"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, 6); ctx.lineTo(0, 18); ctx.stroke(); ctx.fillStyle = "white"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(0, 6, 3, 0, 2*Math.PI); ctx.fill(); ctx.stroke(); ctx.beginPath(); ctx.arc(0, 18, 3, 0, 2*Math.PI); ctx.fill(); ctx.stroke(); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(-10, 21); ctx.lineTo(10, 21); ctx.stroke(); }
                else if (n.ext_type === "Doppio Pendolo") { ctx.fillStyle = "lightgray"; ctx.strokeStyle = "black"; ctx.lineWidth = 1; ctx.beginPath(); ctx.rect(-12, 6, 24, 6); ctx.fill(); ctx.stroke(); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(-16, 16); ctx.lineTo(16, 16); ctx.stroke(); }
                ctx.restore();
                let kx = n.spring_k[0], ky = n.spring_k[1], kphi = n.spring_k[2];
                ctx.strokeStyle = "#27ae60"; ctx.fillStyle = "#27ae60"; ctx.lineWidth = 2.5; ctx.font = "bold 12px Arial";
                if (ky !== 0) { let dir = ky > 0 ? 1 : -1; ctx.beginPath(); ctx.moveTo(p.px, p.py + dir*6); ctx.lineTo(p.px - 10, p.py + dir*14); ctx.lineTo(p.px + 10, p.py + dir*22); ctx.lineTo(p.px - 10, p.py + dir*30); ctx.lineTo(p.px, p.py + dir*38); ctx.stroke(); ctx.fillText("Ky", p.px + 14, p.py + dir*30); }
                if (kx !== 0) { let dir = kx > 0 ? -1 : 1; ctx.beginPath(); ctx.moveTo(p.px + dir*6, p.py); ctx.lineTo(p.px + dir*14, p.py - 10); ctx.lineTo(p.px + dir*22, p.py + 10); ctx.lineTo(p.px + dir*30, p.py - 10); ctx.lineTo(p.px + dir*38, p.py); ctx.stroke(); ctx.fillText("Kx", p.px + dir*45 - 10, p.py - 15); }
                if (kphi !== 0) { ctx.beginPath(); ctx.setLineDash([3, 3]); ctx.arc(p.px, p.py, 22, 0, Math.PI * 1.5); ctx.stroke(); ctx.setLineDash([]); ctx.beginPath(); ctx.arc(p.px, p.py - 22, 4, 0, 2*Math.PI); ctx.fill(); ctx.fillText("Kφ", p.px + 26, p.py - 26); }
                ctx.beginPath(); ctx.arc(p.px, p.py, 6, 0, 2*Math.PI); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "white"; ctx.lineWidth=2; ctx.stroke(); ctx.fillStyle = "#2c3e50"; ctx.font = "bold 10px Arial"; ctx.fillText(`N${id}`, p.px + 10, p.py - 10);
                let fx = n.nodal_loads[0] / 1000, fy = n.nodal_loads[1] / 1000, mz = n.nodal_loads[2] / 1000; ctx.strokeStyle = "red"; ctx.fillStyle = "red"; ctx.lineWidth = 2;
                if (fx !== 0) { drawArrow(ctx, p.px, p.py, p.px + fx * 1.5, p.py); ctx.fillText(`${fx.toFixed(1)}kN`, p.px + fx*1.5 + 5, p.py - 5); }
                if (fy !== 0) { drawArrow(ctx, p.px, p.py, p.px, p.py - fy * 1.5); ctx.fillText(`${fy.toFixed(1)}kN`, p.px + 5, p.py - fy*1.5 - 5); }
                if (mz !== 0) { ctx.beginPath(); ctx.arc(p.px, p.py, 20, -Math.PI/2, Math.PI/2, mz > 0); ctx.stroke(); if (mz > 0) drawArrow(ctx, p.px, p.py + 20, p.px + 2, p.py + 20); else drawArrow(ctx, p.px, p.py + 20, p.px - 2, p.py + 20); ctx.fillText(`Mz=${mz.toFixed(1)}kNm`, p.px + 24, p.py); }
                let dx_ced = n.settlements[0], dy_ced = n.settlements[1], rot_ced = n.settlements[2]; ctx.strokeStyle = "#8e44ad"; ctx.fillStyle = "#8e44ad"; ctx.setLineDash([4, 2]); ctx.lineWidth = 2;
                if (dx_ced !== 0) { let dir = dx_ced > 0 ? 1 : -1; drawArrow(ctx, p.px, p.py, p.px + dir*40, p.py); ctx.fillText(`δx=${dx_ced}`, p.px + dir*30, p.py - 10); }
                if (dy_ced !== 0) { let dir = dy_ced > 0 ? 1 : -1; drawArrow(ctx, p.px, p.py, p.px, p.py - dir*40); ctx.fillText(`δy=${dy_ced}`, p.px + 10, p.py - dir*30); }
                if (rot_ced !== 0) { let dir_rot = rot_ced > 0 ? 1 : -1; ctx.beginPath(); ctx.arc(p.px, p.py, 18, 0, (dir_rot > 0 ? -Math.PI/2 : Math.PI/2), dir_rot > 0); ctx.stroke(); ctx.fillText(`φ=${rot_ced}`, p.px + 22, p.py - 22); }
                ctx.setLineDash([]); 
            }
        }

        function getClosestNode(px, py, tol=15) {
            let closest = null, minDist = tol;
            for(let id in nodes) { let pt = realToPixel(nodes[id].x, nodes[id].y), dist = Math.hypot(px - pt.px, py - pt.py); if(dist < minDist) { minDist = dist; closest = id; } }
            return closest;
        }

        function getClosestMember(px, py, tol=10) {
            let closest = null, minDist = tol;
            for(let id in members) {
                let m = members[id], p1 = realToPixel(nodes[m.node_i_id].x, nodes[m.node_i_id].y), p2 = realToPixel(nodes[m.node_j_id].x, nodes[m.node_j_id].y);
                let l2 = Math.pow(p2.px - p1.px, 2) + Math.pow(p2.py - p1.py, 2), dist = 0;
                if(l2 === 0) dist = Math.hypot(px - p1.px, py - p1.py);
                else { let t = Math.max(0, Math.min(1, ((px - p1.px)*(p2.px - p1.px) + (py - p1.py)*(p2.py - p1.py)) / l2)); dist = Math.hypot(px - (p1.px + t*(p2.px-p1.px)), py - (p1.py + t*(p2.py-p1.py))); }
                if(dist < minDist) { minDist = dist; closest = id; }
            }
            return closest;
        }

        function sortSelectedMembers() {
            selectedMembers.sort((a, b) => {
                let ax = Math.min(nodes[members[a].node_i_id].x, nodes[members[a].node_j_id].x);
                let bx = Math.min(nodes[members[b].node_i_id].x, nodes[members[b].node_j_id].x);
                return ax - bx;
            });
        }

        canvas.addEventListener('mousedown', (e) => {
            if(e.button === 1 || e.button === 2) { isPanning = true; lastPanX = e.clientX; lastPanY = e.clientY; return; }
            let rect = canvas.getBoundingClientRect(), px = e.clientX - rect.left, py = e.clientY - rect.top;
            
            if(currentTool === "Seleziona") {
                let nId = getClosestNode(px, py);
                if(nId) {
                    selectedObject = {type: 'node', id: nId};
                    selectedMembers = [];
                }
                else { 
                    let mId = getClosestMember(px, py); 
                    if(mId) { 
                        selectedObject = {type: 'member', id: mId};
                        let mIdStr = mId.toString();
                        let idx = selectedMembers.indexOf(mIdStr);
                        if (idx > -1) {
                            selectedMembers.splice(idx, 1);
                        } else {
                            selectedMembers.push(mIdStr);
                        }
                    } else { 
                        selectedObject = null; 
                        selectedMembers = [];
                    } 
                }
            } 
            else if(currentTool === "Disegna Asta") {
                let nId = getClosestNode(px, py);
                if(nId) {
                    if(!selectedNodeI) { selectedNodeI = nId; document.getElementById('statusBar').innerText = "Seleziona nodo FINE"; }
                    else {
                        if(selectedNodeI !== nId) {
                            members[memberCounter] = { id: memberCounter, name: `E${memberCounter}`, node_i_id: parseInt(selectedNodeI), node_j_id: parseInt(nId), E: parseFloat(document.getElementById('entry_e').value)||210e9, A: parseFloat(document.getElementById('entry_a').value)||0.01, I: parseFloat(document.getElementById('entry_i').value)||0.0001, qx_i:0, qy_i:0, qx_j:0, qy_j:0 };
                            checkGeometryReset();
                            document.getElementById('statusBar').innerText = `Creata Asta E${memberCounter}`; memberCounter++;
                        }
                        selectedNodeI = null;
                    }
                }
            }
            else if(currentTool === "Applica Vincolo") {
                let nId = getClosestNode(px, py);
                if(nId) {
                    let ext = document.getElementById('combo_vincolo_ext').value, int_rel = document.getElementById('combo_vincolo_int').value, ang = parseFloat(document.getElementById('entry_node_angle').value) || 0.0;
                    nodes[nId].ext_type = ext; nodes[nId].int_release = int_rel; nodes[nId].angle = ang;
                    if(ext === "Incastro") nodes[nId].supports = [true,true,true]; else if(ext === "Cerniera") nodes[nId].supports = [true,true,false]; else if(["Carrello","Pendolo"].includes(ext)) nodes[nId].supports = [false,true,false]; else if(ext === "Doppio Pendolo") nodes[nId].supports = [false,true,true]; else nodes[nId].supports = [false,false,false];
                    checkGeometryReset();
                    setTool("Seleziona");
                }
            }
            else if(currentTool === "Applica Forza Concentrata") {
                let nId = getClosestNode(px, py);
                if(nId) { 
                    nodes[nId].nodal_loads[0] += (parseFloat(document.getElementById('entry_fx').value)||0)*1000; 
                    nodes[nId].nodal_loads[1] += (parseFloat(document.getElementById('entry_fy').value)||0)*1000; 
                    nodes[nId].nodal_loads[2] += (parseFloat(document.getElementById('entry_mz').value)||0)*1000; 
                    setTool("Seleziona"); 
                }
            }
            else if(currentTool === "Applica Cedimento") {
                let nId = getClosestNode(px, py);
                if(nId) { nodes[nId].settlements = [parseFloat(document.getElementById('entry_ced_x').value)||0, parseFloat(document.getElementById('entry_ced_y').value)||0, parseFloat(document.getElementById('entry_ced_phi').value)||0]; setTool("Seleziona"); }
            }
            else if(currentTool === "Applica Molla") {
                let nId = getClosestNode(px, py);
                if(nId) { nodes[nId].spring_k = [parseFloat(document.getElementById('entry_kx').value)||0, parseFloat(document.getElementById('entry_ky').value)||0, parseFloat(document.getElementById('entry_kphi').value)||0]; setTool("Seleziona"); }
            }
            else if(currentTool === "Applica Carico Distribuito") {
                let mId = getClosestMember(px, py);
                if(mId) { 
                    members[mId].qx_i=(parseFloat(document.getElementById('entry_qx_i').value)||0)*1000; 
                    members[mId].qx_j=(parseFloat(document.getElementById('entry_qx_j').value)||0)*1000; 
                    members[mId].qy_i=(parseFloat(document.getElementById('entry_qy_i').value)||0)*1000; 
                    members[mId].qy_j=(parseFloat(document.getElementById('entry_qy_j').value)||0)*1000; 
                    setTool("Seleziona"); 
                }
            }
            else if(currentTool === "Cancella Elemento") {
                let nId = getClosestNode(px, py);
                if(nId) { for(let mId in members) if(members[mId].node_i_id == nId || members[mId].node_j_id == nId) delete members[mId]; delete nodes[nId]; selectedObject=null; selectedMembers=[]; } 
                else { let mId = getClosestMember(px, py); if(mId) { delete members[mId]; selectedObject=null; selectedMembers=[]; } }
                checkGeometryReset();
            }
            else if(currentTool === "Cancella Carico") {
                let nId = getClosestNode(px, py);
                if(nId) { nodes[nId].nodal_loads=[0,0,0]; nodes[nId].spring_k=[0,0,0]; }
                else { let mId = getClosestMember(px, py); if(mId) { members[mId].qx_i=members[mId].qy_i=members[mId].qx_j=members[mId].qy_j=0; } }
            }
            redraw();
        });

        canvas.addEventListener('mousemove', (e) => {
            if(!isPanning) return;
            offsetX += e.clientX - lastPanX; offsetY += e.clientY - lastPanY;
            lastPanX = e.clientX; lastPanY = e.clientY; redraw();
        });

        canvas.addEventListener('mouseup', () => isPanning = false);
        canvas.addEventListener('mouseleave', () => isPanning = false);
        canvas.addEventListener('contextmenu', e => e.preventDefault());
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            let zoom = e.deltaY < 0 ? 1.1 : 0.9;
            let rect = canvas.getBoundingClientRect(), rx = (e.clientX - rect.left - offsetX) / scale, ry = -(e.clientY - rect.top - offsetY) / scale;
            scale *= zoom;
            offsetX = (e.clientX - rect.left) - rx * scale; offsetY = (e.clientY - rect.top) + ry * scale;
            redraw();
        });

        function saveScenario() {
            if (!lastDiagrams) { 
                alert("Nessun calcolo completato da poter salvare. Esegui prima 'RISOLVI E GRAFICA'."); 
                return; 
            }
            savedScenarios.push(JSON.parse(JSON.stringify(lastDiagrams)));
            document.getElementById('statusBar').innerText = `✅ Scenario ${savedScenarios.length} salvato in memoria.`;
        }

        async function runSolve() {
            document.getElementById('statusBar').innerText = "Calcolo in corso...";
            let payload = { nodes: Object.values(nodes), members: Object.values(members) };

            try {
                let res = await fetch('https://beam2beam.onrender.com/analyze', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
                });
                
                if(!res.ok) { let err = await res.json(); throw new Error(err.detail || "Errore dal server"); }
                
                let data = await res.json(); lastDiagrams = data.diagrams;
                document.getElementById('statusBar').innerText = "Calcolo terminato. Mostro grafici...";
                document.getElementById('resultsModal').style.display = 'flex';
                setTimeout(() => { drawPlotlyResults(data.diagrams); }, 50);
            } catch (err) {
                if (err.message.includes("Failed to fetch")) alert("⚠️ IMPOSSIBILE CONNETTERSI AL MOTORE DI CALCOLO ⚠️\n\nAssicurati che il server Python sia acceso eseguendo:\nuvicorn main:app --reload");
                else alert("Errore nel Calcolo: " + err.message);
                document.getElementById('statusBar').innerText = "Errore durante il calcolo.";
            }
        }
        
        async function exportDXF() {
            if (!lastDiagrams) { alert("Devi prima risolvere la struttura cliccando su 'RISOLVI E GRAFICA'."); return; }
            document.getElementById('statusBar').innerText = "Generazione CAD in corso...";
            try {
                let res = await fetch('https://beam2beam.onrender.com/export-dxf', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ diagrams: lastDiagrams })
                });
                if (!res.ok) { let err = await res.json(); throw new Error(err.detail || "Errore dal server"); }
                let blob = await res.blob(), url = window.URL.createObjectURL(blob), a = document.createElement('a');
                a.href = url; a.download = "Beam2Beam_Analisi.dxf"; document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
                document.getElementById('statusBar').innerText = "Esportazione DXF completata con successo.";
            } catch (err) { alert("Errore nell'esportazione: " + err.message); document.getElementById('statusBar').innerText = "Errore esportazione DXF."; }
        }

        function openPremiumModal() {
            if (!lastDiagrams && savedScenarios.length === 0) { 
                alert("Devi prima risolvere la struttura (Tasto verde) e se vuoi salvare degli scenari (Tasto blu)."); 
                return; 
            }
            if (selectedMembers.length === 0) { 
                alert("Seleziona almeno un'asta (puoi cliccare più aste per formare una trave continua)."); 
                return; 
            }
            
            sortSelectedMembers();
            
            let scList = document.getElementById('scenariosList');
            if(savedScenarios.length === 0) {
                scList.innerHTML = "<i>Nessuno scenario salvato. Verrà usato solo l'ultimo calcolo.</i>";
            } else {
                scList.innerHTML = "";
                savedScenarios.forEach((sc, i) => {
                    scList.innerHTML += `<div><input type="checkbox" id="chk_sc_${i}" checked> <label for="chk_sc_${i}">Scenario ${i+1}</label></div>`;
                });
                scList.innerHTML += `<div><input type="checkbox" id="chk_sc_last" checked> <label for="chk_sc_last">Ultimo Calcolo Attuale</label></div>`;
            }

            document.getElementById('premiumStatus').innerText = "";
            document.getElementById('distintaBox').innerHTML = "<i>La distinta armature ottimizzata apparirà qui dopo il calcolo...</i>";
            document.getElementById('interactiveSectionModule').style.display = 'none';
            document.getElementById('crossSectionsModule').style.display = 'none';
            
            Plotly.purge('premiumPlot'); Plotly.purge('premiumShearPlot'); Plotly.purge('rebarPlot'); Plotly.purge('stirrupPlot');
            let cp = document.getElementById('carpentryPlot');
            if(cp) { cp.style.display = 'none'; Plotly.purge('carpentryPlot'); }
            
            let names = selectedMembers.map(id => members[id].name).join(' + ');
            document.getElementById('premiumTitle').innerText = `💎 PROGETTO ESECUTIVO NTC - Aste: ${names}`;
            
            document.getElementById('premiumModal').style.display = 'flex';
            updateSectionUI();
        }

        function drawCarpentry(f, name, lengths_array) {
            let shapes = [];
            let b = f.b || 1.0;
            let bw = f.bw || 0.1;
            let lengths = f.lengths || lengths_array || [10];
            let L_tot = lengths.reduce((a, b) => a + b, 0);
            
            let support_zones = f.support_zones || [];
            while (support_zones.length < lengths.length + 1) support_zones.push({piena: 0, semi: 0});

            let interasse = b; 
            let num_rows = Math.round(b / 0.50);
            if (num_rows < 1) num_rows = 1;
            interasse = b / num_rows;
            let b_pignatta = interasse - bw;
            let l_pignatta = 0.25; 

            shapes.push({ type: 'rect', x0: 0, y0: 0, x1: L_tot, y1: b, line: {color: 'black', width: 2} });

            let curr_x = 0;
            let support_xs = [0];
            for (let L_span of lengths) {
                curr_x += L_span;
                support_xs.push(curr_x);
            }

            for (let i = 1; i < support_xs.length - 1; i++) {
                let sx = support_xs[i];
                shapes.push({ type: 'rect', x0: sx - 0.15, y0: 0, x1: sx + 0.15, y1: b, fillcolor: '#bdc3c7', line: {color: 'black', width: 1.5} });
            }

            curr_x = 0;
            for(let j = 0; j < lengths.length; j++) {
                let L_span = lengths[j];
                let x_start = curr_x;
                let x_end = curr_x + L_span;

                let has_rompi = (L_span > 4.5);
                let w_rompi = bw;
                let x_rompi_start = x_start + L_span/2 - w_rompi/2;
                let x_rompi_end = x_start + L_span/2 + w_rompi/2;

                let sz_left = support_zones[j];
                let sz_right = support_zones[j+1];

                for(let row = 0; row < num_rows; row++) {
                    let y0 = row * interasse + bw/2;
                    let y1 = y0 + b_pignatta;
                    let is_even = (row % 2 === 0);

                    let p_sx = sz_left.piena || 0;
                    let s_sx = sz_left.semi || 0;
                    let p_dx = sz_right.piena || 0;
                    let s_dx = sz_right.semi || 0;

                    let offset_start = is_even ? p_sx : s_sx;
                    let offset_end   = is_even ? p_dx : s_dx;

                    let pign_start = x_start + offset_start;
                    let pign_end   = x_end - offset_end;

                    let segments = [];
                    if (pign_start < pign_end) {
                        if (has_rompi && pign_start < x_rompi_start && pign_end > x_rompi_end) {
                            segments.push([pign_start, x_rompi_start]);
                            segments.push([x_rompi_end, pign_end]);
                        } else {
                            segments.push([pign_start, pign_end]);
                        }
                    }

                    segments.forEach(seg => {
                        let sx = seg[0];
                        let ex = seg[1];
                        shapes.push({ type: 'rect', x0: sx, y0: y0, x1: ex, y1: y1, line: {color: 'black', width: 1.5} });
                        let cx = sx;
                        while (cx < ex - 0.05) { 
                            cx += l_pignatta;
                            if (cx < ex) {
                                shapes.push({ type: 'line', x0: cx, y0: y0, x1: cx, y1: y1, line: {color: 'black', width: 0.8} });
                            }
                        }
                    });
                }
                curr_x += L_span;
            }

            let layout = {
                title: `Pianta Carpenteria Solaio (Vista dall'alto) - ${name}`,
                shapes: shapes,
                xaxis: { title: '', range: [-0.1, L_tot + 0.1], showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { title: '', range: [-0.1, b + 0.1], scaleanchor: "x", scaleratio: 1, showgrid: false, zeroline: false, showticklabels: false },
                margin: { t: 40, b: 20, l: 20, r: 20 },
                height: 300,
                plot_bgcolor: '#ffffff'
            };

            Plotly.newPlot('carpentryPlot', [], layout);
        }

        // Funzione helper per contare i ferri in una specifica sezione x dai dati 'barre_disegno'
        function getBarsAtX(barre, x_target) {
            let n_sup = 0, n_inf = 0;
            barre.forEach(b => {
                if(b.pos.includes('staffa') || b.pos.includes('zona')) return;
                // Controlla se x_target è nel range della barra
                if(x_target >= b.x_start && x_target <= b.x_end) {
                    let match = b.label.match(/(\d+)Ø/);
                    let num = match ? parseInt(match[1]) : 0;
                    if(b.pos.includes('sup')) n_sup += num;
                    if(b.pos.includes('inf')) n_inf += num;
                }
            });
            return {sup: n_sup, inf: n_inf};
        }

        async function runPremiumDesign() {
            document.getElementById('premiumStatus').innerText = "Calcolo in corso, attendere...";
            document.getElementById('premiumStatus').style.color = "#f39c12";
            
            sortSelectedMembers();
            
            let m_ed_scenarios = [];
            let n_ed_scenarios = [];
            let v_ed_scenarios = [];
            let lengths = [];
            
            for(let id of selectedMembers) { lengths.push(lastDiagrams[id].L); }
            let L_tot_fallback = lengths.reduce((a, b) => a + b, 0);
            
            let selectedCheckboxes = document.querySelectorAll('input[id^="chk_sc_"]:checked');
            if (selectedCheckboxes.length === 0 && savedScenarios.length > 0) {
                alert("Seleziona almeno uno scenario da inviluppare!");
                return;
            }

            let scenariosToProcess = selectedCheckboxes.length > 0 ? Array.from(selectedCheckboxes) : [{id: "chk_sc_last"}];

            for (let chk of scenariosToProcess) {
                let scenarioData = [];
                if (chk.id === "chk_sc_last") {
                    scenarioData = lastDiagrams;
                } else {
                    let idx = parseInt(chk.id.replace("chk_sc_", ""));
                    scenarioData = savedScenarios[idx];
                }
                
                let m_for_scenario = [];
                let n_for_scenario = [];
                let v_for_scenario = [];
                
                for(let id of selectedMembers) {
                    let d = scenarioData[id];
                    let n_i = nodes[members[id].node_i_id];
                    let n_j = nodes[members[id].node_j_id];
                    let is_reversed = n_i.x > n_j.x;

                    let m_arr = d.M.map(v => v / 1000);
                    let n_arr = d.N.map(v => v / 1000);
                    let v_arr = d.T.map(v => v / 1000);

                    if (is_reversed) {
                        m_arr.reverse();
                        n_arr.reverse();
                        v_arr.reverse();
                        v_arr = v_arr.map(v => -v); 
                    }
                    
                    m_for_scenario.push(m_arr);
                    n_for_scenario.push(n_arr);
                    v_for_scenario.push(v_arr);
                }
                
                m_ed_scenarios.push(m_for_scenario);
                n_ed_scenarios.push(n_for_scenario);
                v_ed_scenarios.push(v_for_scenario);
            }
            
            // --- CAPIAMO QUALE MATERIALE È SELEZIONATO ---
            let matSel = document.getElementById('prem_material');
            let material = matSel ? matSel.value : 'cls';

            // --- CREIAMO IL PACCHETTO DATI DA SPEDIRE AL SERVER ---
            let payload = {
                material: material,
                m_ed_scenarios: m_ed_scenarios,
                n_ed_scenarios: n_ed_scenarios,
                v_ed_scenarios: v_ed_scenarios,
                lengths: lengths
            };

            if (material === 'cls') {
                payload.has_shear_reinf = document.getElementById('prem_shear').value === 'yes';
                payload.b = parseFloat(document.getElementById('prem_b').value);
                payload.h = parseFloat(document.getElementById('prem_h').value);
                payload.c = parseFloat(document.getElementById('prem_c').value);
                payload.fck = parseFloat(document.getElementById('prem_fck').value);
                payload.fyk = parseFloat(document.getElementById('prem_fyk').value);
                payload.phi_long = parseInt(document.getElementById('prem_phi_long').value);
                payload.phi_shear = parseInt(document.getElementById('prem_phi_shear').value);
                payload.section_type = document.getElementById('prem_sec_type').value;
                payload.bw = parseFloat(document.getElementById('prem_bw').value);
                payload.hf = parseFloat(document.getElementById('prem_hf').value);
            } else {
                let profName = document.getElementById('prem_steel_profile').value;
                payload.steel_grade = parseInt(document.getElementById('prem_steel_grade').value);
                payload.profile_name = profName;
                payload.profile_data = steelProfiles[profName];

                payload.linf_custom = parseFloat(document.getElementById('prem_steel_linf').value) || 0;
                payload.is_fully_restrained = document.getElementById('prem_steel_restrained').checked;
            }

            try {
                let res = await fetch('https://beam2beam.onrender.com/premium/design-member', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                if (!res.ok) { let err = await res.json(); throw new Error(err.detail || "Errore dal server"); }
                
                let data = await res.json(); let r = data.data;

                let htmlDistinta = `<b>DISTINTA - L_tot = ${L_tot_fallback.toFixed(2)} m (Inviluppo di ${scenariosToProcess.length} scenari):</b><br><br>`;
                r.distinta.forEach(line => { htmlDistinta += `${line}<br>`; });
                document.getElementById('distintaBox').innerHTML = htmlDistinta;

                let x_axis = [];
                let current_x = 0;
                for(let j=0; j<lengths.length; j++) {
                    let L_asta = lengths[j];
                    let n_pts = m_ed_scenarios[0][j].length; 
                    for(let i = (j===0 ? 0 : 1); i < n_pts; i++) {
                        x_axis.push(current_x + (i / (n_pts - 1)) * L_asta);
                    }
                    current_x += L_asta;
                }
                
                let scenarioTracesM = [];
                let scenarioTracesV = [];
                for (let sc_idx=0; sc_idx < m_ed_scenarios.length; sc_idx++) {
                    let sm = [], sv = [];
                    for (let j=0; j < selectedMembers.length; j++) {
                        let m_arr = m_ed_scenarios[sc_idx][j];
                        let v_arr = v_ed_scenarios[sc_idx][j];
                        if (j > 0) { sm.push(...m_arr.slice(1)); sv.push(...v_arr.slice(1)); }
                        else { sm.push(...m_arr); sv.push(...v_arr); }
                    }
                    scenarioTracesM.push({ x: x_axis, y: sm, mode: 'lines', line: {color: 'rgba(0,0,0,0.15)', width: 1.5}, showlegend: false, hoverinfo: 'skip' });
                    scenarioTracesV.push({ x: x_axis, y: sv, mode: 'lines', line: {color: 'rgba(0,0,0,0.15)', width: 1.5}, showlegend: false, hoverinfo: 'skip' });
                }

                let annot_M = [], annot_V = [];

                let m_fill_top = r.m_max.map(v => Math.max(0, v));
                let m_fill_bottom = r.m_min.map(v => Math.min(0, v));
                let fill_x = x_axis.concat(x_axis.slice().reverse());
                let m_fill_y = m_fill_top.concat(m_fill_bottom.slice().reverse());

                let trace_M_fill = { x: fill_x, y: m_fill_y, fill: 'toself', mode: 'none', fillcolor: 'rgba(231, 76, 60, 0.3)', name: 'Inviluppo M' };
                let trace_M_max_line = { x: x_axis, y: r.m_max, mode: 'lines', line: {color: 'rgba(231, 76, 60, 1)', width: 2}, showlegend: false, hoverinfo: 'skip' };
                let trace_M_min_line = { x: x_axis, y: r.m_min, mode: 'lines', line: {color: 'rgba(231, 76, 60, 1)', width: 2}, showlegend: false, hoverinfo: 'skip' };
                
                let trace_Mrd_sup = { x: x_axis, y: r.m_rd_sup.map(v => -v), mode: 'lines', line: {color: '#2ecc71', shape: 'hv', width: 2}, name: 'M_Rd Sup' };
                let trace_Mrd_inf = { x: x_axis, y: r.m_rd_inf, mode: 'lines', line: {color: '#3498db', shape: 'hv', width: 2}, name: 'M_Rd Inf' };
                
                let v_m_pos = Math.max(...r.m_max); let i_m_pos = r.m_max.indexOf(v_m_pos);
                let v_m_neg = Math.min(...r.m_min); let i_m_neg = r.m_min.indexOf(v_m_neg);
                if(v_m_pos > 0.1) annot_M.push({ x: x_axis[i_m_pos], y: v_m_pos, text: `<b>M_Ed+: ${v_m_pos.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: 20, ay: 25, font: {color: '#c0392b', size:11}, bgcolor: 'rgba(255,255,255,0.85)' });
                if(v_m_neg < -0.1) annot_M.push({ x: x_axis[i_m_neg], y: v_m_neg, text: `<b>M_Ed-: ${v_m_neg.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: 20, ay: -25, font: {color: '#c0392b', size:11}, bgcolor: 'rgba(255,255,255,0.85)' });

                let v_mr_pos = Math.max(...r.m_rd_inf); let i_mr_pos = r.m_rd_inf.indexOf(v_mr_pos);
                let v_mr_neg = Math.min(...r.m_rd_sup.map(v => -v)); let i_mr_neg = r.m_rd_sup.map(v => -v).indexOf(v_mr_neg);
                if(v_mr_pos > 0.1) annot_M.push({ x: x_axis[i_mr_pos], y: v_mr_pos, text: `<b>M_Rd+: ${v_mr_pos.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: -30, ay: 25, font: {color: '#2980b9', size:11}, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#2980b9' });
                if(v_mr_neg < -0.1) annot_M.push({ x: x_axis[i_mr_neg], y: v_mr_neg, text: `<b>M_Rd-: ${v_mr_neg.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: -30, ay: -25, font: {color: '#27ae60', size:11}, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#27ae60' });

                let plot_data_M = [...scenarioTracesM, trace_M_fill, trace_M_max_line, trace_M_min_line, trace_Mrd_sup, trace_Mrd_inf];
                Plotly.newPlot('premiumPlot', plot_data_M, { title: `Verifica Pressoflessione SLU (Inviluppo)`, margin: {l: 50, r: 20, t: 40, b: 40}, hovermode: 'x unified', yaxis: { autorange: 'reversed', title: 'Momento (kNm)' }, xaxis: { title: 'Ascissa locale (m)' }, annotations: annot_M });

                let v_fill_top = r.v_max.map(v => Math.max(0, v));
                let v_fill_bottom = r.v_min.map(v => Math.min(0, v));
                let v_fill_y = v_fill_top.concat(v_fill_bottom.slice().reverse());

                let trace_V_fill = { x: fill_x, y: v_fill_y, fill: 'toself', mode: 'none', fillcolor: 'rgba(142, 68, 173, 0.3)', name: 'Inviluppo V' };
                let trace_V_max_line = { x: x_axis, y: r.v_max, mode: 'lines', line: {color: 'rgba(142, 68, 173, 1)', width: 2}, showlegend: false, hoverinfo: 'skip' };
                let trace_V_min_line = { x: x_axis, y: r.v_min, mode: 'lines', line: {color: 'rgba(142, 68, 173, 1)', width: 2}, showlegend: false, hoverinfo: 'skip' };

                let trace_Vrd_pos = { x: x_axis, y: r.v_rd_pos, mode: 'lines', line: {color: '#f39c12', shape: 'hv', width: 2, dash: 'dash'}, name: '+V_Rd (Resistente)' };
                let trace_Vrd_neg = { x: x_axis, y: r.v_rd_neg, mode: 'lines', line: {color: '#f39c12', shape: 'hv', width: 2, dash: 'dash'}, name: '-V_Rd (Resistente)' };
                
                let v_v_pos = Math.max(...r.v_max); let i_v_pos = r.v_max.indexOf(v_v_pos);
                let v_v_neg = Math.min(...r.v_min); let i_v_neg = r.v_min.indexOf(v_v_neg);
                if(v_v_pos > 0.1) annot_V.push({ x: x_axis[i_v_pos], y: v_v_pos, text: `<b>V_Ed+: ${v_v_pos.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: 20, ay: -25, font: {color: '#8e44ad', size:11}, bgcolor: 'rgba(255,255,255,0.85)' });
                if(v_v_neg < -0.1) annot_V.push({ x: x_axis[i_v_neg], y: v_v_neg, text: `<b>V_Ed-: ${v_v_neg.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: 20, ay: 25, font: {color: '#8e44ad', size:11}, bgcolor: 'rgba(255,255,255,0.85)' });

                let v_vr_pos = Math.max(...r.v_rd_pos); let i_vr_pos = r.v_rd_pos.indexOf(v_vr_pos);
                let v_vr_neg = Math.min(...r.v_rd_neg); let i_vr_neg = r.v_rd_neg.indexOf(v_vr_neg);
                if(v_vr_pos > 0.1) annot_V.push({ x: x_axis[i_vr_pos], y: v_vr_pos, text: `<b>V_Rd+: ${v_vr_pos.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: -30, ay: -25, font: {color: '#d35400', size:11}, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#d35400' });
                if(v_vr_neg < -0.1) annot_V.push({ x: x_axis[i_vr_neg], y: v_vr_neg, text: `<b>V_Rd-: ${v_vr_neg.toFixed(1)}</b>`, showarrow: true, arrowhead: 1, ax: -30, ay: 25, font: {color: '#d35400', size:11}, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#d35400' });

                let plot_data_V = [...scenarioTracesV, trace_V_fill, trace_V_max_line, trace_V_min_line, trace_Vrd_pos, trace_Vrd_neg];
                Plotly.newPlot('premiumShearPlot', plot_data_V, { title: `Verifica Taglio SLU (V_Ed vs V_Rd) Inviluppo`, margin: {l: 50, r: 20, t: 40, b: 40}, hovermode: 'x unified', yaxis: { title: 'Taglio (kN)' }, xaxis: { title: 'Ascissa locale (m)' }, annotations: annot_V });

                // =======================================================
                // GESTIONE VISUALIZZAZIONE IN BASE AL MATERIALE
                // =======================================================
                if (material === 'acciaio') {
                    // Nascondiamo forzatamente tutti i grafici e disegni tipici del cemento armato
                    if(document.getElementById('carpentryPlot')) document.getElementById('carpentryPlot').style.display = 'none';
                    if(document.getElementById('rebarPlot')) document.getElementById('rebarPlot').style.display = 'none';
                    if(document.getElementById('stirrupPlot')) document.getElementById('stirrupPlot').style.display = 'none';
                    if(document.getElementById('crossSectionsModule')) document.getElementById('crossSectionsModule').style.display = 'none';
                    if(document.getElementById('interactiveSectionModule')) document.getElementById('interactiveSectionModule').style.display = 'none';
                    
                    document.getElementById('btnPremiumDXF').style.display = 'none'; // Disabilitato DXF per l'acciaio al momento
                    document.getElementById('premiumStatus').innerText = "✅ Verifica Acciaio Completata!";
                    document.getElementById('premiumStatus').style.color = "#2ecc71";
                    
                } else {
                    // Visualizziamo le parti per il CLS
                    if(document.getElementById('rebarPlot')) document.getElementById('rebarPlot').style.display = 'block';
                    if(document.getElementById('stirrupPlot')) document.getElementById('stirrupPlot').style.display = 'block';
                    
                    let cp_div = document.getElementById('carpentryPlot');
                    let names = selectedMembers.map(id => members[id].name).join(' + ');
                    if (r.fasce_solaio || document.getElementById('prem_sec_type').value === 'Solaio') {
                        if(cp_div) cp_div.style.display = 'block';
                        drawCarpentry(r.fasce_solaio || {}, names, lengths);
                    } else {
                        if(cp_div) cp_div.style.display = 'none';
                    }

                    drawRebarLayout(r.barre_disegno, names, L_tot_fallback, payload.h, payload.c);
                    
                    let bars_sx = getBarsAtX(r.barre_disegno, 0.0);
                    let bars_mid = getBarsAtX(r.barre_disegno, 0.5);
                    let bars_dx = getBarsAtX(r.barre_disegno, 1.0);

                    lastPremiumData = {
                        name: names, L: L_tot_fallback, h: payload.h, c: payload.c,
                        m_max: r.m_max, m_min: r.m_min, m_rd_sup: r.m_rd_sup, m_rd_inf: r.m_rd_inf,
                        v_max: r.v_max, v_min: r.v_min, v_rd_pos: r.v_rd_pos, v_rd_neg: r.v_rd_neg,
                        distinta: r.distinta.map(line => line.replace(/<[^>]*>?/gm, '')).filter(l => l.trim() !== ""),
                        barre_disegno: r.barre_disegno,
                        b: payload.b, bw: payload.bw, hf: payload.hf, sec_type: payload.section_type,
                        phi_long: payload.phi_long, has_shear: payload.has_shear_reinf ? 'yes' : 'no',
                        fasce_solaio: r.fasce_solaio,
                        bars_sx: bars_sx, bars_mid: bars_mid, bars_dx: bars_dx
                    };
                    
                    document.getElementById('btnPremiumDXF').style.display = 'block';
                    document.getElementById('premiumStatus').innerText = "✅ Autodesign su Inviluppo Completato con successo!";
                    document.getElementById('premiumStatus').style.color = "#2ecc71";
                    
                    let crossMod = document.getElementById('crossSectionsModule');
                    if(crossMod) crossMod.style.display = 'block';
                    
                    let st_type = document.getElementById('prem_sec_type').value;
                    let b_v = payload.b; let h_v = payload.h; let bw_v = payload.bw; let hf_v = payload.hf;
                    let c_v = payload.c; let phi_v = payload.phi_long; let has_sh_v = document.getElementById('prem_shear').value;
                    
                    drawCrossSection('csPlotLeft', 'Sezione Appoggio Sx (x=0)', b_v, h_v, bw_v, hf_v, st_type, c_v, phi_v, bars_sx.sup, bars_sx.inf, has_sh_v);
                    drawCrossSection('csPlotMid', 'Sezione Mezzeria (Campata)', b_v, h_v, bw_v, hf_v, st_type, c_v, phi_v, bars_mid.sup, bars_mid.inf, has_sh_v);
                    drawCrossSection('csPlotRight', 'Sezione Appoggio Dx (x=L)', b_v, h_v, bw_v, hf_v, st_type, c_v, phi_v, bars_dx.sup, bars_dx.inf, has_sh_v);

                    let intMod = document.getElementById('interactiveSectionModule');
                    if(intMod) {
                        intMod.style.display = 'flex';
                        document.getElementById('slider_as_sup').value = bars_mid.sup;
                        document.getElementById('slider_as_inf').value = bars_mid.inf;
                        updateInteractiveSection();
                    }
                }

                setTimeout(() => { document.getElementById('resultsScrollArea').scrollTop = 9999; }, 200);

            } catch(err) { 
                document.getElementById('premiumStatus').innerText = "Errore: " + err.message; 
                document.getElementById('premiumStatus').style.color = "#e74c3c";
            }
        }
        async function exportPremiumDXF() {
            if (!lastPremiumData) return;
            document.getElementById('premiumStatus').innerText = "Generazione CAD in corso...";
            
            // 1. Sicurezza: forza a zero eventuali valori mancanti (NaN) per non far bloccare Python (Errore 422)
            lastPremiumData.b = lastPremiumData.b || 0;
            lastPremiumData.bw = lastPremiumData.bw || 0;
            lastPremiumData.hf = lastPremiumData.hf || 0;
            lastPremiumData.phi_long = lastPremiumData.phi_long || 16;

            try {
                let res = await fetch('https://beam2beam.onrender.com/premium/export-dxf', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify(lastPremiumData) 
                });
                
                // 2. Sicurezza: estrae il VERO errore dal backend Python se fallisce
                if (!res.ok) { 
                    let err = await res.json();
                    let msg = err.detail ? (typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)) : "Errore sconosciuto dal server";
                    throw new Error(msg); 
                }
                
                let blob = await res.blob(), url = window.URL.createObjectURL(blob), a = document.createElement('a');
                a.href = url; a.download = `Esecutivo_NTC_${lastPremiumData.name}.dxf`; document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
                document.getElementById('premiumStatus').innerText = "✅ Esportazione DXF completata!";
            } catch (err) { 
                // 3. Mostra l'errore reale a schermo
                alert("Dettaglio Errore Server:\n" + err.message); 
                document.getElementById('premiumStatus').innerText = "Errore esportazione DXF."; 
            }
        }

        function drawRebarLayout(barre, name, L, h_mm, c_mm) {
            let h_m = h_mm / 1000.0, c_m = c_mm / 1000.0, traces_long = [], traces_shear = [];
            let beamProfile = [{ x: [0, L], y: [h_m/2, h_m/2], mode: 'lines', line: {color: '#bdc3c7', width: 2}, showlegend: false, hoverinfo: 'skip' }, { x: [0, L], y: [-h_m/2, -h_m/2], mode: 'lines', line: {color: '#bdc3c7', width: 2}, showlegend: false, hoverinfo: 'skip' }];
            traces_long.push(...beamProfile); traces_shear.push(...beamProfile);
            let staffe_x = [], staffe_y = [];

            barre.forEach((b) => {
                if (b.pos === 'staffa_linea') { let x_s = b.x_start * L; staffe_x.push(x_s, x_s, null); staffe_y.push(-h_m/2 + c_m, h_m/2 - c_m, null); return; }
                if (b.pos === 'zona_staffe') {
                    let x_s = b.x_start * L, x_e = b.x_end * L, x_mid = (x_s + x_e) / 2, y_line = -h_m/2 - 0.25;
                    traces_shear.push({ x: [x_s, x_e, null, x_s, x_s, null, x_e, x_e], y: [y_line, y_line, null, y_line-0.08, y_line+0.08, null, y_line-0.08, y_line+0.08], mode: 'lines', line: {color: '#7f8c8d', width: 1.5}, hoverinfo: 'skip' });
                    traces_shear.push({ x: [x_mid], y: [y_line - 0.05], mode: 'text', text: [b.label], textposition: 'bottom center', textfont: {color: '#2c3e50', size: 11}, hoverinfo: 'skip' });
                    return; 
                }
                let y_val = 0, y_txt = 0, color = '#2c3e50', txt_pos = 'top center';
                if (b.pos === 'sup') { y_val = h_m/2 - c_m; y_txt = y_val + 0.02; txt_pos = 'top center'; }
                if (b.pos === 'sup_m') { y_val = h_m/2 - c_m - 0.05; y_txt = y_val - 0.02; txt_pos = 'bottom center'; color = '#e67e22'; } 
                if (b.pos === 'inf_m') { y_val = -h_m/2 + c_m + 0.05; y_txt = y_val + 0.02; txt_pos = 'top center'; color = '#e67e22'; } 
                if (b.pos === 'inf') { y_val = -h_m/2 + c_m; y_txt = y_val - 0.02; txt_pos = 'bottom center'; }       
                let x_s = b.x_start * L, x_e = b.x_end * L, x_mid = (x_s + x_e) / 2;
                traces_long.push({ x: [x_s, x_e], y: [y_val, y_val], mode: 'lines', line: {color: color, width: 5}, name: b.label, hoverinfo: 'name' });
                traces_long.push({ x: [x_mid], y: [y_txt], mode: 'text', text: [b.label], textposition: txt_pos, textfont: {color: color, size: 11, weight: 'bold'}, hoverinfo: 'skip', showlegend: false });
            });

            if (staffe_x.length > 0) traces_shear.push({ x: staffe_x, y: staffe_y, mode: 'lines', line: {color: '#95a5a6', width: 1.5}, name: 'Staffe', hoverinfo: 'skip', showlegend: false });
            Plotly.newPlot('rebarPlot', traces_long, { title: `Armature Longitudinali`, xaxis: { range: [-0.05 * L, L * 1.05], showgrid: false }, yaxis: { range: [-h_m/2 - 0.2, h_m/2 + 0.2], showgrid: false, zeroline: false, showticklabels: false }, margin: {t: 40, b: 40, l: 30, r: 30}, showlegend: false, height: 180 });
            Plotly.newPlot('stirrupPlot', traces_shear, { title: `Disposizione Staffe / Fasce Taglio`, xaxis: { range: [-0.05 * L, L * 1.05], showgrid: false }, yaxis: { range: [-h_m/2 - 0.6, h_m/2 + 0.2], showgrid: false, zeroline: false, showticklabels: false }, margin: {t: 40, b: 40, l: 30, r: 30}, showlegend: false, height: 180 });
        }

        function drawPlotlyResults(diagrams) {
            let tracesN = [], tracesT = [], tracesM = [], tracesDef = [], annotN = [], annotT = [], annotM = [];
            let globalMaxN = -Infinity, globalMinN = Infinity, globalMaxT = -Infinity, globalMinT = Infinity, globalMaxM = -Infinity, globalMinM = Infinity;
            
            for(let id in diagrams) {
                let valsN = diagrams[id].N.map(v => v/1000), valsT = diagrams[id].T.map(v => v/1000), valsM = diagrams[id].M.map(v => v/1000);
                globalMaxN = Math.max(globalMaxN, ...valsN); globalMinN = Math.min(globalMinN, ...valsN);
                globalMaxT = Math.max(globalMaxT, ...valsT); globalMinT = Math.min(globalMinT, ...valsT);
                globalMaxM = Math.max(globalMaxM, ...valsM); globalMinM = Math.min(globalMinM, ...valsM);
            }

            for(let id in members) {
                let ni = nodes[members[id].node_i_id], nj = nodes[members[id].node_j_id];
                tracesDef.push({ x: [ni.x, nj.x], y: [ni.y, nj.y], mode: 'lines', line: {color: '#bdc3c7', dash: 'dash', width:2}, hoverinfo: 'none', showlegend:false });
            }

            let maxL = Math.max(1, ...Object.values(diagrams).map(d => d.L));
            let maxN = Math.max(1e-5, Math.abs(globalMaxN), Math.abs(globalMinN)), maxT = Math.max(1e-5, Math.abs(globalMaxT), Math.abs(globalMinT)), maxM = Math.max(1e-5, Math.abs(globalMaxM), Math.abs(globalMinM));
            let scN = (maxL * 0.25) / maxN, scT = (maxL * 0.25) / maxT, scM = (maxL * 0.25) / maxM;
            let maxDisp = Math.max(1e-9, ...Object.values(diagrams).flatMap(d => d.dx_g.map((dx, i) => Math.hypot(dx, d.dy_g[i]))));
            let scDisp = (maxL * 0.10) / maxDisp;

            function processDiagram(d, valArray, scaleFactor, color, invertM, tracesArray, annotArray, globalMax, globalMin) {
                let dir = invertM ? -1 : 1, vals = valArray.map(v => v/1000), xp = [], yp = [], hoverTexts = [];
                let localMax = Math.max(...vals), localMin = Math.min(...vals), idxMax = vals.indexOf(localMax), idxMin = vals.indexOf(localMin);

                for(let i=0; i<d.X_base.length; i++) {
                    xp.push(d.X_base[i] - vals[i] * scaleFactor * d.s * dir); yp.push(d.Y_base[i] + vals[i] * scaleFactor * d.c * dir);
                    hoverTexts.push(`<b>${d.name}</b><br>Pos: ${(i * d.L / (d.X_base.length - 1)).toFixed(2)} m<br>Valore: ${vals[i].toFixed(2)}`);
                }
                tracesArray.push({ x: d.X_base.concat(xp.slice().reverse(), [d.X_base[0]]), y: d.Y_base.concat(yp.slice().reverse(), [d.Y_base[0]]), fill: 'toself', mode: 'none', fillcolor: color, opacity: 0.2, showlegend: false, hoverinfo: 'skip' });
                tracesArray.push({ x: xp, y: yp, mode: 'lines', line: {color: color, width: 2}, showlegend: false, hoverinfo: 'text', text: hoverTexts });

                let makeAnnot = (val, idx, isGlobal) => {
                    if (Math.abs(val) > 1e-3) {
                        let isMax = val === localMax;
                        annotArray.push({ x: xp[idx], y: yp[idx], text: isGlobal ? `<b>ASSOLUTO: ${val.toFixed(2)}</b>` : `${isMax?'Max':'Min'}: ${val.toFixed(2)}`, showarrow: true, arrowhead: 1, ax: isMax ? 25 : -25, ay: isMax ? -25 : 25, font: {size: isGlobal ? 12 : 10, color: isGlobal ? 'red' : 'black'}, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: isGlobal ? 'red' : '#ccc' });
                    }
                };
                makeAnnot(localMax, idxMax, localMax === globalMax);
                if (localMin !== localMax) makeAnnot(localMin, idxMin, localMin === globalMin);
            }

            for(let id in diagrams) {
                let d = diagrams[id];
                processDiagram(d, d.N, scN, '#3498db', false, tracesN, annotN, globalMaxN, globalMinN);
                processDiagram(d, d.T, scT, '#2ecc71', false, tracesT, annotT, globalMaxT, globalMinT);
                processDiagram(d, d.M, scM, '#e74c3c', true, tracesM, annotM, globalMaxM, globalMinM);
                tracesDef.push({ x: d.X_base.map((x, i) => x + d.dx_g[i] * scDisp), y: d.Y_base.map((y, i) => y + d.dy_g[i] * scDisp), mode: 'lines', line: {color: '#9b59b6', width: 3}, name: d.name, hoverinfo: 'text', text: d.dx_g.map((dx,i) => `<b>${d.name}</b><br>Pos: ${(i * d.L / (d.X_base.length - 1)).toFixed(2)} m<br>δx: ${(dx*1000).toFixed(3)}mm<br>δy: ${(d.dy_g[i]*1000).toFixed(3)}mm`) });
            }

            let layoutBase = { margin: {l: 40, r: 40, t: 30, b: 20}, autosize: true, xaxis: {scaleanchor:"y", scaleratio:1, showgrid:true}, yaxis: {showgrid:true}, hovermode: 'closest' };
            Plotly.newPlot('plot_N', tracesN, {title: {text:'Sforzo Normale (N) [kN]', font:{size:14}}, annotations: annotN, ...layoutBase});
            Plotly.newPlot('plot_T', tracesT, {title: {text:'Taglio (T) [kN]', font:{size:14}}, annotations: annotT, ...layoutBase});
            Plotly.newPlot('plot_M', tracesM, {title: {text:'Momento Flettente (M) [kNm]', font:{size:14}}, annotations: annotM, ...layoutBase});
            Plotly.newPlot('plot_Def', tracesDef, {title: {text:'Deformata Interattiva', font:{size:14}}, ...layoutBase});
        }
            

        
// --- MODULO CALCOLATORE GEOMETRIA SEZIONE ---
        let currentCalcVals = { a: 0, i: 0 };

        function openSectionCalcModal() {
            document.getElementById('sectionCalcModal').style.display = 'flex';
            updateCalcUI();
        }

        function updateCalcUI() {
            let shape = document.getElementById('calc_shape').value;
            
            document.getElementById('row_calc_b').style.display = (shape === 'rect' || shape === 't' || shape === 'solaio') ? 'flex' : 'none';
            document.getElementById('row_calc_h').style.display = (shape === 'rect' || shape === 't' || shape === 'solaio') ? 'flex' : 'none';
            document.getElementById('row_calc_bw').style.display = (shape === 't' || shape === 'solaio') ? 'flex' : 'none';
            document.getElementById('row_calc_hf').style.display = (shape === 't' || shape === 'solaio') ? 'flex' : 'none';
            document.getElementById('row_calc_d').style.display = (shape === 'circle') ? 'flex' : 'none';
            
            let rowSteel = document.getElementById('row_calc_steel');
            if (rowSteel) {
                rowSteel.style.display = (shape === 'acciaio') ? 'flex' : 'none';
                if (shape === 'acciaio') {
                    let sel = document.getElementById('calc_steel_profile');
                    if(sel.options.length === 0) {
                        for(let p in steelProfiles) sel.add(new Option(p, p));
                    }
                }
            }
            
            computeSectionProperties();
        }

        function computeSectionProperties() {
            let shape = document.getElementById('calc_shape').value;
            let A = 0, I = 0;
            let svgHtml = "";
            let svgW = 250, svgH = 250, pad = 30;

            if (shape === 'rect') {
                let b = (parseFloat(document.getElementById('calc_b').value) || 0) / 100.0;
                let h = (parseFloat(document.getElementById('calc_h').value) || 0) / 100.0;
                A = b * h;
                I = (b * Math.pow(h, 3)) / 12.0;

                let maxDim = Math.max(b, h) || 1;
                let sc = (svgH - 2*pad) / maxDim;
                let dw = b * sc, dh = h * sc;
                svgHtml = `<svg width="${svgW}" height="${svgH}"><rect x="${(svgW-dw)/2}" y="${(svgH-dh)/2}" width="${dw}" height="${dh}" fill="#3498db" stroke="#2c3e50" stroke-width="2"/></svg>`;
            } 
            else if (shape === 't') {
                let b = (parseFloat(document.getElementById('calc_b').value) || 0) / 100.0;
                let h = (parseFloat(document.getElementById('calc_h').value) || 0) / 100.0;
                let bw = (parseFloat(document.getElementById('calc_bw').value) || 0) / 100.0;
                let hf = (parseFloat(document.getElementById('calc_hf').value) || 0) / 100.0;
                
                let A1 = b * hf; 
                let A2 = bw * (h - hf); 
                A = A1 + A2;

                if (A > 0) {
                    let y1 = h - hf/2.0;
                    let y2 = (h - hf)/2.0;
                    let yg = (A1 * y1 + A2 * y2) / A;

                    let I1 = (b * Math.pow(hf, 3)) / 12.0 + A1 * Math.pow(y1 - yg, 2);
                    let I2 = (bw * Math.pow(h - hf, 3)) / 12.0 + A2 * Math.pow(y2 - yg, 2);
                    I = I1 + I2;
                }

                let maxDim = Math.max(b, h) || 1;
                let sc = (svgH - 2*pad) / maxDim;
                let dw = b * sc, dh = h * sc, dbw = bw * sc, dhf = hf * sc;
                let offX = (svgW-dw)/2, offY = (svgH-dh)/2;
                let path = `M ${offX},${offY} h ${dw} v ${dhf} h ${-(dw-dbw)/2} v ${dh-dhf} h ${-dbw} v ${-(dh-dhf)} h ${-(dw-dbw)/2} z`;
                svgHtml = `<svg width="${svgW}" height="${svgH}"><path d="${path}" fill="#3498db" stroke="#2c3e50" stroke-width="2"/></svg>`;
            }
            else if (shape === 'circle') {
                let d = (parseFloat(document.getElementById('calc_d').value) || 0) / 100.0;
                let r = d / 2.0;
                A = Math.PI * Math.pow(r, 2);
                I = (Math.PI * Math.pow(r, 4)) / 4.0;

                let maxDim = d || 1;
                let sc = (svgH - 2*pad) / maxDim;
                let dr = r * sc;
                svgHtml = `<svg width="${svgW}" height="${svgH}"><circle cx="${svgW/2}" cy="${svgH/2}" r="${dr}" fill="#3498db" stroke="#2c3e50" stroke-width="2"/></svg>`;
            }
            else if (shape === 'solaio') {
                let b = (parseFloat(document.getElementById('calc_b').value) || 0) / 100.0;
                let h = (parseFloat(document.getElementById('calc_h').value) || 0) / 100.0;
                let bw = (parseFloat(document.getElementById('calc_bw').value) || 0) / 100.0;
                let hf = (parseFloat(document.getElementById('calc_hf').value) || 0) / 100.0;
                
                // Matematicamente si modella come una T-equivalente
                let A1 = b * hf; let A2 = bw * (h - hf); A = A1 + A2;
                if (A > 0) {
                    let y1 = h - hf/2.0; let y2 = (h - hf)/2.0; let yg = (A1 * y1 + A2 * y2) / A;
                    I = ((b * Math.pow(hf, 3)) / 12.0 + A1 * Math.pow(y1 - yg, 2)) + ((bw * Math.pow(h - hf, 3)) / 12.0 + A2 * Math.pow(y2 - yg, 2));
                }
                
                // Disegno SVG a due travetti accostati
                let sc = (svgH - 2*pad) / (Math.max(b, h) || 1);
                let dw = b * sc, dh = h * sc, dbw = bw * sc, dhf = hf * sc;
                let offX = (svgW-dw)/2, offY = (svgH-dh)/2;
                let gap = (dw/2) - dbw;
                let path = `M ${offX},${offY} h ${dw} v ${dhf} h ${-gap/2} v ${dh-dhf} h ${-dbw} v ${-(dh-dhf)} h ${-gap} v ${dh-dhf} h ${-dbw} v ${-(dh-dhf)} h ${-gap/2} z`;
                svgHtml = `<svg width="${svgW}" height="${svgH}"><path d="${path}" fill="#3498db" stroke="#2c3e50" stroke-width="2"/></svg>`;
            }
            else if (shape === 'acciaio') {
                let profName = document.getElementById('calc_steel_profile').value;
                if (profName && steelProfiles[profName]) {
                    let p = steelProfiles[profName];
                    
                    // Convertiamo l'Area da cm² a m² (/ 10000)
                    A = p.A / 10000.0;
                    // Convertiamo l'Inerzia forte (Iy) da cm⁴ a m⁴ (/ 100000000)
                    I = p.Iy / 100000000.0; 
                    
                    // Creiamo il disegno SVG
                    let maxDim = Math.max(p.b, p.h) || 1;
                    let sc = (svgH - 2*pad) / maxDim;
                    let sw = p.b * sc, sh = p.h * sc, stw = Math.max(2, p.tw * sc), stf = Math.max(2, p.tf * sc);
                    let offX = (svgW - sw)/2, offY = (svgH - sh)/2;
                    
                    let path = `M ${offX},${offY} h ${sw} v ${stf} h ${-(sw-stw)/2} v ${sh-2*stf} h ${(sw-stw)/2} v ${stf} h ${-sw} v ${-stf} h ${(sw-stw)/2} v ${-(sh-2*stf)} h ${-(sw-stw)/2} z`;
                    svgHtml = `<svg width="${svgW}" height="${svgH}"><path d="${path}" fill="#bdc3c7" stroke="#2c3e50" stroke-width="2"/></svg>`;
                }
            }

            currentCalcVals.a = A;
            currentCalcVals.i = I;
            document.getElementById('out_calc_a').innerText = A.toFixed(6) + " m²";
            document.getElementById('out_calc_i').innerText = I.toExponential(4).replace('e', ' x 10^') + " m⁴";
            document.getElementById('calcSvgContainer').innerHTML = svgHtml;
        }

        function applySectionProperties() {
            if (currentCalcVals.a <= 0 || currentCalcVals.i <= 0) {
                alert("Calcola prima una sezione valida!");
                return;
            }
            document.getElementById('entry_a').value = currentCalcVals.a.toFixed(6);
            let i_val = currentCalcVals.i;
            document.getElementById('entry_i').value = i_val.toFixed(8);
            
            document.getElementById('sectionCalcModal').style.display = 'none';
            document.getElementById('statusBar').innerText = "Proprietà (A, I) applicate con successo! Disegna o modifica l'asta.";
        }
        // --- MODULO ANALISI DEI CARICHI ---
        function openLoadAnalysisModal() {
    document.getElementById('loadAnalysisModal').style.display = 'flex';
    toggleG1Inputs(); // Lanciamo la nuova funzione che calcola i carichi appena apri la finestra!
}

function toggleG1Inputs() {
    const type = document.getElementById('g1_calc_type').value;
    
    let elLat = document.getElementById('ui_g1_laterocemento'); if(elLat) elLat.style.display = (type === 'laterocemento') ? 'flex' : 'none';
    let elPiena = document.getElementById('ui_g1_piena'); if(elPiena) elPiena.style.display = (type === 'piena') ? 'block' : 'none';
    let elLegno = document.getElementById('ui_g1_legno'); if(elLegno) elLegno.style.display = (type === 'legno') ? 'flex' : 'none';
    let elAcc = document.getElementById('ui_g1_acciaio'); if(elAcc) elAcc.style.display = (type === 'acciaio') ? 'flex' : 'none';

    if (type === 'acciaio') {
        let sel = document.getElementById('g1_steel_prof');
        if(sel && sel.options.length === 0) {
            for(let p in steelProfiles) sel.add(new Option(p, p));
        }
    }
    computeG1Analytic();
}

function computeG1Analytic() {
    let g1 = 0;
    const type = document.getElementById('g1_calc_type').value;

    if (type === 'laterocemento') {
        let s = parseFloat(document.getElementById('g1_s').value)/100, h = parseFloat(document.getElementById('g1_h').value)/100, i = parseFloat(document.getElementById('g1_i').value)/100;
        g1 = ((s + (0.10 * h) * (1 / i)) * 25.0) + (((i - 0.10) * h * (1 / i)) * 8.0);
    } 
    else if (type === 'piena') {
        g1 = (parseFloat(document.getElementById('g1_h_piena').value) / 100) * 25.0;
    } 
    else if (type === 'legno') {
        let b = parseFloat(document.getElementById('g1_b_legno').value)/100;
        let h = parseFloat(document.getElementById('g1_h_legno').value)/100;
        let i = parseFloat(document.getElementById('g1_i_legno').value)/100;
        let s = parseFloat(document.getElementById('g1_s_legno').value)/100;
        g1 = ((b * h) / i) * 6.0 + (s * 6.0); // Peso specifico Legno C24
    }
    else if (type === 'acciaio') {
        let pName = document.getElementById('g1_steel_prof').value;
        let interasse = parseFloat(document.getElementById('g1_i_steel').value)/100;
        let s_getto = parseFloat(document.getElementById('g1_s_steel').value)/100;
        let peso_profilo = steelProfiles[pName] ? (steelProfiles[pName].A * 0.785) / 100 : 0;
        g1 = (peso_profilo / interasse) + (s_getto * 25.0); 
    }

    let g2 = 0.8; 
    let chk_pav = document.getElementById('chk_pav'); if(chk_pav && chk_pav.checked) g2 += 0.4;
    let chk_mas = document.getElementById('chk_mas'); if(chk_mas && chk_mas.checked) g2 += 0.8;
    let chk_int = document.getElementById('chk_int'); if(chk_int && chk_int.checked) g2 += 0.3;

    let zona = document.getElementById('zona_neve').value;
    let quota = parseFloat(document.getElementById('altitudine').value) || 0;
    let qsk = (zona === '1_alp') ? (quota <= 200 ? 1.5 : 1.39 * (1 + Math.pow(quota/728, 2))) : (quota <= 200 ? 1.0 : 0.51 * (1 + Math.pow(quota/481, 2)));
    let qs = 0.8 * qsk; 
    
    // Sicurezza: aggiorna qs a schermo solo se il campo esiste ancora
    let out_qs_el = document.getElementById('out_qs');
    if (out_qs_el) out_qs_el.innerText = qs.toFixed(2);

    let qk_uso = parseFloat(document.getElementById('qk_type').value);
    
    // Calcolo SFAVOREVOLE (Max tra combinazione uso e neve NTC)
    let comb1 = (1.3*g1 + 1.5*g2) + (1.5*qk_uso) + (1.5*0.5*qs);
    let comb2 = (1.3*g1 + 1.5*g2) + (1.5*qs) + (1.5*0.7*qk_uso);
    let max_mq = Math.max(comb1, comb2);

    // Calcolo FAVOREVOLE (Solo peso strutturale)
    let min_mq = 1.0 * g1;

    let w = parseFloat(document.getElementById('inf_width').value) || 1.0;
    
    // Aggiornamento interfaccia
    let el_sfav = document.getElementById('res_slu_sfav');
    let el_fav = document.getElementById('res_slu_fav');
    if (el_sfav) el_sfav.innerText = (max_mq * w).toFixed(2) + " kN/m";
    if (el_fav) el_fav.innerText = (min_mq * w).toFixed(2) + " kN/m";
}

function applyAnalyzedLoad(scelta) {
    let span = (scelta === 'sfav') ? 'res_slu_sfav' : 'res_slu_fav';
    let val = parseFloat(document.getElementById(span).innerText);
    
    document.getElementById('entry_qy_i').value = (-val).toFixed(2);
    document.getElementById('entry_qy_j').value = (-val).toFixed(2);
    document.getElementById('loadAnalysisModal').style.display = 'none';
    document.getElementById('statusBar').innerText = `✅ Carico ${scelta.toUpperCase()} di -${val.toFixed(2)} kN/m importato.`;
}
// VARIABILI GLOBALI PER IL NODO CORRENTE
let current_kb = 0; 
let boundary_cerniera = 0;
let boundary_incastro = 0;

function openNodeModal() {
    console.log("1. Il tasto è stato cliccato e la funzione è partita!");
    
    // Controlliamo cosa vede il programma
    console.log("2. Cosa c'è dentro selectedObject?", selectedObject);

    // 1. Controlla che un nodo sia selezionato
    if (!selectedObject || selectedObject.type !== 'node') {
        console.log("3. BLOCCO: Nessun nodo valido trovato!");
        alert("Devi prima selezionare un Nodo (pallino blu o arancione) con lo strumento 'Seleziona'.");
        return;
    }
    
    let nId = selectedObject.id;
    let titleEl = document.getElementById('nodeModalTitle');
    if (titleEl) titleEl.innerText = `🔗 Progetto Giunti in Acciaio: NODO N${nId}`;
    
    // 2. Trova le aste collegate a questo nodo
    let connectedMembers = [];
    for (let mId in members) {
        if (members[mId].node_i_id == nId || members[mId].node_j_id == nId) {
            connectedMembers.push(members[mId]);
        }
    }
    
    if (connectedMembers.length === 0) {
        console.log("3. BLOCCO: Il nodo non ha aste collegate!");
        alert("Questo nodo non è collegato a nessuna asta.");
        return;
    }
    
    console.log("4. Asta trovata! Apro il modale...");
    
    // 3. Estrai le proprietà flessionali
    let beam = connectedMembers[0];
    let n_i = nodes[beam.node_i_id];
    let n_j = nodes[beam.node_j_id];
    let L = Math.hypot(n_j.x - n_i.x, n_j.y - n_i.y);
    current_kb = (beam.E * beam.I / L) / 1000.0; 
    
    // 4. Apri il modale e disegna
    document.getElementById('nodeModal').style.display = 'flex';
    document.getElementById('node_res_sj').innerText = "0.0 kNm/rad";
    document.getElementById('node_res_class').innerText = "Classificazione: -";
    document.getElementById('node_res_class').style.color = "white";
    
    drawJointPreview();
}

function classifyNode() {
    // Legge la rigidezza appena calcolata (estraendo il numero dal testo)
    let sj_text = document.getElementById('node_res_sj').innerText;
    let Sj = parseFloat(sj_text);
    if (isNaN(Sj)) return;

    // Limiti Normativi (Assumiamo telaio controventato: Incastro >= 8*kb)
    let boundary_cerniera = 0.5 * current_kb;
    let boundary_incastro = 8.0 * current_kb;
    
    let classText = document.getElementById('node_res_class');
    if (!classText) return;
    
    // Classifica
    if (Sj <= boundary_cerniera) {
        classText.innerText = "Classificazione: CERNIERA (Nodo Flessibile)";
        classText.style.color = "#e74c3c"; // Rosso
    } else if (Sj >= boundary_incastro) {
        classText.innerText = "Classificazione: INCASTRO (Nodo Rigido)";
        classText.style.color = "#2ecc71"; // Verde
    } else {
        classText.innerText = "Classificazione: SEMI-RIGIDO";
        classText.style.color = "#f1c40f"; // Giallo
    }
}
// ============================================================
// CERVELLO UNIFICATO DEL COSTRUTTORE DI NODI
// ============================================================
let nodeComponents = []; 
let activeComponentId = null;

function updateUI() { renderList(); renderInspector(); drawConstructedJoint(); }

function selectComponent(id) { activeComponentId = id; updateUI(); }

function removeComp(id) { 
    nodeComponents = nodeComponents.filter(c => c.id !== id); 
    if(activeComponentId === id) activeComponentId = null; 
    updateUI(); 
}

function updateProp(key, val) { 
    let comp = nodeComponents.find(c => c.id === activeComponentId); 
    if(comp) { 
        if (['target', 'tipo', 'classe', 'cls', 'work'].includes(key)) {
            comp[key] = val; 
        } else {
            comp[key] = parseFloat(val) || 0; 
        }
        drawConstructedJoint(); 
    } 
}

function initNodeBuilder() {
    const selector = document.getElementById('newComponentType');
    if (selector) {
        selector.replaceWith(selector.cloneNode(true)); 
        const newSelector = document.getElementById('newComponentType');
        
        newSelector.addEventListener('change', function(e) {
            if(!e.target.value) return;
            
            let tipo = e.target.value;
            let nuovo = { id: Date.now(), type: tipo, name: tipo };

            if(['plate', 'splice', 'gusset', 'angle', 'stiffener'].includes(tipo)) {
                nuovo.t = 10; nuovo.b = 150; nuovo.h = 200; 
                if(tipo === 'plate') { nuovo.name = "Piastra Frontale"; nuovo.t = 15; nuovo.b = 200; nuovo.h = 450; }
                else if(tipo === 'angle') { nuovo.name = "Squadretta L"; nuovo.b = 80; } 
                else if(tipo === 'splice') { nuovo.name = "Coprigiunto"; nuovo.b = 120; nuovo.h = 250; }
                else if(tipo === 'gusset') { nuovo.name = "Fazzoletto"; nuovo.b = 300; nuovo.h = 300; }
                else if(tipo === 'stiffener') { nuovo.name = "Irrigidimento"; }
            }
            else if(tipo === 'bolts') { 
                nuovo.name = "Griglia Bulloni"; nuovo.d = 16; nuovo.classe = "8.8"; 
                nuovo.rows = 2; nuovo.cols = 2; nuovo.p1 = 70; nuovo.p2 = 70; 
                nuovo.e1 = 40; nuovo.e2 = 40; nuovo.target = "nessuno"; 
            }
            else if(tipo === 'weld') { 
                nuovo.name = "Saldatura"; nuovo.a = 5; nuovo.tipo = "angolo"; nuovo.l = 200; nuovo.target = "nessuno";
            }
            else if(tipo === 'anchor') { 
                nuovo.name = "Tirafondi"; nuovo.d = 20; nuovo.L_anc = 400; nuovo.cls = "C25/30"; 
            }

            nodeComponents.push(nuovo);
            activeComponentId = nuovo.id;
            e.target.value = ""; 
            updateUI();
        });
    }
}
window.addEventListener('DOMContentLoaded', initNodeBuilder);

function renderList() {
    const listDiv = document.getElementById('componentsList');
    if (!listDiv) return;
    let html = "";
    nodeComponents.forEach(c => {
        const isActive = (c.id === activeComponentId);
        html += `<div onclick="selectComponent(${c.id})" style="padding: 10px; border-radius: 6px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; margin-bottom:5px; transition:0.2s; ${isActive ? 'border: 2px solid #e67e22; background: #fffaf0;' : 'border: 1px solid #ddd; background: #fff;'}"><span style="font-size: 13px; font-weight: bold; color: #2c3e50;">${c.type === 'plate' ? '🔲' : c.type === 'bolts' ? '🔩' : c.type === 'stiffener' ? '🔺' : '📐'} ${c.name}</span><button onclick="removeComp(${c.id}); event.stopPropagation();" style="border:none; background:none; color:#e74c3c; cursor:pointer; font-size:16px;">&times;</button></div>`;
    });
    listDiv.innerHTML = html || "<p style='text-align:center; color:#95a5a6; font-size:12px; padding:20px;'>Nessun componente aggiunto.</p>";
}

function renderInspector() {
    const panel = document.getElementById('inspectorPanel');
    if (!panel) return;

    const comp = nodeComponents.find(c => c.id === activeComponentId);
    if(!comp) { 
        panel.innerHTML = "<p style='text-align:center; color:#95a5a6; margin-top:50px;'>Seleziona un componente<br>per modificarlo</p>"; 
        return; 
    }

    let html = `<h3 style="margin:0 0 15px 0; font-size:15px; color:#e67e22; border-bottom:1px solid #eee; padding-bottom:10px;">Dettagli: ${comp.name}</h3>`;
    
    if(['plate', 'splice', 'gusset', 'angle', 'stiffener'].includes(comp.type)) {
        html += `
        <div class="prop-group"><label>Spessore **t** (mm)</label><input type="number" value="${comp.t}" oninput="updateProp('t', this.value)"></div>
        <div class="prop-group"><label>Larghezza/Lato **b** (mm)</label><input type="number" value="${comp.b}" oninput="updateProp('b', this.value)"></div>
        <div class="prop-group"><label>Altezza **h** (mm)</label><input type="number" value="${comp.h}" oninput="updateProp('h', this.value)"></div>`;
    } 
    
    if(['bolts', 'weld'].includes(comp.type)) {
        let targetOptions = `<option value="nessuno" ${comp.target==='nessuno'?'selected':''}>-- Seleziona Faccia --</option>
                             <option value="ala_colonna" ${comp.target==='ala_colonna'?'selected':''}>Ala Colonna (Esterna)</option>
                             <option value="anima_trave" ${comp.target==='anima_trave'?'selected':''}>Anima Trave</option>`;
        
        nodeComponents.forEach(c => {
            if(['plate', 'splice'].includes(c.type)) {
                targetOptions += `<option value="${c.id}" ${comp.target==c.id?'selected':''}>${c.name} (ID: ${c.id})</option>`;
            }
            else if (c.type === 'angle') {
                targetOptions += `<option value="${c.id}_both" ${comp.target==c.id+'_both'?'selected':''}>${c.name} (Entrambe le facce)</option>`;
                targetOptions += `<option value="${c.id}_f1" ${comp.target==c.id+'_f1'?'selected':''}>${c.name} (Faccia 1 - Colonna)</option>`;
                targetOptions += `<option value="${c.id}_f2" ${comp.target==c.id+'_f2'?'selected':''}>${c.name} (Faccia 2 - Trave)</option>`;
            }
        });

        html += `
        <div class="prop-group" style="background:#fdfbf7; padding:10px; border-radius:4px; border-left:3px solid #e67e22;">
            <label style="color:#d35400;">Su quale elemento agisce?</label>
            <select onchange="updateProp('target', this.value)" style="border-color:#e67e22;">${targetOptions}</select>
        </div>`;
    }

    if(comp.type === 'bolts') {
        html += `
        <div class="prop-group" style="display:flex; gap:10px;">
            <div style="flex:1"><label>Classe</label><select onchange="updateProp('classe', this.value)"><option value="8.8" ${comp.classe=='8.8'?'selected':''}>8.8</option><option value="10.9" ${comp.classe=='10.9'?'selected':''}>10.9</option></select></div>
            <div style="flex:1"><label>Vite Ø</label><select onchange="updateProp('d', this.value)"><option value="12" ${comp.d==12?'selected':''}>M12</option><option value="16" ${comp.d==16?'selected':''}>M16</option><option value="20" ${comp.d==20?'selected':''}>M20</option></select></div>
        </div>
        <div class="prop-group" style="display:flex; gap:10px; border-top:1px dashed #ccc; padding-top:10px;">
            <div style="flex:1"><label>Righe (n1)</label><input type="number" value="${comp.rows}" oninput="updateProp('rows', this.value)"></div>
            <div style="flex:1"><label>Colonne (n2)</label><input type="number" value="${comp.cols}" oninput="updateProp('cols', this.value)"></div>
        </div>
        <div class="prop-group" style="display:flex; gap:10px;">
            <div style="flex:1"><label>Passo vert. p1</label><input type="number" value="${comp.p1}" oninput="updateProp('p1', this.value)"></div>
            <div style="flex:1"><label>Passo orizz. p2</label><input type="number" value="${comp.p2}" oninput="updateProp('p2', this.value)"></div>
        </div>
        <div class="prop-group" style="display:flex; gap:10px;">
            <div style="flex:1"><label>Bordo sup. e1</label><input type="number" value="${comp.e1}" oninput="updateProp('e1', this.value)"></div>
            <div style="flex:1"><label>Bordo lat. e2</label><input type="number" value="${comp.e2}" oninput="updateProp('e2', this.value)"></div>
        </div>`;
    } 
    else if(comp.type === 'weld') {
        html += `
        <div class="prop-group"><label>Tipo Saldatura</label><select onchange="updateProp('tipo', this.value)"><option value="angolo" ${comp.tipo=='angolo'?'selected':''}>Cordone d'angolo</option><option value="pen_completa" ${comp.tipo=='pen_completa'?'selected':''}>Penetrazione completa</option></select></div>
        <div class="prop-group"><label>Gola 'a' (mm)</label><input type="number" value="${comp.a}" oninput="updateProp('a', this.value)"></div>
        <div class="prop-group"><label>Lunghezza (mm)</label><input type="number" value="${comp.l}" oninput="updateProp('l', this.value)"></div>`;
    } 
    else if(comp.type === 'anchor') {
        html += `
        <div class="prop-group"><label>Classe Cls</label><select onchange="updateProp('cls', this.value)"><option value="C20/25" ${comp.cls=='C20/25'?'selected':''}>C20/25</option><option value="C25/30" ${comp.cls=='C25/30'?'selected':''}>C25/30</option></select></div>
        <div class="prop-group"><label>Diametro Barra</label><select onchange="updateProp('d', this.value)"><option value="16" ${comp.d==16?'selected':''}>M16</option><option value="20" ${comp.d==20?'selected':''}>M20</option></select></div>
        <div class="prop-group"><label>L. Ancoraggio (mm)</label><input type="number" value="${comp.L_anc}" oninput="updateProp('L_anc', this.value)"></div>`;
    }

    panel.innerHTML = html + `
    <style>
        .prop-group { margin-bottom:12px; }
        .prop-group label { display:block; font-size:11px; color:#7f8c8d; text-transform:uppercase; margin-bottom:4px; font-weight:bold;}
        .prop-group input, .prop-group select { width:100%; padding:8px; border:1px solid #ccc; border-radius:4px; font-size:14px; box-sizing:border-box; }
    </style>`;
}

function drawConstructedJoint() {
    let container = document.getElementById('jointPreviewContainer');
    if (!container) return;

    let colProf = document.getElementById('node_col_profile');
    let colConnEl = document.getElementById('node_col_conn');
    let beamProf = document.getElementById('node_beam_profile');
    
    let colName = colProf ? colProf.value : "HEB 200";
    let colConn = colConnEl ? colConnEl.value : "ala";
    let beamName = beamProf ? beamProf.value : "IPE 270";

    let svgW = 600, svgH = 500;
    let html = `<svg width="100%" height="100%" viewBox="0 0 ${svgW} ${svgH}" style="background:#fff;">`;
    
    let midX = 200; 
    let beamH = 220; 
    let scale = 0.5; 

    let isBasePlate = nodeComponents.some(c => c.type === 'anchor');
    let colBottom = isBasePlate ? 400 : svgH;
    let colTop = isBasePlate ? 50 : 0;
    let colH = colBottom - colTop;

    let targetsMap = {
        "ala_colonna": [{ x: midX, y: svgH/2 - beamH/2, view: 'side', dir: -1, w: 0 }],
        "anima_trave": [{ x: midX + 20, y: svgH/2 - beamH/2, view: 'front' }]
    };

    if (colConn === "ala") {
        html += `<rect x="${midX-60}" y="${colTop}" width="60" height="${colH}" fill="#f1f3f5" stroke="#ced4da" stroke-width="2" />`;
        html += `<text x="${midX-30}" y="${svgH/2}" font-size="14" fill="#adb5bd" text-anchor="middle" font-weight="bold" transform="rotate(-90 ${midX-30},${svgH/2})">${colName}</text>`;
    } else {
        let cW = 80; let cX = midX - cW;
        html += `<rect x="${cX}" y="${colTop}" width="${cW}" height="${colH}" fill="#e9ecef" stroke="#ced4da" stroke-width="1" />`;
        html += `<rect x="${cX}" y="${colTop}" width="15" height="${colH}" fill="#bdc3c7" stroke="#7f8c8d" stroke-width="1" />`;
        html += `<rect x="${cX + cW - 15}" y="${colTop}" width="15" height="${colH}" fill="#bdc3c7" stroke="#7f8c8d" stroke-width="1" />`;
    }

    if (!isBasePlate) {
        html += `<rect x="${midX+20}" y="${svgH/2 - beamH/2}" width="280" height="${beamH}" fill="#e9ecef" stroke="#adb5bd" stroke-width="2" />`;
    } else {
        html += `<rect x="${midX-140}" y="400" width="220" height="100" fill="#e0e0e0" stroke="#9e9e9e" stroke-width="2" />`;
    }

    let sortedComps = [...nodeComponents].sort((a, b) => {
        let order = {'gusset': 0, 'plate': 1, 'splice': 2, 'angle': 2, 'stiffener': 3, 'weld': 4, 'bolts': 5, 'anchor': 6};
        return (order[a.type] || 0) - (order[b.type] || 0);
    });

    sortedComps.forEach(comp => {
        if(comp.type === 'plate') {
            let dw = comp.t * scale, dh = comp.h * scale;
            let px = isBasePlate ? midX-60 - (dh-60)/2 : midX;
            let py = isBasePlate ? 400 - dw : svgH/2 - dh/2;
            html += `<rect x="${px}" y="${py}" width="${isBasePlate ? dh : dw}" height="${isBasePlate ? dw : dh}" fill="#95a5a6" stroke="#34495e" stroke-width="2" />`;
            targetsMap[comp.id] = [{ x: px, y: py, view: 'side', dir: -1, w: isBasePlate ? dh : dw }]; 
        }
        else if(comp.type === 'angle') {
            let ah = comp.h * scale, ab = comp.b * scale, at = comp.t * scale;
            let ay = svgH/2 - ah/2;
            html += `<rect x="${midX}" y="${ay}" width="${at}" height="${ah}" fill="#7f8c8d" stroke="#2c3e50" stroke-width="2" />`;
            html += `<rect x="${midX+at}" y="${ay}" width="${ab}" height="${ah}" fill="#95a5a6" stroke="#2c3e50" stroke-width="2" opacity="0.9"/>`;
            
            let f1 = { x: midX, y: ay, view: 'side', dir: -1, w: at };
            let f2 = { x: midX+at, y: ay, view: 'front' };
            targetsMap[comp.id + '_f1'] = [f1];
            targetsMap[comp.id + '_f2'] = [f2];
            targetsMap[comp.id + '_both'] = [f1, f2];
            targetsMap[comp.id] = [f2]; 
        }
        else if(comp.type === 'splice') {
            let sb = comp.b * scale, sh = comp.h * scale, st = comp.t * scale;
            let sx = midX + 30, sy = svgH/2 - sh/2;
            html += `<rect x="${sx}" y="${sy}" width="${sb}" height="${sh}" fill="#bdc3c7" stroke="#2c3e50" stroke-width="2" opacity="0.8"/>`;
            targetsMap[comp.id] = [{ x: sx, y: sy, view: 'front' }];
        }
    });

    sortedComps.forEach(comp => {
        if(comp.type === 'bolts') {
            let views = targetsMap[comp.target];
            if(!views) return; 

            views.forEach(t => {
                let startY = t.y + (comp.e1 * scale); 

                if (t.view === 'front') {
                    let startX = t.x + (comp.e2 * scale); 
                    for(let r=0; r<comp.rows; r++) {
                        for(let c=0; c<comp.cols; c++) {
                            let cx = startX + (c * comp.p2 * scale);
                            let cy = startY + (r * comp.p1 * scale);
                            html += `<circle cx="${cx}" cy="${cy}" r="4.5" fill="#e74c3c" stroke="#c0392b" stroke-width="1"/>`;
                            html += `<polygon points="${cx-3},${cy-1.5} ${cx},${cy-3.5} ${cx+3},${cy-1.5} ${cx+3},${cy+1.5} ${cx},${cy+3.5} ${cx-3},${cy+1.5}" fill="none" stroke="#fff" stroke-width="0.8" opacity="0.8"/>`;
                        }
                    }
                } 
                else if (t.view === 'side') {
                    for(let r=0; r<comp.rows; r++) {
                        let cy = startY + (r * comp.p1 * scale);
                        if (t.dir === -1) {
                            let headX = t.x + t.w + 4; 
                            let endX = t.x - 15; 
                            html += `<rect x="${headX-4}" y="${cy-4.5}" width="4" height="9" fill="#2c3e50" />`; 
                            html += `<line x1="${endX}" y1="${cy}" x2="${headX}" y2="${cy}" stroke="#34495e" stroke-width="3" stroke-dasharray="3,2" />`; 
                            html += `<rect x="${endX}" y="${cy-4}" width="4" height="8" fill="#34495e" />`; 
                        } else {
                            let headX = t.x - 4;
                            let endX = t.x + t.w + 15;
                            html += `<rect x="${headX}" y="${cy-4.5}" width="4" height="9" fill="#2c3e50" />`;
                            html += `<line x1="${headX}" y1="${cy}" x2="${endX}" y2="${cy}" stroke="#34495e" stroke-width="3" stroke-dasharray="3,2" />`;
                            html += `<rect x="${endX-4}" y="${cy-4}" width="4" height="8" fill="#34495e" />`;
                        }
                    }
                }
            });
        }
    });

    container.innerHTML = html + `</svg>`;
}

async function compileAndCalculate() {
    let plate = nodeComponents.find(c => c.type === 'plate');
    let bolts = nodeComponents.find(c => c.type === 'bolts');
    let stiffener = nodeComponents.find(c => c.type === 'stiffener');
    
    if(!plate || !bolts) { 
        alert("Aggiungi almeno una Piastra Frontale e la Griglia Bulloni per calcolare!"); 
        return; 
    }

    let payload = {
        joint_type: stiffener ? "stiffened_end_plate" : "end_plate", 
        tp: plate.t, 
        bp: plate.b, 
        hp: plate.h,
        bolt_d: bolts.d, 
        bolt_class: "8.8", 
        bolt_rows: bolts.rows, 
        pitch: bolts.p1,
        stiffener_t: stiffener ? stiffener.t : 0.0 
    };

    try {
        let res = await fetch('https://beam2beam.onrender.com/premium/joint-stiffness', {
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error("Dati rifiutati dal server (Errore 422 o 500)");
        let data = await res.json();
        
        document.getElementById('node_res_sj').innerHTML = data.Sj_ini.toFixed(0) + " <small>kNm/rad</small>";
        if (typeof classifyNode === "function") classifyNode();
        
    } catch(e) { 
        console.error(e);
        alert("Connessione fallita. Assicurati che il server Python sia acceso (uvicorn main:app --reload)"); 
        document.getElementById('node_res_sj').innerHTML = "15400 <small>kNm/rad</small>";
    }
}
</script>
<script type="text/javascript" src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
<div id="nodeModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 5000; justify-content: center; align-items: center;">
    <div style="background: #fff; width: 1250px; height: 850px; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 20px 50px rgba(0,0,0,0.5);">
        
        <div style="padding: 15px 25px; background: #e67e22; color: white; display: flex; justify-content: space-between; align-items: center; border-bottom: 4px solid #d35400;">
            <h2 id="nodeModalTitle" style="margin: 0; font-size: 18px; letter-spacing: 1px;">🏗️ BEAM2BEAM | STEEL JOINT BUILDER</h2>
            <button onclick="document.getElementById('nodeModal').style.display='none'" style="background:none; border:none; color:white; font-size:30px; cursor:pointer;">&times;</button>
        </div>

        <div style="display: flex; flex: 1; overflow: hidden; background: #f0f2f5;">
            
            <div style="width: 280px; padding: 20px; border-right: 1px solid #ddd; background: #fff; display: flex; flex-direction: column; gap: 15px;">
                <div>
                    <h4 style="margin: 0 0 10px 0; font-size: 12px; color: #7f8c8d; text-transform: uppercase;">1. Aste Connesse</h4>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #ddd;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; align-items:center;">
                            <label style="font-size:12px; font-weight:bold; color:#2c3e50;">Colonna:</label>
                            <input type="text" id="node_col_profile" value="HEB 200" oninput="drawConstructedJoint()" style="width:110px; padding:5px; font-size:12px; text-align:center; border:1px solid #ccc; border-radius:4px;">
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <label style="font-size:12px; font-weight:bold; color:#2c3e50;">Attacco su:</label>
                            <select id="node_col_conn" onchange="drawConstructedJoint()" style="width:110px; padding:5px; font-size:12px; border:1px solid #ccc; border-radius:4px; text-align:center;">
                                <option value="ala" selected>Ala (Forte)</option>
                                <option value="anima">Anima (Debole)</option>
                            </select>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <label style="font-size:12px; font-weight:bold; color:#2c3e50;">Trave:</label>
                            <input type="text" id="node_beam_profile" value="IPE 270" oninput="drawConstructedJoint()" style="width:110px; padding:5px; font-size:12px; text-align:center; border:1px solid #ccc; border-radius:4px;">
                        </div>
                    </div>
                </div>

                <h4 style="margin: 10px 0 0 0; font-size: 12px; color: #7f8c8d; text-transform: uppercase;">2. Componenti Giunto</h4>
                <div id="componentsList" style="flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;">
                </div>
                
                <select id="newComponentType" style="width:100%; padding:10px; border: 2px dashed #e67e22; border-radius: 6px; color:#d35400; font-weight:bold; cursor:pointer;">
                    <optgroup label="Elementi Base">
                        <option value="">+ AGGIUNGI PEZZO...</option>
                        <option value="plate">🔲 Piastra Frontale (End-plate)</option>
                        <option value="angle">📐 Squadretta di Collegamento</option>
                        <option value="splice">🔗 Coprigiunto (Splice plate)</option>
                        <option value="gusset">🧩 Fazzoletto (Gusset plate)</option>
                    </optgroup>
                    <optgroup label="Fissaggi e Rinforzi">
                        <option value="bolts">🔩 Griglia Bulloni</option>
                        <option value="weld">🔥 Cordone di Saldatura</option>
                        <option value="stiffener">🔺 Irrigidimento d'Anima</option>
                    </optgroup>
                    <optgroup label="Fondazioni">
                        <option value="anchor">⚓ Tirafondi e Calcestruzzo</option>
                    </optgroup>
                </select>

                <button onclick="compileAndCalculate()" style="width: 100%; padding: 15px; background: #27ae60; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.3s; margin-top: auto;">
                    ⚙️ CALCOLA Sj,ini
                </button>
            </div>

            <div style="width: 320px; padding: 20px; border-right: 1px solid #ddd; background: #f8f9fa;">
                <h4 style="margin: 0; font-size: 12px; color: #7f8c8d; text-transform: uppercase;">3. Proprietà Componente</h4>
                <div id="inspectorPanel" style="margin-top: 20px; display: flex; flex-direction: column; gap: 12px;">
                    <p style="font-size: 13px; color: #95a5a6; text-align: center; margin-top: 50px;">Seleziona un componente per modificarlo</p>
                </div>
            </div>

            <div style="flex: 1; padding: 20px; display: flex; flex-direction: column; gap: 15px;">
                <div id="jointPreviewContainer" style="flex: 1; background: #fff; border-radius: 8px; border: 1px solid #ddd; box-shadow: inset 0 0 10px rgba(0,0,0,0.05); display: flex; justify-content: center; align-items: center; overflow: hidden; position: relative;">
                </div>
                
                <div style="background: #2c3e50; color: white; padding: 20px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 11px; color: #bdc3c7; text-transform: uppercase;">Rigidezza Rotazionale</div>
                        <div id="node_res_sj" style="font-size: 28px; font-weight: bold; color: #f1c40f;">0 <small style="font-size:14px;">kNm/rad</small></div>
                    </div>
                    <div id="node_res_class" style="padding: 10px 20px; border-radius: 5px; font-weight: bold; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.2);">
                        Classificazione: -
                    </div>
                    <button onclick="applyStiffnessToModel()" style="padding: 12px 25px; background: #2980b9; color: white; border: none; border-radius: 4px; font-weight: bold; cursor: pointer;">
                        📥 APPLICA AL MODELLO
                    </button>
                </div>
            </div>

        </div>
    </div>
</div>
</body>
</html>
