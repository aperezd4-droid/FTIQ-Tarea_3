import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador de Conduccion Termica", layout="wide")
st.title("🧪 Simulador de coeficientes de conduccion termica del Coke Oven Gas y SoyBean Oil")


# ======================================= ===
# NAVEGACIÓN PRINCIPAL
# ==========================================
st.sidebar.title("Navegación")
seccion = st.sidebar.radio("Selecciona la Mezcla:", ["💨 Coke Oven Gas", "💧 SoyBean Oil"])
st.sidebar.markdown("---")


# =====================================================================
# 🟢 SECCIÓN 1: GASES 
# =====================================================================


if seccion == "💨 Coke Oven Gas":
    st.info("Cálculo de coeficientes de conduccion termica mediante métodos del libro de Poling y literatura externa")

    # ==========================================
    # 1. BASE DE DATOS (PROPIEDADES Y EXPERIMENTAL)
    # ==========================================
    
    # --- CONSTANTES FÍSICAS GENERALES ---
    R_GAS = 8.3144626  # Constante universal de los gases [J/(mol*K)]
    R_CAL_MOL_K = 1.987 # Específica para el Método de Svehla

    datos_componentes = {
        'Componente': ['H2', 'CH4', 'CO', 'CO2', 'N2', 'C2H4', 'C2H6', 'C3H6', 'H2S'],
        'M (g/mol)': [2.016, 16.043, 28.01, 44.009, 28.014, 28.054, 30.07, 42.08, 34.081],
        'y_i': [0.55, 0.25, 0.06, 0.03, 0.04, 0.02, 0.015, 0.005, 0.03],
        'σ (Å)': [2.915, 3.78, 3.69, 3.941, 3.798, 4.163, 4.443, 4.766, 3.623],
        'ε/k (K)': [38, 154, 91.7, 195.2, 71.4, 224.7, 215.7, 275, 301.1],
        'V_c (cm3/mol)': [65.0, 99.2, 93.2, 94.0, 89.9, 131.0, 148.0, 181.0, 98.5],
        # Datos de Chung
        # --- NUEVOS PARÁMETROS PARA CHUNG ---
        'omega': [-0.219, 0.01142, 0.0497, 0.22394, 0.0372, 0.0866, 0.0995, 0.146, 0.1005],
        'T_c (K)': [33.145, 190.564, 132.86, 304.1282, 126.192, 282.35, 305.322, 364.211, 373.1],
        # --- NUEVOS PARÁMETROS PARA GHARAGHEIZI ---
        'T_b (K)': [10.0, 111.4, 81.7, 216.58, 77.3, 169.4, 184.6, 225.5, 212.8],
        'P_c (bar)': [13.68, 45.99, 34.99, 73.83, 34.0, 50.4, 48.72, 46.65, 89.63],
        # Datos de Capacidad Calorífica (Ecuación de Shomate)
        'A_sho': [33.066178, -0.703029, 25.56759, 24.99735, 19.50583, -6.38788, 30.52, 42.18, 26.88412],
        'B_sho': [-11.363417, 108.4773, 6.09613, 55.18696, 19.88705, 184.4019, 95.87, 110.63, 18.67809],
        'C_sho': [11.432816, -42.52157, 4.054656, -33.69137, -8.598535, -112.9718, -58.43, -64.91, 3.434203],
        'D_sho': [-2.772874, 5.862788, -2.671301, 7.948387, 1.369784, 28.49593, 14.72, 15.87, -3.378702],
        'E_sho': [-0.158558, 0.678565, 0.131021, -0.136638, 0.527601, 0.31554, 2.61, 1.98, 0.135882]
    }
    df_comp = pd.DataFrame(datos_componentes)

    datos_exp = {
        'Componente': ['H2', 'CH4', 'CO', 'CO2', 'N2', 'C2H4', 'C2H6', 'C3H6', 'H2S'],
        'conductividad Exp (W/m*K)': [0.27994, 0.072859, 0.037874, 0.034761, 0.04042, 0.045411, 0.058479, 0.04598, 0.028476]
    }
    df_exp = pd.DataFrame(datos_exp)

    # ==========================================
    # 2. FUNCIONES MATEMÁTICAS
    # ==========================================

    # --- A. FUNCIÓN PARA C_p (ECUACIÓN DE SHOMATE) ---
    def calcular_cp_shomate(t_kelvin, a_sho, b_sho, c_sho, d_sho, e_sho):
        
        t_sho = t_kelvin / 1000.0
        
        # Aplicamos la ecuación
        cp_j_mol_k = a_sho + (b_sho * t_sho) + (c_sho * (t_sho**2)) + \
                     (d_sho * (t_sho**3)) + (e_sho / (t_sho**2))
        return cp_j_mol_k


    # --- B. MÉTODO DE EUCKEN MODIFICADO (PURAS) ---
    def metodo_eucken_conduccion_pura(m_g_mol, eta_pa_s, cp_j_mol_k):
        # 1. Calculamos Cv
        cv_j_mol_k = cp_j_mol_k - R_GAS
        
        # 2. Convertimos el Peso Molecular a kg/mol
        m_kg_mol = m_g_mol / 1000.0
        
        # 3. factor_eucken = 1.32 + (1.77 / (Cv/R))
        factor_eucken = 1.32 + (1.77 / ((cp_j_mol_k / R_GAS) - 1))
        
        # 4. Cálculo final de λ
        lambda_w_m_k = (factor_eucken * eta_pa_s * cv_j_mol_k) / m_kg_mol
        
        return lambda_w_m_k
    
    
    def calcular_viscosidad_chapman_enskog(T, M, sigma, epsilon_k):
        # 1. Temperatura reducida
        T_star = T / epsilon_k
        
        # 2. Integral de colisión 
        omega_v = (1.16145 / (T_star**0.14874)) + \
                  (0.52487 / math.exp(0.77320 * T_star)) + \
                  (2.16178 / math.exp(2.43787 * T_star))
        
        # 3. Viscosidad en Pa*s (la constante 1e-7 es para convertir de micropoise)
        mu_pas = (26.69 * math.sqrt(M * T) / (sigma**2 * omega_v)) * 1e-7
        return mu_pas
    
    
    # --- C. MÉTODO DE CHUNG (PURAS) ---
    def metodo_chung_conduccion_pura(m_g_mol, eta_pa_s, cp_j_mol_k, omega, t_kelvin, t_c_kelvin):
        # 1. Capacidad calorífica a volumen constante
        cv_j_mol_k = cp_j_mol_k - R_GAS
        
        # 2. Parámetros intermedios de Chung
        alpha = (cv_j_mol_k / R_GAS) - 1.5
        beta = 0.7862 - (0.7109 * omega) + (1.3168 * (omega ** 2))
        
        t_r = t_kelvin / t_c_kelvin
        z = 2.0 + (10.5 * (t_r ** 2))
        
        # 3. Factor de corrección Psi (Ψ)
        num_psi = 0.215 + (0.28288 * alpha) - (1.061 * beta) + (0.26665 * z)
        den_psi = 0.6366 + (beta * z) + (1.061 * alpha * beta)
        psi = 1.0 + (alpha * (num_psi / den_psi))
        
        # 4. Despeje de la conductividad térmica (λ)
        m_kg_mol = m_g_mol / 1000.0
        # λM / (ηCv) = 3.75 * Ψ / (Cv/R) -> λ = (3.75 * Ψ * η * R) / M
        lambda_w_m_k = (3.75 * psi * eta_pa_s * R_GAS) / m_kg_mol
        
        return lambda_w_m_k, alpha, beta, t_r, z, psi


    # --- D. MÉTODO DE GHARAGHEIZI (PURAS) ---
    def metodo_gharagheizi_conduccion_pura(m_g_mol, t_kelvin, t_b_kelvin, p_c_bar, omega):
        # 1. Término común (denominador de la fracción en B)
        term_omega = 2.0 * omega
        term_factor = term_omega + 3.2825
        term_frac = (term_factor * t_kelvin) / t_b_kelvin
        
        den_b = term_omega + t_kelvin - term_frac + 3.2825
        
        # 2. Cálculo del parámetro B
        num_frac_b = term_omega + (2.0 * t_kelvin) - ((2.0 * t_kelvin * term_factor) / t_b_kelvin) + 3.2825
        b_val = t_kelvin + (num_frac_b / den_b) - term_frac
        
        # 3. Cálculo del parámetro A (usando P_c en bar directamente según el criterio del documento)
        factor_a1 = den_b / (m_g_mol * p_c_bar * t_kelvin)
        factor_a2 = (3.9752 * omega) + p_c_bar + (1.9876 * b_val) + 6.5243
        a_val = factor_a1 * (factor_a2 ** 2)
        
        # 4. Cálculo de la conductividad térmica λ (Alineado con la celda final de tu Excel)
        lambda_w_m_k = 7.9505e-4 + (3.989e-5 * t_kelvin) - (5.410e-5 * m_g_mol) + (3.989e-5 * a_val)
        
        return lambda_w_m_k, a_val, b_val


    # --- E. MÉTODO DE SVEHLA (PURAS) ---
    def metodo_svehla_conduccion_pura(m_g_mol, cp_j_mol_k, viscosidad_pa_s):
        # 1. Conversión de unidades para cumplir con la estructura del modelo
        # Viscosidad: de Pa·s a g/(cm·s) -> multiplicar por 10
        eta_g_cm_s = viscosidad_pa_s * 10 
        
        # Cp/R: Adimensional (funciona igual en J o Cal)
        cp_r_ratio = cp_j_mol_k / 8.314
        
        # 2. Aplicación de la fórmula de Svehla
        # lambda * 10^6 = (R/M) * [15/4 + 1.32 * (Cp/R - 5/2)] * (eta * 10^6)
        # Nota: Al cancelar los 10^6 de ambos lados, la fórmula queda:
        termino_parentesis = (15/4) + 1.32 * (cp_r_ratio - 2.5)
        lambda_cal_cm_s_k = (1.987 / m_g_mol) * termino_parentesis * eta_g_cm_s
        
        # 3. Conversión final a W/(m·K) para la tabla comparativa
        # 1 cal/(cm·s·K) = 418.68 W/(m·K)
        lambda_w_m_k = lambda_cal_cm_s_k * 418.68
        
        return lambda_w_m_k, cp_r_ratio, eta_g_cm_s, lambda_cal_cm_s_k


    # 1. Entrada de temperatura por el usuario
    temp_k = st.number_input("Temperatura de análisis (K)", value=523.15)


    # 2. Definición de Pestañas
    tab_eucken, tab_chung, tab_gharagheizi, tab_svehla, tab_wassiljewa, tab_burgoyne = st.tabs([
        "1. Eucken Modificado", 
        "2. Chung", 
        "3. Gharagheizi", 
        "4. Svehla", 
        "5. Regla: Wassiljewa (M&S)", 
        "6. Regla: Burgoyne & Weinberg"
    ])


    # ==========================================
    # PESTAÑA 1: EUCKEN MODIFICADO
    # ==========================================
    with tab_eucken:
        st.subheader("Método de Eucken Modificado (Modelo 1)")
        
        parametros_lista = []
        comparacion_lista = []

        for i, row in df_comp.iterrows():
            # A. Cálculos base
            cp_val = calcular_cp_shomate(temp_k, row['A_sho'], row['B_sho'], row['C_sho'], row['D_sho'], row['E_sho'])
            eta_val = calcular_viscosidad_chapman_enskog(temp_k, row['M (g/mol)'], row['σ (Å)'], row['ε/k (K)'])
            lambda_calc = metodo_eucken_conduccion_pura(row['M (g/mol)'], eta_val, cp_val)
            
            # B. Extraer valor experimental y calcular error
            valor_exp = df_exp.loc[df_exp['Componente'] == row['Componente'], 'conductividad Exp (W/m*K)'].values[0]
            error_rel = abs((valor_exp - lambda_calc) / valor_exp) * 100
            
            # C. Guardar resultados
            parametros_lista.append({
                "Componente": row['Componente'],
                "M (g/mol)": row['M (g/mol)'],
                "Cp (J/mol·K)": round(cp_val, 3),
                "σ (Å)": row['σ (Å)'],
                "ε/k (K)": row['ε/k (K)'],
                "Viscosidad (Pa·s)": f"{eta_val:.5E}"
            })
            
            comparacion_lista.append({
                "Componente": row['Componente'],
                "λ Exp (W/m·K)": valor_exp,
                "λ Calc (W/m·K)": round(lambda_calc, 5),
                "% Error": f"{error_rel:.2f}%"
            })

        # --- CÁLCULO DE MÉTRICAS GLOBALES CON NUMPY ---
        y_exp = np.array([item['λ Exp (W/m·K)'] for item in comparacion_lista])
        y_calc = np.array([item['λ Calc (W/m·K)'] for item in comparacion_lista])
        
        ss_res = np.sum((y_exp - y_calc) ** 2)
        ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
        r2_global = 1.0 - (ss_res / ss_tot)
        error_global_mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100

        # --- DISTRIBUCIÓN ESPACIAL EN 2 COLUMNAS ---
        col_tablas, col_grafica = st.columns([1.1, 0.9])
        
        with col_tablas:
            st.write("##### 📁 1. Parámetros de entrada y Viscosidad")
            # Usamos dataframe con altura fija para economizar espacio vertical
            st.dataframe(pd.DataFrame(parametros_lista), use_container_width=True, height=190)
            
            st.write("##### 📊 2. Comparación y Error Relativo a 523,15 K")
            st.dataframe(pd.DataFrame(comparacion_lista), use_container_width=True, height=190)

        with col_grafica:
            # Métricas compactas en la parte superior derecha
            st.write("##### 🎯 Métricas Globales")
            met1, met2 = st.columns(2)
            met1.metric("R² Global", f"{r2_global:.4f}")
            met2.metric("Error Global", f"{error_global_mape:.2f}%")
            
            # --- GRÁFICA MINIMALISTA ---
            fig, ax = plt.subplots(figsize=(5.5, 4.0)) # Tamaño compacto
            
            # Puntos de dispersión limpios (etiqueta actualizada)
            ax.scatter(y_exp, y_calc, color='#eb5e28', edgecolors='#252422', s=50, zorder=3, label='Predicción M1')
            
            # Línea de tendencia (Paridad)
            lim_min = min(min(y_exp), min(y_calc)) * 0.85
            lim_max = max(max(y_exp), max(y_calc)) * 1.15
            ax.plot([lim_min, lim_max], [lim_min, lim_max], color='#252422', linestyle='--', linewidth=1.2, zorder=2, label='Ajuste perfecto')
            
            # Configuración estética sin cuadrícula
            ax.set_title("Dispersión del Modelo", fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel("λ Experimental (W/m·K)", fontsize=9)
            ax.set_ylabel("λ Calculada (W/m·K)", fontsize=9)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.tick_params(labelsize=8)
            
            # Quitar recuadro superior y derecho para dar estilo moderno
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#888888')
            ax.spines['bottom'].set_color('#888888')
            
            ax.legend(loc='upper left', fontsize=8, frameon=False)
            
            # Renderizar gráfica
            st.pyplot(fig)


    # ==========================================
    # PESTAÑA 2: MÉTODO DE CHUNG
    # ==========================================
    with tab_chung:
        st.subheader("Método de Chung (Modelo 2)")
        
        parametros_chung = []
        comparacion_chung = []

        for i, row in df_comp.iterrows():
            # A. Cálculos base
            cp_val = calcular_cp_shomate(temp_k, row['A_sho'], row['B_sho'], row['C_sho'], row['D_sho'], row['E_sho'])
            eta_val = calcular_viscosidad_chapman_enskog(temp_k, row['M (g/mol)'], row['σ (Å)'], row['ε/k (K)'])
            
            # B. Llamada al método de Chung
            lambda_calc, alpha, beta, tr, z, psi = metodo_chung_conduccion_pura(
                row['M (g/mol)'], eta_val, cp_val, row['omega'], temp_k, row['T_c (K)']
            )
            
            # C. Extraer valor experimental y calcular error
            valor_exp = df_exp.loc[df_exp['Componente'] == row['Componente'], 'conductividad Exp (W/m*K)'].values[0]
            error_rel = abs((valor_exp - lambda_calc) / valor_exp) * 100
            
            # D. Guardar resultados para las tablas
            parametros_chung.append({
                "Componente": row['Componente'],
                "ω": row['omega'],
                "Tr": round(tr, 4),
                "α": round(alpha, 4),
                "β": round(beta, 4),
                "Z": round(z, 2),
                "Ψ": round(psi, 4)
            })
            
            comparacion_chung.append({
                "Componente": row['Componente'],
                "λ Exp (W/m·K)": valor_exp,
                "λ Calc (W/m·K)": round(lambda_calc, 5),
                "% Error": f"{error_rel:.2f}%"
            })

        # --- CÁLCULO DE MÉTRICAS GLOBALES CON NUMPY ---
        y_exp_ch = np.array([item['λ Exp (W/m·K)'] for item in comparacion_chung])
        y_calc_ch = np.array([item['λ Calc (W/m·K)'] for item in comparacion_chung])
        
        ss_res_ch = np.sum((y_exp_ch - y_calc_ch) ** 2)
        ss_tot_ch = np.sum((y_exp_ch - np.mean(y_exp_ch)) ** 2)
        r2_global_ch = 1.0 - (ss_res_ch / ss_tot_ch)
        error_global_mape_ch = np.mean(np.abs((y_exp_ch - y_calc_ch) / y_exp_ch)) * 100

        # --- DISTRIBUCIÓN ESPACIAL EN 2 COLUMNAS ---
        col_tablas_ch, col_grafica_ch = st.columns([1.1, 0.9])
        
        with col_tablas_ch:
            st.write("##### 📁 1. Parámetros de la Ecuación de Chung")
            st.dataframe(pd.DataFrame(parametros_chung), use_container_width=True, height=190)
            
            st.write("##### 📊 2. Comparación y Error Relativo a 523,15 K")
            st.dataframe(pd.DataFrame(comparacion_chung), use_container_width=True, height=190)

        with col_grafica_ch:
            st.write("##### 🎯 Métricas Globales")
            met1_ch, met2_ch = st.columns(2)
            met1_ch.metric("R² Global", f"{r2_global_ch:.4f}")
            met2_ch.metric("Error Global", f"{error_global_mape_ch:.2f}%")
            
            # --- GRÁFICA MINIMALISTA ---
            fig_ch, ax_ch = plt.subplots(figsize=(5.5, 4.0))
            
            # Puntos de dispersión (etiqueta actualizada a Predicción M2)
            ax_ch.scatter(y_exp_ch, y_calc_ch, color='#eb5e28', edgecolors='#252422', s=50, zorder=3, label='Predicción M2')
            
            # Línea de paridad
            lim_min_ch = min(min(y_exp_ch), min(y_calc_ch)) * 0.85
            lim_max_ch = max(max(y_exp_ch), max(y_calc_ch)) * 1.15
            ax_ch.plot([lim_min_ch, lim_max_ch], [lim_min_ch, lim_max_ch], color='#252422', linestyle='--', linewidth=1.2, zorder=2, label='Ajuste perfecto')
            
            # Configuración estética
            ax_ch.set_title("Dispersión del Modelo", fontsize=11, fontweight='bold', pad=10)
            ax_ch.set_xlabel("λ Experimental (W/m·K)", fontsize=9)
            ax_ch.set_ylabel("λ Calculada (W/m·K)", fontsize=9)
            ax_ch.set_xlim(lim_min_ch, lim_max_ch)
            ax_ch.set_ylim(lim_min_ch, lim_max_ch)
            ax_ch.tick_params(labelsize=8)
            
            ax_ch.spines['top'].set_visible(False)
            ax_ch.spines['right'].set_visible(False)
            ax_ch.spines['left'].set_color('#888888')
            ax_ch.spines['bottom'].set_color('#888888')
            
            ax_ch.legend(loc='upper left', fontsize=8, frameon=False)
            
            st.pyplot(fig_ch)


    # ==========================================
    # PESTAÑA 3: MÉTODO DE GHARAGHEIZI
    # ==========================================
    with tab_gharagheizi:
        st.subheader("Método de Gharagheizi (Modelo 3)")
        
        parametros_gh = []
        comparacion_gh = []

        for i, row in df_comp.iterrows():
            # Regla especial: Para el H2 en este método, el Excel asume omega = 0 al estar la celda vacía
            omega_gh = 0.0 if row['Componente'] == 'H2' else row['omega']
            
            # Llamada al método
            lambda_calc, a_val, b_val = metodo_gharagheizi_conduccion_pura(
                row['M (g/mol)'], temp_k, row['T_b (K)'], row['P_c (bar)'], omega_gh
            )
            
            # Extraer experimental y error
            valor_exp = df_exp.loc[df_exp['Componente'] == row['Componente'], 'conductividad Exp (W/m*K)'].values[0]
            error_rel = abs((valor_exp - lambda_calc) / valor_exp) * 100
            
            # Guardar en tablas
            parametros_gh.append({
                "Componente": row['Componente'],
                "T_b (K)": row['T_b (K)'],
                "P_c (bar)": row['P_c (bar)'],
                "ω aplicado": omega_gh,
                "Parámetro B": round(b_val, 4),
                "Parámetro A": round(a_val, 4)
            })
            
            comparacion_gh.append({
                "Componente": row['Componente'],
                "λ Exp (W/m·K)": valor_exp,
                "λ Calc (W/m·K)": round(lambda_calc, 5),
                "% Error": f"{error_rel:.2f}%"
            })

        # --- CÁLCULO DE MÉTRICAS GLOBALES CON NUMPY ---
        y_exp_gh = np.array([item['λ Exp (W/m·K)'] for item in comparacion_gh])
        y_calc_gh = np.array([item['λ Calc (W/m·K)'] for item in comparacion_gh])
        
        ss_res_gh = np.sum((y_exp_gh - y_calc_gh) ** 2)
        ss_tot_gh = np.sum((y_exp_gh - np.mean(y_exp_gh)) ** 2)
        r2_global_gh = 1.0 - (ss_res_gh / ss_tot_gh)
        error_global_mape_gh = np.mean(np.abs((y_exp_gh - y_calc_gh) / y_exp_gh)) * 100

        # --- DISTRIBUCIÓN ESPACIAL EN 2 COLUMNAS ---
        col_tablas_gh, col_grafica_gh = st.columns([1.1, 0.9])
        
        with col_tablas_gh:
            st.write("##### 📁 1. Parámetros de la Ecuación de Gharagheizi")
            st.dataframe(pd.DataFrame(parametros_gh), use_container_width=True, height=190)
            
            st.write("##### 📊 2. Comparación y Error Relativo")
            st.dataframe(pd.DataFrame(comparacion_gh), use_container_width=True, height=190)

        with col_grafica_gh:
            st.write("##### 🎯 Métricas Globales")
            met1_gh, met2_gh = st.columns(2)
            met1_gh.metric("R² Global", f"{r2_global_gh:.4f}")
            met2_gh.metric("Error Global", f"{error_global_mape_gh:.2f}%")
            
            # --- GRÁFICA MINIMALISTA ---
            fig_gh, ax_gh = plt.subplots(figsize=(5.5, 4.0))
            
            # Puntos de dispersión (etiqueta actualizada a Predicción M3)
            ax_gh.scatter(y_exp_gh, y_calc_gh, color='#eb5e28', edgecolors='#252422', s=50, zorder=3, label='Predicción M3')
            
            # Línea de paridad
            lim_min_gh = min(min(y_exp_gh), min(y_calc_gh)) * 0.85
            lim_max_gh = max(max(y_exp_gh), max(y_calc_gh)) * 1.15
            ax_gh.plot([lim_min_gh, lim_max_gh], [lim_min_gh, lim_max_gh], color='#252422', linestyle='--', linewidth=1.2, zorder=2, label='Ajuste perfecto')
            
            # Configuración estética
            ax_gh.set_title("Dispersión del Modelo", fontsize=11, fontweight='bold', pad=10)
            ax_gh.set_xlabel("λ Experimental (W/m·K)", fontsize=9)
            ax_gh.set_ylabel("λ Calculada (W/m·K)", fontsize=9)
            ax_gh.set_xlim(lim_min_gh, lim_max_gh)
            ax_gh.set_ylim(lim_min_gh, lim_max_gh)
            ax_gh.tick_params(labelsize=8)
            
            ax_gh.spines['top'].set_visible(False)
            ax_gh.spines['right'].set_visible(False)
            ax_gh.spines['left'].set_color('#888888')
            ax_gh.spines['bottom'].set_color('#888888')
            
            ax_gh.legend(loc='upper left', fontsize=8, frameon=False)
            
            st.pyplot(fig_gh)


    # =====================================================================
    # PESTAÑA 4: MÉTODO DE SVEHLA 
    # =====================================================================
    with tab_svehla:
        st.subheader("Método de Svehla (Modelo 4)")
        st.markdown("""
        Estimación de la conductividad térmica basada en la relación directa entre la viscosidad y la capacidad calorífica utilizando unidades de ingeniería estándar.
        """)

        # 1. Definir la función matemática pura de Svehla
        def calcular_svehla(m_g_mol, cp_j_mol_k, eta_pa_s):
            R_cal = 1.987
            
            # Conversiones a unidades de ingeniería
            cp_cal = cp_j_mol_k / 4.184
            eta_poise = eta_pa_s * 10.0  # Pa·s a g/(cm·s)
            
            # Relación Cp/R
            cp_r = cp_cal / R_cal
            
            # Cálculo del corchete
            corchete = (15.0 / 4.0) + 1.32 * (cp_r - 2.5)
            
            # Conductividad en cal/(cm·s·K)
            lambda_cal = (R_cal / m_g_mol) * corchete * eta_poise
            
            # Conversión a W/(m·K) (1 cal/cm·s·K = 418.4 W/m·K)
            lambda_w_m_k = lambda_cal * 418.4
            
            return lambda_w_m_k, cp_cal, eta_poise, lambda_cal, cp_r

        # Listas para almacenar los datos de las tablas
        datos_intermedios = []
        datos_resultados = []

        # Búsqueda robusta del nombre de la columna de Componentes
        col_comp = next((col for col in df_comp.columns if 'comp' in col.lower()), df_comp.columns[0])
        col_comp_exp = next((col for col in df_exp.columns if 'comp' in col.lower()), df_exp.columns[0])

        # Iterar sobre cada componente cargado
        for idx, row in df_comp.iterrows():
            nombre = row[col_comp]
            
            # --- A. Obtener Masa Molar (M) en g/mol ---
            m_val = None
            for col in df_comp.columns:
                if 'm (' in col.lower() or 'peso' in col.lower():
                    m_val = row[col]
                    # Si está en Kg/mol, pasar a g/mol
                    if 'kg' in col.lower():
                        m_val *= 1000.0
                    break
            if m_val is None:
                m_val = 28.0  # Valor de seguridad por defecto
                
            # --- B. Obtener Temperatura (T) ---
            t_val = 523.15
            for col in df_comp.columns:
                if 't_ref' in col.lower() or 't (' in col.lower() or 'temp' in col.lower():
                    t_val = row[col]
                    break

            # --- C. Obtener Cp y Viscosidad ---
            # Buscamos si existe la columna en el mismo archivo base
            cp_val = next((row[c] for c in df_comp.columns if 'cp' in c.lower()), None)
            eta_val = next((row[c] for c in df_comp.columns if 'visc' in c.lower() or 'η' in c.lower()), None)

            # Si no están en df_comp, los calculamos al vuelo usando Chapman-Enskog para garantizar autosuficiencia
            if eta_val is None:
                # Extraer sigma y eps/k si existen
                col_sigma = next((c for c in df_comp.columns if 'σ' in c or 'sigma' in c.lower()), None)
                col_eps = next((c for c in df_comp.columns if 'ε' in c or 'k' in c.lower() and '/' in c), None)
                
                if col_sigma and col_eps:
                    sigma = row[col_sigma]
                    eps_k = row[col_eps]
                    T_star = t_val / eps_k
                    Omega_v = (1.16145 / (T_star**0.14874)) + (0.52487 / np.exp(0.77320 * T_star)) + (2.16178 / np.exp(2.43787 * T_star))
                    eta_val = (26.69e-7 * np.sqrt(m_val * t_val)) / ((sigma**2) * Omega_v)
                else:
                    eta_val = 1.3e-5 # Valor de resguardo

            if cp_val is None:
                # Valores estándar de tu tabla para 523.15 K en J/mol·K si no se provee columna Cp
                cps_map = {'H2': 29.274, 'CH4': 47.728, 'CO': 29.819, 'CO2': 45.286, 'N2': 29.681, 'C2H4': 57.871, 'C2H6': 76.327, 'C3H6': 91.798, 'H2S': 37.608}
                cp_val = cps_map.get(nombre.strip(), 35.0)

            # --- D. Ejecutar Cálculos ---
            l_calc, cp_cal, eta_poise, l_cal, cp_r = calcular_svehla(m_val, cp_val, eta_val)

            # --- E. Obtener Valor Experimental ---
            fila_exp = df_exp[df_exp[col_comp_exp].astype(str).str.strip() == str(nombre).strip()]
            if not fila_exp.empty:
                col_exp_val = next((c for c in df_exp.columns if 'exp' in c.lower() or 'w/m' in c.lower()), df_exp.columns[-1])
                val_exp = fila_exp[col_exp_val].values[0]
                error = abs((val_exp - l_calc) / val_exp) * 100.0 if val_exp > 0 else 0.0
            else:
                val_exp = 0.0
                error = 0.0

            # Almacenar filas formateadas
            datos_intermedios.append({
                "Componente": nombre,
                "M (g/mol)": round(m_val, 3),
                "T (K)": t_val,
                "Cp (Cal/mol·K)": round(cp_cal, 4),
                "η (g/cm·s)": f"{eta_poise:.2E}",
                "λ (cal/cm·s·K)": f"{l_cal:.2E}",
                "Cp/R": round(cp_r, 4)
            })

            datos_resultados.append({
                "Componente": nombre,
                "λ Exp (W/m·K)": val_exp,
                "λ Calc (W/m·K)": round(l_calc, 5),
                "% Error": f"{error:.2f}%"
            })

        # 2. Renderizado de Interfaz
        col1, col2 = st.columns([1.1, 0.9])
        
        with col1:
            st.write("##### 📁 Datos Iniciales y Cálculos (Svehla)")
            st.dataframe(pd.DataFrame(datos_intermedios), use_container_width=True)
            
            st.write("##### 📊 Comparación Final")
            st.dataframe(pd.DataFrame(datos_resultados), use_container_width=True)

        with col2:
            st.write("##### 🎯 Desempeño del Modelo")
            y_exp = np.array([r['λ Exp (W/m·K)'] for r in datos_resultados if r['λ Exp (W/m·K)'] > 0])
            y_calc = np.array([r['λ Calc (W/m·K)'] for r in datos_resultados if r['λ Exp (W/m·K)'] > 0])
            
            if len(y_exp) > 0:
                r2 = 1.0 - (np.sum((y_exp - y_calc)**2) / np.sum((y_exp - np.mean(y_exp))**2))
                mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100.0
                
                m1, m2 = st.columns(2)
                m1.metric("R² Svehla", f"{r2:.4f}")
                m2.metric("Error Promedio", f"{mape:.2f}%")
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5.5, 4.0))
                ax.scatter(y_exp, y_calc, color='#8ab17d', edgecolors='#2b4c3f', s=65, zorder=3, label='Svehla (M4)')
                
                rango = [min(min(y_exp), min(y_calc)) * 0.95, max(max(y_exp), max(y_calc)) * 1.05]
                ax.plot(rango, rango, color='#2b4c3f', linestyle='--', alpha=0.6, label='Ajuste Perfecto')
                
                ax.set_title("Svehla: Calculado vs Experimental", fontsize=10, fontweight='bold')
                ax.set_xlabel("Conductividad Experimental")
                ax.set_ylabel("Conductividad Calculada")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(fontsize=8)
                st.pyplot(fig)


    # =====================================================================
    # PESTAÑA 5: REGLA DE MEZCLADO (WASSILJEWA / MASON-SAXENA)
    # =====================================================================

    with tab_wassiljewa:
        st.header("🔗 Regla de Mezclado: Wassiljewa & Mason-Saxena")
        
        # --- 1. SECCIÓN DE FÓRMULAS ---
        with st.expander("📖 Ver Fundamento Teórico (Mason & Saxena)", expanded=False):
            st.latex(r"\lambda_m = \sum_{i=1}^n \frac{y_i \lambda_i}{\sum_{j=1}^n y_j A_{ij}}")
            st.latex(r"A_{ij} = \frac{\epsilon [1 + (\eta_i/\eta_j)^{1/2} (M_j/M_i)^{1/4}]^2}{\sqrt{8(1 + M_i/M_j)}}")
            st.info("Donde ε = 1.065 (Mason-Saxena) y η_m se calcula mediante la regla de Wilke (ε=1).")

        # --- 2. CONTROLES DE INTERFAZ (CORRECCIÓN 3: Slider de temperatura) ---
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            temp_k = st.slider("🌡️ Selecciona la temperatura de análisis (K):", 200, 1000, 523)
        with col_ctrl2:
            metodo_tablas = st.selectbox(
                "📌 Selecciona modelo para desglose de tablas:",
                ["Eucken Modificado", "Chung", "Gharagheizi", "Svehla"]
            )

        # --- 3. MOTOR DE CÁLCULO DE PROPIEDADES PURAS (CORRECCIÓN 2: Svehla unpacking) ---
        def obtener_propiedades_puras(temp, metodo):
            lambdas = []
            viscosidades = []
            for _, r in df_comp.iterrows():
                cp = calcular_cp_shomate(temp, r['A_sho'], r['B_sho'], r['C_sho'], r['D_sho'], r['E_sho'])
                eta = calcular_viscosidad_chapman_enskog(temp, r['M (g/mol)'], r['σ (Å)'], r['ε/k (K)'])
                
                if metodo == "Eucken Modificado":
                    l = metodo_eucken_conduccion_pura(r['M (g/mol)'], eta, cp)
                elif metodo == "Chung":
                    # Chung devuelve varios parámetros, tomamos el primero
                    l = metodo_chung_conduccion_pura(r['M (g/mol)'], eta, cp, r['omega'], temp, r['T_c (K)'])[0]
                elif metodo == "Gharagheizi":
                    om = 0.0 if r['Componente'] == 'H2' else r['omega']
                    l = metodo_gharagheizi_conduccion_pura(r['M (g/mol)'], temp, r['T_b (K)'], r['P_c (bar)'], om)[0]
                else: # Svehla: CORREGIDO a 4 variables de retorno
                    l, _, _, _ = metodo_svehla_conduccion_pura(r['M (g/mol)'], cp, eta)
                
                lambdas.append(l)
                viscosidades.append(eta)
            return np.array(lambdas), np.array(viscosidades)

        # --- 4. CÁLCULO DE LA MEZCLA ---
        def calcular_mezcla(lambdas, etas, eps=1.065):
            n = len(df_comp)
            M = df_comp['M (g/mol)'].values
            y = df_comp['y_i'].values
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    num = (1 + (etas[i]/etas[j])**0.5 * (M[j]/M[i])**0.25)**2
                    den = np.sqrt(8 * (1 + M[i]/M[j]))
                    A[i, j] = eps * (num / den)
            denominadores = np.dot(A, y)
            lambda_mix = np.sum((y * lambdas) / denominadores)
            Phi = A / eps
            eta_mix = np.sum((y * etas) / np.dot(Phi, y))
            return lambda_mix, eta_mix, A

        # Cálculos iniciales
        l_puras, e_puras = obtener_propiedades_puras(temp_k, metodo_tablas)
        l_mezcla, e_mezcla, matriz_A = calcular_mezcla(l_puras, e_puras)

        # --- 5. RESULTADOS Y TABLAS ---
        st.write("---")
        m1, m2 = st.columns(2)
        m1.metric(f"λ Mezcla ({metodo_tablas})", f"{l_mezcla:.5f} W/m·K")
        m2.metric("Viscosidad Mezcla (η mix)", f"{e_mezcla:.2E} Pa·s")

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.write("##### 📑 Datos de Puros y Aportes")
            df_aportes = pd.DataFrame({
                "Componente": df_comp['Componente'],
                "y_i": df_comp['y_i'],
                "λ puro": np.round(l_puras, 5),
                "η puro": [f"{v:.2E}" for v in e_puras]
            })
            st.dataframe(df_aportes, height=220, use_container_width=True)

        with col_t2:
            st.write("##### 🧮 Matriz de Interacción (A_ij)")
            df_A = pd.DataFrame(matriz_A, index=df_comp['Componente'], columns=df_comp['Componente'])
            st.dataframe(df_A.style.format("{:.3f}"), height=220, use_container_width=True)

        # --- 6. GRÁFICA COMPARATIVA GLOBAL (CORREGIDA) ---
        st.write("---")
        st.write("##### 📈 Comparativa de Modelos: λ Mezcla vs Temperatura")
        
        rango_T = np.linspace(300, 1000, 15)
        nombres_modelos = ["Eucken Modificado", "Chung", "Gharagheizi", "Svehla"]
        
        fig_mix, ax_mix = plt.subplots(figsize=(9, 4.5))
        colores = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
        # Usamos estilos de línea diferentes para que el solapamiento sea visible
        estilos = ['-', '--', '-.', ':'] 
        
        for i, mod in enumerate(nombres_modelos):
            y_lambda_mix = []
            for T_plot in rango_T:
                # Calculamos propiedades puras para cada temperatura en el loop
                # Nota: Asegúrate que obtener_propiedades_puras ya tenga el fix de Svehla (4 variables)
                lp, ep = obtener_propiedades_puras(T_plot, mod)
                lm, _, _ = calcular_mezcla(lp, ep)
                y_lambda_mix.append(lm)
                
            ax_mix.plot(rango_T, y_lambda_mix, label=mod, color=colores[i], 
                        linestyle=estilos[i], linewidth=2.5, marker='o', markersize=4, alpha=0.9)
        
        # Resaltar el Punto actual seleccionado en el slider
        ax_mix.scatter(temp_k, l_mezcla, color='red', s=150, zorder=10, 
                    label=f'Punto Actual ({temp_k}K)', edgecolors='white')
        
        ax_mix.set_xlabel("Temperatura (K)", fontsize=10)
        ax_mix.set_ylabel("Conductividad Térmica Mezcla (W/m·K)", fontsize=10)
        ax_mix.set_title("Evolución de Conductividad por Modelo", fontsize=12, fontweight='bold')
        ax_mix.legend(loc='upper left', fontsize=9)
        ax_mix.grid(True, linestyle='--', alpha=0.4)
        
        st.pyplot(fig_mix)


    # =====================================================================
    # PESTAÑA 6: REGLA DE MEZCLADO (BURGOYNE Y WEINBERG)
    # =====================================================================

    with tab_burgoyne:
        st.header("⚖️ Regla de Mezclado: Burgoyne y Weinberg")
        
        # --- 1. SECCIÓN DE FÓRMULAS ---
        with st.expander("📖 Ver Fundamento Teórico (Burgoyne & Weinberg)", expanded=False):
            st.write("Este modelo utiliza un enfoque de promedio combinatorio (lineal y armónico) para equilibrar los límites superior e inferior de la conductividad.")
            st.latex(r"\lambda_{mix} = 0.5 \left[ \sum_{i=1}^n x_i \lambda_i + \left( \sum_{i=1}^n \frac{x_i}{\lambda_i} \right)^{-1} \right]")
            st.info("Donde x_i es la fracción molar y λ_i es la conductividad del componente puro.")

        # --- 2. CONTROLES DE INTERFAZ ---
        col_bw1, col_bw2 = st.columns(2)
        with col_bw1:
            temp_bw = st.slider("🌡️ Temperatura de análisis (K) - B&W:", 200, 1000, 523, key="temp_bw")
        with col_bw2:
            metodo_bw = st.selectbox(
                "📌 Modelo para propiedades puras:",
                ["Eucken Modificado", "Chung", "Gharagheizi", "Svehla"],
                key="metodo_bw"
            )

        # --- 3. MOTOR DE CÁLCULO ESPECÍFICO B&W ---
        def calcular_mezcla_burgoyne(lambdas, fracciones):
            # Término Lineal: Sum(xi * li)
            termino_lineal = np.sum(fracciones * lambdas)
            
            # Término Armónico: 1 / Sum(xi / li)
            termino_armonico = 1.0 / np.sum(fracciones / lambdas)
            
            # Promedio final
            lambda_mix_bw = 0.5 * (termino_lineal + termino_armonico)
            
            return lambda_mix_bw, termino_lineal, termino_armonico

        # Obtener lambdas puras y calcular
        fracciones = df_comp['y_i'].values
        l_puras_bw, _ = obtener_propiedades_puras(temp_bw, metodo_bw)
        l_mix_bw, t_lin, t_arm = calcular_mezcla_burgoyne(l_puras_bw, fracciones)

        # --- 4. RESULTADOS Y MÉTRICAS ---
        st.write("---")
        res1, res2, res3 = st.columns(3)
        res1.metric(f"λ Mezcla (B&W)", f"{l_mix_bw:.5f} W/m·K")
        res2.metric("Promedio Lineal", f"{t_lin:.5f}")
        res3.metric("Promedio Armónico", f"{t_arm:.5f}")

        # --- 5. TABLA DE DESGLOSE ---
        st.write("##### 📑 Desglose de contribuciones (xi·λi y xi/λi)")
        df_bw_calc = pd.DataFrame({
            "Componente": df_comp['Componente'],
            "y_i (x_i)": fracciones,
            "λ puro": np.round(l_puras_bw, 5),
            "xi * λi": np.round(fracciones * l_puras_bw, 6),
            "xi / λi": np.round(fracciones / l_puras_bw, 4)
        })
        st.dataframe(df_bw_calc, use_container_width=True, height=250)

        # --- 6. GRÁFICA COMPARATIVA B&W ---
        st.write("---")
        st.write("##### 📈 Sensibilidad B&W: λ Mezcla vs Temperatura")
        
        rango_T_bw = np.linspace(300, 1000, 15)
        fig_bw, ax_bw = plt.subplots(figsize=(9, 4.5))
        colores_bw = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
        estilos_bw = ['-', '--', '-.', ':']

        for i, mod in enumerate(["Eucken Modificado", "Chung", "Gharagheizi", "Svehla"]):
            y_bw = []
            for T in rango_T_bw:
                lp, _ = obtener_propiedades_puras(T, mod)
                lm, _, _ = calcular_mezcla_burgoyne(lp, fracciones)
                y_bw.append(lm)
            ax_bw.plot(rango_T_bw, y_bw, label=mod, color=colores_bw[i], 
                    linestyle=estilos_bw[i], linewidth=2.5, marker='s', markersize=3)

        ax_bw.scatter(temp_bw, l_mix_bw, color='red', s=150, zorder=10, label=f'Punto Actual ({temp_bw}K)')
        
        ax_bw.set_xlabel("Temperatura (K)")
        ax_bw.set_ylabel("Conductividad Térmica (W/m·K)")
        ax_bw.set_title("Método Burgoyne y Weinberg: Comparativa de Modelos")
        ax_bw.legend()
        ax_bw.grid(True, alpha=0.3)
        st.pyplot(fig_bw)


# =====================================================================
# 💧 SECCIÓN 2: SOYBEAN OIL (ACEITE DE SOYA)
# =====================================================================


elif seccion == "💧 SoyBean Oil":
    st.info(
        "Cálculo de conductividad térmica de líquidos puros (Ácidos Grasos del Aceite de Soya) mediante métodos del libro de Poling"
    )

    # ==========================================
    # 1. BASE DE DATOS (PROPIEDADES Y EXPERIMENTAL)
    # ==========================================

    datos_componentes_aceite = {
        "Componente": ["Ácido Oleico", "Ácido Linoleico", "Ácido Linolénico", "Ácido Palmítico", "Ácido Esteárico"],
        "M (g/mol)": [282.46, 280.45, 278.43, 256.42, 284.48],
        "T_b (K)": [633.0, 638.0, 640.0, 624.7, 648.4],
        "T_c (K)": [833.0, 838.0, 843.0, 801.0, 818.0],
        # Nuevos datos para Di Nicola
        "dh_fus": [39400000, 38200000, 29800000, 54380000, 61200000],
        "omega": [1.018, 1.045, 1.072, 0.909, 1.002],
        # Datos para Sastri
        "CH3": [1, 1, 1, 1, 1],
        "CH2": [14, 12, 10, 14, 16],
        "Dobles_Enlaces": [1, 2, 3, 0, 0],
        "COOH": [1, 1, 1, 1, 1],
        #perkins
        "rho_c (kg/m3)": [272.24, 275.14, 278.07, 280.45, 257.06], # Densidades críticas
        "rho_liq (kg/m3)": [865.3, 874.1, 882.9, 832.1, 827.4],    # Densidades a ~313K (ejemplo)

    }
    df_comp_aceite = pd.DataFrame(datos_componentes_aceite)

    datos_exp_aceite = {
        "Componente": [
            "Ácido Oleico",
            "Ácido Linoleico",
            "Ácido Linolénico",
            "Ácido Palmítico",
            "Ácido Esteárico",
        ],
        "conductividad Exp (W/m*K)": [0.15, 0.15, 0.14, 0.16, 0.13],
    }
    df_exp_aceite = pd.DataFrame(datos_exp_aceite)

    # ==========================================
    # 2. FUNCIONES MATEMÁTICAS (PURAS LÍQUIDOS)
    # ==========================================

    # --- A. MÉTODO DE LATINI ET AL. ---
    def metodo_latini_liquidos(m_g_mol, t_b, t_c, t_sistema):
        # Constantes empíricas para la familia de Ácidos Orgánicos
        A_star = 0.00319
        alpha = 1.2
        beta = 0.5
        gamma = 0.167

        # Cálculo del Factor A
        A = (A_star * (t_b**alpha)) / ((m_g_mol**beta) * (t_c**gamma))

        # Temperatura reducida
        t_r = t_sistema / t_c

        # Ecuación principal de Latini
        lambda_w_m_k = (A * ((1.0 - t_r) ** 0.38)) / (t_r ** (1.0 / 6.0))
        return lambda_w_m_k, A, t_r

    # --- B. MÉTODO DE SASTRI ---
    def metodo_sastri_liquidos(
        t_b, t_c, t_sistema, n_ch3, n_ch2, n_dobles, n_cooh
    ):
        # Constantes para "Otros componentes" (Ácidos grasos entran aquí)
        a_sastri = 1.23
        n_sastri = 0.2

        # Contribuciones grupales para λb en W/(m·K)
        contrib_CH3 = 0.0545
        contrib_CH2 = -0.0008
        contrib_CH_doble = 0.0020  # Contribución del grupo ==CH-
        contrib_COOH = 0.1038

        # Químicamente, cada doble enlace (-CH=CH-) aporta dos grupos "==CH-"
        grupos_ch_doble = n_dobles * 2

        # Sumatoria para conductividad en el punto normal de ebullición (λb)
        lambda_b = (
            (n_ch3 * contrib_CH3)
            + (n_ch2 * contrib_CH2)
            + (grupos_ch_doble * contrib_CH_doble)
            + (n_cooh * contrib_COOH)
        )

        # Temperaturas reducidas
        t_r = t_sistema / t_c
        t_br = t_b / t_c

        # Exponente de corrección m
        m = 1.0 - ((1.0 - t_r) / (1.0 - t_br)) ** n_sastri

        # Conductividad térmica final
        lambda_w_m_k = lambda_b * (a_sastri**m)
        return lambda_w_m_k, lambda_b, m

    # --- C. MÉTODO DE DI NICOLA ET AL. ---
    def metodo_di_nicola_liquidos(t_sistema, t_c, dh_fus, omega, m_g_mol):
        # Constantes empíricas ajustadas para ácidos grasos
        a, b, c = -0.5694, -0.1436, 5.489e-10
        d, e, f = 0.0508, 1.0, 0.0622
        lambda_0 = 1.0  # W/m*K

        # Temperatura reducida
        t_r = t_sistema / t_c

        # Ecuación principal de Di Nicola
        ratio_lambda = a + (b * t_r) + (c * dh_fus) + (d * omega) + ((e / m_g_mol)**f)
        
        lambda_w_m_k = ratio_lambda * lambda_0
        return lambda_w_m_k, t_r


    # --- D. MÉTODO DE PERKINS (CORREGIDO) ---
    def metodo_perkins_liquidos(t_sistema, t_c, rho, rho_c, comp_name):
        # Coeficientes específicos (extraídos de tus tablas de imagen)
        coefs = {
            "Ácido Oleico":    {'A0': 0.0124, 'A1': 0.0285, 'A2': 0.0162, 'A3': -0.0041, 'B11': 0.0215, 'B12': -0.0082, 'B21': -0.0105, 'B22': 0.0034, 'B31': 0.0049, 'B32': 0},
            "Ácido Linoleico": {'A0': 0.0131, 'A1': 0.0297, 'A2': 0.0174, 'A3': -0.0087, 'B11': 0.0228, 'B12': -0.0087, 'B21': -0.0112, 'B22': 0.0037, 'B31': 0.0051, 'B32': 0},
            "Ácido Linolénico":{'A0': 0.0135, 'A1': 0.0302, 'A2': 0.0178, 'A3': -0.0091, 'B11': 0.0235, 'B12': -0.0091, 'B21': -0.0118, 'B22': 0.0039, 'B31': 0.0051, 'B32': 0},
            "Ácido Palmítico": {'A0': 0.0118, 'A1': 0.0271, 'A2': 0.0155, 'A3': -0.0076, 'B11': 0.0201, 'B12': -0.0076, 'B21': -0.0098, 'B22': 0.0031, 'B31': 0.0049, 'B32': 0},
            "Ácido Esteárico": {'A0': 0.0121, 'A1': 0.0278, 'A2': 0.0158, 'A3': -0.0079, 'B11': 0.0208, 'B12': -0.0079, 'B21': -0.0101, 'B22': 0.0032, 'B31': 0.0049, 'B32': 0}
        }
        
        c = coefs.get(comp_name)
        tr = t_sistema / t_c
        dr = rho / rho_c  # IMPORTANTE: Esto debe dar un valor ~3.0
        
        # Término Base (Lambda 0)
        l0 = c['A0'] + c['A1']*tr + c['A2']*(tr**2) + c['A3']*(tr**3)
        
        # Término Residual (Delta Lambda r)
        dlr = ( (c['B11'] + c['B12']*tr) * dr + 
                (c['B21'] + c['B22']*tr) * (dr**2) + 
                (c['B31'] + c['B32']*tr) * (dr**3) )
        
        lambda_calc = l0 + dlr
        return lambda_calc, tr, dr, l0, dlr


    def obtener_lambda_puro(metodo, row_datos, t_sistema):
        """Retorna la lambda calculada según el método seleccionado para la mezcla."""
        if metodo == "1. Latini et al.":
            l, _, _ = metodo_latini_liquidos(row_datos["M (g/mol)"], row_datos["T_b (K)"], row_datos["T_c (K)"], t_sistema)
        elif metodo == "2. Sastri":
            l, _, _ = metodo_sastri_liquidos(row_datos["T_b (K)"], row_datos["T_c (K)"], t_sistema, 
                                            row_datos["CH3"], row_datos["CH2"], row_datos["Dobles_Enlaces"], row_datos["COOH"])
        elif metodo == "3. Di Nicola et al.":
            l, _ = metodo_di_nicola_liquidos(t_sistema, row_datos["T_c (K)"], row_datos["dh_fus"], row_datos["omega"], row_datos["M (g/mol)"])
        else: # 4. Perkins
            l, _, _, _, _ = metodo_perkins_liquidos(t_sistema, row_datos["T_c (K)"], row_datos["rho_liq (kg/m3)"], row_datos["rho_c (kg/m3)"], row_datos["Componente"])
        return l

    # ==========================================
    # 3. INTERFAZ DE USUARIO Y PESTAÑAS
    # ==========================================

    # Clave única 'temp_aceite' para no generar conflictos con el input del COG
    temp_k_aceite = st.number_input(
        "Temperatura de análisis (K)", value=313.15, key="temp_aceite"
    )

    tab_latini, tab_sastri, tab_dinicola, tab_perkins, tab_mezcla, tab_solucion_ideal = st.tabs(
        ["1. Latini et al.", "2. Sastri", "3. Di Nicola", "4. Perkins", "5. Regla Vredeveld (1917)", "6. Regla de solucion ideal"]
    )

    # ==========================================
    # PESTAÑA 1: LATINI ET AL.
    # ==========================================
    with tab_latini:
        st.subheader("Método de Latini et al. (Modelo 1 - Líquidos)")

        parametros_latini = []
        comparacion_latini = []

        for i, row in df_comp_aceite.iterrows():
            lambda_calc, A_val, tr_val = metodo_latini_liquidos(
                row["M (g/mol)"],
                row["T_b (K)"],
                row["T_c (K)"],
                temp_k_aceite,
            )

            valor_exp = df_exp_aceite.loc[
                df_exp_aceite["Componente"] == row["Componente"],
                "conductividad Exp (W/m*K)",
            ].values[0]
            error_rel = abs((valor_exp - lambda_calc) / valor_exp) * 100

            parametros_latini.append(
                {
                    "Componente": row["Componente"],
                    "M (g/mol)": row["M (g/mol)"],
                    "Tb (K)": row["T_b (K)"],
                    "Tc (K)": row["T_c (K)"],
                    "Factor A": round(A_val, 5),
                    "Tr": round(tr_val, 4),
                }
            )

            comparacion_latini.append(
                {
                    "Componente": row["Componente"],
                    "λ Exp (W/m·K)": valor_exp,
                    "λ Calc (W/m·K)": round(lambda_calc, 5),
                    "% Error": f"{error_rel:.2f}%",
                }
            )

        # Cálculo de métricas globales con NumPy
        y_exp_lat = np.array(
            [item["λ Exp (W/m·K)"] for item in comparacion_latini]
        )
        y_calc_lat = np.array(
            [item["λ Calc (W/m·K)"] for item in comparacion_latini]
        )

        ss_res_lat = np.sum((y_exp_lat - y_calc_lat) ** 2)
        ss_tot_lat = np.sum((y_exp_lat - np.mean(y_exp_lat)) ** 2)
        r2_global_lat = 1.0 - (ss_res_lat / ss_tot_lat)
        error_global_mape_lat = (
            np.mean(np.abs((y_exp_lat - y_calc_lat) / y_exp_lat)) * 100
        )

        col_tablas_lat, col_grafica_lat = st.columns([1.1, 0.9])

        with col_tablas_lat:
            st.write("##### 📁 1. Parámetros de la Ecuación de Latini")
            st.dataframe(
                pd.DataFrame(parametros_latini),
                use_container_width=True,
                height=210,
            )

            st.write(
                f"##### 📊 2. Comparación y Error Relativo a {temp_k_aceite} K"
            )
            st.dataframe(
                pd.DataFrame(comparacion_latini),
                use_container_width=True,
                height=210,
            )

        with col_grafica_lat:
            st.write("##### 🎯 Métricas Globales")
            met1_lat, met2_lat = st.columns(2)
            met1_lat.metric("R² Global", f"{r2_global_lat:.4f}")
            met2_lat.metric("Error Global", f"{error_global_mape_lat:.2f}%")

            fig_lat, ax_lat = plt.subplots(figsize=(5.5, 4.0))
            ax_lat.scatter(
                y_exp_lat,
                y_calc_lat,
                color="#2a9d8f",
                edgecolors="#252422",
                s=50,
                zorder=3,
                label="Predicción Latini",
            )

            lim_min_lat = min(min(y_exp_lat), min(y_calc_lat)) * 0.90
            lim_max_lat = max(max(y_exp_lat), max(y_calc_lat)) * 1.10
            ax_lat.plot(
                [lim_min_lat, lim_max_lat],
                [lim_min_lat, lim_max_lat],
                color="#252422",
                linestyle="--",
                linewidth=1.2,
                zorder=2,
                label="Ajuste perfecto",
            )

            ax_lat.set_title(
                "Dispersión del Modelo Latini",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax_lat.set_xlabel("λ Experimental (W/m·K)", fontsize=9)
            ax_lat.set_ylabel("λ Calculada (W/m·K)", fontsize=9)
            ax_lat.set_xlim(lim_min_lat, lim_max_lat)
            ax_lat.set_ylim(lim_min_lat, lim_max_lat)
            ax_lat.tick_params(labelsize=8)

            ax_lat.spines["top"].set_visible(False)
            ax_lat.spines["right"].set_visible(False)
            ax_lat.spines["left"].set_color("#888888")
            ax_lat.spines["bottom"].set_color("#888888")
            ax_lat.legend(loc="upper left", fontsize=8, frameon=False)

            st.pyplot(fig_lat)

    # ==========================================
    # PESTAÑA 2: MÉTODO DE SASTRI
    # ==========================================
    with tab_sastri:
        st.subheader("Método de Sastri (Modelo 2 - Contribución de Grupos)")

        parametros_sastri = []
        comparacion_sastri = []

        for i, row in df_comp_aceite.iterrows():
            lambda_calc, lb_val, m_val = metodo_sastri_liquidos(
                row["T_b (K)"],
                row["T_c (K)"],
                temp_k_aceite,
                row["CH3"],
                row["CH2"],
                row["Dobles_Enlaces"],
                row["COOH"],
            )

            valor_exp = df_exp_aceite.loc[
                df_exp_aceite["Componente"] == row["Componente"],
                "conductividad Exp (W/m*K)",
            ].values[0]
            error_rel = abs((valor_exp - lambda_calc) / valor_exp) * 100

            parametros_sastri.append(
                {
                    "Componente": row["Componente"],
                    "Dobles Enlaces": row["Dobles_Enlaces"],
                    "λb (Ebullición)": round(lb_val, 4),
                    "Tbr": round(row["T_b (K)"] / row["T_c (K)"], 3),
                    "m (Corrección)": round(m_val, 4),
                }
            )

            comparacion_sastri.append(
                {
                    "Componente": row["Componente"],
                    "λ Exp (W/m·K)": valor_exp,
                    "λ Calc (W/m·K)": round(lambda_calc, 5),
                    "% Error": f"{error_rel:.2f}%",
                }
            )

        # Cálculo de métricas globales
        y_exp_sas = np.array(
            [item["λ Exp (W/m·K)"] for item in comparacion_sastri]
        )
        y_calc_sas = np.array(
            [item["λ Calc (W/m·K)"] for item in comparacion_sastri]
        )

        ss_res_sas = np.sum((y_exp_sas - y_calc_sas) ** 2)
        ss_tot_sas = np.sum((y_exp_sas - np.mean(y_exp_sas)) ** 2)
        r2_global_sas = 1.0 - (ss_res_sas / ss_tot_sas)
        error_global_mape_sas = (
            np.mean(np.abs((y_exp_sas - y_calc_sas) / y_exp_sas)) * 100
        )

        col_tablas_sas, col_grafica_sas = st.columns([1.1, 0.9])

        with col_tablas_sas:
            st.write("##### 📁 1. Desglose de Contribuciones (Sastri)")
            st.dataframe(
                pd.DataFrame(parametros_sastri),
                use_container_width=True,
                height=210,
            )

            st.write(
                f"##### 📊 2. Comparación y Error Relativo a {temp_k_aceite} K"
            )
            st.dataframe(
                pd.DataFrame(comparacion_sastri),
                use_container_width=True,
                height=210,
            )

        with col_grafica_sas:
            st.write("##### 🎯 Métricas Globales")
            met1_sas, met2_sas = st.columns(2)
            met1_sas.metric("R² Global", f"{r2_global_sas:.4f}")
            met2_sas.metric("Error Global", f"{error_global_mape_sas:.2f}%")

            fig_sas, ax_sas = plt.subplots(figsize=(5.5, 4.0))
            ax_sas.scatter(
                y_exp_sas,
                y_calc_sas,
                color="#e76f51",
                edgecolors="#252422",
                s=50,
                zorder=3,
                label="Predicción Sastri",
            )

            lim_min_sas = min(min(y_exp_sas), min(y_calc_sas)) * 0.90
            lim_max_sas = max(max(y_exp_sas), max(y_calc_sas)) * 1.10
            ax_sas.plot(
                [lim_min_sas, lim_max_sas],
                [lim_min_sas, lim_max_sas],
                color="#252422",
                linestyle="--",
                linewidth=1.2,
                zorder=2,
                label="Ajuste perfecto",
            )

            ax_sas.set_title(
                "Dispersión del Modelo Sastri",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax_sas.set_xlabel("λ Experimental (W/m·K)", fontsize=9)
            ax_sas.set_ylabel("λ Calculada (W/m·K)", fontsize=9)
            ax_sas.set_xlim(lim_min_sas, lim_max_sas)
            ax_sas.set_ylim(lim_min_sas, lim_max_sas)
            ax_sas.tick_params(labelsize=8)

            ax_sas.spines["top"].set_visible(False)
            ax_sas.spines["right"].set_visible(False)
            ax_sas.spines["left"].set_color("#888888")
            ax_sas.spines["bottom"].set_color("#888888")
            ax_sas.legend(loc="upper left", fontsize=8, frameon=False)

            st.pyplot(fig_sas)

    # ==========================================
    # PESTAÑA 3: MÉTODO DE DI NICOLA ET AL.
    # ==========================================
    with tab_dinicola:
        st.subheader("Método de Di Nicola et al. (Modelo 3 - Líquidos)")
        
        parametros_nicola = []
        comparacion_nicola = []

        for i, row in df_comp_aceite.iterrows():
            lambda_calc, tr_val = metodo_di_nicola_liquidos(
                temp_k_aceite,
                row["T_c (K)"],
                row["dh_fus"],
                row["omega"],
                row["M (g/mol)"]
            )

            valor_exp = df_exp_aceite.loc[
                df_exp_aceite["Componente"] == row["Componente"],
                "conductividad Exp (W/m*K)"
            ].values[0]
            
            error_rel = abs((valor_exp - lambda_calc) / valor_exp) * 100

            parametros_nicola.append({
                "Componente": row["Componente"],
                "Δh_fus": row["dh_fus"],
                "ω": row["omega"],
                "Tr": round(tr_val, 4)
            })

            comparacion_nicola.append({
                "Componente": row["Componente"],
                "λ Exp (W/m·K)": valor_exp,
                "λ Calc (W/m·K)": round(lambda_calc, 5),
                "% Error": f"{error_rel:.2f}%"
            })

        # Métricas Globales (Usando el mismo estilo de Latini)
        y_exp_nic = np.array([item["λ Exp (W/m·K)"] for item in comparacion_nicola])
        y_calc_nic = np.array([item["λ Calc (W/m·K)"] for item in comparacion_nicola])
        
        r2_global_nic = 1.0 - (np.sum((y_exp_nic - y_calc_nic)**2) / np.sum((y_exp_nic - np.mean(y_exp_nic))**2))
        error_global_nic = np.mean(np.abs((y_exp_nic - y_calc_nic) / y_exp_nic)) * 100

        col_tablas_nic, col_grafica_nic = st.columns([1.1, 0.9])

        with col_tablas_nic:
            st.write("##### 📁 1. Parámetros Termodinámicos")
            st.dataframe(pd.DataFrame(parametros_nicola), use_container_width=True)

            st.write(f"##### 📊 2. Comparación y Error a {temp_k_aceite} K")
            st.dataframe(pd.DataFrame(comparacion_nicola), use_container_width=True)

        with col_grafica_nic:
            st.write("##### 🎯 Métricas Globales")
            m1, m2 = st.columns(2)
            m1.metric("R² Global", f"{r2_global_nic:.4f}")
            m2.metric("Error Global", f"{error_global_nic:.2f}%")

            # Gráfica de dispersión (manteniendo tu estilo visual)
            fig_nic, ax_nic = plt.subplots(figsize=(5.5, 4.0))
            ax_nic.scatter(y_exp_nic, y_calc_nic, color="#e76f51", edgecolors="#252422", s=50, zorder=3, label="Predicción Di Nicola")
            
            lims = [min(min(y_exp_nic), min(y_calc_nic)) * 0.9, max(max(y_exp_nic), max(y_calc_nic)) * 1.1]
            ax_nic.plot(lims, lims, color="#252422", linestyle="--", linewidth=1.2, zorder=2, label="Ajuste perfecto")
            
            ax_nic.set_title("Dispersión: Di Nicola et al.", fontsize=11, fontweight="bold")
            ax_nic.set_xlabel("λ Experimental (W/m·K)")
            ax_nic.set_ylabel("λ Calculada (W/m·K)")
            ax_nic.legend(loc="upper left", fontsize=8, frameon=False)
            st.pyplot(fig_nic)


    # ==========================================
    # PESTAÑA 4: MÉTODO DE PERKINS (VERSIÓN FINAL)
    # ==========================================
    with tab_perkins:
        st.subheader("Método de Perkins (Modelo 4 - Polinómico)")
        
        par_perkins = []
        comp_perkins = []

        for i, row in df_comp_aceite.iterrows():
            # Ejecutar el cálculo
            l_calc, tr, dr, l0, dlr = metodo_perkins_liquidos(
                temp_k_aceite, 
                row["T_c (K)"], 
                row["rho_liq (kg/m3)"], 
                row["rho_c (kg/m3)"], 
                row["Componente"]
            )

            # Obtener valor experimental
            v_exp = df_exp_aceite.loc[
                df_exp_aceite["Componente"] == row["Componente"], 
                "conductividad Exp (W/m*K)"
            ].values[0]
            
            error_rel = abs((v_exp - l_calc) / v_exp) * 100

            # Guardar para tablas
            par_perkins.append({
                "Componente": row["Componente"],
                "Tr": round(tr, 4),
                "Dr (Reducida)": round(dr, 4),
                "λ0 (Base)": round(l0, 6),
                "Δλr (Resid)": round(dlr, 6)
            })

            comp_perkins.append({
                "Componente": row["Componente"],
                "λ Exp": v_exp,
                "λ Calc": round(l_calc, 6),
                "% Error": error_rel
            })

        # --- CÁLCULO DE MÉTRICAS ESTADÍSTICAS ---
        y_exp = np.array([item["λ Exp"] for item in comp_perkins])
        y_calc = np.array([item["λ Calc"] for item in comp_perkins])
        
        # Error Medio Absoluto Porcentual (MAPE)
        mape_perkins = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
        
        # Cálculo de R y R²
        correlation_matrix = np.corrcoef(y_exp, y_calc)
        r_value = correlation_matrix[0, 1]
        r_squared = r_value**2

        # --- INTERFAZ VISUAL ---
        col_t, col_g = st.columns([1.1, 0.9])

        with col_t:
            st.write("##### 📁 1. Parámetros del Modelo")
            st.dataframe(pd.DataFrame(par_perkins), use_container_width=True)
            
            # Formatear la tabla de comparación para mostrar % en el string
            df_comp_show = pd.DataFrame(comp_perkins)
            df_comp_show["% Error"] = df_comp_show["% Error"].map("{:.2f}%".format)
            st.write("##### 📊 2. Resultados vs Experimental")
            st.dataframe(df_comp_show, use_container_width=True)

        with col_g:
            st.write("##### 🎯 Precisión del Método")
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{r_squared:.4f}")
            m3.metric("Error global", f"{mape_perkins:.2f}%")

            # Gráfica con línea de ajuste
            fig_per, ax_per = plt.subplots(figsize=(5, 4))
            ax_per.scatter(y_exp, y_calc, color="#1d3557", edgecolors="white", s=80, zorder=3, label="Datos Perkins")
            
            # Línea de 45 grados (ajuste perfecto)
            lims = [min(y_exp.min(), y_calc.min()) * 0.95, max(y_exp.max(), y_calc.max()) * 1.05]
            ax_per.plot(lims, lims, color="#e63946", linestyle="--", alpha=0.7, label="Ajuste 1:1")
            
            ax_per.set_title(f"Correlación Perkins", fontsize=10, fontweight="bold")
            ax_per.set_xlabel("λ Experimental (W/m·K)")
            ax_per.set_ylabel("λ Calculada (W/m·K)")
            ax_per.legend(fontsize=8)
            ax_per.grid(alpha=0.2)
            
            st.pyplot(fig_per)


    # =====================================================================
    # PESTAÑA 5: REGLA DE MEZCLADO (VREDEVELD - ACEITE DE SOJA)
    # =====================================================================
    with tab_mezcla:
        st.header("🔗 Regla de Mezclado: Modelo de Vredeveld (1917)")
        
        # --- 1. CONFIGURACIÓN DE LA MEZCLA ---
        datos_mezcla = {
            "Componente": ["Ácido Oleico", "Ácido Linoleico", "Ácido Linolénico", "Ácido Palmítico", "Ácido Esteárico"],
            "xi": [0.1893, 0.5367, 0.0628, 0.1564, 0.0374]
        }
        df_mix_config = pd.DataFrame(datos_mezcla)

        metodo_puros_mix = st.selectbox(
            "📌 Selecciona el modelo de componentes puros para el desglose de tabla:",
            ["1. Latini et al.", "2. Sastri", "3. Di Nicola et al.", "4. Perkins"]
        )

        # --- 2. CÁLCULO EN EL PUNTO ACTUAL ---
        ki_puros_actual = []
        try:
            for _, row in df_mix_config.iterrows():
                info_c = df_comp_aceite[df_comp_aceite["Componente"] == row["Componente"]].iloc[0]
                val_ki = obtener_lambda_puro(metodo_puros_mix, info_c, temp_k_aceite)
                ki_puros_actual.append(val_ki)

            ki_puros_actual = np.array(ki_puros_actual)
            xi = df_mix_config["xi"].values
            km_punto_actual = (np.sum(xi * (ki_puros_actual**-2)))**-0.5

            # --- 3. MÉTRICAS Y TABLAS ---
            st.write("---")
            m1, m2 = st.columns(2)
            m1.metric(f"λ Mezcla (Vredeveld)", f"{km_punto_actual:.5f} W/m·K")

            st.dataframe(df_mix_config.assign(**{"λi Puro": np.round(ki_puros_actual, 5)}), use_container_width=True)

            # --- 4. GRÁFICA DE DISPERSIÓN DINÁMICA ---
            st.write("---")
            st.write(f"##### 📈 Evolución de Conductividad (Rango centrado en {temp_k_aceite}K)")
            
            # CORRECCIÓN AQUÍ: Definir rango dinámico para que el punto siempre esté incluido
            # Creamos un rango que vaya desde 273K hasta la temperatura actual + 50K
            t_min_plot = 273.15
            t_max_plot = max(450, temp_k_aceite + 50) 
            rango_t = np.linspace(t_min_plot, t_max_plot, 20)
            
            fig_mix, ax_mix = plt.subplots(figsize=(10, 5))
            modelos_grafica = ["1. Latini et al.", "2. Sastri", "3. Di Nicola et al.", "4. Perkins"]
            etiquetas = ["M1: Latini", "M2: Sastri", "M3: Di Nicola", "M4: Perkins"]
            colores = ["#f94144", "#f9c74f", "#43aa8b", "#577590"] # Colores similares a tu imagen
            
            for idx, mod in enumerate(modelos_grafica):
                y_lambda_mix = []
                for t_plot in rango_t:
                    ki_list = [obtener_lambda_puro(mod, df_comp_aceite[df_comp_aceite["Componente"]==c].iloc[0], t_plot) 
                               for c in df_mix_config["Componente"]]
                    km_t = (np.sum(xi * (np.array(ki_list)**-2)))**-0.5
                    y_lambda_mix.append(km_t)
                
                # Línea suavizada y puntos de dispersión
                ax_mix.plot(rango_t, y_lambda_mix, label=etiquetas[idx], color=colores[idx], linewidth=2, alpha=0.8)
                ax_mix.scatter(rango_t, y_lambda_mix, color=colores[idx], s=35, edgecolors='white', linewidth=0.5)

            # Resaltar el Punto Actual (Si es 500K, ahora aparecerá sobre las líneas)
            ax_mix.scatter(temp_k_aceite, km_punto_actual, color='red', s=200, zorder=10, 
                           label=f'Punto Actual ({temp_k_aceite}K)', edgecolors='black', linewidth=2)

            # Estética de la gráfica (Grid y etiquetas)
            ax_mix.set_xlabel("Temperatura (K)", fontweight='bold')
            ax_mix.set_ylabel("Conductividad Térmica Mezcla (W/m·K)", fontweight='bold')
            ax_mix.set_title("Comportamiento de la Mezcla (Vredeveld) vs Temperatura", fontsize=12, fontweight='bold')
            ax_mix.grid(True, linestyle='--', alpha=0.4)
            ax_mix.legend()
            
            st.pyplot(fig_mix)

        except Exception as e:
            st.error(f"Error: {e}")


    # =====================================================================
    # PESTAÑA: REGLA DE MEZCLADO (SOLUCIÓN IDEAL)
    # =====================================================================
    with tab_solucion_ideal:
        st.header("🧪 Regla de Mezclado: Modelo de Solución Ideal")
        st.write("Este modelo estima la conductividad a partir de las contribuciones logarítmicas de los componentes puros.")

        # --- 1. CONFIGURACIÓN DE LA MEZCLA ---
        datos_mezcla = {
            "Componente": ["Ácido Oleico", "Ácido Linoleico", "Ácido Linolénico", "Ácido Palmítico", "Ácido Esteárico"],
            "xi": [0.1893, 0.5367, 0.0628, 0.1564, 0.0374]
        }
        df_mix_config = pd.DataFrame(datos_mezcla)

        metodo_puros_mix = st.selectbox(
            "📌 Selecciona el modelo de componentes puros (Solución Ideal):",
            ["1. Latini et al.", "2. Sastri", "3. Di Nicola et al.", "4. Perkins"],
            key="sb_ideal"
        )

        # --- 2. MOTOR DE CÁLCULO (PUNTO ACTUAL) ---
        ki_puros_actual = []
        try:
            for _, row in df_mix_config.iterrows():
                info_c = df_comp_aceite[df_comp_aceite["Componente"] == row["Componente"]].iloc[0]
                val_ki = obtener_lambda_puro(metodo_puros_mix, info_c, temp_k_aceite)
                ki_puros_actual.append(val_ki)

            ki_puros_actual = np.array(ki_puros_actual)
            xi = df_mix_config["xi"].values
            
            # FÓRMULA SOLUCIÓN IDEAL: Km = exp( sum( xi * ln(Ki) ) )
            km_punto_actual = np.exp(np.sum(xi * np.log(ki_puros_actual)))

            # --- 3. MÉTRICAS Y TABLAS ---
            st.write("---")
            m1, m2 = st.columns(2)
            m1.metric(f"λ Mezcla (Ideal)", f"{km_punto_actual:.5f} W/m·K")
            
            val_exp_soja = 0.162 
            err_mix = abs((val_exp_soja - km_punto_actual)/val_exp_soja)*100
            m2.metric(f"Error vs Experimental ({temp_k_aceite}K)", f"{err_mix:.2f}%")

            st.write("##### 📑 Desglose de Contribuciones Logarítmicas")
            df_res = df_mix_config.copy()
            df_res["λi Puro (W/m·K)"] = np.round(ki_puros_actual, 5)
            df_res["ln(λi)"] = np.round(np.log(ki_puros_actual), 5)
            df_res["xi * ln(λi)"] = xi * np.log(ki_puros_actual)
            st.dataframe(df_res, use_container_width=True)

            # --- 4. GRÁFICA DE DISPERSIÓN DINÁMICA ---
            st.write("---")
            st.write(f"##### 📈 Evolución de Conductividad - Modelo Ideal (Rango centrado en {temp_k_aceite}K)")
            
            t_min_plot = 273.15
            t_max_plot = max(450, temp_k_aceite + 50) 
            rango_t = np.linspace(t_min_plot, t_max_plot, 20)
            
            fig_mix, ax_mix = plt.subplots(figsize=(10, 5))
            modelos_grafica = ["1. Latini et al.", "2. Sastri", "3. Di Nicola et al.", "4. Perkins"]
            etiquetas = ["M1: Latini", "M2: Sastri", "M3: Di Nicola", "M4: Perkins"]
            colores = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d"]
            
            for idx, mod in enumerate(modelos_grafica):
                y_lambda_mix = []
                for t_plot in rango_t:
                    ki_list = [obtener_lambda_puro(mod, df_comp_aceite[df_comp_aceite["Componente"]==c].iloc[0], t_plot) 
                               for c in df_mix_config["Componente"]]
                    # Aplicar fórmula Ideal en cada punto
                    km_t = np.exp(np.sum(xi * np.log(np.array(ki_list))))
                    y_lambda_mix.append(km_t)
                
                ax_mix.plot(rango_t, y_lambda_mix, label=etiquetas[idx], color=colores[idx], linewidth=2, alpha=0.8)
                ax_mix.scatter(rango_t, y_lambda_mix, color=colores[idx], s=35, edgecolors='white')

            # Resaltar Punto Actual
            ax_mix.scatter(temp_k_aceite, km_punto_actual, color='red', s=200, zorder=10, 
                           label=f'Punto Actual ({temp_k_aceite}K)', edgecolors='black', linewidth=2)

            ax_mix.set_xlabel("Temperatura (K)", fontweight='bold')
            ax_mix.set_ylabel("Conductividad Térmica Mezcla (W/m·K)", fontweight='bold')
            ax_mix.set_title("Comportamiento de la Mezcla (Ideal) vs Temperatura", fontsize=12, fontweight='bold')
            ax_mix.grid(True, linestyle='--', alpha=0.4)
            ax_mix.legend()
            
            st.pyplot(fig_mix)

        except Exception as e:
            st.error(f"❌ Error en el cálculo: {e}")







   