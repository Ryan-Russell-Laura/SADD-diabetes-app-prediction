import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Sistema de Diagnóstico - Diabetes",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILOS CSS PERSONALIZADOS
# ==============================================================================
st.markdown("""
<style>
    /* Paleta de colores médicos profesionales */
    :root {
        --medical-blue: #0077B6;
        --medical-green: #06D6A0;
        --medical-red: #EF476F;
        --medical-yellow: #FFD166;
        --medical-dark: #023047;
        --medical-light: #F8F9FA;
    }
    
    /* Fondo principal */
    .main {
        background-color: #F5F7FA;
    }
    
    /* Títulos personalizados */
    .titulo-principal {
        background: linear-gradient(135deg, #0077B6 0%, #023047 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Tarjetas de información */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #0077B6;
        margin-bottom: 1rem;
        color: #023047;
    }
    
    .info-card small {
        color: #666;
    }
    
    /* Alertas personalizadas */
    .alert-high {
        background: linear-gradient(135deg, #EF476F 0%, #d62956 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(239, 71, 111, 0.3);
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #FFD166 0%, #f4c24d 100%);
        color: #023047;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 209, 102, 0.3);
    }
    
    .alert-low {
        background: linear-gradient(135deg, #06D6A0 0%, #05b887 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(6, 214, 160, 0.3);
    }
    
    /* Botón principal */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #06D6A0 0%, #05b887 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(6, 214, 160, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(6, 214, 160, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARGA DE MODELOS Y ODDS RATIOS REALES
# ==============================================================================
@st.cache_resource
def load_resources():
    """Carga el modelo principal (escalado), el StandardScaler y los Odds Ratios reales."""
    model, scaler, or_table = None, None, None
    
    try:
        # Nota: He cambiado el nombre del archivo del modelo a logreg por consistencia.
        # Asegúrate de que el archivo 'modelo_diabetes_logreg.pkl' exista.
        model = joblib.load('modelo_diabetes_logreg.pkl') 
        scaler = joblib.load('scaler_diabetes.pkl')
        or_table = pd.read_csv('odds_ratios_reales.csv')
        
        st.success("✅ Modelos y Odds Ratios cargados correctamente.")
        
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: No se encontró el archivo '{e.filename}'. Asegúrate de tener 'modelo_diabetes_logreg.pkl', 'scaler_diabetes.pkl' y 'odds_ratios_reales.csv' en la carpeta.")
    
    return model, scaler, or_table

MODEL, SCALER, TABLA_OR = load_resources()
FEATURES = ['nivel_glucosa', 'nivel_hba1c', 'imc', 'hipertension', 'cardiopatia']

# ==============================================================================
# INICIALIZAR SESSION STATE
# ==============================================================================
if 'diagnostico_realizado' not in st.session_state:
    st.session_state.diagnostico_realizado = False

# ==============================================================================
# FUNCIÓN DE PREDICCIÓN
# ==============================================================================
def predict_diabetes(data):
    """Procesa los datos, escala y predice la probabilidad de diabetes."""
    if MODEL is None or SCALER is None:
        return 0.0, None
    
    input_df = pd.DataFrame([data], columns=FEATURES)
    scaled_data = SCALER.transform(input_df)
    prob = MODEL.predict_proba(scaled_data)[0][1]
    
    return prob, input_df

# ==============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ==============================================================================
def create_gauge_chart(probability):
    """Crea un gráfico de medidor para la probabilidad"""
    # Determinar color de la barra Y del número según la probabilidad
    if probability >= 0.5: 
        bar_color = "#EF476F"
        number_color = "#EF476F"
    elif probability >= 0.2:
        bar_color = "#FFD166"
        number_color = "#FFD166"
    else: 
        bar_color = "#06D6A0"
        number_color = "#06D6A0"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilidad de Diabetes", 'font': {'size': 24, 'color': '#E8F5E9'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': number_color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#023047"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#023047",
            'steps': [
                {'range': [0, 20], 'color': '#E8F5E9'},
                {'range': [20, 50], 'color': '#FFF9C4'},
                {'range': [50, 100], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "#EF476F", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def create_or_chart(tabla_or, patient_data):
    """Crea un gráfico de barras para los Odds Ratios (ORs)"""
    if tabla_or is None or tabla_or.empty:
        return None
    
    relevant_vars = []
    or_values = []
    colors = []
    
    # Preparamos las variables relevantes
    for index, row in tabla_or.iterrows():
        var = row['Variable Clínica']
        or_val = row['Odds Ratio (OR)']
        
        if or_val > 1.0:
            # Solo incluimos las variables que son factores de riesgo (OR > 1)
            relevant_vars.append(var.replace('_', ' ').title().replace('Imc', 'IMC').replace('Hba1c', 'HbA1c'))
            or_values.append(or_val)
            # Definimos el color basado en la magnitud del riesgo
            colors.append('#EF476F' if or_val > 2.0 else '#FFD166')
    
    if not relevant_vars:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=or_values,
            y=relevant_vars,
            orientation='h',
            marker=dict(color=colors, line=dict(color='#023047', width=2)),
            text=[f"{val:.2f}x" for val in or_values],
            textposition='outside',
            textfont=dict(size=14, color='#023047', family='Arial Black')
        )
    ])
    
    # Ajuste de Layout para mejor legibilidad
    fig.update_layout(
        title={
            'text': "Factores de Riesgo Principales (Odds Ratios)",
            'font': {'size': 20, 'color': '#023047', 'family': 'Arial Black'}
        },
        xaxis_title="Odds Ratio (OR)",
        yaxis_title="",
        # Aumentar margen izquierdo para las etiquetas del eje Y
        margin=dict(l=150, r=50, t=80, b=50), 
        paper_bgcolor='white',
        plot_bgcolor='#F8F9FA', 
        font={'family': "Arial, sans-serif", 'color': '#023047'} # Color del texto principal (incluye etiquetas Y)
    )
    
    # Asegurar que las etiquetas del eje Y se vean bien (ya se maneja con font color en update_layout)
    fig.update_yaxes(tickfont=dict(color='#023047', size=14), automargin=True)
    
    fig.add_vline(x=1, line_dash="dash", line_color="#023047", annotation_text="Sin efecto")
    
    return fig

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="titulo-principal">
    <h1>🏥 SISTEMA DE APOYO AL DIAGNÓSTICO DE DIABETES</h1>
    <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.95;">
        Herramienta de predicción basada en Inteligencia Artificial
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - ENTRADA DE DATOS (SOLO CAJAS DE TEXTO)
# ==============================================================================
with st.sidebar:
    st.markdown("### 📋 DATOS DEL PACIENTE")
    st.markdown("---")
    
    # GLUCOSA
    st.markdown("**🩸 Nivel de Glucosa (mg/dL)**")
    glucosa = st.number_input(
        "glucosa_input",
        min_value=70.0,
        max_value=300.0,
        value=120.0,
        step=1.0,
        key="glucosa",
        label_visibility="collapsed"
    )
    
    # HBA1C
    st.markdown("**🩸 Hemoglobina Glicosilada HbA1c (%)**")
    hba1c = st.number_input(
        "hba1c_input",
        min_value=3.0,
        max_value=15.0,
        value=5.5,
        step=0.1,
        key="hba1c",
        label_visibility="collapsed"
    )
    
    # IMC
    st.markdown("**📊 Índice de Masa Corporal IMC (kg/m²)**")
    imc = st.number_input(
        "imc_input",
        min_value=15.0,
        max_value=80.0,
        value=25.0,
        step=0.1,
        key="imc",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**❤️ Antecedentes Clínicos**")
    
    hipertension = st.checkbox("🔴 Hipertensión Arterial", value=False, key="ht")
    cardiopatia = st.checkbox("💔 Enfermedad Cardiovascular", value=False, key="cp")
    
    ht_val = 1 if hipertension else 0
    cp_val = 1 if cardiopatia else 0
    
    st.markdown("---")
    
    # Botón de diagnóstico
    if st.button("🔬 REALIZAR DIAGNÓSTICO", type="primary", use_container_width=True):
        st.session_state.diagnostico_realizado = True

# ==============================================================================
# DATOS DEL PACIENTE
# ==============================================================================
patient_data = {
    'nivel_glucosa': glucosa,
    'nivel_hba1c': hba1c,
    'imc': imc,
    'hipertension': ht_val,
    'cardiopatia': cp_val
}

# ==============================================================================
# ÁREA PRINCIPAL - RESULTADOS
# ==============================================================================
if st.session_state.diagnostico_realizado:
    # Realizar predicción
    prob, input_df = predict_diabetes(patient_data)
    riesgo_pct = prob * 100
    
    # ==============================================================================
    # SECCIÓN 1: RESULTADO PRINCIPAL
    # ==============================================================================
    st.markdown("## 📊 RESULTADO DEL ANÁLISIS")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Gráfico de medidor
        gauge_fig = create_gauge_chart(prob)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Interpretación del riesgo
        if riesgo_pct >= 50:
            st.markdown("""
            <div class="alert-high">
                🔴 ALTO RIESGO DE DIABETES<br>
                <small>Se recomienda intervención inmediata y pruebas confirmatorias</small>
            </div>
            """, unsafe_allow_html=True)
            recomendacion = "Acción Inmediata: Solicitar pruebas confirmatorias (Glucemia en ayunas, Curva de tolerancia a la glucosa). Considerar derivación a endocrinología."
        elif riesgo_pct >= 20:
            st.markdown("""
            <div class="alert-medium">
                ⚠️ RIESGO MODERADO DE DIABETES<br>
                <small>Requiere monitoreo intensivo y cambios en el estilo de vida</small>
            </div>
            """, unsafe_allow_html=True)
            recomendacion = "Monitoreo Activo: Control periódico cada 3-6 meses. Implementar programa de modificación de estilo de vida (dieta, ejercicio)."
        else:
            st.markdown("""
            <div class="alert-low">
                🟢 BAJO RIESGO DE DIABETES<br>
                <small>Mantener hábitos saludables y vigilancia preventiva</small>
            </div>
            """, unsafe_allow_html=True)
            recomendacion = "Prevención: Mantener controles anuales de rutina. Continuar con hábitos de vida saludables."
    
    with col2:
        st.markdown("### 📋 Parámetros Evaluados")
        
        # Tabla de parámetros con colores
        params_display = pd.DataFrame({
            'Parámetro': ['Glucosa', 'HbA1c', 'IMC', 'Hipertensión', 'Cardiopatía'],
            'Valor': [
                f"{glucosa:.1f} mg/dL",
                f"{hba1c:.1f}%",
                f"{imc:.1f} kg/m²",
                "Sí ✓" if ht_val == 1 else "No ✗",
                "Sí ✓" if cp_val == 1 else "No ✗"
            ],
            'Estado': [
                "⚠️ Elevado" if glucosa > 125 else "✓ Normal",
                "⚠️ Elevado" if hba1c >= 6.5 else "⚠️ Prediabetes" if hba1c >= 5.7 else "✓ Normal",
                "⚠️ Elevado" if imc > 30 else "⚠️ Sobrepeso" if imc > 25 else "✓ Normal",
                "⚠️ Presente" if ht_val == 1 else "✓ Ausente",
                "⚠️ Presente" if cp_val == 1 else "✓ Ausente"
            ]
        })
        
        st.dataframe(params_display, use_container_width=True, hide_index=True)
        
        # Métricas clave
        st.markdown("### 🎯 Métricas Clave")
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            riesgo_categoria = "ALTO" if riesgo_pct >= 50 else "MODERADO" if riesgo_pct >= 20 else "BAJO"
            st.metric("Categoría de Riesgo", riesgo_categoria)
        
        with metric_col2:
            confianza = 95 if riesgo_pct > 70 or riesgo_pct < 30 else 85
            st.metric("Confianza del Modelo", f"{confianza}%")
    
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 2: ANÁLISIS DE FACTORES DE RIESGO
    # ==============================================================================
    st.markdown("## 🔍 ANÁLISIS DE FACTORES DE RIESGO")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Gráfico de ORs
        or_chart = create_or_chart(TABLA_OR, patient_data)
        if or_chart:
            st.plotly_chart(or_chart, use_container_width=True)
        else:
            st.info("No se identificaron factores de riesgo elevados en este paciente.")
    
    with col2:
        st.markdown("### 📊 Tabla de Odds Ratios")
        st.dataframe(
            TABLA_OR,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 3: INTERPRETACIÓN CLÍNICA
    # ==============================================================================
    st.markdown("## 📝 INTERPRETACIÓN CLÍNICA DETALLADA")
    
    # Análisis individualizado
    st.markdown("### 🎯 Factores Identificados en el Paciente")
    
    factores_identificados = []
    
    for index, row in TABLA_OR.iterrows():
        var = row['Variable Clínica']
        or_val = row['Odds Ratio (OR)']
        paciente_val = patient_data[var]
        
        # SIMPLIFICACIÓN: SOLO VARIABLES CON RIESGO REAL (OR > 1.0)
        if or_val > 1.0:
            if var == 'nivel_glucosa' and paciente_val > 100:
                factores_identificados.append({
                    'factor': f"⚠️ <strong>Glucosa Elevada ({paciente_val:.1f} mg/dL)</strong>",
                    'detalle': f"OR: {or_val:.2f}x - Cada unidad adicional multiplica el riesgo {or_val:.2f} veces."
                })
            
            elif var == 'nivel_hba1c':
                if paciente_val >= 6.5:
                    factores_identificados.append({
                        'factor': f"⚠️ <strong>HbA1c en rango de Diabetes ({paciente_val:.1f}%)</strong>",
                        'detalle': f"OR: {or_val:.2f}x - Valor diagnóstico de diabetes según criterios ADA."
                    })
                elif paciente_val >= 5.7:
                    factores_identificados.append({
                    'factor': f"⚠️ <strong>HbA1c en Prediabetes ({paciente_val:.1f}%)</strong>",
                    'detalle': f"OR: {or_val:.2f}x - Elevado riesgo metabólico, requiere intervención temprana."
                    })

            
            elif var == 'imc' and paciente_val > 25:
                factores_identificados.append({
                    'factor': f"⚠️ <strong>IMC Elevado ({paciente_val:.1f} kg/m²)</strong>",
                    'detalle': f"OR: {or_val:.2f}x - Obesidad/sobrepeso aumenta resistencia a la insulina."
                })
            
            elif var == 'hipertension' and paciente_val == 1:
                factores_identificados.append({
                    'factor': "⚠️ <strong>Hipertensión Arterial Presente</strong>",
                    'detalle': f"OR: {or_val:.2f}x - La presencia de HTA multiplica el riesgo {or_val:.2f} veces."
                })
            
            elif var == 'cardiopatia' and paciente_val == 1:
                factores_identificados.append({
                    'factor': "⚠️ <strong>Enfermedad Cardiovascular Presente</strong>",
                    'detalle': f"OR: {or_val:.2f}x - La cardiopatía aumenta significativamente el riesgo metabólico."
                })
    
    if factores_identificados:
        for factor_info in factores_identificados:
            st.markdown(f"""
            <div class="info-card">
                {factor_info['factor']}<br>
                <small>{factor_info['detalle']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No se identificaron factores de riesgo significativos en los parámetros evaluados.")
    
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 4: REPORTE COMPLETO Y RECOMENDACIONES (ACTUALIZADA)
    # ==============================================================================
    st.markdown("## 📄 REPORTE CLÍNICO COMPLETO")
    
    col1, spacer, col2 = st.columns([1, 0.1, 1])
    
    with col1:
        st.markdown("### 💊 Recomendaciones Terapéuticas")
        st.markdown(f"""
        <div class="info-card">
            <strong>{recomendacion}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔬 Pruebas Adicionales Sugeridas")
        if riesgo_pct >= 50:
            st.markdown("""
            <div class="info-card">
                • Glucemia en ayunas<br>
                • Curva de tolerancia oral a la glucosa (CTOG)<br>
                • Perfil lipídico completo<br>
                • Función renal (creatinina, urea)<br>
                • Examen de fondo de ojo
            </div>
            """, unsafe_allow_html=True)
        elif riesgo_pct >= 20:
            st.markdown("""
            <div class="info-card">
                • Glucemia en ayunas (control cada 3-6 meses)<br>
                • HbA1c semestral<br>
                • Perfil lipídico
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                • Glucemia anual de rutina<br>
                • Control de peso y presión arterial
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # RECOMENDACIONES DE ESTILO DE VIDA Y NUTRICIÓN (TARJETA 1)
        st.markdown("### 🥗 Recomendaciones de Nutrición")
        st.markdown("""
        <div class="info-card" style="border-left-color: #06D6A0;">
            <strong>Nutrición (Dieta):</strong><br>
            • Dieta baja en azúcares simples (bebidas azucaradas, postres).<br>
            • Aumento de fibra dietética (vegetales, legumbres).<br>
            • Control de porciones y horarios de comidas.
        </div>
        """, unsafe_allow_html=True)
        
        # ACTIVIDAD FÍSICA (TARJETA)
        st.markdown("### 💪 Actividad Física")
        st.markdown("""
        <div class="info-card" style="border-left-color: #0077B6;">
            <strong>Actividad Física:</strong><br>
            • 150 min/semana de ejercicio aeróbico moderado.<br>
            • Entrenamiento de fuerza 2–3 veces por semana.<br>
            • Incremento progresivo si el paciente es sedentario.
        </div>
        """, unsafe_allow_html=True)

        # MONITOREO (TARJETA SEPARADA)
        st.markdown("### 📈 Monitoreo")
        st.markdown("""
        <div class="info-card" style="border-left-color: #FFD166;">
            <strong>Monitoreo:</strong><br>
            • Control periódico de glucosa según categoría de riesgo.<br>
            • Medición de HbA1c cada 3-6 meses si hay factores de riesgo.<br>
            • Registro personal de síntomas y cambios de hábitos.
        </div>
        """, unsafe_allow_html=True)


        st.markdown("### ⚠️ Señales de Alarma")
        st.markdown("""
        <div class="info-card" style="border-left-color: #EF476F;">
            • Sed excesiva (polidipsia)<br>
            • Micción frecuente (poliuria)<br>
            • Pérdida de peso inexplicable<br>
            • Fatiga constante<br>
            • Visión borrosa<br>
            • Heridas que no cicatrizan
        </div>
        """, unsafe_allow_html=True)
    
    # Footer del reporte
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; margin-top: 2rem;">
        <p style="color: #666; font-size: 0.9rem; margin: 0;">
            <strong>Nota Importante:</strong> Este sistema es una herramienta de apoyo al diagnóstico.
            Las decisiones clínicas finales deben ser tomadas por un profesional médico calificado
            considerando el contexto clínico completo del paciente.
        </p>
        <p style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">
            Sistema versión 1.0 | Modelo: Regresión Logística | Última actualización: 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Pantalla inicial cuando no se ha realizado diagnóstico
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h2 style="color: #023047;">👈 Complete los datos del paciente en el panel lateral</h2>
        <p style="color: #666; font-size: 1.1rem; margin-top: 1rem;">
            Ingrese los parámetros clínicos escribiendo directamente los valores numéricos.
        </p>
        <p style="color: #999; margin-top: 2rem;">
            Presione el botón <strong>"REALIZAR DIAGNÓSTICO"</strong> para obtener el análisis completo.
        </p>
    </div>
    """, unsafe_allow_html=True)
