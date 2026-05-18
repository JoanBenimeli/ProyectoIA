import streamlit as st
import joblib
import re
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

st.set_page_config(
    page_title="Análisis de Sentimiento Pro", 
    page_icon="📊", 
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>Analizador de Sentimientos</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Arquitectura Avanzada: Doc2Vec + Random Forest Classifier</p>", unsafe_allow_html=True)
st.divider()

def extract_words(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'<[^>]+>', ' ', sentence)
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    words = sentence.strip().split()
    return words

@st.cache_resource
def cargar_modelos():
    try:
        modelo_d2v = Doc2Vec.load("modelo_sentimientos.d2v")
        modelo_rf = joblib.load('modelo_rf_final.pkl')
        return modelo_d2v, modelo_rf
    except Exception as e:
        st.error(f"Error al cargar los archivos de los modelos: {e}")
        return None, None

modelo_doc2vec, modelo_rf = cargar_modelos()

def predecir_sentimiento(texto, modelo_d2v, modelo_rf):
    palabras_limpias = extract_words(texto)
    vector = modelo_d2v.infer_vector(palabras_limpias)
    vector_modelo = vector.reshape(1, -1)
    
    prediccion = modelo_rf.predict(vector_modelo)[0]
    probabilidades = modelo_rf.predict_proba(vector_modelo)[0]
    
    prob_neg = probabilidades[0]
    prob_pos = probabilidades[1]
    
    resultado = "Positivo" if prediccion == 1 else "Negativo"
    return resultado, prob_pos, prob_neg

if modelo_doc2vec is not None and modelo_rf is not None:
    
    review_input = st.text_area(
        "Introduce la reseña o comentario a analizar:", 
        placeholder="Escribe aquí tu texto en inglés...", 
        height=120
    )

    if st.button("Analizar Texto", type="primary", use_container_width=True):
        if review_input.strip() == "":
            st.warning("Por favor, escribe un texto válido antes de analizar.")
        else:
            with st.spinner("Analizando patrones de lenguaje..."):
                sentimiento, p_pos, p_neg = predecir_sentimiento(review_input, modelo_doc2vec, modelo_rf)
                
                st.markdown("### 📊 Resultado del Análisis")
                
                if sentimiento == "Positivo":
                    st.success(f"### Resultado: **{sentimiento}**")
                else:
                    st.error(f"### Resultado: **{sentimiento}**")
                
                st.write("")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Confianza Positiva", value=f"{p_pos * 100:.1f}%")
                    st.progress(float(p_pos))
                    
                with col2:
                    st.metric(label="Confianza Negativa", value=f"{p_neg * 100:.1f}%")
                    st.progress(float(p_neg))
                
                st.divider()
                
                st.markdown("#### 📈 Distribución de Probabilidad")
                
                df_grafico = pd.DataFrame({
                    'Polaridad': ['Negativo 🔴', 'Positivo 🟢'],
                    'Porcentaje': [p_neg * 100, p_pos * 100]
                })
                
                st.bar_chart(data=df_grafico, x='Polaridad', y='Porcentaje', use_container_width=True)

else:
    st.info("Esperando que los modelos locales estén disponibles en la raíz del proyecto.")