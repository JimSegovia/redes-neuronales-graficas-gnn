# Estructura del Notebook — SentinelGrafo

## Formato de entrega
Un solo archivo `.ipynb` con dos partes claramente separadas.

---

## PARTE I: INFORME DEL PROYECTO
(Celdas markdown — sin código ejecutable, figuras como imágenes estáticas o referencias a la Parte II)

### 1. Portada
- Título: **SentinelGrafo — Clasificación de mensajes críticos en plataformas digitales mediante Redes Neuronales Gráficas**
- Curso: Inteligencia Artificial 2026-1, Sección 3, EPISW-FISI-UNMSM
- Profesor: Rolando A. Maguiña Pérez
- Integrantes (con código):
  - Segovia Valencia Jim Bryan
  - Morales Usca Andres
  - Chavez Cerna Joshua

### 2. Resumen
- Síntesis del proyecto (200-300 palabras): qué se hizo, con qué técnicas, sobre qué datasets, principales resultados.

### 3. Objetivo
- Objetivo general: aplicar GNNs a clasificación de texto, comparando con baseline clásico.
- Objetivos específicos (2-3).

### 4. Marco Teórico
- **4.1 Redes Neuronales Gráficas (GNNs)**: definición, paradigma de Message Passing (AGGREGATE + UPDATE).
- **4.2 GraphSAGE**: fundamentos, muestreo de vecinos, función de agregación, carácter inductivo.
- **4.3 GCN (Graph Convolutional Networks)**: convolución espectral de primer orden, diferencias con GraphSAGE.
- **4.4 Construcción de grafos textuales**: TF-IDF, similitud coseno, k-NN.
- **4.5 Comparativa con técnicas similares**: GAT (solo mención teórica), por qué se eligió GraphSAGE como principal.

### 5. Desarrollo y Aplicación
- **5.1 Pipeline general**: texto → limpieza → TF-IDF → grafo k-NN → GNN → clasificación.
- **5.2 Caso de estudio 1 — Disaster Tweets**:
  - Descripción del problema (clasificar tweets como desastre real vs figurado).
  - Origen del dataset (Kaggle NLP with Disaster Tweets).
  - Características: 6,542 muestras balanceadas, 2 clases, textos cortos en inglés.
- **5.3 Caso de estudio 2 — AG News**:
  - Descripción del problema (clasificar noticias en 4 categorías: World, Sports, Business, Sci/Tech).
  - Origen del dataset (AG News corpus).
  - Características: 24,000 muestras (6,000/clase), 4 clases, textos de mediana longitud.

### 6. Experimentos Computacionales
- Tabla comparativa de configuración por caso:

| Parámetro | Caso 1 (Disaster) | Caso 2 (AG News) |
|-----------|--------------------|-------------------|
| Muestras totales | 6,542 | 24,000 |
| Features TF-IDF | 3,000 | 5,000 |
| k vecinos (k-NN) | 5 | 5 |
| Número de clases | 2 | 4 |
| Modelos evaluados | GraphSAGE + GCN | GraphSAGE + GCN |
| Baseline | Regresión Logística sobre TF-IDF | Regresión Logística sobre TF-IDF |
| Split train/test | 80/20 estratificado | 80/20 estratificado |
| Capas GNN | 2 | 2 |
| Dimensión oculta | 64 | 64 |
| Dropout | 0.5 | 0.5 |
| Optimizador | Adam (lr=0.01, weight_decay=5e-4) | Adam (lr=0.01, weight_decay=5e-4) |
| Early stopping | paciencia=30, máx 200 épocas | paciencia=30, máx 200 épocas |
| Métricas | Accuracy, Precision, Recall, F1-macro | Accuracy, Precision, Recall, F1-macro |

### 7. Análisis de Resultados
- **7.1 Caso 1 — Disaster Tweets**:
  - Tabla de métricas (Baseline vs GraphSAGE vs GCN).
  - Matriz de confusión (imagen).
  - Curvas de entrenamiento (imagen).
  - Proyección t-SNE (imagen).
  - Interpretación: ¿el grafo ayudó? ¿qué modelo fue mejor?
- **7.2 Caso 2 — AG News**:
  - Tabla de métricas (Baseline vs GraphSAGE vs GCN).
  - Matriz de confusión (imagen).
  - Curvas de entrenamiento (imagen).
  - Proyección t-SNE (imagen).
  - Interpretación.
- **7.3 Comparativa global**:
  - Gráfico de barras comparativo Baseline vs GraphSAGE vs GCN para ambos casos (imagen).
  - Discusión: ¿cuándo conviene usar GNN sobre métodos clásicos en NLP?

### 8. Dificultades encontradas y soluciones
- Dificultad 1 + cómo se resolvió.
- Dificultad 2 + cómo se resolvió.
- (Inferencia con subgrafo local, balanceo del dataset AG News, etc.)

### 9. Conclusiones
- Qué se logró, qué se aprendió, limitaciones, trabajo futuro.

### 10. Bibliografía
- Hamilton et al. "Inductive Representation Learning on Large Graphs" (GraphSAGE).
- Veličković et al. "Graph Attention Networks" (GAT).
- Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks" (GCN).
- Documentación de PyTorch Geometric.
- Dataset Disaster Tweets (Kaggle).
- Dataset AG News.

---

## PARTE II: CÓDIGO FUENTE DOCUMENTADO
(Celdas markdown + código — el notebook ejecutable)

### Celda introductoria (markdown)
"Anexo: Código Fuente Documentado — A continuación se presenta la implementación completa del software SentinelGrafo..."

### Índice del código
1. Instalación de dependencias
2. Definición de rutas (auto-detección)
3. Pipeline común: limpieza de texto y construcción de grafos
4. Carga y preprocesamiento: Caso 1 — Disaster Tweets
5. Carga y preprocesamiento: Caso 2 — AG News
6. Visualización de la topología de los grafos
7. Baseline: Logistic Regression sobre vectores TF-IDF
8. Definición de modelos GNN: GraphSAGE y GCN
9. Función unificada de entrenamiento y evaluación
10. Entrenamiento de modelos para ambos casos
11. Visualización 1: Curvas de entrenamiento
12. Visualización 2: Matrices de confusión
13. Visualización 3: Proyección t-SNE de embeddings
14. Visualización 4: Diagrama esquemático de Message Passing
15. Visualización 5: Gráfico de barras comparativo
16. Dashboard interactivo — Demo de predicción en vivo
17. Tabla resumen final de resultados

---

## Notas
- La Parte I usa las **mismas figuras** que genera la Parte II. Para la entrega, se recomienda ejecutar primero todo el código y luego incrustar las imágenes resultantes en las celdas markdown del informe.
- El notebook se nombrará: `solTC1_Segovia-Jim_Morales-Andres_Chavez-Joshua.ipynb`.
- Las primeras celdas deben consignar título del proyecto + nombres con código (según instrucciones del TC1).
