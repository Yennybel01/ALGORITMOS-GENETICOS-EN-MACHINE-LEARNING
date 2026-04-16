# 🧬 Algoritmos Genéticos en Machine Learning

Implementación de tres aplicaciones de Algoritmos Genéticos (AG) en el contexto del aprendizaje automático, usando el dataset **Mushroom Classification** (UCI/Kaggle).

> **Dataset:** 8 124 registros · 22 características categóricas · Clasificación binaria: comestible vs. venenoso

---

## 📁 Estructura del repositorio

```
├── feature_selection.py       # AG para selección de características
├── hyperparameter_opt.py      # AG para optimización de hiperparámetros
├── neuroevolution.py          # AG para neuroevolución (arquitectura MLP)
└── README.md
```

---

## 🔬 Experimentos

### 1. Selección de Características (`feature_selection.py`)

Identifica el subconjunto óptimo de las 22 features que maximiza la precisión de un Random Forest, penalizando el exceso de variables.

| Parámetro | Valor |
|-----------|-------|
| Cromosoma | Binario (22 genes) |
| Función de aptitud | Accuracy CV-5 con penalización α·(k/n) |
| Población / Generaciones | 20 / 15 |
| Cruzamiento / Mutación | 0.80 / 0.05 (bit-flip) |

**Resultado esperado:** subconjunto de 6–10 features con reducción de dimensionalidad del 50–70 %.

---

### 2. Optimización de Hiperparámetros (`hyperparameter_opt.py`)

Optimiza automáticamente 4 hiperparámetros del Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `max_features`) explorando 1 680 combinaciones posibles.

| Parámetro | Valor |
|-----------|-------|
| Cromosoma | Entero (4 genes) |
| Función de aptitud | Accuracy CV-5 |
| Población / Generaciones | 20 / 15 |
| Cruzamiento / Mutación | 0.80 / 0.20 |

**Resultado esperado:** configuración con accuracy ≥ defaults de scikit-learn.

---

### 3. Neuroevolución (`neuroevolution.py`)

Usa el AG como meta-optimizador para encontrar la arquitectura óptima de una red MLP, sin acceder a gradientes — solo observa la métrica final (caja negra). Incluye caché para evitar re-evaluar arquitecturas repetidas.

| Parámetro | Valor |
|-----------|-------|
| Cromosoma | Entero (7 genes) |
| Espacio de búsqueda | 5 184 arquitecturas |
| Función de aptitud | Accuracy CV-3 |
| Población / Generaciones | 10 / 8 |
| Cruzamiento / Mutación | 0.85 / 0.15 |

**Baseline:** red (100,) relu, lr=0.001, max_iter=200 → accuracy ≈ 100 % en test.

---

## ⚙️ Ciclo evolutivo común

Todos los experimentos comparten el mismo pipeline AG:

```
Inicialización → Evaluación (aptitud) → Selección por torneo (k=3)
      → Cruzamiento de un punto → Mutación → Elitismo → Nueva generación
```

---

## 🚀 Instalación y uso

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Instalar dependencias
pip install scikit-learn pandas numpy

# Descargar el dataset (Kaggle)
# https://www.kaggle.com/datasets/uciml/mushroom-classification
# Guardar como mushrooms.csv en la raíz del proyecto

# Ejecutar cada experimento
python feature_selection.py
python hyperparameter_opt.py
python neuroevolution.py
```

---

## 📊 Comparativa general

| Criterio | Feature Selection | Hyperparameter Opt. | Neuroevolution |
|----------|:-----------------:|:-------------------:|:--------------:|
| Cromosoma | Binario (22) | Entero (4) | Entero (7) |
| Modelo | Random Forest | Random Forest | MLP |
| Validación | CV-5 | CV-5 | CV-3 |
| Generaciones | 15 | 15 | 8 |
| Población | 20 | 20 | 10 |
| Beneficio | Reduce dimensiones | Mejor config RF | Mejor arquitectura |

---

## 👩‍💻 Autoras

- Centeno Mamani, Mirian Lucero  
- Colquehuanca Colquehuanca, Jhosemy Brissett  
- Hancco Vilca, Yennybel Rocio  
- Mamani Apaza, Sadith Leidy  

**Curso:** Aprendizaje de Máquina — FIEES, Universidad Nacional del Altiplano  
**Docente:** Ing. Fernandez Chambi, Mayenka · Puno, Perú · 2026
