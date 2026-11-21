"""Clasificador académico de hongos con Streamlit y un modelo de Orange."""
from pathlib import Path
import pickle

import numpy as np
import streamlit as st

try:
    import Orange
    from Orange.base import Model
    # Registrar clases que pueden estar referenciadas por el archivo serializado.
    import Orange.classification.random_forest  # noqa: F401
    import Orange.preprocess.impute  # noqa: F401
    import Orange.preprocess.transformation  # noqa: F401
except Exception as exc:
    Orange = None
    Model = None
    ORANGE_IMPORT_ERROR = exc
else:
    ORANGE_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo.pkcls"

# Límites observados en Mushroom_Dataset.xlsx (1489 registros).
NUMERIC_CONFIG = {
    "cap-diameter": {"min": 2.19, "max": 18.89, "default": 7.09},
    "stem-height": {"min": 4.78, "max": 19.46, "default": 7.97},
    "stem-width": {"min": 5.47, "max": 19.96, "default": 13.02},
}
LABELS = {
    "cap-diameter": "Diámetro del sombrero",
    "cap-shape": "Forma del sombrero",
    "cap-surface": "Superficie del sombrero",
    "cap-color": "Color del sombrero",
    "gill-attachment": "Unión de las láminas",
    "gill-color": "Color de las láminas",
    "stem-height": "Altura del tallo",
    "stem-width": "Ancho del tallo",
    "stem-color": "Color del tallo",
    "ring-type": "Tipo de anillo",
    "season": "Estación del año",
}

@st.cache_resource(show_spinner="Cargando el modelo Random Forest...")
def load_model(path_string):
    if Orange is None:
        raise RuntimeError(
            "No se pudo importar Orange3. Comprueba las dependencias. "
            f"Detalle técnico: {ORANGE_IMPORT_ERROR}"
        )
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError("No se encontró modelo.pkcls junto a app.py.")
    # Cargar únicamente el archivo .pkcls de confianza exportado por el propio proyecto.
    with path.open("rb") as file:
        model = pickle.load(file)
    if not hasattr(model, "original_domain") or not hasattr(model, "domain"):
        raise TypeError("El archivo no tiene la estructura esperada de un modelo Orange.")
    if model.domain.class_var is None:
        raise ValueError("El modelo no contiene una variable objetivo.")
    if not model.original_domain.attributes:
        raise ValueError("El modelo no contiene atributos de entrada.")
    return model


def get_input_schema(model):
    """Obtiene las categorías discretas del dominio del modelo, no de listas inventadas."""
    schema = {}
    for var in model.original_domain.attributes:
        name = var.name
        if name in LABELS:
            if var.is_discrete:
                schema[name] = {"kind": "categorical", "options": list(var.values)}
            else:
                schema[name] = {"kind": "numeric"}
        elif "=" in name:
            base, category = name.split("=", 1)
            if base in LABELS:
                schema.setdefault(base, {"kind": "categorical", "options": []})
                if category not in schema[base]["options"]:
                    schema[base]["options"].append(category)
        else:
            raise ValueError(
                f"El atributo '{name}' del modelo no coincide con las 11 variables "
                "del formulario. Revisa el preprocesamiento exportado desde Orange."
            )
    missing = set(LABELS) - set(schema)
    if missing:
        raise ValueError("El modelo no contiene estas variables: " + ", ".join(sorted(missing)))
    return schema


def build_input_row(model, user_values):
    row = []
    for var in model.original_domain.attributes:
        name = var.name
        if name in user_values:
            value = user_values[name]
            if var.is_discrete:
                if value not in var.values:
                    raise ValueError(
                        f"'{value}' no es una categoría válida para '{name}'. "
                        f"Opciones del modelo: {', '.join(var.values)}"
                    )
                value = var.to_val(value)
            else:
                value = float(value)
            row.append(value)
            continue
        if "=" in name:
            base, category = name.split("=", 1)
            if base in user_values:
                row.append(float(user_values[base] == category))
                continue
        raise ValueError(f"No se pudo completar el atributo del modelo: {name}")
    return row


def predict(model, user_values):
    row = build_input_row(model, user_values)
    input_domain = Orange.data.Domain(model.original_domain.attributes)
    data = Orange.data.Table.from_list(input_domain, [row])
    predictions, probabilities = model(data, ret=Model.ValueProbs)

    class_var = model.domain.class_var
    classes = list(class_var.values)
    pred_array = np.asarray(predictions).reshape(-1)
    predicted_index = int(round(float(pred_array[0])))
    if not 0 <= predicted_index < len(classes):
        raise ValueError("El modelo devolvió un índice de clase no válido.")

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    probs = probs[0]
    if len(probs) != len(classes):
        raise ValueError("Las probabilidades no coinciden con las clases del modelo.")
    return classes[predicted_index], dict(zip(classes, probs.tolist()))


st.set_page_config(page_title="Clasificador de Hongos", page_icon="🍄", layout="wide")
st.title("🍄 Clasificador de Hongos")
st.subheader("Clasificación académica: Comestible o Venenoso")
st.write(
    "Ingresa las características físicas para obtener la predicción del modelo "
    "Random Forest entrenado en Orange Data Mining."
)
st.warning(
    "Uso exclusivamente académico. Esta aplicación no identifica de forma fiable "
    "hongos silvestres ni debe utilizarse para decidir si son seguros para consumir."
)

try:
    model = load_model(str(MODEL_PATH))
    schema = get_input_schema(model)
except Exception as error:
    st.error("No se pudo preparar el modelo para la clasificación.")
    st.exception(error)
    st.stop()

st.sidebar.success("Modelo cargado")
st.sidebar.write(f"**Archivo:** {MODEL_PATH.name}")
st.sidebar.write(f"**Objetivo:** {model.domain.class_var.name}")
st.sidebar.write("**Clases:** " + ", ".join(model.domain.class_var.values))

st.header("1. Características del hongo")
with st.form("mushroom_form"):
    cols = st.columns(3)
    user_values = {}
    groups = [
        ["cap-diameter", "cap-shape", "cap-surface", "cap-color"],
        ["gill-attachment", "gill-color", "stem-height", "stem-width"],
        ["stem-color", "ring-type", "season"],
    ]
    for col, fields in zip(cols, groups):
        with col:
            for field in fields:
                label = LABELS[field]
                info = schema[field]
                if info["kind"] == "categorical":
                    options = info["options"]
                    if not options:
                        st.error(f"El modelo no ofrece categorías para {label}.")
                        st.stop()
                    user_values[field] = st.selectbox(label, options, key=f"input_{field}")
                else:
                    cfg = NUMERIC_CONFIG[field]
                    user_values[field] = st.number_input(
                        label, min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                        value=float(cfg["default"]), step=0.01, format="%.2f",
                        key=f"input_{field}"
                    )
    submitted = st.form_submit_button("🍄 Clasificar hongo", type="primary", use_container_width=True)

if submitted:
    try:
        with st.spinner("Analizando las características..."):
            predicted_class, class_probabilities = predict(model, user_values)
        st.header("2. Resultado")
        normalized = str(predicted_class).strip().casefold()
        if normalized == "comestible":
            st.success("Clasificación del modelo: COMESTIBLE")
        elif normalized == "venenoso":
            st.error("Clasificación del modelo: VENENOSO")
        else:
            st.info(f"Clasificación del modelo: {str(predicted_class).upper()}")
        st.subheader("Probabilidades estimadas")
        result_cols = st.columns(len(class_probabilities))
        for col, (class_name, probability) in zip(result_cols, class_probabilities.items()):
            pct = float(probability) * 100
            with col:
                st.metric(str(class_name).upper(), f"{pct:.2f}%")
                st.progress(min(max(float(probability), 0.0), 1.0))
        st.caption("Las probabilidades son estimaciones del modelo, no garantías de certeza ni de seguridad alimentaria.")
        st.subheader("Datos ingresados")
        st.dataframe(
            [{"Característica": LABELS[key], "Valor": value} for key, value in user_values.items()],
            use_container_width=True, hide_index=True
        )
    except Exception as error:
        st.error("No se pudo completar la predicción. No se mostrará un resultado no validado.")
        st.exception(error)

st.divider()
st.caption("Proyecto académico de Inteligencia Artificial Aplicada | Dataset: Mushroom_Dataset.xlsx | Modelo: Random Forest (Orange)")
