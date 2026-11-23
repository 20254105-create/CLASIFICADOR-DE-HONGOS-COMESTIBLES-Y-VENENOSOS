"""
Clasificador de Hongos: Comestibles y Venenosos
Aplicación académica con Streamlit y un modelo Random Forest exportado desde Orange.

IMPORTANTE:
- El archivo modelo.pkcls debe ser el modelo original exportado desde Orange Data Mining.
- El modelo se carga mediante pickle. Solo debe utilizarse con archivos de modelo de confianza.
- Las predicciones son exclusivamente educativas y no deben utilizarse para decidir si un hongo
  silvestre es seguro para el consumo.
"""

from pathlib import Path
import pickle

import numpy as np
import streamlit as st

# Importaciones de Orange necesarias para registrar las clases utilizadas
# al serializar el modelo .pkcls.
try:
    import Orange
    from Orange.base import Model
    import Orange.classification.random_forest  # noqa: F401
    import Orange.preprocess.impute  # noqa: F401
    import Orange.preprocess.transformation  # noqa: F401
except Exception as exc:
    Orange = None
    Model = None
    ORANGE_IMPORT_ERROR = exc
else:
    ORANGE_IMPORT_ERROR = None


DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
RUTA_MODELO = DIRECTORIO_PROYECTO / "modelo.pkcls"

# Categorías observadas en Mushroom_Dataset.xlsx.
OPCIONES_CATEGORICAS = {
    "cap-shape": ["Convexa", "Plana", "Campana"],
    "cap-surface": ["Estriada", "Brillante", "Pegajosa", "Escamosa", "Lisa"],
    "cap-color": ["Naranja", "Blanco", "Amarillo"],
    "gill-attachment": ["Libre", "Adherido", "Adnato", "Poros"],
    "gill-color": ["Blanco", "Amarillo", "Negro"],
    "stem-color": ["Blanco", "Amarillo"],
    "ring-type": ["Estriado", "Colgante", "Grande", "Zona"],
    "season": ["Invierno", "Verano", "Otoño", "Primavera"],
}

# Límites y valores iniciales calculados a partir del Excel entregado.
# Se limita el formulario al intervalo observado en el conjunto de datos.
CONFIG_NUMERICA = {
    "cap-diameter": {
        "min": 2.19, "max": 18.89, "default": 7.09, "step": 0.01
    },
    "stem-height": {
        "min": 4.78, "max": 19.46, "default": 7.97, "step": 0.01
    },
    "stem-width": {
        "min": 5.47, "max": 19.96, "default": 13.02, "step": 0.01
    },
}

NOMBRES_ES = {
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
def cargar_modelo(ruta_modelo: str):
    """Carga el modelo Orange exportado en formato .pkcls."""
    if Orange is None:
        raise RuntimeError(
            "No se pudo importar Orange3. Revisa la instalación de las dependencias. "
            f"Detalle: {ORANGE_IMPORT_ERROR}"
        )

    ruta = Path(ruta_modelo)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo '{ruta.name}' en la carpeta de la aplicación."
        )

    # pickle.load es necesario porque .pkcls es el formato de serialización de Orange.
    # No se debe cargar aquí ningún archivo .pkcls de procedencia desconocida.
    with ruta.open("rb") as archivo:
        modelo = pickle.load(archivo)

    if not hasattr(modelo, "skl_model") or not hasattr(modelo, "original_domain"):
        raise TypeError(
            "El archivo no parece contener el clasificador Random Forest esperado de Orange."
        )

    if not hasattr(modelo, "domain") or modelo.domain.class_var is None:
        raise ValueError("El modelo no contiene una variable objetivo de clasificación.")

    return modelo


def construir_fila_entrada(modelo, valores_usuario):
    """Codifica y normaliza las entradas usando las transformaciones guardadas en Orange."""
    atributos = list(modelo.original_domain.attributes)
    if not atributos:
        raise ValueError("El dominio original del modelo no contiene variables predictoras.")

    def buscar_normalizador(variable):
        """Localiza el objeto Normalizer dentro de la cadena de transformaciones Orange."""
        pendientes = [getattr(variable, "compute_value", None)]
        visitados = set()
        while pendientes:
            actual = pendientes.pop()
            if actual is None or id(actual) in visitados:
                continue
            visitados.add(id(actual))
            if actual.__class__.__name__ == "Normalizer":
                return actual
            for atributo in ("variable", "compute_value"):
                siguiente = getattr(actual, atributo, None)
                if siguiente is not None:
                    pendientes.append(siguiente)
        return None

    fila = []
    for variable in atributos:
        nombre = variable.name
        if nombre not in valores_usuario:
            raise ValueError(f"No se encontró el valor ingresado para '{nombre}'.")

        normalizador = buscar_normalizador(variable)
        if normalizador is None:
            raise ValueError(
                f"El modelo no conserva un normalizador identificable para '{nombre}'. "
                "No se realizará una conversión aproximada."
            )

        # La variable que alimenta al Normalizer conserva el tipo original.
        variable_intermedia = getattr(normalizador, "variable", None)
        identidad = getattr(variable_intermedia, "compute_value", None)
        variable_original = getattr(identidad, "variable", None)
        valor_usuario = valores_usuario[nombre]

        # Si la variable original tiene 'values', era categórica en el dataset.
        categorias = getattr(variable_original, "values", None)
        if categorias is not None:
            categorias = list(categorias)
            if valor_usuario not in categorias:
                raise ValueError(
                    f"La categoría '{valor_usuario}' no es válida para '{nombre}'. "
                    f"Categorías del modelo: {', '.join(categorias)}"
                )
            valor_base = float(categorias.index(valor_usuario))
        else:
            valor_base = float(valor_usuario)

        offset = float(normalizador.offset)
        factor = float(normalizador.factor)
        fila.append((valor_base - offset) * factor)

    return fila


def obtener_prediccion(modelo, valores_usuario):
    """Ejecuta la predicción y obtiene las probabilidades si están disponibles."""
    fila = construir_fila_entrada(modelo, valores_usuario)

    # Se crea una tabla sin variable objetivo. Orange se encarga de transformar
    # los datos desde el dominio original hasta el dominio usado por el modelo.
    dominio_entrada = Orange.data.Domain(modelo.original_domain.attributes)
    tabla_entrada = Orange.data.Table.from_list(dominio_entrada, [fila])

    prediccion, probabilidades = modelo(tabla_entrada, ret=Model.ValueProbs)

    variable_objetivo = modelo.domain.class_var
    clases = list(variable_objetivo.values)

    prediccion_array = np.asarray(prediccion).reshape(-1)
    indice_predicho = int(round(float(prediccion_array[0])))

    if indice_predicho < 0 or indice_predicho >= len(clases):
        raise ValueError("El modelo devolvió una clase que no coincide con su variable objetivo.")

    nombre_clase = clases[indice_predicho]

    probabilidades_array = np.asarray(probabilidades, dtype=float)
    if probabilidades_array.ndim == 1:
        probabilidades_array = probabilidades_array.reshape(1, -1)

    probabilidades_fila = probabilidades_array[0]

    if len(probabilidades_fila) != len(clases):
        raise ValueError(
            "La cantidad de probabilidades devueltas no coincide con las clases del modelo."
        )

    return nombre_clase, dict(zip(clases, probabilidades_fila))


# ---------------------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Clasificador de Hongos",
    page_icon="🍄",
    layout="wide",
)

st.title("🍄 Clasificador de Hongos")
st.subheader("Identificación académica: Comestible o Venenoso")

st.markdown(
    """
    Esta aplicación utiliza un modelo **Random Forest entrenado en Orange Data Mining**
    para clasificar hongos a partir de sus características físicas.
    """
)

st.info(
    "Aviso académico: esta herramienta es exclusivamente educativa. "
    "Sus resultados no garantizan la identificación correcta de un hongo y "
    "**no deben utilizarse para decidir si un hongo silvestre es seguro para consumir**."
)

with st.expander("¿Cómo funciona esta aplicación?"):
    st.write(
        "El usuario ingresa las características físicas del hongo. La aplicación "
        "envía esos datos al modelo Random Forest exportado desde Orange. El modelo "
        "conserva su dominio y sus transformaciones para preparar los datos antes "
        "de realizar la predicción."
    )

# Carga del modelo y comprobación inicial.
try:
    modelo = cargar_modelo(str(RUTA_MODELO))
    st.sidebar.success("Modelo Random Forest cargado correctamente")
    st.sidebar.write(f"**Archivo:** {RUTA_MODELO.name}")
    st.sidebar.write(f"**Algoritmo:** {getattr(modelo, 'name', 'Random Forest')}")
    st.sidebar.write(
        f"**Variable objetivo:** {modelo.domain.class_var.name}"
    )
    st.sidebar.write(
        f"**Clases:** {', '.join(modelo.domain.class_var.values)}"
    )
except Exception as error:
    st.error("No se pudo cargar el modelo de clasificación.")
    st.exception(error)
    st.stop()

st.divider()
st.header("1. Ingrese las características del hongo")
st.caption(
    "Complete los campos según las características observadas. "
    "Los valores numéricos están limitados al intervalo encontrado en el dataset."
)

with st.form("formulario_hongo"):
    columna_izquierda, columna_central, columna_derecha = st.columns(3)

    with columna_izquierda:
        st.markdown("**Sombrero**")
        cap_diameter = st.number_input(
            "Diámetro del sombrero",
            min_value=CONFIG_NUMERICA["cap-diameter"]["min"],
            max_value=CONFIG_NUMERICA["cap-diameter"]["max"],
            value=CONFIG_NUMERICA["cap-diameter"]["default"],
            step=CONFIG_NUMERICA["cap-diameter"]["step"],
            format="%.2f",
            help="Variable cap-diameter del conjunto de datos.",
        )
        cap_shape = st.selectbox(
            "Forma del sombrero",
            OPCIONES_CATEGORICAS["cap-shape"],
        )
        cap_surface = st.selectbox(
            "Superficie del sombrero",
            OPCIONES_CATEGORICAS["cap-surface"],
        )
        cap_color = st.selectbox(
            "Color del sombrero",
            OPCIONES_CATEGORICAS["cap-color"],
        )

    with columna_central:
        st.markdown("**Láminas y tallo**")
        gill_attachment = st.selectbox(
            "Unión de las láminas",
            OPCIONES_CATEGORICAS["gill-attachment"],
        )
        gill_color = st.selectbox(
            "Color de las láminas",
            OPCIONES_CATEGORICAS["gill-color"],
        )
        stem_height = st.number_input(
            "Altura del tallo",
            min_value=CONFIG_NUMERICA["stem-height"]["min"],
            max_value=CONFIG_NUMERICA["stem-height"]["max"],
            value=CONFIG_NUMERICA["stem-height"]["default"],
            step=CONFIG_NUMERICA["stem-height"]["step"],
            format="%.2f",
            help="Variable stem-height del conjunto de datos.",
        )
        stem_width = st.number_input(
            "Ancho del tallo",
            min_value=CONFIG_NUMERICA["stem-width"]["min"],
            max_value=CONFIG_NUMERICA["stem-width"]["max"],
            value=CONFIG_NUMERICA["stem-width"]["default"],
            step=CONFIG_NUMERICA["stem-width"]["step"],
            format="%.2f",
            help="Variable stem-width del conjunto de datos.",
        )

    with columna_derecha:
        st.markdown("**Otras características**")
        stem_color = st.selectbox(
            "Color del tallo",
            OPCIONES_CATEGORICAS["stem-color"],
        )
        ring_type = st.selectbox(
            "Tipo de anillo",
            OPCIONES_CATEGORICAS["ring-type"],
        )
        season = st.selectbox(
            "Estación del año",
            OPCIONES_CATEGORICAS["season"],
        )

    enviar = st.form_submit_button(
        "🍄 Clasificar hongo",
        type="primary",
        use_container_width=True,
    )

if enviar:
    datos_ingresados = {
        "cap-diameter": cap_diameter,
        "cap-shape": cap_shape,
        "cap-surface": cap_surface,
        "cap-color": cap_color,
        "gill-attachment": gill_attachment,
        "gill-color": gill_color,
        "stem-height": stem_height,
        "stem-width": stem_width,
        "stem-color": stem_color,
        "ring-type": ring_type,
        "season": season,
    }

    try:
        with st.spinner("Analizando las características con Random Forest..."):
            clase, probabilidades = obtener_prediccion(modelo, datos_ingresados)

        st.divider()
        st.header("2. Resultado de la clasificación")

        clase_normalizada = str(clase).strip().lower()

        if clase_normalizada == "comestible":
            st.success("Clasificación obtenida: COMESTIBLE")
        elif clase_normalizada == "venenoso":
            st.error("Clasificación obtenida: VENENOSO")
        else:
            st.warning(f"Clasificación obtenida: {str(clase).upper()}")

        st.markdown("### Porcentajes de predicción")

        columnas_resultado = st.columns(len(probabilidades))
        for columna, (nombre, probabilidad) in zip(columnas_resultado, probabilidades.items()):
            porcentaje = float(probabilidad) * 100
            with columna:
                st.metric(nombre.upper(), f"{porcentaje:.2f}%")
                st.progress(
                    min(max(float(probabilidad), 0.0), 1.0),
                    text=f"{porcentaje:.2f}%"
                )

        st.caption(
            "Los porcentajes corresponden a las probabilidades estimadas por el "
            "clasificador Random Forest. No representan una garantía de certeza "
            "ni una evaluación de seguridad alimentaria."
        )

        st.markdown("### Características ingresadas")
        resumen = {
            NOMBRES_ES[nombre]: valor
            for nombre, valor in datos_ingresados.items()
        }
        st.dataframe(
            [{"Característica": nombre, "Valor ingresado": valor}
             for nombre, valor in resumen.items()],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:
        st.error(
            "No se pudo realizar la predicción. No se ha generado una clasificación "
            "porque los datos no pudieron asociarse de forma segura con el dominio del modelo."
        )
        st.exception(error)

st.divider()
st.caption(
    "Proyecto académico de Inteligencia Artificial Aplicada | "
    "Dataset: Mushroom_Dataset.xlsx | Modelo: Random Forest (Orange Data Mining)"
)
