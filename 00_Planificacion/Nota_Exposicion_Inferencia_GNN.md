# Nota de Exposicion: Inferencia en Vivo con GNN

## Idea corta

En el notebook, un texto nuevo se convierte con el mismo TF-IDF usado en entrenamiento, se conecta a sus `k` vecinos mas similares del grafo y luego la GNN predice su clase.

## Como explicarlo en clase

1. Se limpia el texto nuevo.
2. Se vectoriza con el mismo `TfidfVectorizer` del entrenamiento.
3. Se buscan los `k` vecinos mas similares en el grafo textual.
4. Se agrega el nuevo nodo al grafo conectado con esos vecinos.
5. La GNN hace message passing y produce la clase del nodo nuevo.

## Pregunta tipica: "¿Hay que pasar por todo el grafo?"

La respuesta correcta es: no necesariamente.

Si el modelo tiene 2 capas GraphSAGE, la prediccion del nodo nuevo depende sobre todo de su vecindario relevante hasta 2 saltos. Por eso:

- Recalcular todo el grafo es una implementacion simple y valida.
- Usar solo el subgrafo local tambien es correcto.
- El enfoque local suele ser mejor en eficiencia y es mas cercano a como se implementaria en un entorno real.

## Frase segura para decirle al profesor

"Mi version base puede rehacer la inferencia sobre el grafo completo por simplicidad conceptual, pero teoricamente no es necesario. Como GraphSAGE propaga informacion por vecinos y mi modelo tiene 2 capas, basta con inferir sobre el subgrafo local alcanzable en 2 saltos desde el nodo nuevo. Eso mantiene la logica de la GNN y reduce mucho el costo computacional."

## Por que este approach es defendible

- Sigue siendo una inferencia GNN real.
- No reemplaza el modelo por una heuristica distinta.
- Respeta el alcance del message passing segun el numero de capas.
- Explica bien la diferencia entre una version didactica y una version eficiente.

## Respuesta tecnica corta

"En una GNN de 2 capas, el embedding final de un nodo depende de su entorno de hasta 2 hops. Por eso, para un nodo nuevo conectado por k-NN, se puede extraer su ego-subgrafo relevante y correr la inferencia solo ahi, sin perder correccion respecto a la dependencia estructural del modelo."
