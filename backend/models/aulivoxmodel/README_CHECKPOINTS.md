Este directorio almacena checkpoints generados durante el entrenamiento.
Se sugiere no se subirlos al repositorio debido a por su tamaño.

Por lo que se declara en gitignore. la omisión del siguiente contenido:
* Directorios "checkpoint-x"
* Archivos en formato:
	*.ckpt
	*.pt
	*.pth
	*.bin