# Projeto ia de reconhecimento

Projeto para criação de uma ia que detecte objetos e pessoas usando webcam.

## Instalar dependencias 
```bash
    pip install -r requirements.txt
```

### tambem e preciso baixar um modelo de deteção, no caso estou usando versao 512x512.

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## Para compilar 
crie uma pasta **tensor** para compilar
```bash
    python setup.py build_ext --inplace
```

## run 
```bash
    python .
```
