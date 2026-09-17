"""Desktop bundle helpers (ONNX encoder + numpy index).

Этот файл делает каталог desktop/ импортируемым как пакет, чтобы
app/backends.py мог использовать OnnxEncoder в light-режиме без
дублирования кода. Содержимое используется и для сборки exe.
"""
