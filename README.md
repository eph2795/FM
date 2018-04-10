# FM
Simple Factorization Machine implementation

Использование:

cd src

make all

./main --data ../../datasets/rcv1/rcv1.vw --test ../../datasets/rcv1/rcv1.test.vw --model fm --loss logistic --factors_size 5 --use_offset --passes 10 --learning_rate 0.01

# Что нужно сделать

В техническом плане:
* Feature hashing(альтернатива для OHE)
* Dump модели
* Запись predictions в текстовый файл
* Ускорить парсинг входного файла
* Арифметические операции над sparse-векторами

В алгоритмическом плане:
* Поддержка регуляризации
* Кеширование(?) скалярных произведений
* Добавление абстракции в класс оптимизации
* Инициализация весов
* Более хитрые sparse-векторы

В содержательном плане:
* Ещё методы оптимизации
* Повышение стабильности обучения
* Hogwild?
* FFM?
