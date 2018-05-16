# FM
Реализация Factorization Machine

### Структура репозитория

#### Materials:
> Статьи по теме

#### src:
> Исходный код и makefile

#### utils: 
> Python-скрипты для экспериментов и конвертации форматов(чисто для удобства)

Реализовано:
* Парсинг vw формата, упрощённый вариант: у метки нет веса и лейбла, один namespace, фиче должно соответствовать float значение
* Индексация в матрице признаков двух типов: OHE и hashing с задаваемым количеством фичей
* Две модели - линейная регрессия и FM https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
* Две функции потерь - mse и logistic
* Независимая sparse регуляризация l1 и l2 для разных видов параметров(w0, w, v)
* Оптимизация с помощью SGD
* Оптимизация с помощью ALS https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2011-Context_Aware.pdf
* Адаптивная регуляризация https://github.com/buptjz/Factorization-Machine/blob/master/paper/Steffen%20Rendle%20(2012)%20Learning%20Recommender%20Systems%20with%20Adaptive%20Regularization.pdf


### Используемые в коде абстракции

#### Классы:
> SparseVector, X, Y - хранят выборку и метки

> DataReader - считывает входной файл в оперативную память, для OHE составляют пары фича-индекс(для хеширования препроцесса входного файла не будет)

> Optimizer - абстракция для метода оптимизации, реализован только SGD и ALS

> Model - абстракция для модели, предоставляет интерфейс для подсчета градиента и предикта, реализованы две модели - линейная и FM

> Loss - абстракция для функции потерь, предоставляет интерфейс для подсчета лосса и градиента

> SparseWeights и Weights - абстракция для весов, отвечает за подсчет update'а в методе оптимизации и обновление весов модели

> Regularizer - отвечает за нужную функцию регуляризации

#### Нюансы и недостатки реализации:

> Используются sparse апдейты градиeнтов и sparse регуляризация.

> Для ускорения вычисления fm-предиктов используется предподсчет скалярных произведений.

> Веса для квадратичных членов инициализируются из нормального распределения N(0, 0.001).

> Я использовал абстракции только для наследования интерфейсов. Для удобной работы нужно оперировать указателями абстрактных классов(например, внутри класса LinearModel должен бы быть *Weights, а не LinearWeights). Из-за этого, например, не получается отнаследовать FMModel: LinearModel, и приходится дублировать часть кода линейной модели.

> Изначально взаимодействия классов планировались только для градиентов. С методами, отличными от градиентных, всё получается сложнее. Было бы эффективнее написать код в процедурном стиле.

> В рамках Hogwild! требуется, чтобы потоки вычисляли свои градиенты. У меня градиент хранится внутри модели, для реализации Hogwild! требовалось бы копировать модели, что странно и расходует память.

> В ALS для оптимизации весов требуются вычисления и по фичам(формат входных данных csc) и вычисление предиктов(для линейности по времени нужен формат csr), значит, данные дублируются.


### Пример использования

cd src

make all

./main --train ../../datasets/rcv1/rcv1.vw --validation ../../datasets/rcv1/rcv1.test.vw --test ../../datasets/rcv1/rcv1.test.vw --dump ../../datasets/model.bin --predict ../../datasets/rcv1/pred.txt --model fm --loss mse --factors_size 10 --use_offset --passes 50 --optimizer sgd --learning_rate 0.001 --reg_type l2 -C0 1 -Cw 0.000001 -Cv 0.001 --index_type hash --bits_number 15 --adaptive_reg


### Сравнение c vw 

#### Avazu

Исходныцй train был разделен в пропорции 3:1:1 и сформированы train, val и test выборки. Качество измерялось для test. 

Обучение линейной модели:

./main --train ../../datasets/avazu/train.vw --validation ../../datasets/avazu/val.vw --test ../../datasets/avazu/test.vw --dump ../../datasets/model.bin --predict ../../datasets/avazu/pred.txt --model linear --loss logistic --use_offset --passes 5 --optimizer sgd --learning_rate 0.005 --reg_type l2 -C0 1 -Cw 0.000001 --index_type hash --bits_number 15 --adaptive_reg

Результат:
0.39977

Обучение FM:

./main --train ../../datasets/avazu/train.vw --validation ../../datasets/avazu/val.vw --test ../../datasets/avazu/test.vw --dump ../../datasets/model.bin --predict ../../datasets/avazu/pred.txt --model fm --loss logistic --factors_size 20 --use_offset --passes 5 --optimizer sgd --learning_rate 0.001 --reg_type l2 -C0 1 -Cw 0.000001 -Cv 0.00001 --index_type hash --bits_number 15

Результат:
0.399265

Обучение vw:

vw -b 15 --passes 5 --sgd --l2 0.000001 -d ../../datasets/avazu/train.vw --link logistic --loss_function=logistic -f ../../datasets/avazu/model.vw --cache_file ../../datasets/avazu/cache.vw

vw -d ../../datasets/avazu/test.vw -t -i ../../datasets/model.vw --loss_function=logistic -r ../../datasets/avazu/preds_vw.txt

Результат:
0.406582

#### Movielens

Исходныцй train был разделен в пропорции 3:1:1 и сформированы train, val и test выборки. Качество измерялось для test. 

Обучение линейной модели:

./main --train ../../datasets/ml-20m/train.vw --validation ../../datasets/ml-20m/val.vw --test ../../datasets/ml-20m/test.vw --dump ../../datasets/model.bin --predict ../../datasets/ml-20m/pred.txt --model linear --loss mse --use_offset --passes 5 --optimizer sgd --learning_rate 0.005 --reg_type l2 -C0 1 -Cw 0.000001 --index_type hash --bits_number 15 --adaptive_reg

Результат:
0.39977

Обучение FM:

./main --train ../../datasets/ml-20m/train.vw --validation ../../datasets/ml-20m/val.vw --test ../../datasets/ml-20m/test.vw --dump ../../datasets/model.bin --predict ../../datasets/ml-20m/pred.txt --model fm --loss mse --factors_size 10 --use_offset --passes 5 --optimizer sgd --learning_rate 0.001 --reg_type l2 -C0 1 -Cw 0.000001 -Cv 0.00001 --index_type hash --bits_number 15

Результат:
0.399265

Обучение vw:

vw -b 15 --passes 5 --sgd --l2 0.000001 -d ../../datasets/ml-20m/train.vw --loss_function=mse -f ../../datasets/ml-20m/model.vw --cache_file ../../datasets/ml-20m/cache.vw

vw -d ../../datasets/ml-20mtest.vw -t -i ../../datasets/model.vw --loss_function=mse -r ../../datasets/ml-20m/preds_vw.txt

Результат:
0.406582

