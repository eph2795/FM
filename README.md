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

./main --train ../../datasets/rcv1/rcv1.vw --validation ../../datasets/rcv1/rcv1.test.vw --test ../../datasets/rcv1/rcv1.test.vw --dump ../../datasets/model.bin --predict ../../datasets/rcv1/pred.txt --model fm --loss mse --factors_size 20 --use_offset --passes 50 --optimizer sgd --learning_rate 0.001 --reg_type l2 -C0 1 -Cw 0.000001 -Cv 0.001 --index_type hash --bits_number 15


### Сравнение c vw - TODO

Обучение FM:

./main --train ../../datasets/avazu/train.vw --validation ../../datasets/avazu/val.vw --test ../../datasets/avazu/test.vw --dump ../../datasets/model.bin --predict ../../datasets/avazu/pred.txt --model linear --loss mse --factors_size 10 --use_offset --passes 1 --optimizer sgd --learning_rate 0.005 --reg_type l2 -C0 1 -Cw 0.000001 -Cv 0.001 --index_type hash --bits_number 15

Обучение vw:

vw -b 15 --passes 1 --sgd --l2 0.001 -d ../../datasets/avazu/train.vw --link logistic --loss_function=logistic -f ../../datasets/avazu/model.vw

vw -d ../../datasets/avazu/test.vw -t -i ../../datasets/model.vw --loss_function=logistic -r ../../datasets/avazu/preds_vw.txt

Для примерочного сравнения я выбрал датасет rcv1, потому что в нём 700к примеров(достаточно много, чтобы проверка несла какую-то информацию, и достаточно мало, чтобы быстро получать результат), есть категориальные и численные фичи, он доступен в vw формате. 

Обучал и в случае vw, и в случае своей реализации 10 итераций обычного sgd с mse-loss и learning_rate=0.1. 

Для vw получилось ~0.042 на train и ~0.044 на test.
Для моей линейной регрессии ~0.054 на train и ~0.063 на test.
Для моей FM ~0.051 и ~0.136 на test.

Сейчас линейная модель(без учета чтения данных) тратит на 10 итераций SGD порядка 10сек, а FM c 5 факторами - 180сек, что много, требуется оптимизация кода.

Понятно, что FM даёт прирост качества на train, но переобучается, поэтому проводить более подробный анализ без регуляризации смысла нет. Так же для FM требуется некоторая инициализация весов(с нулевыми не обучается, с неправильными константами расходится) и достаточно маленький lr(я брал 0.01, c 0.1 расходится). Это решается адаптивными методами и более аккуратной инициализацией весов.
