# ML1

- [Метод опорных методов](#Метод-опорных-методов)
- [Метод главных компонент](#Метод-главных-компонент)
- [Нелинейные методы восстановления регрессии](#Нелинейные-методы-восстановления-регрессии)

# Логические алгоритмы классификаци

Пусть **ϕ: X → {0, 1}** — некоторый предикат, определённый на множестве объектов
X. Говорят, что предикат **ϕ** выделяет или покрывает (cover) объект **x**, если
**ϕ(x) = 1**. 

Предикат называют **закономерностью**, если он выделяет достаточно много
объектов какого-то одного класса **c**, и практически не выделяет объекты других
классов (более строгое определение будет дано ниже).
Особую ценность представляют закономерности, которые описываются простой
логической формулой. Их называют **правилами (rules)**. Процесс поиска правил по выборке
называют **извлечением знаний из данных (knowledge discovery)**. К знаниям
предъявляется особое требование — они должны быть интерпретируемы, то есть
понятны людям. На практике логические закономерности часто ищут в виде конъюнкций
небольшого числа элементарных высказываний. Именно в такой форме люди
привыкли выражать свой житейский и профессиональный опыт.

### Пример
Решается вопрос о целесообразности хирургической
операции. Закономерность: если возраст пациента выше 60 лет и ранее он
перенёс инфаркт, то операцию не делать — риск отрицательного исхода велик.

## Понятие информативности

### Эвристическое определение информативности

Интуитивно предикат **ϕ** тем более информативен, чем больше он выделяет объектов
«своего» класса **c ∈ Y** по сравнению с объектами всех остальных «чужих»
классов. Свои объекты называют также **позитивными (positive)**, а чужие — **негативными
(negative)**. 

Введём следующие обозначения:
**Pc** — число объектов класса c в выборке **Xℓ**;
**pc(ϕ)** — из них число объектов, для которых выполняется условие **ϕ(x) = 1**;
**Nc** — число объектов всех остальных классов **Y \ {c}** в выборке **Xℓ**;
**nc(ϕ)** — из них число объектов, для которых выполняется условие **ϕ(x) = 1**.
Введём обозначение **Ec** для доли негативных среди всех выделяемых объектов,
и **Dc** для доли выделяемых позитивных объектов:

![](11.png)

Предикат **ϕ(x)** будем называть **логической ε, δ-закономерностью** для
класса **c ∈ Y**, если **Ec(ϕ, Xℓ)<=ε** и **Dc(ϕ, Xℓ) >= δ** при заданных достаточно малом **ε**
и достаточно большом **δ** из отрезка **[0, 1]**.
Если **nc(ϕ) = 0**, то закономерность **ϕ** называется чистой или непротиворечивой.
Если **nc(ϕ) > 0**, то закономерность **ϕ** называется частичной

### Статистическое определение информативности

Пусть X — вероятностное пространство, выборка Xℓ — простая, то есть случайная,
независимая, одинаково распределённая (independent, identically distributed), y∗(x) и ϕ(x) — случайные величины. Допустим, справедлива гипотеза о независимости
событий {x: y∗(x) = c} и {x: ϕ(x) = 1}. Тогда вероятность реализации
пары (p, n) подчиняется гипергеометрическому распределению

![](12.png)

Информативность предиката **ϕ(x)** относительно класса **c ∈ Y** по выборке **Xℓ**

![](13.png)

Предикат ϕ(x) будем называть статистической закономерностью для класса **c**, если
**Ic(ϕ, Xℓ) >= I0** при заданном достаточно большом **I0**.

При разумных сочетаниях параметров **ε** и **I0** эвристический критерий практически
всегда оказывается строже статистического. Имеется довольно
обширная область статистических закономерностей, для которых вероятность случайной
реализации крайне низка, в то же время, они допускают слишком много
ошибок и не являются логическими закономерностями в смысле **ε**, **δ**-критерия.

![](14.png)

### Энтропийное определение информативности

Предикат **ϕ** является закономерностью по энтропийному критерию информативности,
если **IGainc(ϕ, Xℓ) > G0** при некотором достаточно большом **G0**.

Энтропийный критерий IGainc асимптотически эквивалентен статистическому Ic:

![](15.png)

### Многоклассовая информативность

Статистический критерий обобщает статистическое определение информативности на случай произвольного числа классов **Y = {1, . . ., M}**:

![](16.png)

Энтропийный критерий для случая большого числа классов является асимптотическим приближением
статистического:

![](17.png)

### Взвешенная информативность

Цена ошибки на разных объектах может быть различной. Например, она может
отличаться для разных классов: при выдаче кредитов «пропуск цели» (т. е. ненадёжного
заёмщика) обходится банку существенно дороже, чем «ложная тревога» (отказ
хорошему заёмщику). Обучающие объекты из класса «плохие заёмщики» должны
учитываться с большим весом при поиске логических закономерностей.

![](18.png)

### Методы поиска информативных закономерностей

Бинаризация количественных признаков
Произвольный признак **f : X → Df** порождает семейство предикатов, проверяющих
попадание значения **f(x)** в определённые подмножества множества Df . Ниже
перечисляются наиболее типичные конструкции такого вида.

![](19.png)

![](20.png)

### Поиск закономерностей в форме конъюнкций

![](21.png)

Число термов k в конъюнкции называется её рангом.

**Градиентный алгоритм*** синтеза конъюнкций. Поставим каждой конъюнкции
**ϕ** в соответствие её окрестность — множество конъюнкций **V (ϕ)**, получаемых
из **ϕ** путём элементарных модификаций: добавлением, изъятием или модификацией
одного из термов конъюнкции.
Начиная с заданной конъюнкции **ϕ0** (например, пустой), строится последовательность
конъюнкций **ϕ0, ϕ1, . . ., ϕt, . . .** , в которой каждая следующая конъюнкция
ϕt выбирается из окрестности предыдущей **Vt = V (ϕt−1)** по критерию максимума
информативности (шаг 3).

**Жадный алгоритм синтеза конъюнкции** использует только операцию добавления
термов. Начальным приближением является пустая конъюнкция (не содержащая
термов). Недостаток жадной стратегии в том, что она может уводить в сторону
от глобального максимума информативности. Терм, найденный на k-м шаге, перестаёт
быть оптимальным после добавления последующих термов. Тем не менее, в ряде
практических задач эта простая эвристика демонстрирует способность находить
неплохие закономерности.

![](22.png)

**Стохастический локальный поиск** (stochastic local search, SLS) также начинает
с пустой конъюнкции, но использует полный набор возможных модификаций. Это
преимущество по сравнению с жадным алгоритмом, так как появляется возможность
удалять и заменять неоптимальные термы. С другой стороны, мощность окрестности
**|V (ϕ)|** может оказаться настолько большой, что перебор всех допустимых модификаций
станет нерентабельным. Поэтому в SLS строится не вся окрестность, а только
некоторое её случайное подмножество. Максимальная допустимая мощность этого
подмножества задаётся как дополнительный параметр алгоритма **Vmax**.

Процедура стабилизации пытается улучшить конъюнкцию ϕ, удаляя или заменяя
по одному терму. В отличие от SLS, перебираются все возможные удаления и замены.
Модификации производятся до тех пор, пока возрастает информативность конъюнкции
Ic(ϕ, Xℓ
). Стабилизация повышает устойчивость алгоритма относительно малых
вариаций обучающей выборки или других условий обучения (например, генератора
псевдослучайной последовательности в SLS)

**Процедура редукции** отличается тем, что термы только удаляются, а информативность
вычисляется по независимой контрольной выборке Xk
, составленной из объектов,
не участвовавших в построении конъюнкции ϕ. Контрольную выборку формируют
до начала обучения, выделяя из массива исходных данных около 30% объектов,
как правило, случайным образом. При этом объекты разных классов распределяются
в той же пропорции, что и во всей выборке (этот принцип отбора называется стратификацией
выборки).

**Генетический алгоритм синтеза конъюнкций** (Genetic Algorithm, GA) можно
рассматривать как дальнейшее усовершенствование SLS на основе идей дарвиновской
эволюции. Главное отличие GA от SLS в том, что на каждом шаге отбирается
не одна наилучшая конъюнкция, а целое множество лучших конъюнкций, называемое
популяцией. Из них порождается большое количество конъюнкций-потомков
с помощью двух генетических операций — скрещивания и мутации. Скрещивание
(crossingover) — это образование новой конъюнкции путём обмена термами между
двумя членами популяции. В роли мутаций выступают уже знакомые операции добавления,
замещения и удаления термов. Таким способом можно получить огромное
количество потомков, но на практике строят лишь ограниченное число, не более T1
потомков путём случайных скрещиваний и мутаций. Затем производится естественный
отбор, в результате которого в следующее поколение переходят не более T0 наиболее
информативных потомков. Обычно берут T0 ≪ T1.

 В общем случае к форме
предъявляются два требования.

• Интерпретируемость. Условие **ϕ(x) = 1** должно выражаться на естественном
языке в форме, понятной эксперту. Для этого, в частности, закономерность ϕ(x)
должна зависеть от небольшого числа признаков **ω ⊆ {1, . . ., n}**. Обычно **|ω|**
ограничивают сверху числом от 2 до 7. Найденные сочетания признаков **ω** могут
либо подтверждать существующие представления о взаимосвязях между
признаками, либо указывать на существование ранее неизвестных взаимосвязей,
и тогда можно говорить о получении новых знаний из данных (knowledge
discovery). Некоторые из найденных сочетаний признаков ω могут оказаться
не интерпретируемыми с содержательной точки зрения; такие закономерности
отвергаются экспертами как «ложные».
• Эффективность поиска. Должны существовать эффективные методы поиска
закономерностей по конечной выборке U в рамках выбранного семейства предикатов
**Φ: I(ϕ, U) → max ϕ∈Φ**
. Как правило, наиболее трудоёмким является поиск
информативных наборов признаков **ω ⊆ {1, . . ., n}**.

**Решающий список** — это алгоритм классификации a: X → Y , который
задаётся набором закономерностей **ϕ1(x), . . ., ϕT (x)**, приписанных к классам
c1, . . ., cT ∈ Y соответственно, и вычисляется согласно Алгоритму 1.3.

![](23.png)

![](24.png)


**Деревом** называется конечный связный граф с множеством вершин V , не содержащий циклов и имеющий выделенную вершину v0 ∈ V , в которую не входит ни одно ребро. Эта вершина называется корнем дерева. Вершина, не имеющая выходящих рёбер, называется терминальной или листом. Остальные вершины называются внутренними. Дерево называется бинарным, если из любой его внутренней вершины выходит ровно два ребра. Выходящие рёбра связывают каждую внутреннюю вершину v с левой дочерней вершиной Lv и с правой дочерней вершиной Rv.

**Бинарное решающее дерево** — это алгоритм классификации, задающийся бинарным деревом, в котором каждой внутренней вершине v ∈ V приписан предикат βv : X → {0, 1}, каждой терминальной вершине v ∈ V приписано имя класса cv ∈ Y . При классификации объекта x ∈ X он проходит по дереву путь от корня до некоторого листа, в соответствии с алгоритмом.

1: v := v0;

2: пока вершина v внутренняя

3: если βv(x) = 1 то

4: v := Rv; (переход вправо)

5: иначе

6: v := Lv; (переход влево)

7: вернуть cv.

Решающее дерево никогда не отказывается от классификации, в отличие от решающего списка. Отсюда также следует, что алгоритм классификации a: X → Y , реализуемый бинарным решающим деревом, можно представить в виде простого голосования конъюнкций.

# Метод опорных методов

D некоторых случаях более естественно использовать кусочно-линейную функцию **ε**-чувствительности, которая
не считает за ошибки отклонения **a(xi)** от **yi**, меньшие **ε**. Предполагается, что значение параметра **ε** задаёт эксперт, исходя из априорных соображений.

![](2.png)

С этой функцией потерь функционал принимает вид

![](3.png)

Легко обнаруживается сходство данной задачи с задачей классификации.
Покажем, что минимизация финкионала эквивалентна некоторой задаче квадратичного
программирования с линейными ограничениями типа неравенств. При этом также
возникает двойственная задача, зависящая только от двойственных переменных;
также достаточно оставить в выборке только опорные объекты; также решение
выражается через скалярные произведения объектов, а не сами объекты; и также
можно использовать ядра. Иными словами, SVM-регрессия отличается от SVM классификации
только в технических деталях, основные идеи остаются теми же. Положим **С=1/2r**. 

Введём дополнительные переменные 

![](4.png)

Тогда задача минимизации может быть переписана в эквивалентной форме
как задача квадратичного программирования с линейными ограничениями неравенствами

![](5.png)

Как и в предыдущих случаях, лагранжиан этой задачи выражается через двойственные
переменные, а скалярные произведения можно заменить ядром.

![](6.png)

![](7.png)

Объекты типов 2–5 являются опорными и учитываются при определении вектора
весов. При этом только на объектах типов 4 и 5 возникает ненулевая ошибка.

Уравнение регрессии также выражается через двойственные переменные:

![](8.png)

![](9.png)

Как и раньше, чтобы избежать численной неустойчивости, имеет смысл взять медиану
множества значений **w0**, вычисленных по всем опорным векторам.
В этом методе есть два управляющих параметра. Параметр точности **ε** задаётся
из априорных соображений. Параметр регуляризации **C** подбирается, как правило,
по скользящему контролю, что является вычислительно трудоёмкой процедурой.

# Метод главных компонент

Ещё одно решение проблемы мультиколлинеарности заключается в том, чтобы подвергнуть исходные признаки некоторому функциональному преобразованию, гарантировав линейную независимость новых признаков, и, возможно, сократив их количество, то есть уменьшив размерность задачи.

В методе главных компонент (principal component analysis, PCA) строится минимальное число новых признаков, по которым исходные признаки восстанавливаются линейным преобразованием с минимальными погрешностями. PCA относится к методам обучения без учителя (unsupervised learning), поскольку матрица «объекты–признаки» _F_ преобразуется без учёта целевого вектора _y_.

**Постановка задачи.** Пусть имеется n числовых признаков ![f_j (x), j = 1,\dots,n](https://i.upmath.me/svg/f_j%20(x)%2C%20j%20%3D%201%2C%5Cdots%2Cn). Как обычно, будем отождествлять объекты обучающей выборки и их признаковые описания: ![x_i ≡ (f_1(x_i),\dots,f_n(x_i)), i = 1,\dots, ℓ.](https://i.upmath.me/svg/x_i%20%E2%89%A1%20(f_1(x_i)%2C%5Cdots%2Cf_n(x_i))%2C%20i%20%3D%201%2C%5Cdots%2C%20%E2%84%93.) 

Рассмотрим матрицу F, строки которой соответствуют признаковым описаниям обучающих объектов: 

![F_{l\times n} = \begin{pmatrix}f_1(x_1)&\dots&f_n(x_1)\\\dots&\dots&\dots\\f_1(x_l)&\dots&f_n(x_l)\end{pmatrix} =  \begin{pmatrix}x_{1}\\\dots\\x_{l}\end{pmatrix}](https://i.upmath.me/svg/F_%7Bl%5Ctimes%20n%7D%20%3D%20%5Cbegin%7Bpmatrix%7Df_1(x_1)%26%5Cdots%26f_n(x_1)%5C%5C%5Cdots%26%5Cdots%26%5Cdots%5C%5Cf_1(x_l)%26%5Cdots%26f_n(x_l)%5Cend%7Bpmatrix%7D%20%3D%20%20%5Cbegin%7Bpmatrix%7Dx_%7B1%7D%5C%5C%5Cdots%5C%5Cx_%7Bl%7D%5Cend%7Bpmatrix%7D)

Обозначим через ![z_i = (g_1(x_i),\dots,g_m(x_i))](https://i.upmath.me/svg/z_i%20%3D%20(g_1(x_i)%2C%5Cdots%2Cg_m(x_i))) признаковые описания тех же объектов в новом пространстве ![Z = \mathbb{R}_m](https://i.upmath.me/svg/Z%20%3D%20%5Cmathbb%7BR%7D_m) меньшей размерности, ![m<n](https://i.upmath.me/svg/m%3Cn): 

![G_{l\times M} = \begin{pmatrix}g_1(x_1)&\dots&g_m(x_1)\\\dots&\dots&\dots\\g_1(x_l)&\dots&g_m(x_l)\end{pmatrix} =  \begin{pmatrix}z_{1}\\\dots\\z_{l}\end{pmatrix}](https://i.upmath.me/svg/G_%7Bl%5Ctimes%20M%7D%20%3D%20%5Cbegin%7Bpmatrix%7Dg_1(x_1)%26%5Cdots%26g_m(x_1)%5C%5C%5Cdots%26%5Cdots%26%5Cdots%5C%5Cg_1(x_l)%26%5Cdots%26g_m(x_l)%5Cend%7Bpmatrix%7D%20%3D%20%20%5Cbegin%7Bpmatrix%7Dz_%7B1%7D%5C%5C%5Cdots%5C%5Cz_%7Bl%7D%5Cend%7Bpmatrix%7D)

Потребуем, чтобы исходные признаковые описания можно было восстановить по новым описаниям с помощью некоторого линейного преобразования, определяемого матрицей ![U = (u_{js})_{n\times m}](https://i.upmath.me/svg/U%20%3D%20(u_%7Bjs%7D)_%7Bn%5Ctimes%20m%7D): 

![\widehat{f}_j(x) =  \sum_{s=1}^{m}g_s(x)u_{js}, \ j = 1,\dots, n, \ x  \in X](https://i.upmath.me/svg/%5Cwidehat%7Bf%7D_j(x)%20%3D%20%20%5Csum_%7Bs%3D1%7D%5E%7Bm%7Dg_s(x)u_%7Bjs%7D%2C%20%5C%20j%20%3D%201%2C%5Cdots%2C%20n%2C%20%5C%20x%20%20%5Cin%20X)

или в векторной записи: ![\widehat{x} = zU^T](https://i.upmath.me/svg/%5Cwidehat%7Bx%7D%20%3D%20zU%5ET). 

Восстановленное описание ![\widehat{x}](https://i.upmath.me/svg/%5Cwidehat%7Bx%7D) не обязано в точности совпадать с исходным описанием ![x](https://i.upmath.me/svg/x), но их отличие на объектах обучающей выборки должно быть как можно меньше при выбранной размерности ![m](https://i.upmath.me/svg/m). Будем искать одновременно и матрицу новых признаковых описаний ![G](https://i.upmath.me/svg/G), и матрицу линейного преобразования ![U](https://i.upmath.me/svg/U), при которых суммарная невязка восстановленных описаний минимальна: 

![\Delta^2(G,U) =  \sum_{i=1}^{l} ||\widehat{x}_i - x_i||^2 = \sum_{i=1}^{l} ||z_iU^T - x_i||^2 = ||GU^T - F||^2 \rightarrow \min_{G,U}](https://i.upmath.me/svg/%5CDelta%5E2(G%2CU)%20%3D%20%20%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%20%7C%7C%5Cwidehat%7Bx%7D_i%20-%20x_i%7C%7C%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%20%7C%7Cz_iU%5ET%20-%20x_i%7C%7C%5E2%20%3D%20%7C%7CGU%5ET%20-%20F%7C%7C%5E2%20%5Crightarrow%20%5Cmin_%7BG%2CU%7D)

где все нормы евклидовы. Напомним, что ![||A||^2 = trAA^T = tr A^TA](https://i.upmath.me/svg/%7C%7CA%7C%7C%5E2%20%3D%20trAA%5ET%20%3D%20tr%20A%5ETA), где ![tr](https://i.upmath.me/svg/tr) — операция следа матрицы.

### Теорема
Если **m <= rk F**, то минимум **∆<sup>2</sup>(G, U)** достигается, когда столбцы матрицы **U** есть собственные векторы **F<sup>T</sup>F**, соответствующие **m** максимальным собственным значениям. При этом **G = FU**, матрицы **U** и **G** ортогональны.

### Связь с сингулярным разложением
Если **m = n**, то **∆<sup>2</sup>(G, U) = 0**.
В этом случае представление **F = GU<sup>T</sup>** является точным и совпадает с сингулярным разложением: 

**F = GU<sup>T</sup> = VDU<sup>T</sup>**, если положить **G = VD** и **Λ = D<sup>2</sup>**. При этом матрица **V** ортогональна: **V<sup>T</sup>V = I<sub>m</sub>**.

Если **m < n**, то представление **F≈GU<sup>T</sup>** является приближённым.
Сингулярное разложение матрицы **GU<sup>T</sup>** получается из сингулярного разложения матрицы **F** путём обнуления **n − m** минимальных собственных значений.

### Преобразование Карунена–Лоэва.
Диагональность матрицы **G<sup>T</sup>G = Λ** означает, что новые признаки **g<sub> 1 .. m</sub>** не коррелируют на обучающих объектах.
Ортогональное преобразование **U** называют декоррелирующим или преобразованием Карунена–Лоэва.
Если **m = n**, то прямое и обратное преобразование вычисляются с помощью одной и той же матрицы **U**: **F = GU<sup>T</sup>** и **G = FU**.

### Эффективная размерность.
Главные компоненты содержат основную информацию о матрице **F**.
Число главных компонент **m** называют также эффективной размерностью задачи.
На практике её определяют следующим образом.
Все собственные значения матрицы **F<sup>T</sup>F** упорядочиваются по убыванию:
**λ<sub>1</sub> >= ... >= λ<sup>n</sup> >= 0**.
Задаётся пороговое значение **ε** из **[0, 1]**, достаточно близкое к нулю, и определяется наименьшее целое **m**, при котором относительная погрешность приближения матрицы **F** не превышает **ε**:

![](1.png)

# Нелинейные методы восстановления регрессии

Общая идея в том, что задача сводится к решению последовательности более простых линейных задач.

![](https://latex.codecogs.com/gif.latex?f%28x%2C%5Calpha%29) нелинейная модель регрессии.
Требуется минимизировать функционал качества по вектору параметров ![](https://latex.codecogs.com/gif.latex?%5Calpha%5Csubset%5Cmathbb%7BR%7D%5Ep):

![](https://latex.codecogs.com/gif.latex?Q%28%5Calpha%2CX%5El%29%3D%5Csum_%7Bi%3D1%7D%5El%28f%28x_i%2C%5Calpha%29-y_i%29%5E2) .

### Метод Ньютона–Рафсона. 
Выберем начальное приближение ![](https://latex.codecogs.com/gif.latex?%5Calpha%5E0%3D%5C%7B%5Calpha_1%5E0%2C...%2C%5Calpha_p%5E0%20%5C%7D)и организуем итерационный процесс:

![](https://latex.codecogs.com/gif.latex?%5Calpha%5E%7Bt&plus;1%7D%3A%3D%5Calpha%5Et%20-%20h_t%28Q%27%27%28%5Calpha%5Et%29%29%5E%7B-1%7DQ%27%28%5Calpha%5Et%29%2C)

где ![](https://latex.codecogs.com/gif.latex?Q%27%28%5Calpha%5Et%29) — градиент функционала Q в точке ![](https://latex.codecogs.com/gif.latex?%5Calpha%5Et) 

![](https://latex.codecogs.com/gif.latex?Q%27%27%28%5Calpha%5Et%29)  — гессиан(матрица вторых производных) функционала Q в точке ![](https://latex.codecogs.com/gif.latex?%5Calpha%5Et)

![](https://latex.codecogs.com/gif.latex?h_t)— величина шага,который можнорегулировать,а в простейшем варианте просто полагать равным единице.
  Запишем компоненты градиента:
  
  ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20%5Calpha_j%20%7DQ%28%5Calpha%29%3D2%5Csum%20_%7Bi%3D1%7D%5El%28f%28x_i%2C%5Calpha%20%29-y_i%29%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha%20_j%7D%28x_i%2C%5Calpha%29)

Запишем компоненты гессиана:

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%5E2%20%7D%7B%5Cpartial%20%5Calpha_j%5Cpartial%20%5Calpha%20_k%20%7DQ%28%5Calpha%29%3D2%5Csum%20_%7Bi%3D1%7D%5El%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha_j%7D%28x_i%2C%5Calpha%20%29%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha%20_k%7D%28x_i%2C%5Calpha%20%29-2%5Csum%20_%7Bi%3D1%7D%5El%28f%28x_i%2C%5Calpha%20%29-y_i%29%5Cfrac%7B%5Cpartial%5E2%20f%20%7D%7B%5Cpartial%20%5Calpha_j%5Cpartial%20%5Calpha%20_k%20%7D%28x_i%2C%5Calpha%20%29)


### Основная сложность метода Ньютона–Рафсона заключается в обращении гессианана каждой итерации.

## Метод Ньютона-Гаусса

Если функция f достаточно гладкая(дважды непрерывно дифференцируема),то её можно линеаризовать в окрестности текущего значения вектора коэффициентов ![](https://latex.codecogs.com/gif.latex?%5Calpha%20_t):

![](https://latex.codecogs.com/gif.latex?f%28x_i%2C%5Calpha%20%29%3D%20f%28x_i%2C%5Calpha%5Et%29&plus;%5Csum_%7Bj%3D1%7D%5Ep%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha%20_j%7D%28x_i%2C%5Calpha_j%29%28%5Calpha_j-%5Calpha_j%5Et%29)

Заменим в гессиане функцию f на её линеаризацию.Это всё равно,что положить второе слагаемое в гессиане равным нулю.Тогда не нужно будет вычислять вторые производные. 

Основная сложность метода Ньютона–Рафсона заключается в обращении гессиана на каждой итерации.

Введём матричные обозначения: 

![](https://latex.codecogs.com/gif.latex?F_t%3D%28%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20%5Calpha%20_j%20%7D%28x_i%2C%5Calpha%20%5Et%29%20%29%5E%7Bj%3D1%2Cp%7D_%7Bi%3D1%2Cl%7D) — матрица первых производных.

![](https://github.com/null32/ML1/blob/master/nlr/def2.png) — вектор значений аппроксимирующей функции на t-й итерации

Тогда формула t-й итерации метода Ньютона–Гаусса в матричной записи примет вид:

![](https://latex.codecogs.com/gif.latex?%5Calpha%20%5E%7Bt&plus;1%7D%3A%3D%5Calpha%20%5Et-h_t%28F%5ET_tF_t%29%5E%7B-1%7DF%5ET_t%28f%5Et-y%29)

