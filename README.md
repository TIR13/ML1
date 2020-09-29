# ML1

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

