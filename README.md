# Генератор датасета

Генерируются массивы частотных характеристик линейных звеньев вида

$$\begin{equation}
    H(s) = G \frac{\sum_{m=1}^{M}1 + \frac{s}{\omega_{z(m)}}}{s^K \sum_{n=1}^{N} 1 + \frac{s}{\omega_{p(n)}}},
\end{equation}$$

где
- ${G}$ - пропорциональный коэффициент; принят ${G=1}$,
- ${\omega_z}$ - нули передаточной функции в количестве ${M}$,
- ${\omega_p}$ - полюса передаточной функции в количестве ${N}$,
- ${K}$ - количество интеграторов.

В [general_functions.py](utils/general_functions.py) представлены функции:
- [transfer_function](utils/general_functions.py) - вычисление массива комплексного коэффициента передачи,
- [generate_masks](utils/general_functions.py) - генерация int-координат нулей и полюсов,
- [generate_freq_zeros_poles](utils/general_functions.py) - преобразование int-координат нулей и полюсов в частоты нулей/полюсов для использования в [transfer_function](utils/general_functions.py).

## Конфигурирование

Входные параметры генерации задаются в [config.json](config/config.json):

- `split`: `str` - `train`, `val`, `test`.
- `size`: `int` - количество генерируемых данных на каждый набор полюсов-нулей.
- `seed`: `int` или `null`.
- `length`: размер одного элемента.
- `fmin`: `List[float, float]` - диапазон для нижней границы частот.
- `fmax`: `List[float, float]` - диапазон для верхней границы частот.
- `Nzp_max`: `int` - максимальное количество интеграторов.
- `Nlp_max`: `int` - максимальное количество полюсов левых.
- `Nrp_max`: `int` - максимальное количество полюсов правых.
- `Nlz_max`: `int` - максимальное количество нулей левых.
- `Nrz_max`: `int` - максимальное количество нулей правых.

## Структура датасета

Каждой конфигурации `[Nzp, Nlp, Nrp, Nlz, Nrz]` соответствуют `size`- элементов с разным расположением нулей-полюсов и разным диапазоном частоты. Количество интеграторов (`zp`), полюсов (`nlp`, `nrp`) и нулей (`nlz`, `nrz`)  зашифровано в имени файла:

`nzp_nlp_nrp_nlz_nrz_xxx.csv`

где `xxx` - порядковый номер из `size`.

```
project/
├── dataset/
│   └── train/
│       └── 0zp0lp0rp0lz0rz_000.csv
│       └── 0zp0lp0rp0lz0rz_001.csv
│       └── ...
│   └── train_masks.json
│   └── val/
│       └── 0zp0lp0rp0lz0rz_000.csv
│       └── 0zp0lp0rp0lz0rz_001.csv
│       └── ...
│   └── val_masks.json
│   └── test/
│       └── 0zp0lp0rp0lz0rz_000.csv
│       └── 0zp0lp0rp0lz0rz_001.csv
│       └── ...
│   └── test_masks.json
```

Каждый элемент датасета представляет собой csv-таблицу с тремя столбцами:

- Frequency (Hz),
- Gain (Real),
- Gain (Imag).

Маски агрегированы в json-файлы - по одному файлу на каждый `split`: `train`, `val`, `test`. В файлах каждому элементу датасета приведены соответствующие:
- "zero_poles": `int` - количество интеграторов,
- "left_poles": `List[int]` - координаты полюсов левых,
- "right_poles": `List[int]` - координаты полюсов правых,
- "left_zeros": `List[int]` - координаты нулей левых,
- "right_zeros": `List[int]` - координаты нулей правых.

## Запуск на генерацию

Запуск генерации датасета осуществляется через [main.py](main.py). При необходимости сгенерировать все 3 набора `split`: `str` - `train`, `val`, `test` необходимо последовательно задать в [config.json](config/config.json) соответствующие `split` и `size` и каждый раз выполнять

```
python src/main.py
```

## Даталоудер

Создан датакласс [ZerosPolesDataset.py](utils/ZerosPolesDataset.py), наследующий от `torch.utils.data.Dataset` следующие методы:
- `__init__`: инициализация путей к данным и маскам;
- `__len__`: возврат количества примеров в датасете;
- `__getitem__`: загрузка и возврат одного примера в виде `(data_tensor, masks_tensor, freq)`.

Каждый элемент представлен следующими объектами:
- `data_tensor` - тензор размера `[2, length]` - канал `real` и канал `imag`,
- `masks_tensor` - тензор размера `[4, length]` - по своей маске для полюсов левых, полюсов правых, нулей левых, нулей правых. Каждая маска формируется функцией [positions_to_mask](utils/ZerosPolesDataset.py) - преобразование координат нулей/полюсов в маски - массивы длины `length`, содержащие `1` в координатах нулей/полюсов и `0` в остальных,
- `freq` - numpy-массив частот размера `(length,)`.

Пример использования даталоудера приведен в [debug_notebook.ipynb](debug_notebook.ipynb).

<p align="center" width="100%">
  <img src="./readme_img/dataset_samples.png"
  style="background-color: white; padding: 0;
  width="100%" />
</p>