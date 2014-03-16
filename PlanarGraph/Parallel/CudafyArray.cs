using System;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PlanarGraph.Parallel
{
    /// <summary>
    ///     Класс работы с массивом
    ///     Данный класс реализует модель специализированного вычислительного устройства
    ///     с фиксированным набором элементарных операций и использует параллельные вычисления CUDA
    ///     для реализации этой модели
    /// </summary>
    public struct CudafyArray
    {
        /// <summary>
        ///     Семафор для блокирования одновременного доступа к данному статичному классу
        ///     из разных параллельных процессов
        ///     Надеюсь CUDAfy тоже заботится о блокировании одновременного доступа к видеокарточке ;)
        /// </summary>
        public static readonly object Semaphore = new Object();

        private static dim3 _gridSize0;
        private static dim3 _blockSize0;
        private static dim3 _gridSize1;
        private static dim3 _blockSize1;
        private static dim3 _gridSize2;
        private static dim3 _blockSize2;
        private static int _m;
        private static int _m1;

        #region Регистры класса

        [Cudafy] private static int[] _a;
        [Cudafy] private static int[] _b;
        [Cudafy] private static int[] _b1;
        [Cudafy] private static int[] _c;
        [Cudafy] private static int[] _c0;
        [Cudafy] private static int[] _c1;
        [Cudafy] private static readonly int[] D = new int[1];
        [Cudafy] private static int[,] _compare;

        #endregion

        public static int GetHash()
        {
            return D[0];
        }

        /// <summary>
        ///     Вызов и исполнение одной элементарной функции по имени функции
        /// </summary>
        /// <param name="function"></param>
        public static void Execute(string function)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b1);
            int[] devC = gpu.Allocate(_c1);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            gpu.Launch(1, 1).Split(devA, devB, devC, _m1);
            gpu.Launch(_gridSize1, _blockSize1, function, devA, devB, devC, devD, 1);
            gpu.Launch(1, 1, function, devA, devB, devC, devD, 2);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Вызов и исполнение функции проверки что массив отсортирован
        /// </summary>
        public static void ExecuteSorted(int direction = 1)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);


            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b1);
            int[] devC = gpu.Allocate(_c1);
            int[] devD = gpu.Allocate(D);
            int[,] devCompare = gpu.Allocate(_compare);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_compare, devCompare);

            gpu.Launch(1, 1).Split(devA, devB, devC, _m1);
            gpu.Launch(_gridSize1, _blockSize1).Sorted(devA, devB, devC, devD, devCompare, 0, direction);
            gpu.Launch(1, 1).Sorted(devA, devB, devC, devD, devCompare, 1, direction);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выполнение сортировки слияниями
        ///     Пример использования:
        ///     CudafySequencies.SetSequencies(arrayOfArray,arrayOfArray);
        ///     CudafySequencies.Execute("Compare");
        ///     var compare = CudafySequencies.GetMartix();
        ///     CudafyArray.SetArray(Enumerable.Range(0,n).ToArray());
        ///     CudafyArray.SetCompare(compare);
        ///     CudafyArray.MergeSort();
        ///     var indexesOfSorted = CudafyArray.GetArray();
        /// </summary>
        public static void MergeSort(int direction = 1)
        {
            Debug.Assert(_compare.GetLength(0) == _compare.GetLength(1));
            Debug.Assert(_a.GetLength(0) == _compare.GetLength(1));
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[,] devCompare = gpu.Allocate(_compare);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_compare, devCompare);

            gpu.Launch(1, 1).Split(devA, devB, devC, 0);

            gpu.Launch(_gridSize0, _blockSize0).Merge(devA, devB, devC, devCompare, _m, 0, direction);

            gpu.CopyFromDevice((_m & 1) == 0 ? devA : devB, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выполнение чётно-нечётной сортировки
        ///     Пример использования:
        ///     CudafySequencies.SetSequencies(arrayOfArray,arrayOfArray);
        ///     CudafySequencies.Execute("Compare");
        ///     var compare = CudafySequencies.GetMartix();
        ///     CudafyArray.SetArray(Enumerable.Range(0,n).ToArray());
        ///     CudafyArray.SetCompare(compare);
        ///     CudafyArray.OddEvenSort();
        ///     var indexesOfSorted = CudafyArray.GetArray();
        /// </summary>
        public static void OddEvenSort(int direction = 1)
        {
            /*
            Для каждой итерации алгоритма операции сравнения-обмена для всех пар элементов независимы и
            выполняются одновременно. Рассмотрим случай, когда число процессоров равно числу элементов, т.е. p=n -
            число процессоров (сортируемых элементов). Предположим, что вычислительная система имеет топологию
            кольца. Пусть элементы ai (i = 1, .. , n), первоначально расположены на процессорах pi (i = 1, ... , n). В нечетной
            итерации каждый процессор с нечетным номером производит сравнение-обмен своего элемента с элементом,
            находящимся на процессоре-соседе справа. Аналогично в течение четной итерации каждый процессор с четным
            номером производит сравнение-обмен своего элемента с элементом правого соседа.
            На каждой итерации алгоритма нечетные и четные процессоры выполняют шаг сравнения-обмена с их
            правыми соседями за время Q(1). Общее количество таких итераций – n; поэтому время выполнения
            параллельной сортировки – Q(n).
            Когда число процессоров p меньше числа элементов n, то каждый из процессов получает свой блок
            данных n/p и сортирует его за время Q((n/p)·log(n/p)). Затем процессоры проходят p итераций (р/2 и чётных, и
            нечётных) и делают сравнивания-разбиения: смежные процессоры передают друг другу свои данные, а
            внутренне их сортируют (на каждой паре процессоров получаем одинаковые массивы). Затем удвоенный
            массив делится на 2 части; левый процессор обрабатывает далее только левую часть (с меньшими значениями
            данных), а правый – только правую (с большими значениями данных). Получаем отсортированный массив
            после p итераций, выполняя в каждой такие шаги:
            23)узлы считывают входные данные из хранилища;
            24)хост распределяет данные между клонами;
            25)клоны сортируют свою часть данных;
            26)клоны обмениваются частями, при этом объединяя четные и нечетные части;
            27)если количество перестановок равно количеству клонов, то алгоритм завершен;
            28)хост формирует отсортированный массив.
            */
            Debug.Assert(_compare.GetLength(0) == _compare.GetLength(1));
            Debug.Assert(_a.GetLength(0) == _compare.GetLength(1));
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c1);
            int[,] devCompare = gpu.Allocate(_compare);

            // Шаг первый - копируем исходный массив в память GPU 

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_compare, devCompare);

            // Шаг второй - разделим все элементы массива между блоками
            // соответственно они будут содержать данные разной длины, 
            gpu.Launch(1, 1).Split(devA, devB, devC, _m1);

            // запускаем задачи сортировки блоков
            // при этом применяем битоническую сортировку
            // На выходе - отсортированные массивы размера до 1<<m1

            gpu.Launch(_gridSize1, _blockSize1).Bitonic(devA, devB, devC, devCompare, _m1, direction);

            // запускаем задачи сортировки данных в двух соседних блоках
            // чередуя соседние блоки

            for (int i = 0; i < (1 << (_m - _m1)) + _m1; i++)
            {
                gpu.Launch(_gridSize1, _blockSize1).Merge(devA, devB, devC, devCompare, 1, i & 1, direction);
                int[] tmp = devA;
                devA = devB;
                devB = tmp;
            }

            gpu.CopyFromDevice(devA, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выполнение битонической сортировки
        ///     Пример использования:
        ///     CudafySequencies.SetSequencies(arrayOfArray,arrayOfArray);
        ///     CudafySequencies.Execute("Compare");
        ///     var compare = CudafySequencies.GetMartix();
        ///     CudafyArray.SetArray(Enumerable.Range(0,n).ToArray());
        ///     CudafyArray.SetCompare(compare);
        ///     CudafyArray.BitonicSort();
        ///     var indexesOfSorted = CudafyArray.GetArray();
        /// </summary>
        public static void BitonicSort(int direction = 1)
        {
            /*
            В основе этой сортировки лежит операция Bn(полуочиститель, half - cleaner) над массивом, параллельно
            упорядочивающая элементы пар xi и xi + n / 2.На рис. 1 полуочиститель может упорядочивать элементы пар как по
            возрастанию, так и по убыванию.Сортировка основана на понятии битонической последовательности и
            утверждении : если набор полуочистителей правильно сортирует произвольную последовательность нулей и
            единиц, то он корректно сортирует произвольную последовательность.
            Последовательность a0, a1, …, an - 1 называется битонической, если она или состоит из двух монотонных
            частей(т.е.либо сначала возрастает, а потом убывает, либо наоборот), или получена путем циклического
            сдвига из такой последовательности.Так, последовательность 5, 7, 6, 4, 2, 1, 3 битоническая, поскольку
            получена из 1, 3, 5, 7, 6, 4, 2 путем циклического сдвига влево на два элемента.
            Доказано, что если применить полуочиститель Bn к битонической последовательности a0, a1, …, an - 1,
            то получившаяся последовательность обладает следующими свойствами :
            • обе ее половины также будут битоническими.
            • любой элемент первой половины будет не больше любого элемента второй половины.
            • хотя бы одна из половин является монотонной.
            Применив к битонической последовательности a0, a1, …, an - 1 полуочиститель Bn, получим две
            последовательности длиной n / 2, каждая из которых будет битонической, а каждый элемент первой не превысит
            каждый элемент второй.Далее применим к каждой из получившихся половин полуочиститель Bn / 2.Получим
            уже четыре битонические последовательности длины n / 4.Применим к каждой из них полуочиститель Bn / 2 и
            продолжим этот процесс до тех пор, пока не придем к n / 2 последовательностей из двух элементов.Применив к
            каждой из них полуочиститель B2, отсортируем эти последовательности.Поскольку все последовательности
            уже упорядочены, то, объединив их, получим отсортированную последовательность.
            Итак, последовательное применение полуочистителей Bn, Bn / 2, …, B2 сортирует произвольную
            битоническую последовательность.Эту операцию называют битоническим слиянием и обозначают Mn.
            Например, к последовательности из 8 элементов a 0, a1, …, a7 применим полуочиститель B2, чтобы на
            соседних парах порядок сортировки был противоположен.На рис. 2 видно, что первые четыре элемента
            получившейся последовательности образуют битоническую последовательность.Аналогично последние
            четыре элемента также образуют битоническую последовательность.Поэтому каждую из этих половин можно
            отсортировать битоническим слиянием, однако проведем слияние таким образом, чтобы направление
            сортировки в половинах было противоположным.В результате обе половины образуют вместе битоническую
            Битоническая сортировка последовательности из n элементов разбивается пополам и каждая из
            половин сортируется в своем направлении.После этого полученная битоническая последовательность
            сортируется битоническим слиянием.
            */
            Debug.Assert(_compare.GetLength(0) == _compare.GetLength(1));
            Debug.Assert(_a.GetLength(0) == _compare.GetLength(1));
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c0);
            int[,] devCompare = gpu.Allocate(_compare);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_compare, devCompare);

            gpu.Launch(1, 1).Split(devA, devB, devC, _m);

            // Число n представимо в виде суммы степеней двойки,
            // Поэтому, разбиваем исходные данные на подмассивы с длинами равными слагаемым этой суммы
            // и сортируем каждый подмассив битоническим алгоритмом 
            // В разультате получим равное числу слагаеммых отсортированных массивов длинами равным степеням двойки

            gpu.Launch(_gridSize2, _blockSize2).Bitonic(devA, devB, devC, devCompare, _m, direction);

            // Теперь надо произвести слияние уже отсортированных массивов

            gpu.Launch(_gridSize0, _blockSize0).Merge(devA, devB, devC, devCompare, _m, 0, direction);

            gpu.CopyFromDevice((_m & 1) == 0 ? devA : devB, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        [Cudafy]
        public static void Split(GThread thread, int[] a, int[] b, int[] c, int j)
        {
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                c[0] = 0;
                int length = a.Length;
                int n = length & ((1 << j) - 1);
                for (int k = 0; k < j; k++)
                    c[k + 1] = (n & (1 << k));
                for (int i = 1; i <= (length >> j); i++)
                    c[j + i] = (n + (i << j));
            }
        }

        [Cudafy]
        public static void Merge(GThread thread, int[] a, int[] b, int[] c, int[,] compare,
            int k, int parity, int direction)
        {
            for (int j = 0; j < k; j++)
            {
                int step = 1 << j;
                for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x) << (j + 1);
                    tid < c.Length;
                    tid += thread.blockDim.x * thread.gridDim.x << (j + 1))
                {
                    int index0;
                    int index1;
                    int index2;
                    if (tid + (0 - parity)*step < 0) index0 = c[0];
                    else if (tid + (0 - parity)*step >= c.Length) index0 = c[c.Length - 1];
                    else index0 = c[tid + (0 - parity)*step];
                    if (tid + (1 - parity)*step < 0) index1 = c[0];
                    else if (tid + (1 - parity)*step >= c.Length) index1 = c[c.Length - 1];
                    else index1 = c[tid + (1 - parity)*step];
                    if (tid + (2 - parity)*step < 0) index2 = c[0];
                    else if (tid + (2 - parity)*step >= c.Length) index2 = c[c.Length - 1];
                    else index2 = c[tid + (2 - parity)*step];
                    int n0 = index1 - index0;
                    int n1 = index2 - index1;
                    int total = index2 - index0;
                    while (n0 > 0 && n1 > 0)
                    {
                        if (direction*
                            compare[((j & 1) == 0 ? a : b)[index0 + n0 - 1], ((j & 1) == 0 ? a : b)[index1 + n1 - 1]] >
                            0)
                            ((j & 1) != 0 ? a : b)[index0 + --total] = ((j & 1) == 0 ? a : b)[index0 + --n0];
                        else
                            ((j & 1) != 0 ? a : b)[index0 + --total] = ((j & 1) == 0 ? a : b)[index1 + --n1];
                    }
                    while (n0 > 0) ((j & 1) != 0 ? a : b)[index0 + --total] = ((j & 1) == 0 ? a : b)[index0 + --n0];
                    while (n1 > 0) ((j & 1) != 0 ? a : b)[index0 + --total] = ((j & 1) == 0 ? a : b)[index1 + --n1];
                }
            }
        }

        [Cudafy]
        public static void Bitonic(GThread thread, int[] a, int[] b, int[] c, int[,] compare,
            int k, int direction)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < c.Length;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int index = c[tid];
                int length = c[tid + 1] - c[tid];
                for (int i = 1; i < k && (1 << i) <= length; i++)
                {
                    for (int j = i; j >= 0; j--)
                    {
                        int step = 1 << j;
                        for (int id = 0; id < length/2; id++)
                        {
                            int offset = ((id >> j) << (j + 1)) + (id & ((1 << j) - 1));
                            int parity = ((tid & ((1 << k) - 1)) >> i);
                            while (parity > 1) parity = (parity >> 1) ^ (parity & 1);
                            parity = 1 - (parity << 1); // теперь переменная parity может иметь только 2 значения 1 и -1
                            int value = parity*direction*compare[a[index + offset], a[index + offset + step]];
                            if (value <= 0) continue;
                            int tmp = a[index + offset];
                            a[index + offset] = a[index + offset + step];
                            a[index + offset + step] = tmp;
                        }
                    }
                }
            }
        }

        [Cudafy]
        public static void Hash(GThread thread, int[] a, int[] b, int[] c, int[] d, int step)
        {
            switch (step)
            {
                case 1:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < b.Length;
                        tid += thread.blockDim.x * thread.gridDim.x)
                    {
                        int index0 = c[tid];
                        int index1 = c[tid + 1];
                        b[tid] = a[index0];
                        for (int i = index0 + 1; i < index1; i++)
                            b[tid] = (b[tid] << 1) ^ (b[tid] >> (8*sizeof (int) - 1)) ^ a[i];
                    }
                    break;
                case 2:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < 1;
                        tid += thread.blockDim.x * thread.gridDim.x)
                    {
                        d[0] = b[0];
                        for (int i = 1; i < b.Length; i++)
                            d[0] = (d[0] << 1) ^ (d[0] >> (8*sizeof (int) - 1)) ^ b[i];
                    }
                    break;
            }
        }

        [Cudafy]
        public static void Sorted(GThread thread, int[] a, int[] b, int[] c, int[] d, int[,] compare, int step,
            int direction)
        {
            switch (step)
            {
                case 1:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < b.Length;
                        tid += thread.blockDim.x * thread.gridDim.x)
                    {
                        int index0 = c[tid];
                        int index1 = c[tid + 1];
                        b[tid] = 1;
                        for (int i = index0; i < index1 && b[tid] != 0; i++)
                            b[tid] = (direction*compare[a[i], a[i + 1]] <= 0) ? 1 : 0;
                    }
                    break;
                case 2:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < 1;
                        tid += thread.blockDim.x * thread.gridDim.x)
                    {
                        d[0] = b[0];
                        for (int i = 1; i < b.Length && d[0] != 0; i++)
                            d[0] = b[i];
                    }
                    break;
            }
        }

        public static void SetArray(int[] array)
        {
            _m = (int) Math.Ceiling(Math.Log(array.Length, 2));
            _m1 = _m/3;
            _a = array;
            _b = new int[array.Length];
            _b1 = new int[(1 << (_m - _m1))];
            _c = new int[array.Length + 1];
            _c0 = new int[1 + _m];
            _c1 = new int[(1 << (_m - _m1)) + _m1];
            _gridSize0 = Math.Min(15, (int) Math.Pow(array.Length, 0.333333333333));
            _blockSize0 = Math.Min(15, (int) Math.Pow(array.Length, 0.333333333333));
            _gridSize1 = Math.Min(15, (int) Math.Pow((1 << (_m - _m1)) + _m1, 0.333333333333));
            _blockSize1 = Math.Min(15, (int) Math.Pow((1 << (_m - _m1)) + _m1, 0.333333333333));
            _gridSize2 = Math.Min(15, (int) Math.Pow((1 << (_m - _m)) + _m, 0.333333333333));
            _blockSize2 = Math.Min(15, (int) Math.Pow((1 << (_m - _m)) + _m, 0.333333333333));
        }

        public static void SetCompare(int[,] compare)
        {
            _compare = compare;
        }

        public static int[] GetArray()
        {
            return _a;
        }

        public static int GetSorted()
        {
            return D[0];
        }
    }
}