using System;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PlanarGraph.Parallel
{
    /// <summary>
    ///     Класс работы с матрицей
    ///     Данный класс реализует модель специализированного вычислительного устройства
    ///     с фиксированным набором элементарных операций и использует параллельные вычисления CUDA
    ///     для реализации этой модели
    /// </summary>
    public struct CudafyMatrix
    {
        /// <summary>
        ///     Семафор для блокирования одновременного доступа к данному статичному классу
        ///     из разных параллельных процессов
        ///     Надеюсь CUDAfy тоже заботится о блокировании одновременного доступа к видеокарточке ;)
        /// </summary>
        public static readonly object Semaphore = new Object();

        #region Регистры класса

        [Cudafy] private static int[,] _a;
        [Cudafy] private static int[,] _b;
        [Cudafy] private static int[] _c;
        [Cudafy] private static readonly int[] D = new int[1];

        #endregion

        #region Установка текущих значений в регистрах (setter)

        public static void SetMatrix(int[,] value)
        {
            int rows = value.GetLength(0);
            int columns = value.GetLength(1);
            _a = value;
            _b = new int[rows, columns];
            _c = new int[rows];
        }

        public static int[,] GetMatrix()
        {
            return _a;
        }

        public static void SetIndexes(int[] value)
        {
            _c = value;
        }

        #endregion

        /// <summary>
        ///     Вычисление целевой функции для симплекс-метода
        ///     Функция Мак-Лейна вычисляется после применения к данной матрице обратимого преобразования,
        ///     то есть умножение данной матрицы на обратимую матрицу, задаваемую списком индексов,
        ///     где строка i исходной матрицы должна быть добавлена к строкам i и indexes[i] результирующей матрицы, если
        ///     i!=indexes[i], и только к строке i результирующей матрицы, если i==indexes[i]
        ///     Таким образом, каждый базис в этом пространстве получается из данного базиса при помощи цепочки элементарных
        ///     преобразований. А на матричном языке проблема распознавания планарности сводится к нахождению такой матрицы в
        ///     классе эквивалентных матриц (т.е. матриц, которые получаются друг из друга при помощи элементарных преобразований
        ///     над строками), у которой в каждом столбце содержится не более двух единиц [6].
        ///     Указанный критерий позволяет разработать методику определения планарности графа, сводя проблему планарности к
        ///     отысканию минимума некоторого функционала на множестве базисов подпространства квазициклов. Определим следующий
        ///     функционал на матрице С, соответствующий базису подпространства квазициклов (и будем его впредь называть
        ///     функционалом Мак-Лейна)
        ///     Очевидно, что матрица С соответствует базису Мак-Лейна (т.е. базису, удовлетворяющему условию Мак-Лейна) тогда и
        ///     только тогда, когда F(С) = 0.
        /// </summary>
        public static void ExecuteMacLane()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_c, devC);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));

            gpu.Launch(gridSize, blockSize).Push(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).MultiplyBtoAbyC(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).SumAbyColumn(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).CalculateMacLane(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).SumRowAtoD(devA, devB, devC, devD);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Вычисление матрицы, получаемой в результате применения обратимого преобразования,
        ///     то есть умножение данной матрицы на обратимую матрицу, задаваемую списком индексов,
        ///     где строка i исходной матрицы должна быть добавлена к строкам i и indexes[i] результирующей матрицы, если
        ///     i!=indexes[i], и только к строке i результирующей матрицы, если i==indexes[i]
        /// </summary>
        public static void ExecuteUpdate()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_c, devC);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));

            gpu.Launch(gridSize, blockSize).Push(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).MultiplyBtoAbyC(devA, devB, devC, devD);

            gpu.CopyFromDevice(devA, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Приведение матрицы к "каноническому" виду, то есть к матрице, получаемой в результате эквивалентных преобразований
        ///     над строками, и у которой выполнено следующее - если i - индекс первого ненулевого значения в строке, то во всех
        ///     остальных строках матрицы по индексу i содержится только ноль.
        ///     Очевидно, что если индекса первого нулевого значения нет (-1), то вся строка нулевая.
        ///     Приведение матрицы к каноническому виду используется при решении систем линейных уравнений и при поиске
        ///     фундаментальной системы решений системы линейных уравнений.
        ///     В данной реализации используется матрица на полем GF(2), то есть булева матрица.
        /// </summary>
        public static void ExecuteCanonical()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));

            for (int i = 0; i < rows - 1; i++)
            {
                gpu.Launch(gridSize, blockSize).IndexOfNonZero(devA, devB, devC, devD);
                gpu.Launch(gridSize, blockSize).XorAifC(devA, devB, devC, devD);
            }
            gpu.Launch(gridSize, blockSize).IndexOfNonZero(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).XorAifCne(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).XorAifCeq(devA, devB, devC, devD);

            gpu.CopyFromDevice(devA, _a);
            gpu.CopyFromDevice(devC, _c);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Вычисление суммы всех элементов матрицы.
        ///     Сперва производится суммирование по строкам, затем суммируется полученный столбец.
        /// </summary>
        public static void ExecuteCount()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));

            gpu.Launch(gridSize, blockSize).SumToC(devA, devB, devC, devD);

            gpu.CopyFromDevice(devC, _c);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Вычисление минимальной суммы элементов в строках.
        ///     Одновременно вычисляется сумма элементов в строках.
        /// </summary>
        public static void ExecuteMinCount()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));

            gpu.Launch(gridSize, blockSize).SumToC(devA, devB, devC, devD);
            gpu.Launch(gridSize, blockSize).MinFromCtoD(devA, devB, devC, devD);

            gpu.CopyFromDevice(devC, _c);
            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Копирование регистра _a (матрица) в регистр _b (матрица)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Push(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                b[row, column] = a[row, column];
            }
        }

        /// <summary>
        ///     Копирование регистра _b (матрица) в регистр _a (матрица)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Pop(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                a[row, column] = b[row, column];
            }
        }

        /// <summary>
        ///     Прибавление к строкам регистра _a (матрица) строк регистра _b (матрица), задаваемых индексами строк
        ///     в регистре _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void MultiplyBtoAbyC(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int column = tid;
                for (int row = 0; row < rows; row++)
                {
                    if (c[row] == row) continue;
                    a[c[row], column] ^= b[row, column];
                }
            }
        }

        /// <summary>
        ///     Суммирование элементов строк регистра _a (матрица) в регистр _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void SumToC(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid;
                c[row] = 0;
                for (int column = 0; column < columns; column++)
                    c[row] += a[row, column];
            }
        }


        [Cudafy]
        public static void MinFromCtoD(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                d[0] = c[0];
                for (int row = 1; row < rows; row++)
                    if (d[0] > c[row])
                        d[0] = c[row];
            }
        }


        [Cudafy]
        public static void SumRowAtoD(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                d[0] = a[0, 0];
                for (int column = 1; column < columns; column++)
                    d[0] += a[0, column];
            }
        }

        [Cudafy]
        public static void SumAbyColumn(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int column = tid;
                for (int row = 1; row < rows; row++)
                    a[0, column] += a[row, column];
             }
        }

        [Cudafy]
        public static void CalculateMacLane(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int column = tid;
                a[0, column] = (a[0, column] - 1)*(a[0, column] - 2);
            }
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

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(rows*columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(rows*columns)));

            gpu.Launch(gridSize, blockSize, function, devA, devB, devC, devD);

            gpu.CopyFromDevice(devC, _c);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        #region Нахождение индекса первого элемента в строке

        /// <summary>
        ///     Нахождение индекса первого ненулевого элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfNonZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid;
                c[row] = -1;
                for (int column = 0; column < columns; column++)
                    if (a[row, column] != 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        /// <summary>
        ///     Нахождение индекса первого нулевого элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid;
                c[row] = -1;
                for (int column = 0; column < columns; column++)
                    if (a[row, column] == 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        /// <summary>
        ///     Нахождение индекса первого неотрицательного элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfNonNegative(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid;
                c[row] = -1;
                for (int column = 0; column < columns; column++)
                    if (a[row, column] >= 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        /// <summary>
        ///     Нахождение индекса первого неположительного элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfNonPositive(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int row = tid;
                c[row] = -1;
                for (int column = 0; column < columns; column++)
                    if (a[row, column] <= 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        #endregion

        #region Получение текущих значений в регистрах (getter)

        public static int[] GetIndexes()
        {
            return _c;
        }

        public static int[] GetCounts()
        {
            return _c;
        }

        public static int GetMacLane()
        {
            return D[0];
        }

        public static int GetMinCount()
        {
            return D[0];
        }

        #endregion

        #region

        [Cudafy]
        public static void XorAifC(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int column = tid;
                for (int i = 1; i < rows; i++)
                    for (int j = i; j-- > 0;)
                        if (c[i] == c[j])
                        {
                            a[j, column] ^= a[i, column];
                            break;
                        }
            }
        }

        [Cudafy]
        public static void XorAifCne(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int column = tid;
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < rows; j++)
                        if (i != j && c[i] != column && a[j, c[i]] != 0)
                            a[j, column] ^= a[i, column];
            }
        }

        [Cudafy]
        public static void XorAifCeq(GThread thread, int[,] a, int[,] b, int[] c, int[] d)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int column = tid;
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < rows; j++)
                        if (i != j && c[i] == column && a[j, c[i]] != 0)
                            a[j, column] ^= a[i, column];
            }
        }

        #endregion
    }
}