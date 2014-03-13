using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PlanarGraph.Parallel
{
    public struct CudafyMatrix
    {
        public static readonly object Semaphore = new Object();

        [Cudafy] private static int _rows;
        [Cudafy] private static int _columns;
        [Cudafy] private static int[] _a;
        [Cudafy] private static int[] _b;
        [Cudafy] private static int[] _c;
        [Cudafy] private static readonly int[] D = new int[1];

        public static void SetMatrix(int[][] value)
        {
            _rows = value.Count();
            _columns = value[0].Count();
            _a = value.SelectMany(v => v).ToArray();
            _b = new int[_rows*_columns];
            _c = new int[_rows];
        }

        public static int[][] GetMatrix()
        {
            return Enumerable.Range(0, _rows)
                .Select(i => Enumerable.Range(0, _columns)
                    .Select(j => _a[i*_columns + j])
                    .ToArray())
                .ToArray();
        }

        public static void SetIndexes(int[] value)
        {
            _c = value;
        }

        public static void ExecuteHash()
        {
            //Debug.WriteLine("Begin {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            gpu.Launch(gridSize, blockSize).HashToC(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).HashFromCtoD(devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));
            //Debug.WriteLine(D[0]);
            //Debug.WriteLine("End {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
        }

        /// <summary>
        ///     Вычисление целевой функции для симплекс-метода
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
            //Debug.WriteLine("Begin {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_c, devC);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            gpu.Launch(gridSize, blockSize).Push(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).MultiplyBtoAbyC(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).SumAbyColumn(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).CalculateMacLane(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).SumRowAtoD(devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(D[0]);
            //Debug.WriteLine("End {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
        }

        public static void ExecuteUpdate()
        {
            //Debug.WriteLine("Begin {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);
            gpu.CopyToDevice(_c, devC);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            gpu.Launch(gridSize, blockSize).Push(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).MultiplyBtoAbyC(devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devA, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));
            //Debug.WriteLine("End {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
        }

        /// <summary>
        ///     Приведение матрицы к каноническому виду
        /// </summary>
        public static void ExecuteCanonical()
        {
            //Debug.WriteLine("Begin {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            for (int i = 0; i < _rows - 1; i++)
            {
                gpu.Launch(gridSize, blockSize).IndexOfNonZero(devA, devB, devC, devD, _rows, _columns);
                gpu.Launch(gridSize, blockSize).XorAifC(devA, devB, devC, devD, _rows, _columns);
            }
            gpu.Launch(gridSize, blockSize).IndexOfNonZero(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).XorAifCne(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).XorAifCeq(devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devA, _a);
            gpu.CopyFromDevice(devC, _c);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));
            //Debug.WriteLine("End {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
        }

        public static void ExecuteCount()
        {
            //Debug.WriteLine("Begin {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            gpu.Launch(gridSize, blockSize).SumToC(devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devC, _c);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));
            //Debug.WriteLine("End {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
        }

        public static void ExecuteMinCount()
        {
            //Debug.WriteLine("Begin {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            gpu.Launch(gridSize, blockSize).SumToC(devA, devB, devC, devD, _rows, _columns);
            gpu.Launch(gridSize, blockSize).MinFromCtoD(devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devC, _c);
            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));
            //Debug.WriteLine("End {0}::{1}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name);
        }

        [Cudafy]
        public static void Push(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m*n)
            {
                b[tid] = a[tid];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void Pop(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m*n)
            {
                a[tid] = b[tid];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void MultiplyBtoAbyC(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 0; i < m; i++)
                {
                    if (c[i] == i) continue;
                    a[c[i]*n + tid] ^= b[i*n + tid];
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void SumToC(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                c[tid] = 0;
                for (int i = 0; i < n; i++)
                    c[tid] += a[tid*n + i];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void HashToC(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                c[tid] = 0;
                for (int i = 0; i < n; i++)
                {
                    c[tid] = (c[tid] << 1) ^ (c[tid] >> (8*sizeof (int) - 1)) ^ a[tid*n + i] ^ i;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void HashFromCtoD(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < 1)
            {
                d[0] = 0;
                for (int i = 0; i < m; i++)
                    d[0] = (d[0] << 1) ^ (d[0] >> (8*sizeof (int) - 1)) ^ c[i] ^ i;
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void MinFromCtoD(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < 1)
            {
                d[0] = c[0];
                for (int i = 1; i < m; i++)
                    if (d[0] > c[i])
                        d[0] = c[i];
                tid += thread.gridDim.x;
            }
        }


        [Cudafy]
        public static void SumRowAtoD(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < 1)
            {
                d[0] = 0;
                for (int i = 0; i < n; i++)
                    d[0] += a[tid*n + i];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void SumAbyColumn(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 1; i < m; i ++)
                    a[0*n + tid] += a[i*n + tid];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void CalculateMacLane(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                a[tid] = a[tid]*a[tid] - 3*a[tid] + 2;
                tid += thread.gridDim.x;
            }
        }

        #region 

        [Cudafy]
        public static void XorAifC(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 1; i < m; i++)
                    for (int j = i; j-- > 0;)
                        if (c[i] == c[j])
                        {
                            a[j*n + tid] ^= a[i*n + tid];
                            break;
                        }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void XorAifCne(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < m; j++)
                        if (i != j && c[i] != tid && a[j*n + c[i]] != 0)
                            a[j*n + tid] ^= a[i*n + tid];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void XorAifCeq(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < m; j++)
                        if (i != j && c[i] == tid && a[j*n + c[i]] != 0)
                            a[j*n + tid] ^= a[i*n + tid];
                tid += thread.gridDim.x;
            }
        }


        #endregion

        public static void Execute(string function)
        {
            //Debug.WriteLine("Begin {0}::{1}::{2}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name,
            //    function);
            //Debug.WriteLine(string.Join(Environment.NewLine,
            //    Enumerable.Range(0, _rows)
            //        .Select(row => string.Join("", Enumerable.Range(0, _columns)
            //            .Select(column => _a[row*_columns + column].ToString()).ToList())).ToList()));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_rows*_columns)));

            gpu.Launch(gridSize, blockSize, function, devA, devB, devC, devD, _rows, _columns);

            gpu.CopyFromDevice(devC, _c);

            // free the memory allocated on the GPU
            gpu.FreeAll();

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
            //Debug.WriteLine(string.Join(",", Enumerable.Range(0, _rows).Select(i => _c[i].ToString()).ToList()));
            //Debug.WriteLine("End {0}::{1}::{2}", typeof (CudafyMatrix).Name, MethodBase.GetCurrentMethod().Name,
            //    function);
        }

        #region

        [Cudafy]
        public static void IndexOfNonZero(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                c[tid] = -1;
                for (int i = 0; i < n; i++)
                    if (a[tid*n + i] != 0)
                    {
                        c[tid] = i;
                        break;
                    }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void IndexOfZero(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                c[tid] = -1;
                for (int i = 0; i < n; i++)
                    if (a[tid*n + i] == 0)
                    {
                        c[tid] = i;
                        break;
                    }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void IndexOfNonNegative(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                c[tid] = -1;
                for (int i = 0; i < n; i++)
                    if (a[tid*n + i] >= 0)
                    {
                        c[tid] = i;
                        break;
                    }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void IndexOfNonPositive(GThread thread, int[] a, int[] b, int[] c, int[] d, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                c[tid] = -1;
                for (int i = 0; i < n; i++)
                    if (a[tid*n + i] <= 0)
                    {
                        c[tid] = i;
                        break;
                    }
                tid += thread.gridDim.x;
            }
        }

        #endregion

        #region

        public static int[] GetIndexes()
        {
            return _c;
        }

        public static int[] GetCounts()
        {
            return _c;
        }

        #endregion

        #region

        public static int GetMacLane()
        {
            return D[0];
        }

        public static int GetMinCount()
        {
            return D[0];
        }

        public static int GetHash()
        {
            return D[0];
        }

        #endregion
    }
}