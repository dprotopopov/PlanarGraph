using System;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PlanarGraph.Parallel
{
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
    [Cudafy]
    public struct CudifyBooleanMatrixMacLane
    {
        private const int Msize = 64;
        private const int Nsize = 512;

        [Cudafy] private static int m;
        [Cudafy] private static int n;
        [Cudafy] private static int[] a = new int[Nsize*Msize];
        [Cudafy] private static int[] b = new int[Msize];

        public static void SetBooleanMatrix(int[] value, int rows, int columns)
        {
            m = rows;
            n = columns;
            a = value;
        }

        public static void SetIndexes(int[] value)
        {
            b = value.ToArray();
        }

        public static int GetMacLane()
        {
            return b[0];
        }


        public static void Execute()
        {

            CudafyModes.Target = eGPUType.Cuda; // To use OpenCL, change this enum
            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;
            CudafyModule km = CudafyTranslator.Cudafy();
            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);

            int[] dev_a = gpu.Allocate<int>(m*n);
            int[] dev_b = gpu.Allocate<int>(m);

            gpu.CopyToDevice(a, dev_a);
            gpu.CopyToDevice(b, dev_b);

            dim3 gridSize = (int) Math.Sqrt(Math.Sqrt(n)) + 1;
            dim3 blockSize = (int) Math.Sqrt(Math.Sqrt(n)) + 1;

            gpu.Launch(gridSize, blockSize, "MultiplyToMatrix", dev_a, dev_b, m, n);
            gpu.Launch(gridSize, blockSize, "SumByColumn", dev_a, dev_b, m, n);
            gpu.Launch(gridSize, blockSize, "CalculateMacLane", dev_a, dev_b, m, n);
            gpu.Launch(gridSize, blockSize, "SumByRow", dev_a, dev_b, m, n);
            gpu.Launch(gridSize, blockSize, "CopyToResult", dev_a, dev_b, m, n);

            gpu.CopyFromDevice(dev_b, b);

            // free the memory allocated on the GPU
            gpu.Free(dev_a);
            gpu.Free(dev_b);

            // free the memory we allocated on the CPU
            // Not necessary, this is .NET
        }

        [Cudafy]
        public static void MultiplyToMatrix(GThread thread, int[] a, int[] b, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 0; i < m; i++)
                {
                    if (b[i] == i) continue;
                    a[b[i]*m + tid] ^= a[i*m + tid];
                    for (int j = 0; j < i; j++) if (b[j] == i) a[a[i]*n + tid] ^= a[b[j]*m + tid];
                }
                tid += thread.gridDim.x;
            }
        }


        [Cudafy]
        public static void CopyToResult(GThread thread, int[] a, int[] b, int m, int n)
        {
            int tid = thread.blockIdx.x;
            if (tid == 0) b[0] = a[0];
        }

        [Cudafy]
        public static void SumByRow(GThread thread, int[] a, int[] b, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < m)
            {
                for (int i = 0; i < n; i++)
                    a[tid*n + 0] += a[tid*n + i];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void SumByColumn(GThread thread, int[] a, int[] b, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                for (int i = 0; i < m; i ++)
                    a[0*n + tid] += a[i*n + tid];
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void CalculateMacLane(GThread thread, int[] a, int[] b, int m, int n)
        {
            int tid = thread.blockIdx.x;
            while (tid < n)
            {
                a[tid] = a[tid]*a[tid] - 3*a[tid] + 2;
                tid += thread.gridDim.x;
            }
        }
    }
}