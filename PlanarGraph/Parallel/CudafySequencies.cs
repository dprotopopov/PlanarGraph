using System;
using System.Collections.Generic;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PlanarGraph.Parallel
{
    /// <summary>
    ///     Класс множеств последовательностей
    /// </summary>
    public static class CudafySequencies
    {
        public static object Semaphore = new Object();

        private const int MaxBufferSize = 32 * 1024;
        private const int MaxIndexCount = 512;
        [Cudafy] private static int _counts1 = MaxIndexCount;
        [Cudafy] private static int _counts2 = MaxIndexCount;
        [Cudafy] private static int[] _indexes1 = new int[MaxIndexCount];
        [Cudafy] private static int[] _indexes2 = new int[MaxIndexCount];
        [Cudafy] private static int[] _sequencies1 = new int[MaxBufferSize];
        [Cudafy] private static int[] _sequencies2 = new int[MaxBufferSize];
        [Cudafy] private static int[] _matrix = new int[MaxIndexCount*MaxIndexCount];

        public static void SetSequencies(int[][] value1, int[][] value2)
        {
            _counts1 = value1.Length;
            _counts2 = value2.Length;
            var list1 = new List<int> {0};
            foreach (var value in value1) list1.Add(list1.Last() + value.Length);
            _indexes1 = list1.ToArray();
            var list2 = new List<int> {0};
            foreach (var value in value2) list2.Add(list2.Last() + value.Length);
            _indexes2 = list2.ToArray();
            _sequencies1 = value1.SelectMany(seq => seq).ToArray();
            _sequencies2 = value2.SelectMany(seq => seq).ToArray();
        }

        public static int[][] GetMatrix()
        {
            return
                Enumerable.Range(0, _counts1)
                    .Select(i => Enumerable.Range(0, _counts2)
                        .Select(j => _matrix[i*_counts2 + j])
                        .ToArray())
                    .ToArray();
        }

        public static void ExecuteIsIntersected()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);

            _matrix = new int[_counts1*_counts2];
            // copy the arrays 'a' and 'b' to the GPU
            int[] devIndexes1 = gpu.CopyToDevice(_indexes1);
            int[] devIndexes2 = gpu.CopyToDevice(_indexes2);
            int[] devSequencies1 = gpu.CopyToDevice(_sequencies1);
            int[] devSequencies2 = gpu.CopyToDevice(_sequencies2);
            int[] devMatrix = gpu.Allocate(_matrix);


            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            gpu.Launch(gridSize, blockSize).ClearMatrix(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );
            gpu.Launch(gridSize, blockSize).IsIntersected(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(devMatrix, _matrix);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        public static void ExecuteIsFromTo()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);

            _matrix = new int[_counts1*_counts2];
            // copy the arrays 'a' and 'b' to the GPU
            int[] devIndexes1 = gpu.CopyToDevice(_indexes1);
            int[] devIndexes2 = gpu.CopyToDevice(_indexes2);
            int[] devSequencies1 = gpu.CopyToDevice(_sequencies1);
            int[] devSequencies2 = gpu.CopyToDevice(_sequencies2);
            int[] devMatrix = gpu.Allocate(_matrix);


            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            gpu.Launch(gridSize, blockSize).ClearMatrix(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );
            gpu.Launch(gridSize, blockSize).IsFromTo(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(devMatrix, _matrix);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        public static void ExecuteIsContains()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);

            _matrix = new int[_counts1*_counts2];
            // copy the arrays 'a' and 'b' to the GPU
            int[] devIndexes1 = gpu.CopyToDevice(_indexes1);
            int[] devIndexes2 = gpu.CopyToDevice(_indexes2);
            int[] devSequencies1 = gpu.CopyToDevice(_sequencies1);
            int[] devSequencies2 = gpu.CopyToDevice(_sequencies2);
            int[] devMatrix = gpu.Allocate(_matrix);


            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            gpu.Launch(gridSize, blockSize).ClearMatrix(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );
            gpu.Launch(gridSize, blockSize).IsContains(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(devMatrix, _matrix);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        public static void ExecuteIndexOfFirstIntersect()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);

            _matrix = new int[_counts1*_counts2];
            // copy the arrays 'a' and 'b' to the GPU
            int[] devIndexes1 = gpu.CopyToDevice(_indexes1);
            int[] devIndexes2 = gpu.CopyToDevice(_indexes2);
            int[] devSequencies1 = gpu.CopyToDevice(_sequencies1);
            int[] devSequencies2 = gpu.CopyToDevice(_sequencies2);
            int[] devMatrix = gpu.Allocate(_matrix);


            dim3 gridSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            dim3 blockSize = Math.Min(15, (int) Math.Sqrt(Math.Sqrt(_counts1*_counts2)));
            gpu.Launch(gridSize, blockSize).SetOneMatrix(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );
            gpu.Launch(gridSize, blockSize).IndexOfFirstIntersect(
                devSequencies1, devIndexes1, _counts1,
                devSequencies2, devIndexes2, _counts2,
                devMatrix
                );

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(devMatrix, _matrix);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        [Cudafy]
        public static void ClearMatrix(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                matrix[tid] = 0;
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void SetMinusOneMatrix(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                matrix[tid] = -1;
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void SetOneMatrix(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                matrix[tid] = 1;
                tid += thread.gridDim.x;
            }
        }

        public static void IsIntersected(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                int count1 = tid/counts2;
                int count2 = tid%counts2;
                for (int i = indexes1[count1]; i < indexes1[count1 + 1] && matrix[tid] == 0; i++)
                    for (int j = indexes2[count2]; j < indexes2[count2 + 1] && matrix[tid] == 0; j++)
                        if (sequencies1[i] == sequencies2[j])
                            matrix[tid] = 1;
                tid += thread.gridDim.x;
            }
        }

        public static void IsFromTo(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                int count1 = tid/counts2;
                int count2 = tid%counts2;
                int b = 0;
                for (int i = indexes1[count1]; i < indexes1[count1] + 1 && b == 0; i++)
                    for (int j = indexes2[count2]; j < indexes2[count2 + 1] && b == 0; j++)
                        if (sequencies1[i] == sequencies2[j])
                            b = 1;
                for (int i = indexes1[count1 + 1] - 1; i < indexes1[count1 + 1] && b != 0 && matrix[tid] == 0; i++)
                    for (int j = indexes2[count2]; j < indexes2[count2 + 1] && b != 0 && matrix[tid] == 0; j++)
                        if (sequencies1[i] == sequencies2[j])
                            matrix[tid] = 1;
                tid += thread.gridDim.x;
            }
        }

        public static void IsContains(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                int count1 = tid/counts2;
                int count2 = tid%counts2;
                for (int j = indexes2[count2]; j < indexes2[count2 + 1] && matrix[tid] == 0; j++)
                {
                    int b = 0;
                    for (int i = indexes1[count1]; i < indexes1[count1 + 1] && b == 0 && matrix[tid] == 0; i++)
                        b = (sequencies1[i] == sequencies2[j]) ? 1 : 0;
                    matrix[tid] = (matrix[tid] != 0 && b != 0) ? 1 : 0;
                }
                tid += thread.gridDim.x;
            }
        }

        public static void IndexOfFirstIntersect(GThread thread,
            int[] sequencies1, int[] indexes1, int counts1,
            int[] sequencies2, int[] indexes2, int counts2,
            int[] matrix)
        {
            int tid = thread.blockIdx.x;
            while (tid < counts1*counts2)
            {
                int count1 = tid/counts2;
                int count2 = tid%counts2;
                for (int i = indexes1[count1]; i < indexes1[count1 + 1] && matrix[tid] < 0; i++)
                    for (int j = indexes2[count2]; j < indexes2[count2 + 1] && matrix[tid] < 0; j++)
                        if (sequencies1[i] == sequencies2[j])
                            matrix[tid] = i;
                tid += thread.gridDim.x;
            }
        }
    }
}