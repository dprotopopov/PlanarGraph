using System;
using System.Collections.Generic;
using System.Linq;
using MyCudafy;
using MyCudafy.Collections;

namespace PlanarGraph.Comparer
{
    public abstract class StackListQueueComparer<T> : IComparer<StackListQueue<T>>
    {
        public int Compare(StackListQueue<T> x, StackListQueue<T> y)
        {
            int[][] list1 = x.Select(x.GetInts).Select(i => i.ToArray()).ToArray();
            int[][] list2 = y.Select(x.GetInts).Select(i => i.ToArray()).ToArray();
            int value = list1.Length - list2.Length;
            if (value != 0) return value;
            var list11 = new StackListQueue<StackListQueue<int>>(list1.Select(i => new StackListQueue<int>(i.Length)));
            var list22 = new StackListQueue<StackListQueue<int>>(list2.Select(i => new StackListQueue<int>(i.Length)));
            try
            {
                int[,] matrix;
                int first;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(list11.Select(i => i.ToArray()).ToArray(),
                        list22.Select(i => i.ToArray()).ToArray());
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.ExecuteRangeSelectFirstIndexOfNonNegative();
                    first = CudafyMatrix.GetFirst();
                }
                if (first >= 0) return matrix[first, first];
            }
            catch (Exception exception)
            {
                value = list11.Select((c, i) => c[0] - list22[i][0]).FirstOrDefault(compare => compare != 0);
                if (value != 0) return value;
            }
            try
            {
                int[,] matrix;
                int first;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(list1, list2);
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.ExecuteRangeSelectFirstIndexOfNonNegative();
                    first = CudafyMatrix.GetFirst();
                }
                return (first < 0) ? 0 : matrix[first, first];
            }
            catch (Exception exception)
            {
                return
                    list1.Select((t, i) => t.Select((item, j) => item - list2[i][j])
                        .FirstOrDefault(compare => compare != 0))
                        .FirstOrDefault(compare => compare != 0);
            }
        }
    }
}