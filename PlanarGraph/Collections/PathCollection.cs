using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MyCudafy;
using MyCudafy.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class PathCollection : SortedStackListQueue<Path>
    {
        private static readonly PathComparer PathComparer = new PathComparer();

        public PathCollection(IEnumerable<Path> paths)
            : base(paths)
        {
            Comparer = PathComparer;
        }

        public PathCollection()
        {
            Comparer = PathComparer;
        }

        public PathCollection(Path path)
            : base(path)
        {
            Comparer = PathComparer;
        }

        public override StackListQueue<int> GetInts(Path values)
        {
            return new StackListQueue<int>(values.Select(value => value.Id));
        }

        public new IEnumerable<Path> Distinct()
        {
            if (Count == 0) return new StackListQueue<Path>();
            var list =
                new StackListQueue<StackListQueue<int>>(
                    this.Select(path => new StackListQueue<int>(path.Select(vertex => vertex.Id))));
            int[][] arr = list.Select(item => item.ToArray()).ToArray();
            Debug.Assert(Count == arr.Length);
            int[,] matrix;
            int[] indexes;
            lock (CudafySequencies.Semaphore)
            {
                CudafySequencies.SetSequencies(arr, arr);
                CudafySequencies.Execute("Compare");
                matrix = CudafySequencies.GetMatrix();
            }
            Debug.Assert(matrix.GetLength(0) == arr.Length);
            Debug.Assert(matrix.GetLength(1) == arr.Length);
            lock (CudafyMatrix.Semaphore)
            {
                CudafyMatrix.SetMatrix(matrix);
                CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                indexes = CudafyMatrix.GetIndexes();
            }
            Debug.Assert(indexes.GetLength(0) == arr.Length);
            IEnumerable<Path> paths1 = indexes.Where((value, index) => value == index)
                .Select(this.ElementAt);
            var list1 = new StackListQueue<StackListQueue<int>>(paths1.Select(path => new StackListQueue<int>(path.Select(vertex => vertex.Id))));
            var list2 = new StackListQueue<StackListQueue<int>>(paths1.Select(path => new StackListQueue<int>(path.GetReverse().Select(vertex => vertex.Id))));
            lock (CudafySequencies.Semaphore)
            {
                CudafySequencies.SetSequencies(
                    list1.Select(item => item.ToArray()).ToArray(),
                    list2.Select(item => item.ToArray()).ToArray()
                    );
                CudafySequencies.Execute("Compare");
                matrix = CudafySequencies.GetMatrix();
            }
            Debug.Assert(matrix.GetLength(0) == paths1.Count());
            Debug.Assert(matrix.GetLength(1) == paths1.Count());
            lock (CudafyMatrix.Semaphore)
            {
                CudafyMatrix.SetMatrix(matrix);
                CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                indexes = CudafyMatrix.GetIndexes();
            }
            Debug.Assert(indexes.GetLength(0) == paths1.Count());
            return new PathCollection(paths1.Where((value, index) => indexes[index] == -1 || indexes[index] >= index));
        }
    }
}