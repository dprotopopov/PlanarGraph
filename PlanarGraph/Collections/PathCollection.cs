using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Comparer;
using PlanarGraph.Data;
using PlanarGraph.Parallel;

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
            var list = new StackListQueue<StackListQueue<int>>();
            list.AddRange(this.Select(path => new StackListQueue<int>(path.Select(vertex => vertex.Id))));
            list.AddRange(this.Select(path => new StackListQueue<int>(path.GetReverse().Select(vertex => vertex.Id))));
            int[,] matrix;
            int[] indexes;
            lock (CudafySequencies.Semaphore)
            {
                CudafySequencies.SetSequencies(
                    list.Select(item => item.ToArray()).ToArray(),
                    list.Select(item => item.ToArray()).ToArray()
                    );
                CudafySequencies.Execute("Compare");
                matrix = CudafySequencies.GetMatrix();
            }
            lock (CudafyMatrix.Semaphore)
            {
                CudafyMatrix.SetMatrix(matrix);
                CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                indexes = CudafyMatrix.GetIndexes();
            }
            return indexes.Where((value, index) => value == index && index < Count)
                .Select(index => this[index]);
        }
    }
}