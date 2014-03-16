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

        public override IEnumerable<int> GetInts(Path values)
        {
            return values.Select(value => value.Id).ToList();
        }

        public new IEnumerable<Path> Distinct()
        {
            if (Count == 0) return new List<Path>();
            var list = new List<IEnumerable<int>>();
            foreach (Path path in this)
            {
                list.Add(path.Select(vertex => vertex.Id));
                list.Add(path.GetReverse().Select(vertex => vertex.Id));
            }
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
                CudafyMatrix.Execute("IndexOfZero");
                indexes = CudafyMatrix.GetIndexes();
            }
            return indexes.Where((value, index) => value == index && index%2 == 0)
                .Select(index => this[index/2]);
        }
    }
}