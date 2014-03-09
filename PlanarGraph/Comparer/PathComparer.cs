using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class PathComparer : IComparer<Path>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();
        private static readonly GraphComparer GraphComparer = new GraphComparer();

        public int Compare(Path x, Path y)
        {
            int value = x.Count - y.Count;
            if (value != 0) return value;
            int count = x.Count;
            for (int k = 0; k < 2; k++)
            {
                if (count == k) return 0;
                var list1 = new List<Vertex> {x[k], x[count - k - 1]};
                var list2 = new List<Vertex> {y[k], y[count - k - 1]};
                list1.Sort(VertexComparer);
                list2.Sort(VertexComparer);
                value =
                    list1.Select((t, i) => VertexComparer.Compare(t, list2[i])).FirstOrDefault(compare => compare != 0);
                if (value != 0) return value;
            }
            return x.First().Equals(x.Last())
                ? GraphComparer.Compare(new Graph(new Path(x.GetRange(1, count - 2))),
                    new Graph(new Path(y.GetRange(1, count - 2))))
                : GraphComparer.Compare(new Graph(x), new Graph(y));
        }
    }
}