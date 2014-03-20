using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class EdgeComparer : IComparer<Edge>
    {
        private static readonly GraphComparer GraphComparer = new GraphComparer();

        public int Compare(Edge x, Edge y)
        {
            int value = x.Count - y.Count;
            if (value != 0) return value;
            int count = x.Count;
            if (count == 0) return 0;
            List<int> l1 = x.Select(v => v.Id).ToList();
            List<int> l2 = y.Select(v => v.Id).ToList();
            l1.Sort();
            l2.Sort();
            value = l1.Select((i, j) => i - l2[j]).FirstOrDefault(v => v != 0);
            return value != 0 ? value : GraphComparer.Compare(new Graph(x), new Graph(y));
        }
    }
}