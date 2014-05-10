using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class CircleComparer : IComparer<Circle>, IEqualityComparer<Circle>
    {
        private static readonly GraphComparer GraphComparer = new GraphComparer();

        public int Compare(Circle x, Circle y)
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

        public bool Equals(Circle x, Circle y)
        {
            return Compare(x, y) == 0;
        }

        public int GetHashCode(Circle obj)
        {
            return GraphComparer.GetHashCode(new Graph(obj));
        }
    }
}