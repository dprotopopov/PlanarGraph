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
            var value = x.Count() - y.Count;
            return value != 0 ? value : GraphComparer.Compare(new Graph(x), new Graph(y));
        }
    }
}