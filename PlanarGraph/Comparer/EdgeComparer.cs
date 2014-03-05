using System.Collections.Generic;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class EdgeComparer : IComparer<Edge>
    {
        public int Compare(Edge x, Edge y)
        {
            return x.GetHashCode() - y.GetHashCode();
        }
    }
}