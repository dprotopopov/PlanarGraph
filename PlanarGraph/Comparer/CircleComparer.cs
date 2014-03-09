using System.Collections.Generic;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class CircleComparer : IComparer<Circle>
    {
        private static readonly GraphComparer GraphComparer = new GraphComparer();

        public int Compare(Circle x, Circle y)
        {
            int value = x.Count - y.Count;
            return value != 0 ? value : GraphComparer.Compare(new Graph(x), new Graph(y));
        }
    }
}