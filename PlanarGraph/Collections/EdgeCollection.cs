using System.Collections.Generic;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class EdgeCollection : SortedStackListQueue<Edge>
    {
        private static readonly EdgeComparer EdgeComparer = new EdgeComparer();

        public EdgeCollection(IEnumerable<Edge> edges)
            : base(edges)
        {
            Comparer = EdgeComparer;
        }

        public EdgeCollection()
        {
            Comparer = EdgeComparer;
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}