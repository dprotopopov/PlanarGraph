using System.Collections.Generic;
using System.Linq;
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
        public override IEnumerable<int> GetInts(Edge values)
        {
            return values.Select(value => value.Id).ToList();
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