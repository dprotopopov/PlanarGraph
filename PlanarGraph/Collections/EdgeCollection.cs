using System.Collections.Generic;
using System.Linq;
using MyCudafy.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class EdgeCollection : MyLibrary.Collections.SortedStackListQueue<Edge>
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

        public override MyLibrary.Collections.StackListQueue<int> GetInts(Edge values)
        {
            return new MyLibrary.Collections.StackListQueue<int>(values.Select(value => value.Id));
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