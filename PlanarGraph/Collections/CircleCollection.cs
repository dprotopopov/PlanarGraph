using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    internal class CircleCollection : SortedStackListQueue<Circle>
    {
        public CircleCollection()
        {
            Comparer = new CircleComparer();
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override IEnumerable<int> GetInts(Circle values)
        {
            return values.Select(value => value.Id).ToList();
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}