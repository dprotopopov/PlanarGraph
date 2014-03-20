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

        public override StackListQueue<int> GetInts(Circle values)
        {
            return new StackListQueue<int>(values.Select(value => value.Id));
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}