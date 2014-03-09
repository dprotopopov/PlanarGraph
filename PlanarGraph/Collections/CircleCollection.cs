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

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}