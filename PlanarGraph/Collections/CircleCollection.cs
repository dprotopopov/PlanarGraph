using System.Linq;
using MyCudafy.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    internal class CircleCollection : MyLibrary.Collections.SortedStackListQueue<Circle>
    {
        public CircleCollection()
        {
            Comparer = new CircleComparer();
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override MyLibrary.Collections.StackListQueue<int> GetInts(Circle values)
        {
            return new MyLibrary.Collections.StackListQueue<int>(values.Select(value => value.Id));
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}