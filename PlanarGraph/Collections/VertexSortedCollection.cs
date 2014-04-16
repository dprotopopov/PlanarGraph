using System.Collections.Generic;
using System.Linq;
using MyCudafy.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class VertexSortedCollection : SortedStackListQueue<Vertex>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public VertexSortedCollection(IEnumerable<Vertex> vertices)
        {
            Comparer = VertexComparer;
            AddRange(vertices.ToList());
        }

        public VertexSortedCollection()
        {
            Comparer = VertexComparer;
        }

        public VertexSortedCollection(Vertex vertix)
            : base(vertix)
        {
            Comparer = VertexComparer;
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override StackListQueue<int> GetInts(Vertex values)
        {
            return new StackListQueue<int> {values.Id};
        }
    }
}