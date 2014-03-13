using System.Collections.Generic;
using System.Linq;
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

        public override IEnumerable<int> GetInts(Vertex values)
        {
            return new List<int> {values.Id};
        }
    }
}