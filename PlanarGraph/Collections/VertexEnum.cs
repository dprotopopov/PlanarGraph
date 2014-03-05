using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class VertexEnum : Enum<Vertex>
    {
        public VertexEnum(IEnumerable<Vertex> vertices)
        {
            Comparer = new VertexComparer();
            AddRange(vertices.ToList());
        }

        public VertexEnum()
        {
            Comparer = new VertexComparer();
        }

        public VertexEnum(Vertex vertix)
        {
            Comparer = new VertexComparer();
            Add(vertix);
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